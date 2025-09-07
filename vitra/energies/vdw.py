#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
#  Vdw.py
#
#  Copyright 2019 Gabriele Orlando <orlando.gabriele89@gmail.com>
#
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 2 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software
#  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
#  MA 02110-1301, USA.
#
#

import torch

from vitra.sources import hashings
from vitra.sources.globalVariables import PADDING_INDEX

class Vdw(torch.nn.Module):
    r"""
    This module calculates an energy term that appears to be related to Van der Waals interactions,
    but it does not implement the pairwise Lennard-Jones potential described in the README.md.

    Instead, it applies a per-atom energy contribution that is looked up from a table
    (hashings.atom_Properties) and scaled by the solvent-accessible surface area (SASA) of the atom.
    This suggests it might be a component of a solvation energy model or a simplified VdW term,
    rather than a direct implementation of the pairwise Lennard-Jones formula.

    The README.md describes the VdW energy as:
    \[
    \Delta G_{vdw}^{ij} = \epsilon_{ij}\Big[\Big(\frac{R_{min,ij}}{d_{ij}}\Big)^{12} - 2\Big(\frac{R_{min,ij}}{d_{ij}}\Big)^{6}\Big]
    \]
    The code in this file does not perform a pairwise distance calculation.
    """

    def __init__(self, name='Vdw', dev='cpu', backbone_atoms=[]):
        """
        Initializes the Vdw module.

        Args:
            name (str): Name of the module.
            dev (str): Device to run the calculations on ('cpu' or 'cuda').
            backbone_atoms (list): A list of backbone atom names.
        """
        self.name = name
        self.backbone_atoms = backbone_atoms
        self.dev = dev
        super(Vdw, self).__init__()
        self.weight = torch.nn.Parameter(torch.tensor([0.0], device=self.dev))
        self.weight.requires_grad = True

    def bindToAtoms(self, atom_description, alternMask, facc, minSaCoefficient=0.0):
        """
        Calculates the per-atom VdW-like energy.

        The energy is calculated by looking up a pre-calculated value for each atom type and
        scaling it by the solvent-accessible surface area (SASA).

        Args:
            atom_description (torch.Tensor): Tensor containing atom properties.
            alternMask (torch.Tensor): Mask for alternative conformations.
            facc (torch.Tensor): Solvent-accessible surface area for each atom.
            minSaCoefficient (float): Minimum value for the SASA coefficient.

        Returns:
            torch.Tensor: Per-atom energy values.
        """
        saCoefficient = torch.max(facc, torch.tensor(minSaCoefficient,
                                                     device=self.dev, dtype=torch.float))
        vdw_tab_value = hashings.atom_Properties[(atom_description[:, hashings.atom_description_hash['at_name']].long(),
                                                  hashings.property_hashings['solvenergy_props']['VdW'])]
        atomEnergy = torch.zeros((atom_description.shape[0]), (alternMask.shape[-1]),
                                 dtype=torch.float, device=self.dev)

        for alt in range(alternMask.shape[1]):
            mask = alternMask[:, alt]
            atomEnergy[(mask, alt)] = vdw_tab_value[mask] * saCoefficient[(mask, alt)]

        return atomEnergy * (1 - torch.tanh(self.weight)) * 0.3

    def bindToResi(self, atomEnergy, atom_description, is_backbone_mask):
        """
        Aggregates per-atom energies to per-residue energies for both main-chain (MC) and side-chain (SC).

        Args:
            atomEnergy (torch.Tensor): Per-atom energy values.
            atom_description (torch.Tensor): Tensor containing atom properties.
            is_backbone_mask (torch.Tensor): Mask to identify backbone atoms.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple containing the aggregated main-chain and side-chain energies.
        """
        batch = torch.max(atom_description[:, hashings.atom_description_hash['batch']]) + 1
        nres = torch.max(atom_description[:, hashings.atom_description_hash['resnum']]) + 1
        naltern = atomEnergy.shape[-1]
        nchains = torch.max(atom_description[:, hashings.atom_description_hash['chain']]) + 1
        resIndex = atom_description[:, hashings.atom_description_hash['resnum']].long().unsqueeze(-1).expand(
            atom_description.shape[0], naltern)
        chaIndex = atom_description[:, hashings.atom_description_hash['chain']].long().unsqueeze(-1).expand(
            atom_description.shape[0], naltern)
        alt_index = torch.arange(0, naltern, dtype=torch.long, device=self.dev).unsqueeze(0).expand(
            atom_description.shape[0], naltern)
        batchIndex = atom_description[:, hashings.atom_description_hash['batch']].long().unsqueeze(-1).expand(
            atom_description.shape[0], naltern)
        mask_padding = ~resIndex.eq(PADDING_INDEX)
        is_backbone_mask = is_backbone_mask.unsqueeze(-1).expand(-1, naltern)
        finalMC = torch.zeros((batch, nchains, nres, naltern), dtype=torch.float, device=self.dev)
        finalSC = torch.zeros((batch, nchains, nres, naltern), dtype=torch.float, device=self.dev)
        mask = is_backbone_mask & mask_padding
        indices = (batchIndex[mask], chaIndex.long()[mask], resIndex.long()[mask], alt_index[mask])
        finalMC.index_put_(indices, (atomEnergy[mask]), accumulate=True)
        mask = mask_padding & ~is_backbone_mask
        indices = (batchIndex[mask], chaIndex[mask], resIndex[mask].long(), alt_index[mask])
        finalSC.index_put_(indices, (atomEnergy[mask]), accumulate=True)
        return finalMC, finalSC

    def forward(self, coords, atom_description, alternativeMask, facc):
        """
        Forward pass for the Vdw module.

        Args:
            coords (torch.Tensor): Atomic coordinates.
            atom_description (torch.Tensor): Tensor containing atom properties.
            alternativeMask (torch.Tensor): Mask for alternative conformations.
            facc (torch.Tensor): Solvent-accessible surface area for each atom.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple containing the main-chain and side-chain VdW-like energies.
        """
        for i, bb_atom in enumerate(self.backbone_atoms):
            if i == 0:
                is_backbone_mask = atom_description[:, hashings.atom_description_hash['at_name']].eq(bb_atom)
            else:
                is_backbone_mask += atom_description[:, hashings.atom_description_hash['at_name']].eq(bb_atom)

        atomEnergy = self.bindToAtoms(atom_description, alternativeMask, facc)
        residueEnergyMC, residueEnergySC = self.bindToResi(atomEnergy, atom_description, is_backbone_mask)
        return residueEnergyMC.unsqueeze(-1), residueEnergySC.unsqueeze(-1)

    def getWeights(self):
        pass

    def getNumParams(self):
        p = []
        for i in self.parameters():
            p += list(i.data.cpu().numpy().flat)

        print('Number of parameters=', len(p))
