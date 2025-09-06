#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
#  Electrostatics.py
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
from vitra.sources.globalVariables import *

class Vdw(torch.nn.Module):

    def __init__(self, name='Vdw', dev='cpu', backbone_atoms=[]):
        self.name = name
        self.backbone_atoms = backbone_atoms
        self.dev = dev
        super(Vdw, self).__init__()
        self.weight = torch.nn.Parameter(torch.tensor([0.0], device=self.dev))
        self.weight.requires_grad = True

    def bindToAtoms(self, atom_description, alternMask, facc, minSaCoefficient=0.0):
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
        for i, bb_atom in enumerate(self.backbone_atoms):
            if i == 0:
                is_backbone_mask = atom_description[:, hashings.atom_description_hash['at_name']].eq(bb_atom)
            else:
                is_backbone_mask += atom_description[:, hashings.atom_description_hash['at_name']].eq(bb_atom)

        atomEnergy = self.bindToAtoms(atom_description, alternativeMask, facc)
        residueEnergyMC, residueEnergySC = self.bindToResi(atomEnergy, atom_description, is_backbone_mask)
        return residueEnergyMC, residueEnergySC

    def getWeights(self):
        pass

    def getNumParams(self):
        p = []
        for i in self.parameters():
            p += list(i.data.cpu().numpy().flat)

        print('Number of parameters=', len(p))
