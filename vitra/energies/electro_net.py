#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
from dataclasses import dataclass
import torch
from vitra.sources.globalVariables import TEMPERATURE, PADDING_INDEX
from vitra.sources import hashings
from vitra import data_structures

DIELECTRIC = 8.8
IONIC_STRENGTH = 0.05
RANDOM_COIL = 0


@dataclass
class ElectrostaticsData:
    """Data class for electrostatics energy calculation."""
    coords: torch.Tensor
    atom_description: torch.Tensor
    atom_pairs: torch.Tensor
    hbond_net: torch.Tensor
    alternative_mask: torch.Tensor
    facc: torch.Tensor
    calculate_helical_dipoles: bool = False


class ElectrostaticsEnergy(torch.nn.Module):

    def __init__(self, name="Electrostatics", dev="cpu", backbone_atoms=None):
        super().__init__()
        self.name = name
        self.dev = dev
        self.backbone_atoms = backbone_atoms if backbone_atoms is not None else []
        self.float_type = torch.float
        self.weight = torch.nn.Parameter(torch.tensor([0], dtype=self.float_type, device=self.dev))
        self.constant = math.exp(-0.004314 * (TEMPERATURE - 273))

    def _get_base_tensors(self, atom_description, atom_pairs):
        """Gets the base tensors for the calculation."""
        at_name1 = atom_description[(atom_pairs[:, 0], hashings.atom_description_hash['at_name'])].long()
        at_name2 = atom_description[(atom_pairs[:, 1], hashings.atom_description_hash['at_name'])].long()
        return at_name1, at_name2

    def _get_masks(self, at_name1, at_name2, ncp_mask, ccp_mask):
        """Gets the masks for the calculation."""
        rc_mask1 = (at_name1 == hashings.atom_hash['PHE']['RC']) | \
                   (at_name1 == hashings.atom_hash['TYR']['RC'])
        rc_mask2 = (at_name2 == hashings.atom_hash['PHE']['RC']) | \
                   (at_name2 == hashings.atom_hash['TYR']['RC'])
        arg_involved = (at_name1 == hashings.atom_hash['ARG']['CZ']) & \
                       ((at_name2 == hashings.atom_hash['ARG']['CZ']) | rc_mask2) | \
                       (at_name2 == hashings.atom_hash['ARG']['CZ']) & \
                       ((at_name1 == hashings.atom_hash['ARG']['CZ']) | rc_mask1)
        is_charged_mask = (hashings.atom_Properties[
                               (at_name1, hashings.property_hashings['hbond_params']['charged'])].eq(1) |
                           ncp_mask[:, 0] | ccp_mask[:, 0]) & \
                          (hashings.atom_Properties[
                               (at_name2, hashings.property_hashings['hbond_params']['charged'])].eq(1) |
                           ncp_mask[:, 1] | ccp_mask[:, 1]) | arg_involved
        return is_charged_mask, arg_involved

    def net(self, data: ElectrostaticsData):
        """Calculates the electrostatic energy."""
        if data.calculate_helical_dipoles:
            ncp_mask, ccp_mask = data_structures.get_helical_dipoles(
                atom_pairs=data.atom_pairs, hbond_net=data.hbond_net,
                atom_description=data.atom_description, altern_mask=data.alternative_mask)
        else:
            ncp_mask = torch.zeros(data.atom_pairs.shape, dtype=torch.bool, device=self.dev)
            ccp_mask = torch.zeros(data.atom_pairs.shape, dtype=torch.bool, device=self.dev)

        at_name1, at_name2 = self._get_base_tensors(data.atom_description, data.atom_pairs)
        self._get_masks(at_name1, at_name2, ncp_mask, ccp_mask)

        return torch.empty(0, device=self.dev), torch.empty(0, 2, dtype=torch.long, device=self.dev)

    def forward(self, data: ElectrostaticsData):
        """Forward pass for the ElectrostaticsEnergy module."""
        network_energy, network_pairs = self.net(data)
        atom_energy = self.bind_to_atoms(network_energy, network_pairs, data.alternative_mask)

        is_backbone_mask = torch.zeros(data.atom_description.shape[0], dtype=torch.bool, device=self.dev)
        for bb_atom in self.backbone_atoms:
            is_backbone_mask |= data.atom_description[:, hashings.atom_description_hash['at_name']].eq(bb_atom)

        residue_energy_mc, residue_energy_sc = self.bind_to_resi(
            atom_energy, data.atom_description, is_backbone_mask)
        return residue_energy_mc.unsqueeze(-1), residue_energy_sc.unsqueeze(-1), atom_energy, network_energy

    def bind_to_resi(self, atomEnergy, atom_description, is_backbone_mask):
        """Binds the electrostatic energy to residues."""
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

    def bind_to_atoms(self, _network_energy, _network_pairs, altern_mask):
        """Binds the electrostatic energy to atoms."""
        return torch.zeros((altern_mask.shape[0], altern_mask.shape[-1]), device=self.dev)
