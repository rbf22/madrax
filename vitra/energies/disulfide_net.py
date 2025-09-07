#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from dataclasses import dataclass
from vitra.sources import hashings
from vitra.sources.globalVariables import TEMPERATURE


@dataclass
class DisulfideData:
    """Data class for disulfide energy calculation."""
    coords: torch.Tensor
    atom_description: torch.Tensor
    atom_pairs: torch.Tensor
    partners: torch.Tensor
    alternative_mask: torch.Tensor
    facc: torch.Tensor


class DisulfideEnergy(torch.nn.Module):
    """
    This module calculates the energy of disulfide bonds (Cys-Cys bridges).
    """

    def __init__(self, name='disulfide_net', dev='cpu'):
        """
        Initializes the DisulfideEnergy module.
        """
        super().__init__()
        self.name = name
        self.dev = dev
        self.float_type = torch.float
        self.weight = torch.nn.Parameter(torch.tensor([0.0], device=self.dev))

    def _calculate_disulfide_energy(self, dist, residue_distance):
        """Calculates the energy of a single disulfide bond."""
        distance_correction = 5 * torch.abs(dist - 2.04)
        energy = -0.001 * TEMPERATURE * (2.1 + 2.9823825 *
                                         torch.log(torch.abs(residue_distance))) + distance_correction
        return energy

    def net(self, data: DisulfideData):
        """
        Calculates the disulfide bond energy for all potential Cys-Cys pairs.
        """
        at_name1 = data.atom_description[(data.atom_pairs[:, 0],
                                          hashings.atom_description_hash['at_name'])]
        at_name2 = data.atom_description[(data.atom_pairs[:, 1],
                                          hashings.atom_description_hash['at_name'])]
        sulfur_mask = at_name1.eq(hashings.atom_hash['CYS']['SG']) & \
                      at_name2.eq(hashings.atom_hash['CYS']['SG'])

        if not sulfur_mask.any():
            return torch.empty(0, device=self.dev), \
                   torch.empty(0, 2, dtype=torch.long, device=self.dev), \
                   torch.zeros(data.atom_pairs.shape[0], device=self.dev, dtype=torch.bool)

        distmat = torch.pairwise_distance(data.coords[data.atom_pairs[:, 0]],
                                          data.coords[data.atom_pairs[:, 1]])
        dist = distmat[sulfur_mask]

        atom_pairs_sulfur = data.atom_pairs[sulfur_mask]
        residue_distance = torch.abs(
            data.atom_description[(atom_pairs_sulfur[:, 0],
                                   hashings.atom_description_hash['resnum'])] -
            data.atom_description[(atom_pairs_sulfur[:, 1],
                                   hashings.atom_description_hash['resnum'])]).float()

        energy = self._calculate_disulfide_energy(dist, residue_distance)

        return energy, atom_pairs_sulfur, sulfur_mask

    def forward(self, data: DisulfideData):
        """
        Forward pass for the DisulfideEnergy module.
        """
        disulfide_energy, disulfide_atom_pairs, disulf_network = self.net(data)
        atom_energy = self.bind_to_atoms(
            disulfide_energy, disulfide_atom_pairs, data.atom_description,
            data.facc, data.alternative_mask)
        residue_energy = self.bind_to_resi(atom_energy, data.atom_description)
        return residue_energy, atom_energy, disulf_network

    def bind_to_atoms(self, disulfide_energy, disulfide_atom_pairs,
                      atom_description, _facc, altern_mask, _min_sa_coefficient=1.0):
        """Binds the disulfide energy to atoms."""
        if len(disulfide_energy) == 0:
            return torch.zeros((atom_description.shape[0], altern_mask.shape[-1]),
                               dtype=self.float_type, device=self.dev)

        net_energy = disulfide_energy * 0.5
        energy_atom_atom = torch.zeros((altern_mask.shape[0], altern_mask.shape[-1]),
                                       dtype=torch.float, device=self.dev)
        for alt in range(altern_mask.shape[-1]):
            mask = altern_mask[(disulfide_atom_pairs[:, 0], alt)] & \
                   altern_mask[(disulfide_atom_pairs[:, 1], alt)]
            alt_index = torch.full(mask.shape, alt, device=self.dev, dtype=torch.long)[mask]
            atom_pair_alter = disulfide_atom_pairs[mask]
            energy_atom_atom.index_put_((atom_pair_alter[:, 0], alt_index),
                                       (net_energy[mask]), accumulate=True)
            energy_atom_atom.index_put_((atom_pair_alter[:, 1], alt_index),
                                       (net_energy[mask]), accumulate=True)

        return energy_atom_atom

    def bind_to_resi(self, atom_energy, atom_description):
        """Binds the disulfide energy to residues."""
        if atom_energy is None:
            return None
        naltern = atom_energy.shape[-1]
        batch_ind = atom_description[:, hashings.atom_description_hash['batch']].long()
        resnum = atom_description[:, hashings.atom_description_hash['resnum']].long()
        chain_ind = atom_description[:, hashings.atom_description_hash['chain']].long()

        batch = batch_ind.max() + 1 if len(batch_ind) > 0 else 1
        nres = torch.max(resnum) + 1 if len(resnum) > 0 else 1
        nchains = chain_ind.max() + 1 if len(chain_ind) > 0 else 1

        resi_energy = torch.zeros((batch, nchains, nres, naltern),
                                  dtype=self.float_type, device=self.dev)

        for alt in range(naltern):
            alt_mask = torch.ones_like(atom_energy[:, alt], dtype=torch.bool)
            batch_idx = batch_ind[alt_mask]
            chain_idx = chain_ind[alt_mask]
            res_idx = resnum[alt_mask]

            indices = (batch_idx, chain_idx, res_idx, torch.full_like(batch_idx, alt))
            resi_energy.index_put_(indices, atom_energy[alt_mask, alt], accumulate=True)

        return resi_energy

    def get_weights(self):
        """Returns the weights of the module."""

    def get_num_params(self):
        """Returns the number of parameters in the module."""
        p = []
        for i in self.parameters():
            p += list(i.data.cpu().numpy().flat)
        print(f'Number of parameters= {len(p)}')
