#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import torch

from vitra.sources import hashings, math_utils
from vitra.sources.globalVariables import PADDING_INDEX, EPS


class BondLengthConstraintEnergy(torch.nn.Module):
    r"""
    This module calculates a penalty for violations of ideal peptide bond geometry.
    """

    def __init__(self, name="BondLengthConstraint", dev="cpu"):
        super().__init__()
        self.name = name
        self.dev = dev
        self.float_type = torch.float
        self.weight = torch.nn.Parameter(torch.tensor([0.0], device=self.dev))
        self.mean = torch.tensor([[1.33, 121.7, 116.2]] * 20, device=self.dev)
        self.std = torch.tensor([[0.02, 3.0, 3.0]] * 20, device=self.dev)

    def _score_distribution(self, inputs, seq):
        """
        Calculates the negative log-likelihood of a given geometry.
        """
        scores = []
        for i in range(self.std.shape[1]):
            var = self.std[seq, i] ** 2
            denom = (2 * math.pi * var) ** 0.5
            num = torch.exp(-(inputs[:, i] - self.mean[seq, i]) ** 2 / (2 * var))
            norm_factor = 1.0 / denom
            log_prob = -((num / denom).clamp(min=EPS).log() - torch.log(norm_factor))
            scores.append(log_prob.unsqueeze(-1))
        return torch.cat(scores, dim=-1)

    def _find_peptide_bonds(self, atom_description):
        """Finds peptide bonds between C and N atoms."""
        at_name = atom_description[:, hashings.atom_description_hash['at_name']]
        resnum = atom_description[:, hashings.atom_description_hash['resnum']].long()
        chain_ind = atom_description[:, hashings.atom_description_hash['chain']].long()

        c_atoms = (at_name == hashings.atom_hash['ALA']['C'])
        n_atoms = (at_name == hashings.atom_hash['ALA']['N'])

        c_indices = torch.where(c_atoms)[0]
        n_indices = torch.where(n_atoms)[0]

        peptide_bonds = []
        for c_idx in c_indices:
            for n_idx in n_indices:
                if resnum[c_idx] + 1 == resnum[n_idx] and chain_ind[c_idx] == chain_ind[n_idx]:
                    peptide_bonds.append((c_idx, n_idx))
        return torch.tensor(peptide_bonds, device=self.dev) if peptide_bonds else None

    def _get_ca_indices(self, peptide_bonds, atom_description):
        """Gets the indices of the C-alpha atoms for the peptide bonds."""
        if peptide_bonds is None:
            return None, None

        resnum = atom_description[:, hashings.atom_description_hash['resnum']].long()
        chain_ind = atom_description[:, hashings.atom_description_hash['chain']].long()
        at_name = atom_description[:, hashings.atom_description_hash['at_name']]
        ca_atoms = at_name == hashings.atom_hash['ALA']['CA']

        try:
            c_resnum = resnum[peptide_bonds[:, 0]]
            n_resnum = resnum[peptide_bonds[:, 1]]
            c_chain = chain_ind[peptide_bonds[:, 0]]
            n_chain = chain_ind[peptide_bonds[:, 1]]
        except (TypeError, IndexError):
            return None, None

        ca_indices_c_res, ca_indices_n_res = [], []
        for i in range(len(peptide_bonds)):
            ca_c = torch.where((resnum == c_resnum[i]) & (chain_ind == c_chain[i]) & ca_atoms)[0]
            ca_indices_c_res.append(ca_c[0] if len(ca_c) > 0 else PADDING_INDEX)
            ca_n = torch.where((resnum == n_resnum[i]) & (chain_ind == n_chain[i]) & ca_atoms)[0]
            ca_indices_n_res.append(ca_n[0] if len(ca_n) > 0 else PADDING_INDEX)

        return torch.tensor(ca_indices_c_res, device=self.dev), \
               torch.tensor(ca_indices_n_res, device=self.dev)

    def _calculate_geometry_and_scores(self, scored_bonds, ca_indices_c, ca_indices_n, coords, resname):
        """Calculates bond lengths, angles, and scores."""
        if scored_bonds is None or ca_indices_c is None or ca_indices_n is None:
            return None, None

        try:
            c_coords = coords[scored_bonds[:, 0]]
            n_coords = coords[scored_bonds[:, 1]]
            ca_c_coords = coords[ca_indices_c]
            ca_n_coords = coords[ca_indices_n]
        except (TypeError, IndexError):
            return None, None

        v_cn = n_coords - c_coords
        v_nca_n = ca_n_coords - n_coords
        v_cac_c = c_coords - ca_c_coords

        c_n_ca_angle, mask1, _ = math_utils.angle2dVectors(v_cn, v_nca_n)
        ca_c_n_angle, mask2, _ = math_utils.angle2dVectors(v_cac_c, -v_cn)
        valid_angles = mask1.squeeze(-1) & mask2.squeeze(-1)

        peptide_bond_length = torch.norm(v_cn, dim=1)
        distro_input = torch.cat([peptide_bond_length[valid_angles].unsqueeze(-1),
                                  c_n_ca_angle[valid_angles],
                                  ca_c_n_angle[valid_angles]], dim=1)

        if distro_input.shape[0] > 0:
            seq = resname[scored_bonds[valid_angles][:, 0]].long()
            scores = self._score_distribution(distro_input, seq).sum(-1)
            return scores, scored_bonds[valid_angles]
        return None, None

    def forward(self, atom_description, coords, alternatives):
        """
        Forward pass for the BondLengthConstraintEnergy module.
        """
        batch_ind = atom_description[:, hashings.atom_description_hash['batch']].long()
        resnum = atom_description[:, hashings.atom_description_hash['resnum']].long()
        chain_ind = atom_description[:, hashings.atom_description_hash['chain']].long()
        resname = atom_description[:, hashings.atom_description_hash['resname']]

        batch_size = batch_ind.max() + 1 if len(batch_ind) > 0 else 1
        n_res = torch.max(resnum) + 1 if len(resnum) > 0 else 1
        n_chains = chain_ind.max() + 1 if len(chain_ind) > 0 else 1
        resi_energy = torch.zeros((batch_size, n_chains, n_res, alternatives.shape[-1]),
                                  dtype=self.float_type, device=self.dev)

        peptide_bonds = self._find_peptide_bonds(atom_description)
        if peptide_bonds is None:
            return resi_energy

        ca_indices_c, ca_indices_n = self._get_ca_indices(peptide_bonds, atom_description)
        if ca_indices_c is None:
            return resi_energy

        valid_mask = (ca_indices_c != PADDING_INDEX) & (ca_indices_n != PADDING_INDEX)

        if peptide_bonds is not None:
            scores, scored_bonds = self._calculate_geometry_and_scores(
                peptide_bonds[valid_mask], ca_indices_c[valid_mask],
                ca_indices_n[valid_mask], coords, resname)
        else:
            scores, scored_bonds = None, None

        if scores is not None and scored_bonds is not None:
            batch_ind_c = batch_ind[scored_bonds[:, 0]]
            chain_ind_c = chain_ind[scored_bonds[:, 0]]
            resnum_c = resnum[scored_bonds[:, 0]]

            alt = 0
            indices = (batch_ind_c, chain_ind_c, resnum_c,
                       torch.full_like(batch_ind_c, alt))
            resi_energy.index_put_(indices, scores * (1 - torch.tanh(-self.weight)))

        return resi_energy

    def get_weights(self):
        """Returns the weights of the module."""
        return self.weight

    def get_num_params(self):
        """Returns the number of parameters in the module."""
        return sum(p.numel() for p in self.parameters())
