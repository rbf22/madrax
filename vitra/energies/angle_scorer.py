#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from dataclasses import dataclass
import torch

from vitra.sources import hashings
from vitra.sources.globalVariables import PADDING_INDEX
from vitra.sources.kde import realNVP

@dataclass
class ScoreData:
    """Data class for score calculation."""
    aa_idx: int
    angles: torch.Tensor
    batch_idx: torch.Tensor
    chain_idx: torch.Tensor
    resnum_idx: torch.Tensor

class AngleScorerEnergy(torch.nn.Module):
    r"""
    This module calculates the backbone and side-chain entropy terms based on torsional angles.
    """

    def __init__(self, name='AngleScorer', dev='cpu'):
        """
        Initializes the AngleScorer module.
        """
        super().__init__()
        self.name = name
        self.dev = dev
        self.float_type = torch.float
        self.weight_omega = torch.nn.Parameter(torch.tensor([0.0], device=self.dev))
        self.weight_bb = torch.nn.Parameter(torch.tensor([0.0], device=self.dev))
        self.weight_sc = torch.nn.Parameter(torch.zeros(20, device=self.dev))
        self.kde_bb = {}
        self.kde_omega = {}
        self.kde_sc = {}
        self.load()

    def _get_scores(self, data: ScoreData):
        """Calculates the scores for a given amino acid."""
        mask_j = data.resnum_idx == data.aa_idx
        batch_j = data.batch_idx[mask_j].long()
        chain_j = data.chain_idx[mask_j].long()
        resnum_j = data.resnum_idx[mask_j].long()

        inp_bb = data.angles[batch_j, chain_j, resnum_j, :, :3]
        inp_bb_reshaped = inp_bb.reshape(-1, 3)
        bb_angle = [0, 1]

        bb_prob = (self.kde_bb[data.aa_idx].log_prob(inp_bb_reshaped[:, bb_angle]) *
                   (1 - torch.tanh(-self.weight_bb))).clamp(max=5.0)
        omega_prob = self.kde_omega[data.aa_idx].log_prob(inp_bb_reshaped[:, [2]]) * \
                    (1 - torch.tanh(-self.weight_omega))

        sc_prob = torch.zeros_like(bb_prob)
        if data.aa_idx in self.kde_sc:
            n_chi = self.kde_sc[data.aa_idx].s[0][0].in_features
            inp_sc = data.angles[batch_j, chain_j, resnum_j, :, 3:3 + n_chi]
            sc_prob = (self.kde_sc[data.aa_idx].log_prob(inp_sc.reshape(-1, n_chi)) *
                       (1 - torch.tanh(-self.weight_sc[data.aa_idx]))).clamp(max=5.0)

        return bb_prob, omega_prob, sc_prob, mask_j

    def forward(self, atom_description, angles, alternatives, return_raw_values=False):
        """
        Calculates the backbone and side-chain entropy scores.
        """
        naltern = alternatives.shape[-1]
        batch_ind = atom_description[:, hashings.atom_description_hash['batch']].long()
        resnum = atom_description[:, hashings.atom_description_hash['resnum']].long()
        chain_ind = atom_description[:, hashings.atom_description_hash['chain']].long()

        batch = batch_ind.max() + 1 if len(batch_ind) > 0 else 1
        nres = torch.max(resnum) + 1 if len(resnum) > 0 else 1
        nchains = chain_ind.max() + 1 if len(chain_ind) > 0 else 1

        bb_score = torch.zeros((batch, nchains, nres, naltern),
                               dtype=self.float_type, device=self.dev)
        rotamer_violation = torch.zeros((batch, nchains, nres, naltern),
                                        dtype=self.float_type, device=self.dev)

        res_info = torch.unique(atom_description[:, [0, 1, 2, 3]], dim=0)
        batch_idx, chain_idx, resnum_idx, resname_idx = res_info.T
        batch_idx = batch_idx.long()
        chain_idx = chain_idx.long()
        resnum_idx = resnum_idx.long()

        unique_aa = torch.unique(resname_idx[resname_idx != PADDING_INDEX])

        for aa_idx in unique_aa.tolist():
            score_data = ScoreData(aa_idx, angles, batch_idx, chain_idx, resnum_idx)
            bb_prob, omega_prob, sc_prob, mask_j = self._get_scores(score_data)

            fullmask = torch.zeros_like(bb_score, dtype=torch.bool)
            fullmask[batch_idx[mask_j], chain_idx[mask_j], resnum_idx[mask_j], :] = True

            score = ((-1.0 * (bb_prob + omega_prob + sc_prob) + 0.0).clamp(0, 5))
            bb_score[fullmask] = score.reshape(fullmask.sum().item())

        if return_raw_values:
            return bb_score
        return bb_score, rotamer_violation

    def load(self):
        """
        Loads the KDE models from files.
        """
        self.kde_bb = {}
        self.kde_sc = {}
        self.kde_omega = {}
        nfea_hashing = {
            'GLN': 3, 'VAL': 1, 'ASN': 2, 'THR': 1, 'ASP': 2, 'PHE': 2, 'LEU': 2,
            'SER': 1, 'CYS': 1, 'ILE': 1, 'TRP': 2, 'ARG': 5, 'LYS': 4, 'TYR': 2,
            'GLU': 3, 'MET': 3, 'HIS': 2
        }
        for i in range(20):
            base_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                     '..', 'parameters', 'weightsKDE')

            bb_path = os.path.join(base_path, f'kdeBB_{i}.weights')
            if os.path.exists(bb_path):
                bb_model = realNVP.RealNVP(nfea=2, device=self.dev)
                bb_model.load_state_dict(torch.load(bb_path, map_location=torch.device(self.dev)))
                self.kde_bb[i] = bb_model

            omega_path = os.path.join(base_path, f'kdeOmega_{i}.weights')
            if os.path.exists(omega_path):
                omega_model = realNVP.RealNVP(nfea=1, device=self.dev)
                omega_model.load_state_dict(torch.load(omega_path,
                                                 map_location=torch.device(self.dev)))
                self.kde_omega[i] = omega_model

            res_name = hashings.resi_hash_inverse[i]
            if res_name in nfea_hashing:
                sc_path = os.path.join(base_path, f'kdeSC_{i}.weights')
                if os.path.exists(sc_path):
                    sc_model = realNVP.RealNVP(nfea=nfea_hashing[res_name], device=self.dev)
                    sc_model.load_state_dict(torch.load(sc_path,
                                                  map_location=torch.device(self.dev)))
                    self.kde_sc[i] = sc_model
        return True
