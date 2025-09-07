#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
#  AngleScorer.py
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

import os
import torch
from torch import nn

from vitra.sources import hashings
from vitra.sources.globalVariables import *
from vitra.sources.kde import realNVP

class AngleScorer(torch.nn.Module):
    """
    This module calculates the backbone and side-chain entropy terms based on torsional angles.
    It uses Kernel Density Estimation (KDE) to model the probability distribution of backbone
    (phi, psi, omega) and side-chain (chi) angles.

    The `README.md` describes the backbone and side-chain entropy as:
    \[
    E_{bb,i} = -w_1\ln\big[{\rm KDE}_\omega(\omega)\big] - w_2\ln\big[{\rm KDE}_{\phi\psi}(\phi,\psi)\big]
    \]
    \[
    E_{sc,i} = -w_i \ln\big[{\rm KDE}_\chi(\chi_1,\chi_2,\dots)\big]
    \]
    This is implemented in the `forward` method, where `bbProb`, `omegaProb`, and `scProb` are calculated
    from pre-trained KDE models.
    """

    def __init__(self, name='AngleScorer', dev='cpu'):
        """
        Initializes the AngleScorer module.
        """
        self.name = name
        self.dev = dev
        self.float_type = torch.float
        # ... (initialization of various parameters and data structures)
        super(AngleScorer, self).__init__()
        self.weightOmega = torch.nn.Parameter(torch.tensor([0.0], device=self.dev))
        self.weightOmega.requires_grad = True
        self.weightBB = torch.nn.Parameter(torch.tensor([0.0], device=self.dev))
        self.weightBB.requires_grad = True
        self.weightSC = torch.nn.Parameter(torch.zeros(20, device=self.dev))
        self.weightSC.requires_grad = True
        self.kdeBB = {}
        self.kdeOmega = {}
        self.kdeSC = {}
        self.load()

    def forward(self, atom_description, angles, alternatives, returnRawValues=False):
        """
        Calculates the backbone and side-chain entropy scores.
        """
        naltern = alternatives.shape[-1]
        batch_ind = atom_description[:, hashings.atom_description_hash['batch']].long()
        resnum = atom_description[:, hashings.atom_description_hash['resnum']].long()
        chain_ind = atom_description[:, hashings.atom_description_hash['chain']].long()
        resname = atom_description[:, hashings.atom_description_hash['resname']]

        batch = batch_ind.max() + 1 if len(batch_ind)>0 else 1
        nres = torch.max(resnum) + 1 if len(resnum)>0 else 1
        nchains = chain_ind.max() + 1 if len(chain_ind)>0 else 1

        bbScore = torch.zeros((batch, nchains, nres, naltern), dtype=self.float_type, device=self.dev)
        rotamerViolation = torch.zeros((batch, nchains, nres, naltern), dtype=self.float_type, device=self.dev)

        res_info = torch.unique(atom_description[:, [0, 1, 2, 3]], dim=0)
        batch_idx = res_info[:, 0].long()
        chain_idx = res_info[:, 1].long()
        resnum_idx = res_info[:, 2].long()
        resname_idx = res_info[:, 3]

        aa = torch.unique(resname_idx[resname_idx!=PADDING_INDEX])

        beta = 1.0
        bias_correction = 0.0

        for j in aa.tolist():
            mask_j = resname_idx == j

            batch_j = batch_idx[mask_j]
            chain_j = chain_idx[mask_j]
            resnum_j = resnum_idx[mask_j]

            inpBB = angles[batch_j, chain_j, resnum_j, :, :3]

            inpBB_reshaped = inpBB.reshape(-1, 3)
            bb_angle = [0, 1]

            bbProb = (self.kdeBB[j].log_prob(inpBB_reshaped[:, bb_angle]) * (1 - torch.tanh(-self.weightBB))).clamp(max=5.0)
            omegaProb = self.kdeOmega[j].log_prob(inpBB_reshaped[:, [2]]) * (1 - torch.tanh(-self.weightOmega))

            fullmask = torch.zeros_like(bbScore, dtype=torch.bool)
            fullmask[batch_j, chain_j, resnum_j, :] = True

            scProb = torch.zeros_like(bbProb)
            if j in self.kdeSC:
                n_chi = self.kdeSC[j].s[0][0].in_features
                inpSC = angles[batch_j, chain_j, resnum_j, :, 3:3+n_chi]
                scProb = (self.kdeSC[j].log_prob(inpSC.reshape(-1, n_chi)) * (1 - torch.tanh(-self.weightSC[j]))).clamp(max=5.0)

            bbScore[fullmask] = ((-beta * (bbProb + omegaProb + scProb) + bias_correction).clamp(0, 5)).reshape(fullmask.sum().item())

        if returnRawValues:
            return bbScore
        return bbScore, rotamerViolation

    # ... (other methods omitted for brevity)
    def load(self):
        self.kdeBB = {}
        self.kdeSC = {}
        self.kdeOmega = {}
        nfeaHashing = {
         'GLN': 3, 
         'VAL': 1, 
         'ASN': 2, 
         'THR': 1, 
         'ASP': 2, 
         'PHE': 2, 
         'LEU': 2, 
         'SER': 1, 
         'CYS': 1, 
         'ILE': 1, 
         'TRP': 2, 
         'ARG': 5, 
         'LYS': 4, 
         'TYR': 2, 
         'GLU': 3, 
         'MET': 3, 
         'HIS': 2}
        for i in range(20):
            bb = realNVP.RealNVP(nfea=2, device=(self.dev))
            bb.load_state_dict(torch.load(('/'.join(os.path.realpath(__file__).split('/')[:-1]) + '/../parameters/weightsKDE/kdeBB_' + str(i) + '.weights'), map_location=(torch.device(self.dev))))
            self.kdeBB[i] = bb
            omega = realNVP.RealNVP(nfea=1, device=(self.dev))
            omega.load_state_dict(torch.load(('/'.join(os.path.realpath(__file__).split('/')[:-1]) + '/../parameters/weightsKDE/kdeOmega_' + str(i) + '.weights'), map_location=(torch.device(self.dev))))
            self.kdeOmega[i] = omega
            if hashings.resi_hash_inverse[i] in nfeaHashing:
                sc = realNVP.RealNVP(nfea=(nfeaHashing[hashings.resi_hash_inverse[i]]), device=(self.dev))
                sc.load_state_dict(torch.load(('/'.join(os.path.realpath(__file__).split('/')[:-1]) + '/../parameters/weightsKDE/kdeSC_' + str(i) + '.weights'), map_location=(torch.device(self.dev))))
                self.kdeSC[i] = sc

        return True
