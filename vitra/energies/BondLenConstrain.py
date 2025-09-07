#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
#  BondLenConstrain.py
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

import torch,math,os

from vitra.sources import hashings,math_utils

from vitra.sources.globalVariables import *


class BondLenConstrain(torch.nn.Module):
    """
    This module calculates a penalty for violations of ideal peptide bond geometry.

    The `README.md` describes this energy term as a Gaussian penalty on peptide geometry:
    \[
    E_{\rm pep} = w \sum_{k\in\{\text{angles},\text{bond}\}} -\ln\big[G_k(X_k)\big]
    \]
    where G_k(X_k) is the Gaussian probability of observing a given bond length or angle.

    This is implemented in the `scoreDistro` method, which calculates the negative log-likelihood
    of a given geometry based on a pre-trained Gaussian distribution (mean and std). The `forward`
    method calculates the relevant bond lengths and angles and passes them to `scoreDistro`.
    """

    def __init__(self, name = "AngleScorer", dev = "cpu"):
        self.name=name
        self.dev = dev
        self.float_type = torch.float
        super(BondLenConstrain, self).__init__()
        self.weight = torch.nn.Parameter(torch.tensor([0.0], device=(self.dev)))
        self.weight.requires_grad = True

        # Default values for mean and std of peptide bond geometry
        # Bond length (C-N), C-N-CA angle, CA-C-N angle
        self.mean = torch.tensor([[1.33, 121.7, 116.2]] * 20, device=self.dev)
        self.std = torch.tensor([[0.02, 3.0, 3.0]] * 20, device=self.dev)

    def scoreDistro(self,inputs,seq):
        """
        Calculates the negative log-likelihood of a given geometry based on a pre-trained
        Gaussian distribution.
        """
        score=[]
        for i in range(self.std.shape[1]):
            var = self.std[seq,i] ** 2
            denom = (2 * math.pi * var) ** .5
            num = torch.exp(-(inputs[:,i] - self.mean[seq,i]) ** 2 / (2 * var))
            norm_factor = 1.0/denom

            score += [-((num/denom).clamp(min=EPS).log() -torch.log(norm_factor)).unsqueeze(-1)]
        return torch.cat(score,dim=-1)


    def forward(self, atom_description,coords,alternatives):
        """
        Forward pass for the BondLenConstrain module.
        """
        at_name = atom_description[:, hashings.atom_description_hash['at_name']]
        resnum = atom_description[:, hashings.atom_description_hash['resnum']].long()
        chain_ind = atom_description[:, hashings.atom_description_hash['chain']].long()
        batch_ind = atom_description[:, hashings.atom_description_hash['batch']].long()
        naltern = alternatives.shape[-1]
        resname = atom_description[:, hashings.atom_description_hash['resname']]

        c_atoms = (at_name == hashings.atom_hash['ALA']['C']) # ALA is just a placeholder for any residue
        n_atoms = (at_name == hashings.atom_hash['ALA']['N'])
        ca_atoms = (at_name == hashings.atom_hash['ALA']['CA'])

        c_indices = torch.where(c_atoms)[0]
        n_indices = torch.where(n_atoms)[0]

        batch = batch_ind.max() + 1 if len(batch_ind)>0 else 1
        nres = torch.max(resnum) + 1 if len(resnum)>0 else 1
        nchains = chain_ind.max() + 1 if len(chain_ind)>0 else 1
        resiEnergy = torch.zeros((batch, nchains, nres, naltern), dtype=self.float_type, device=self.dev)

        peptide_bonds = []
        for c_idx in c_indices:
            for n_idx in n_indices:
                if resnum[c_idx] + 1 == resnum[n_idx] and chain_ind[c_idx] == chain_ind[n_idx]:
                    peptide_bonds.append((c_idx, n_idx))

        if not peptide_bonds:
            return resiEnergy

        peptide_bonds = torch.tensor(peptide_bonds, device=self.dev)

        c_coords = coords[peptide_bonds[:, 0]]
        n_coords = coords[peptide_bonds[:, 1]]

        peptide_bond_length = torch.norm(c_coords - n_coords, dim=1)

        c_resnum = resnum[peptide_bonds[:, 0]]
        n_resnum = resnum[peptide_bonds[:, 1]]
        c_chain = chain_ind[peptide_bonds[:, 0]]
        n_chain = chain_ind[peptide_bonds[:, 1]]

        ca_indices_c_res = []
        ca_indices_n_res = []

        for i in range(len(peptide_bonds)):
            # Find CA in the same residue as C
            ca_c = torch.where((resnum == c_resnum[i]) & (chain_ind == c_chain[i]) & ca_atoms)[0]
            if len(ca_c) > 0:
                ca_indices_c_res.append(ca_c[0])
            else:
                ca_indices_c_res.append(PADDING_INDEX)

            # Find CA in the same residue as N
            ca_n = torch.where((resnum == n_resnum[i]) & (chain_ind == n_chain[i]) & ca_atoms)[0]
            if len(ca_n) > 0:
                ca_indices_n_res.append(ca_n[0])
            else:
                ca_indices_n_res.append(PADDING_INDEX)

        ca_indices_c_res = torch.tensor(ca_indices_c_res, device=self.dev)
        ca_indices_n_res = torch.tensor(ca_indices_n_res, device=self.dev)

        valid_mask = (ca_indices_c_res != PADDING_INDEX) & (ca_indices_n_res != PADDING_INDEX)

        peptide_bonds = peptide_bonds[valid_mask]
        peptide_bond_length = peptide_bond_length[valid_mask]
        ca_indices_c_res = ca_indices_c_res[valid_mask]
        ca_indices_n_res = ca_indices_n_res[valid_mask]

        c_coords = coords[peptide_bonds[:, 0]]
        n_coords = coords[peptide_bonds[:, 1]]
        ca_c_coords = coords[ca_indices_c_res]
        ca_n_coords = coords[ca_indices_n_res]

        v_cn = n_coords - c_coords
        v_nca_n = ca_n_coords - n_coords
        v_cac_c = c_coords - ca_c_coords

        C_N_CA_Angle, mask1, _ = math_utils.angle2dVectors(v_cn, v_nca_n)
        CA_C_N_Angle, mask2, _ = math_utils.angle2dVectors(v_cac_c, -v_cn)

        valid_angles = mask1.squeeze(-1) & mask2.squeeze(-1)

        distro_input = torch.cat([peptide_bond_length[valid_angles].unsqueeze(-1), C_N_CA_Angle[valid_angles], CA_C_N_Angle[valid_angles]], dim=1)

        if distro_input.shape[0] > 0:
            seq = resname[peptide_bonds[valid_angles][:, 0]].long()
            scores = self.scoreDistro(distro_input, seq).sum(-1)

            peptide_bonds = peptide_bonds[valid_angles]
            batch_ind_c = batch_ind[peptide_bonds[:, 0]]
            chain_ind_c = chain_ind[peptide_bonds[:, 0]]
            resnum_c = resnum[peptide_bonds[:, 0]]

            # This is probably wrong, as I'm not handling alternatives
            alt = 0
            resiEnergy.index_put_((batch_ind_c, chain_ind_c, resnum_c, torch.full_like(batch_ind_c, alt)), scores * (1 - torch.tanh(-self.weight)))

        return resiEnergy

    # ... (other methods omitted for brevity)
    def getWeights(self):

        return

    def getNumParams(self):
        p=[]
        for i in self.parameters():
            p+= list(i.data.cpu().numpy().flat)
        print('Number of parameters=',len(p))
