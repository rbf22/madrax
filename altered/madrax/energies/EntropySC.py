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

import torch, math
from madrax.sources import hashings
from madrax.sources.globalVariables import *
letters = {
 'CYS': 'C', 'ASP': 'D', 'SER': 'S', 'GLN': 'Q', 'LYS': 'K', 'ASN': 'N', 
 'PRO': 'P', 'THR': 'T', 'PHE': 'F', 'ALA': 'A', 'HIS': 'H', 'GLY': 'G', 
 'ILE': 'I', 'LEU': 'L', 'ARG': 'R', 'TRP': 'W', 'VAL': 'V', 'GLU': 'E', 
 'TYR': 'Y', 'MET': 'M'}
inverse_letters = {
 'C': 'CYS', 'D': 'ASP', 'S': 'SER', 'Q': 'GLN', 'K': 'LYS', 'N': 'ASN', 
 'P': 'PRO', 'T': 'THR', 'F': 'PHE', 'A': 'ALA', 'H': 'HIS', 'G': 'GLY', 
 'I': 'ILE', 'L': 'LEU', 'R': 'ARG', 'W': 'TRP', 'V': 'VAL', 'E': 'GLU', 
 'Y': 'TYR', 'M': 'MET'}

class EntropySC(torch.nn.Module):

    def __init__(self, name='EntropySC', dev='cpu', command='SequenceDetail', properties_hashing={}, backbone_atoms=[], donor=[], acceptor=[], hbond_ar=[], hashing={}):
        self.name = name
        self.backbone_atoms = backbone_atoms
        self.donor = donor
        self.dev = dev
        self.acceptor = acceptor
        self.hashing = hashing
        self.properties_hashing = properties_hashing
        super(EntropySC, self).__init__()
        self.entropy_abayan = {
         'ALA': 0, 
         'CYS': 0.00375, 
         'ASP': 0.0020333, 
         'GLU': 0.0055, 
         'PHE': 0.0029333, 
         'GLY': 0, 
         'HIS': 0.0033, 
         'ILE': 0.00375, 
         'LYS': 0.0073667, 
         'LEU': 0.00375, 
         'MET': 0.0051, 
         'ASN': 0.0027, 
         'PRO': 0.001, 
         'GLN': 0.0067333, 
         'ARG': 0.0071, 
         'SER': 0.0020667, 
         'THR': 0.0020333, 
         'VAL': 0.0016667, 
         'TRP': 0.0032333, 
         'TYR': 0.0033}
        self.CAatoms = []
        for res1 in hashings.resi_hash.keys():
            self.CAatoms += [hashings.atom_hash[res1]['CA']]

        self.CAatoms = list(set(self.CAatoms))
        self.weight = torch.nn.Parameter(torch.tensor([0.0], device=(self.dev)))
        self.weight.requires_grad = True

    def forward(self, atom_description, saSC, hbond, vdw, electro, alternatives):
        a = atom_description[:, hashings.atom_description_hash['at_name']]
        for i, caatom in enumerate(self.CAatoms):
            if i == 0:
                caatom_mask = a.eq(caatom)
            else:
                caatom_mask += a.eq(caatom)

        del a
        seq = atom_description[caatom_mask][:, hashings.atom_description_hash['resname']]
        batch = saSC.shape[0]
        nchains = saSC.shape[1]
        nresi = saSC.shape[2]
        naltern = saSC.shape[-1]
        lookup_entrop_sc = torch.zeros((seq.shape), dtype=(torch.float), device=(self.dev))
        for i in inverse_letters.keys():
            m = seq.eq(hashings.resi_hash[inverse_letters[i]])
            if True in m:
                lookup_entrop_sc[m] = (1 - torch.tanh(-self.weight)) * TEMPERATURE * self.entropy_abayan[inverse_letters[i]]

        batch_ind = atom_description[:, hashings.atom_description_hash['batch']][caatom_mask].long()
        resnum = atom_description[:, hashings.atom_description_hash['resnum']][caatom_mask].long()
        chain_ind = atom_description[:, hashings.atom_description_hash['chain']][caatom_mask].long()
        final_entropy_sc = torch.zeros((batch, nchains, nresi, naltern), device=(self.dev), dtype=(torch.float))
        mask_padding = ~seq.eq(PADDING_INDEX)
        for alt in range(naltern):
            maskAlt = alternatives[caatom_mask][:, alt] & mask_padding
            alt_index = torch.full((maskAlt.shape), alt, device=(self.dev), dtype=(torch.long))
            index = (
             batch_ind[maskAlt], chain_ind[maskAlt], resnum[maskAlt], alt_index[maskAlt])
            final_entropy_sc.index_put_(index, lookup_entrop_sc[maskAlt])

        lookup_entrop_sc = final_entropy_sc.clone()
        final_entropy_sc = final_entropy_sc * torch.nn.functional.relu(saSC)
        corr = torch.ones(electro.shape).type_as(electro)
        corr[electro > 0] = 0.2
        residue_energy = torch.abs(hbond + vdw + electro * corr)
        mask1 = lookup_entrop_sc < residue_energy
        mask2 = ~mask1 & (final_entropy_sc < residue_energy)
        final_entropy_sc[mask1] = lookup_entrop_sc[mask1]
        final_entropy_sc[mask2] = residue_energy[mask2]
        return final_entropy_sc

    def getWeights(self):
        pass

    def getNumParams(self):
        p = []
        for i in self.parameters():
            p += list(i.data.cpu().numpy().flat)

        print('Number of parameters=', len(p))
# okay decompiling EntropySC37.pyc
