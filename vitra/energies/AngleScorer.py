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

import os
import torch
from torch import nn

from vitra.sources import hashings
from vitra.sources.globalVariables import *
from vitra.sources.kde import realNVP

letters = {
 'CYS': "'C'", 'ASP': "'D'", 'SER': "'S'", 'GLN': "'Q'", 'LYS': "'K'", 'ASN': "'N'", 
 'PRO': "'P'", 'THR': "'T'", 'PHE': "'F'", 'ALA': "'A'", 'HIS': "'H'", 'GLY': "'G'", 
 'ILE': "'I'", 'LEU': "'L'", 'ARG': "'R'", 'TRP': "'W'", 'VAL': "'V'", 'GLU': "'E'", 
 'TYR': "'Y'", 'MET': "'M'"}
inverse_letters = {
 'C': "'CYS'", 'D': "'ASP'", 'S': "'SER'", 'Q': "'GLN'", 'K': "'LYS'", 'N': "'ASN'", 
 'P': "'PRO'", 'T': "'THR'", 'F': "'PHE'", 'A': "'ALA'", 'H': "'HIS'", 'G': "'GLY'", 
 'I': "'ILE'", 'L': "'LEU'", 'R': "'ARG'", 'W': "'TRP'", 'V': "'VAL'", 'E': "'GLU'", 
 'Y': "'TYR'", 'M': "'MET'"}


class AngleScorer(torch.nn.Module):

    def __init__(self, name='AngleScorer', dev='cpu'):
        self.name = name
        self.dev = dev
        self.float_type = torch.float
        self.anglesOfResidues = {'PRO':[],  'GLY':[],  'GLN':[
          3, 4, 5], 
         'VAL':[
          3], 
         'ASN':[
          3, 4], 
         'THR':[
          3], 
         'ALA':[],  'ASP':[
          3, 4], 
         'PHE':[
          3, 4], 
         'LEU':[
          3, 4], 
         'SER':[
          3], 
         'CYS':[
          3], 
         'ILE':[
          3], 
         'TRP':[
          3, 4], 
         'ARG':[
          3, 4, 5, 6, 7], 
         'LYS':[
          3, 4, 5, 6], 
         'TYR':[
          3, 4], 
         'GLU':[
          3, 4, 5], 
         'MET':[
          3, 4, 5], 
         'HIS':[
          3, 4]}
        self.CAatoms = []
        for res1 in hashings.resi_hash.keys():
            self.CAatoms += [hashings.atom_hash[res1]['CA']]

        self.CAatoms = list(set(self.CAatoms))
        super(AngleScorer, self).__init__()
        self.weightOmega = torch.nn.Parameter(torch.tensor([0.0], device=self.dev))
        self.weightOmega.requires_grad = True
        self.weightBB = torch.nn.Parameter(torch.tensor([0.0], device=self.dev))
        self.weightBB.requires_grad = True
        self.weightSC = torch.nn.Parameter(torch.zeros(20, device=self.dev))
        self.weightSC.requires_grad = True
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.load()
        self.kdeBB = {}
        self.kdeOmega = {}
        self.kdeSC = {}

    def forward(self, atom_description, angles, alternatives, returnRawValues=False):
        a = atom_description[:, hashings.atom_description_hash['at_name']]
        for i, caatom in enumerate(self.CAatoms):
            if i == 0:
                caatom_mask = a.eq(caatom)
            else:
                caatom_mask += a.eq(caatom)

        del a
        seq = atom_description[caatom_mask][:, hashings.atom_description_hash['resname']]
        naltern = alternatives.shape[-1]
        beta = 0.5
        aa = range(20)
        padding_mask = ~seq.eq(PADDING_INDEX).to(self.dev)
        batch_ind = atom_description[:, hashings.atom_description_hash['batch']][caatom_mask].long()
        resnum = atom_description[:, hashings.atom_description_hash['resnum']][caatom_mask].long()
        chain_ind = atom_description[:, hashings.atom_description_hash['chain']][caatom_mask].long()
        chainMax = chain_ind.max() + 1
        resMax = resnum.max() + 1
        batch = batch_ind.max() + 1
        seqalt = torch.full((batch, chainMax, resMax, naltern), (int(PADDING_INDEX)), device=(self.dev))
        bbScore = torch.zeros((batch, chainMax, resMax, naltern), dtype=(torch.float), device=(self.dev))
        rotamerViolation = torch.zeros((batch, chainMax, resMax, naltern), dtype=(torch.float), device=(self.dev))
        for alt in range(naltern):
            maskAlt = alternatives[caatom_mask][:, alt] & padding_mask
            alt_index = torch.full((maskAlt.shape), alt, device=(self.dev), dtype=(torch.long))
            index = (
             batch_ind[maskAlt], chain_ind[maskAlt], resnum[maskAlt], alt_index[maskAlt])
            seqalt.index_put_(index, seq[maskAlt].long().to(self.dev))

        train_diz = {'bb':{},  'omega':{},  'sc':{}}
        for j in aa:
            j = int(j)
            if j == PADDING_INDEX:
                continue
            mask = seqalt.eq(j)
            inp = angles[mask]
            bb_angle = (0, 1)
            missingMask = ~inp[:, bb_angle].eq(PADDING_INDEX).sum(-1).bool()
            fullmask = mask.clone()
            fullmaskSC = mask.clone()
            missingMaskSC = ~inp[:, self.anglesOfResidues[hashings.resi_hash_inverse[j]]].eq(PADDING_INDEX).sum(-1).bool()
            fullmask[fullmask == True] = missingMask
            fullmaskSC[fullmaskSC == True] = missingMaskSC
            if True in fullmask and True in missingMaskSC:
                inpBB = inp[missingMask]
                inpSC = inp[missingMaskSC]
            if returnRawValues:
                train_diz['bb'][j] = inpBB[:, bb_angle]
                train_diz['omega'][j] = inpBB[:, [2]]
                if self.anglesOfResidues[hashings.resi_hash_inverse[j]] != []:
                    train_diz['sc'][j] = inpSC[:, self.anglesOfResidues[hashings.resi_hash_inverse[j]]]
                    continue
                bbProb = (self.kdeBB[j].log_prob(inpBB[:, bb_angle]) * (1 - torch.tanh(-self.weightBB))).clamp(max=5.0)
                omegaProb = self.kdeOmega[j].log_prob(inpBB[:, [2]]) * (1 - torch.tanh(-self.weightOmega))
                bias_correction = 0
                if self.anglesOfResidues[hashings.resi_hash_inverse[j]] != []:
                    scProb = (self.kdeSC[j].log_prob(inpBB[:, self.anglesOfResidues[hashings.resi_hash_inverse[j]]]) * (1 - torch.tanh(-self.weightSC[j]))).clamp(max=5.0)
                    rotamerViolation[fullmask] = (-beta * scProb + bias_correction).clamp(0, 5)
                bbScore[fullmask] = (-beta * (bbProb + omegaProb) + bias_correction).clamp(0, 5)

        if returnRawValues:
            return train_diz
        return bbScore, rotamerViolation

    def fit(self, pdb_dir):
        from vitra import utils, dataStructures
        from vitra.sources import math_utils
        device = 'cuda'
        train_diz = {'bb':{},  'omega':{},  'sc':{}}
        coo, atnames = utils.parsePDB(pdb_dir)
        split_numb = 50
        max_prots = len(atnames)
        for split in range(split_numb, max_prots, split_numb):
            info_tensors = dataStructures.create_info_tensors((atnames[split - split_numb:split]), device=device, verbose=True)
            _, atom_description, coordsIndexingAtom, _, angle_indices, _ = info_tensors
            coords = coo[split - split_numb:split][(atom_description[:, 0].long(), coordsIndexingAtom)].to(self.dev)
            existentAnglesMask = (angle_indices != PADDING_INDEX).prod(-1).bool()
            flat_angle_indices = angle_indices[existentAnglesMask]
            flat_angles, _ = math_utils.dihedral2dVectors(coords[flat_angle_indices[:, 0]], coords[flat_angle_indices[:, 1]], coords[flat_angle_indices[:, 2]], coords[flat_angle_indices[:, 3]])
            angles = torch.full(existentAnglesMask.shape, (float(PADDING_INDEX)), device=self.dev)
            angles[existentAnglesMask] = flat_angles.squeeze(-1)
            alternatives = torch.ones((atom_description.shape[0], 1), device=(self.dev), dtype=(torch.bool))
            try:
                train_dizTMP = self.forward(atom_description, angles, alternatives, returnRawValues=True)
            except:
                continue

            for kdeType in train_dizTMP.keys():
                for res in train_dizTMP[kdeType].keys():
                    if res not in train_diz[kdeType]:
                        train_diz[kdeType][res] = train_dizTMP[kdeType][res]
                    else:
                        train_diz[kdeType][res] = torch.cat([train_dizTMP[kdeType][res], train_diz[kdeType][res]], dim=0)

            torch.save(train_diz, 'marshalled/angles_train.m')

        train_diz = torch.load('marshalled/angles_train.m')

        for j in train_diz['bb'].keys():
            print('kde', j, len(train_diz['bb'][j]))
            self.kdeBB[j] = realNVP.train_kde((train_diz['bb'][j]), epochs=1500)
            self.kdeOmega[j] = realNVP.train_kde((train_diz['omega'][j]), epochs=1500)
            if j in train_diz['sc']:
                self.kdeSC[j] = realNVP.train_kde((train_diz['sc'][j]), epochs=1500)
            print('done', j)

        torch.save(self.kdeBB, '/'.join(os.path.realpath(__file__).split('/')[:-1]) + '/../parameters/kdeBB.m')
        torch.save(self.kdeOmega, '/'.join(os.path.realpath(__file__).split('/')[:-1]) + '/../parameters/kdeOmega.m')
        torch.save(self.kdeSC, '/'.join(os.path.realpath(__file__).split('/')[:-1]) + '/../parameters/kdeSC.m')

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
# okay decompiling AngleScorer37.pyc
