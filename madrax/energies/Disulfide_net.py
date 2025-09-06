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

import torch, numpy as np
from madrax.sources import hashings
from madrax.sources.math_utils import angle2dVectors, dihedral2dVectors
from madrax.sources.globalVariables import *

class Disulfide_net(torch.nn.Module):

    def __init__(self, name='disulfide_net', dev='cpu'):
        self.name = name
        self.dev = dev
        super(Disulfide_net, self).__init__()
        self.weight = torch.nn.Parameter(torch.tensor([0.0], device=(self.dev)))
        self.weight.requires_grad = True
        self.float_type = torch.float

    def net(self, coords, atom_description, atomPairs, partners):
        atName1 = atom_description[(atomPairs[:, 0], hashings.atom_description_hash['at_name'])]
        atName2 = atom_description[(atomPairs[:, 1], hashings.atom_description_hash['at_name'])]
        sulfur_mask = atName1.eq(hashings.atom_hash['CYS']['SG']) & atName2.eq(hashings.atom_hash['CYS']['SG'])
        if sulfur_mask.sum().eq(0):
            return ([], [], torch.zeros((atomPairs.shape[0]), device=(self.dev), dtype=(torch.bool)))
        atomPairs = atomPairs[sulfur_mask]
        sg1sg2_vector = coords[atomPairs[:, 0]] - coords[atomPairs[:, 1]]
        sg1cb1_vector = partners[(atomPairs[:, 0], 0)] - coords[atomPairs[:, 0]]
        sg2cb2_vector = partners[(atomPairs[:, 1], 0)] - coords[atomPairs[:, 1]]
        dist = torch.norm(sg1sg2_vector, dim=(-1))
        angle1, badangle_m1, _ = angle2dVectors(sg1cb1_vector, -sg1sg2_vector)
        angle2, badangle_m2, _ = angle2dVectors(sg2cb2_vector, sg1sg2_vector)
        diehdral, badangle_d = dihedral2dVectors(partners[(atomPairs[:, 0], 0)], coords[atomPairs[:, 0]], coords[atomPairs[:, 1]], partners[(atomPairs[:, 1], 0)])
        bad_angle_mask = badangle_m1 & badangle_m2 & badangle_d
        tolerance = 20
        angleMin = np.radians(90.0 - tolerance)
        angleMax = np.radians(120.0 + tolerance)
        diSDihedMin = np.radians(60.0 - tolerance)
        diSDihedMax = np.radians(150.0 + tolerance)
        disulf_bond_dist = 3.0
        geometricAngles_mask = (angle1.ge(angleMin) & angle1.le(angleMax) & angle2.ge(angleMin) & angle2.le(angleMax) & (diehdral.le(diSDihedMax) & diehdral.ge(diSDihedMin) | diehdral.ge(-diSDihedMax) & diehdral.le(-diSDihedMin)) & bad_angle_mask).squeeze(-1)
        distance_mask = dist.le(disulf_bond_dist)
        geometric_mask = geometricAngles_mask & distance_mask
        atomPairs = atomPairs[geometric_mask]
        dist = dist[geometric_mask]
        residue_distance = torch.abs(atom_description[(atomPairs[:, 0], hashings.atom_description_hash['resnum'])] - atom_description[(atomPairs[:, 1], hashings.atom_description_hash['resnum'])]).float()
        distance_correction = 5 * torch.abs(dist - 2.04)
        energy = -0.001 * TEMPERATURE * (2.1 + 2.9823825 * torch.log(torch.abs(residue_distance))) + distance_correction
        full_mask = sulfur_mask.clone()
        full_mask[full_mask == True] = geometric_mask
        return (
         energy, atomPairs, full_mask)

    def forward(self, coords, atom_description, atom_number, atomPairs, alternativeMask, partners, facc):
        disulfideEnergy, disulfideAtomPairs, disulfNetwork = self.net(coords, atom_description, atomPairs, partners)
        atomEnergy = self.bindToAtoms(disulfideEnergy, disulfideAtomPairs, atom_description, facc, alternativeMask)
        residueEnergy = self.bindToResi(atomEnergy, atom_description)
        return (
         residueEnergy, atomEnergy, disulfNetwork)

    def bindToAtoms(self, disulfideEnergy, disulfideAtomPairs, atom_description, facc, alternMask, minSaCoefficient=1.0):
        energy_atoms = torch.zeros((atom_description.shape[0], alternMask.shape[-1]), dtype=(torch.float), device=(self.dev))
        if len(disulfideAtomPairs) == 0:
            return energy_atoms
        for alt in range(alternMask.shape[-1]):
            mask = alternMask[(disulfideAtomPairs[:, 0], alt)] & alternMask[(disulfideAtomPairs[:, 1], alt)]
            alt_index = torch.full((mask.shape), alt, device=(self.dev), dtype=(torch.long))[mask]
            atom_numberAlt = disulfideAtomPairs[mask]
            indices1 = (
             atom_numberAlt[:, 0], alt_index)
            energy_atoms = energy_atoms.index_put(indices1, (disulfideEnergy[mask] * 0.5), accumulate=True)
            indices2 = (
             atom_numberAlt[:, 1], alt_index)
            energy_atoms = energy_atoms.index_put(indices2, (disulfideEnergy[mask] * 0.5), accumulate=True)

        saCoefficient = torch.max(1 - facc[:, :], torch.tensor(minSaCoefficient, device=(self.dev), dtype=(self.float_type)))
        return energy_atoms * saCoefficient * (1 - torch.tanh(-self.weight))

    def bindToResi(self, atomEnergy, atom_description):
        nres = atom_description[:, hashings.atom_description_hash['resnum']].max() + 1
        nchains = atom_description[:, hashings.atom_description_hash['chain']].max() + 1
        naltern = atomEnergy.shape[-1]
        batchInd = atom_description[:, hashings.atom_description_hash['batch']].unsqueeze(-1).expand(-1, naltern).long()
        batch = batchInd.max() + 1
        chain_ind = atom_description[:, hashings.atom_description_hash['chain']].unsqueeze(-1).expand(-1, naltern).long()
        resIndex = atom_description[:, hashings.atom_description_hash['resnum']].unsqueeze(-1).expand(-1, naltern).long()
        alt_index = torch.arange(0, naltern, dtype=(torch.long), device=(self.dev)).unsqueeze(0).expand(atom_description.shape[0], -1)
        energyResi = torch.zeros((batch, nchains, nres, naltern), dtype=(self.float_type), device=(self.dev))
        bbmask = resIndex != PADDING_INDEX
        index = tuple([batchInd[bbmask], chain_ind[bbmask], resIndex[bbmask], alt_index[bbmask]])
        energyResi.index_put_(index, (atomEnergy[bbmask]), accumulate=True)
        return energyResi

    def getWeights(self):
        pass

    def getNumParams(self):
        p = []
        for i in self.parameters():
            p += list(i.data.cpu().numpy().flat)

        print('Number of parameters=', len(p))
# okay decompiling Disulfide_net37.pyc
