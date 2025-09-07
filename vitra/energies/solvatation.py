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

import math
import torch

from vitra.sources import hashings
from vitra.sources.globalVariables import PADDING_INDEX, TEMPERATURE, EPS


class Solvatation(torch.nn.Module):

    def __init__(self, name='Solvatation', dev='cpu', float_type=torch.float, backbone_atoms=[], acceptor=[], donor=[]):
        self.name = name
        self.backbone_atoms = backbone_atoms
        self.dev = dev
        self.float_type = float_type
        super(Solvatation, self).__init__()
        self.solvHydroWeight = torch.nn.Parameter(torch.tensor([0.0], device=self.dev))
        self.solvHydroWeight.requires_grad = True
        self.solvPolarWeight = torch.nn.Parameter(torch.tensor([0.0], device=self.dev))
        self.solvPolarWeight.requires_grad = True
        self.solvPolarWithWaterWeight = torch.nn.Parameter(torch.tensor([0.0], device=self.dev))
        self.solvPolarWithWaterWeight.requires_grad = True
        self.Natoms = []

        for res1 in hashings.resi_hash.keys():
            self.Natoms += [hashings.atom_hash[res1]['N']]
            self.Natoms += [hashings.atom_hash[res1]['tN']]

        self.Natoms = list(set(self.Natoms))
        self.acceptor = acceptor
        self.donor = donor
        IONSTRENGTH = 0.05
        self.DCp = (0.008 - 5e-05 * (TEMPERATURE - 273.0)) * math.log(TEMPERATURE / 273)
        self.corr_ionic = math.sqrt(IONSTRENGTH) / 3.9

    def bindToAtoms(self, atom_description, facc, faccPol, solvPCorrectionMask, atomHbonds, atomDisul):
        nalter = facc.shape[1]
        atNames = atom_description[:, hashings.atom_description_hash['at_name']].long()
        enerG = hashings.atom_Properties[(atNames, hashings.property_hashings['solvenergy_props']['enerG'])]
        padding_mask = hashings.atom_Properties[(atNames, hashings.property_hashings['other_params']['virtual'])].eq(0)
        hydrophobic_mask = enerG.le(0) & padding_mask
        nonHydrophobic_mask = enerG.ge(0) & padding_mask
        solvH = torch.zeros((atom_description.shape[0], nalter), dtype=self.float_type, device=self.dev)
        solvP = torch.zeros((atom_description.shape[0], nalter), dtype=self.float_type, device=self.dev)

        solvH[hydrophobic_mask] = (1 - torch.tanh(-self.solvHydroWeight)) * \
                                  ((enerG[hydrophobic_mask].unsqueeze(-1) - self.DCp) * (1 + self.corr_ionic)) * \
                                  facc[hydrophobic_mask]

        for i, donor_atom in enumerate(self.donor):
            if i == 0:
                donor_mask = atNames.eq(donor_atom)
            else:
                donor_mask += atNames.eq(donor_atom)

        for i, acceptor_atom in enumerate(self.acceptor):
            if i == 0:
                acceptor_mask = atNames.eq(acceptor_atom)
            else:
                acceptor_mask += atNames.eq(acceptor_atom)

        # donorsToWater = donor_mask.unsqueeze(-1).expand(-1, nalter) & \
        #                 (atomHbonds > bondEnergyTreshold) & (atomDisul > bondEnergyTreshold)
        # acceptorsToWater = acceptor_mask.unsqueeze(-1).expand(-1, nalter) & \
        #                    (atomHbonds > bondEnergyTreshold) & (atomDisul > bondEnergyTreshold)
        # interactingWithWater = donorsToWater | acceptorsToWater

        solvP[nonHydrophobic_mask.unsqueeze(-1).expand(-1, nalter) & solvPCorrectionMask] = \
            (1 - torch.tanh(-self.solvPolarWithWaterWeight)) * \
            faccPol[nonHydrophobic_mask.unsqueeze(-1).expand(-1, nalter) & solvPCorrectionMask]

        solvP[nonHydrophobic_mask.unsqueeze(-1).expand(-1, nalter) & ~solvPCorrectionMask] = \
            (1 - torch.tanh(-self.solvPolarWeight)) * \
            facc[nonHydrophobic_mask.unsqueeze(-1).expand(-1, nalter) & ~solvPCorrectionMask]
        solvP = solvP * enerG.unsqueeze(-1)

        return solvP, solvH

    def bindToResi(self, atomEnergy, atom_description):
        naltern = atomEnergy.shape[-1]
        batchIndex = atom_description[:, hashings.atom_description_hash['batch']].unsqueeze(-1).expand(-1, naltern).long()
        resIndex = atom_description[:, hashings.atom_description_hash['resnum']].unsqueeze(-1).expand(-1, naltern).long()
        chaIndex = atom_description[:, hashings.atom_description_hash['chain']].unsqueeze(-1).expand(-1, naltern).long()
        alt_index = torch.arange(0, (atomEnergy.shape[-1]), dtype=torch.long, device=self.dev).unsqueeze(0).expand(
            atomEnergy.shape[0], -1)

        batch = batchIndex.max() + 1
        nres = torch.max(resIndex) + 1
        nchains = chaIndex.max() + 1
        mask_padding = ~resIndex.eq(PADDING_INDEX)

        final = torch.zeros((batch, nchains, nres, naltern), dtype=torch.float, device=self.dev)

        mask = mask_padding
        indices = (batchIndex[mask], chaIndex[mask], resIndex[mask].long(), alt_index[mask])
        final.index_put_(indices, (atomEnergy[mask]), accumulate=True)
        return final

    def getFaccPol(self, contRatPol, contRat, atom_description, atomHbonds):
        atName = atom_description[:, hashings.atom_description_hash['at_name']].long()
        for i, natom in enumerate(self.Natoms):
            if i == 0:
                is_nitrogen_mask = atName.eq(natom)
            else:
                is_nitrogen_mask += atName.eq(natom)

        for i, acceptor_atom in enumerate(self.acceptor):
            if i == 0:
                flat_acceptor_mask = atName.eq(acceptor_atom)
            else:
                flat_acceptor_mask += atName.eq(acceptor_atom)

        fake_atomsRot = hashings.fake_atom_Properties[atName, :, hashings.property_hashingsFake['rotation']].long()
        existingFakeAts = fake_atomsRot != PADDING_INDEX
        for k in range(fake_atomsRot[existingFakeAts].max() + 1):
            right_rotation = fake_atomsRot.eq(k).unsqueeze(1)
            all_fake_with_occupancyTMP = right_rotation & contRatPol.gt(0) | ~right_rotation
            if k == 0:
                all_fake_with_occupancy = all_fake_with_occupancyTMP.prod(-1).bool() & right_rotation.sum(-1).bool()
            else:
                all_fake_with_occupancy += all_fake_with_occupancyTMP.prod(-1).bool() & right_rotation.sum(-1).bool()

        linear_CR_dependence_mask = (hashings.atom_Properties[(atName, hashings.property_hashings['hbond_params'
            ]['charged'])] != PADDING_INDEX) & ~is_nitrogen_mask & flat_acceptor_mask

        contRatPol[~linear_CR_dependence_mask] = contRatPol[~linear_CR_dependence_mask] * 2
        usecontrat_Gt_0 = contRatPol.sum(dim=(-1)).gt(0)
        usecontrat = contRatPol.sum(dim=(-1)) + contRat
        nAltern = contRatPol.shape[1]
        faccPol = torch.zeros((atom_description.shape[0], nAltern), dtype=self.float_type, device=self.dev)

        zero_denom = hashings.atom_Properties[(atName, hashings.property_hashings['solvenergy_props']['Occmax'])] != \
                     hashings.atom_Properties[(atName, hashings.property_hashings['solvenergy_props']['Occ'])]

        solvPCorrectionMask = usecontrat_Gt_0 & zero_denom.unsqueeze(-1) & (hashings.atom_Properties[
            (atName, hashings.property_hashings['hbond_params']['charged'])] != PADDING_INDEX).unsqueeze(-1) & \
            all_fake_with_occupancy & atomHbonds.eq(0) & ~(existingFakeAts.unsqueeze(1) &
            contRatPol.eq(0)).prod(-1).bool()

        usecontrat[~solvPCorrectionMask] = 0
        expandedAt_names = atName.unsqueeze(1).expand(solvPCorrectionMask.shape)[solvPCorrectionMask]
        val = ((usecontrat[solvPCorrectionMask] - hashings.atom_Properties[(expandedAt_names,
            hashings.property_hashings['solvenergy_props']['Occ'])]) / (hashings.atom_Properties[(expandedAt_names,
            hashings.property_hashings['solvenergy_props']['Occmax'])] - hashings.atom_Properties[(expandedAt_names,
            hashings.property_hashings['solvenergy_props']['Occ'])])).clamp(min=EPS, max=2.0)

        faccPol[solvPCorrectionMask] = val
        return faccPol, solvPCorrectionMask

    def forward(self, atom_description, facc, contRat, contRatPol, atomHbonds, atomDisul):
        faccPol, solvPCorrectionMask = self.getFaccPol(contRatPol, contRat, atom_description, atomHbonds)
        atomEnergyP, atomEnergyH = self.bindToAtoms(atom_description, facc, faccPol, solvPCorrectionMask, atomHbonds, atomDisul)
        residueEnergyP = self.bindToResi(atomEnergyP, atom_description)
        residueEnergyH = self.bindToResi(atomEnergyH, atom_description)
        return residueEnergyP, residueEnergyH

    def getWeights(self):
        pass

    def getNumParams(self):
        p = []
        for i in self.parameters():
            p += list(i.data.cpu().numpy().flat)

        print('Number of parameters=', len(p))
