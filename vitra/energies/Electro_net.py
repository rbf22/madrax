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

import torch
import math
from torch import nn
from vitra.sources.globalVariables import *
from vitra.sources import math_utils, hashings
from vitra import dataStructures
import numpy as np
dielec = 8.8
constant = math.exp(-0.004314 * (TEMPERATURE - 273))
IonStrength = 0.05
random_coil = 0

class Electro_net(torch.nn.Module):
    
    def __init__(self = None, name = None, dev = None, backbone_atoms = None):
        self.name = name
        self.dev = dev
        self.backbone_atoms = backbone_atoms
        self.float_type = torch.float
        super(Electro_net, self).__init__()
        self.MAX_DISTANCE = 15
        self.MIN_LEN_HBOND = 3.5
        self.weight = torch.nn.Parameter(torch.tensor([0], dtype=self.float_type, device=self.dev))
        self.weight.requires_grad = True
        self.weightNonVirt = torch.nn.Parameter(torch.tensor([0], dtype=self.float_type, device=self.dev))
        self.weightNonVirt.requires_grad = True
        self.weightargStack = torch.nn.Parameter(torch.tensor([0], dtype=self.float_type, device=self.dev))
        self.weightargStack.requires_grad = True
        self.weightotherVirtuals = torch.nn.Parameter(torch.tensor([0], dtype=self.float_type, device=self.dev))
        self.weightotherVirtuals.requires_grad = True
        self.weightKon = torch.nn.Parameter(torch.tensor([0], dtype=self.float_type, device=self.dev))
        self.weightKon.requires_grad = True
        self.threshold28 = torch.nn.Threshold(2.8, 2.8)
        self.dipoled_dipoled_dist_threshold = 4
        self.charge_dipoled_dist_threshold = 6
        self.Ncp_charge = 0.5
        self.Ccp_charge = -0.5

    
    def net(self, coords, partners, atom_description, atomPairs, hbondNet, alternativeMask, calculate_helical_dipoles = (False,)):
        if calculate_helical_dipoles:
            (NcpMask, CcpMask) = dataStructures.getHelicalDipoles(atomPairs, hbondNet, atom_description, alternativeMask)
        else:
            NcpMask = torch.zeros(atomPairs.shape, torch.bool, self.dev, **('dtype', 'device'))
            CcpMask = torch.zeros(atomPairs.shape, torch.bool, self.dev, **('dtype', 'device'))
        distance_dipole = 1
        Ionic_corrected = 0.02 + IonStrength / 1.4
        atName1 = atom_description[(atomPairs[:, 0], hashings.atom_description_hash['at_name'])].long()
        atName2 = atom_description[(atomPairs[:, 1], hashings.atom_description_hash['at_name'])].long()
        RCMask1 = (atName1 == hashings.atom_hash['PHE']['RC']) | (atName1 == hashings.atom_hash['TYR']['RC'])
        RCMask2 = (atName2 == hashings.atom_hash['PHE']['RC']) | (atName2 == hashings.atom_hash['TYR']['RC'])
        argInvolved = (atName1 == hashings.atom_hash['ARG']['CZ']) & ((atName2 == hashings.atom_hash['ARG']['CZ']) | RCMask2) | (atName2 == hashings.atom_hash['ARG']['CZ']) & ((atName1 == hashings.atom_hash['ARG']['CZ']) | RCMask1)
        is_charged_mask = (hashings.atom_Properties[(atName1, hashings.property_hashings['hbond_params']['charged'])].eq(1) | 
                NcpMask[:, 0] | CcpMask[:, 0]) & (hashings.atom_Properties[(atName2, hashings.property_hashings['hbond_params']['charged'])].eq(1) | 
                NcpMask[:, 1] | CcpMask[:, 1]) | argInvolved
        atomPairs = atomPairs[is_charged_mask]
        NcpMask = NcpMask[is_charged_mask]
        CcpMask = CcpMask[is_charged_mask]
        argInvolved = argInvolved[is_charged_mask]
        distmat = torch.pairwise_distance(coords[atomPairs[:, 0]], coords[atomPairs[:, 1]])
        RCMask1 = RCMask1[is_charged_mask]
        RCMask2 = RCMask2[is_charged_mask]
        RCMask = torch.cat([RCMask1.unsqueeze(-1), RCMask2.unsqueeze(-1)], dim=-1)
        mask_long = distmat.le(self.MAX_DISTANCE)
        same_residue_mask = (atom_description[(atomPairs[:, 0], hashings.atom_description_hash['resnum'])] != atom_description[(atomPairs[:, 1], hashings.atom_description_hash['resnum'])]) | (atom_description[(atomPairs[:, 0], hashings.atom_description_hash['chain'])] != atom_description[(atomPairs[:, 1], hashings.atom_description_hash['chain'])])
        first_mask = mask_long & same_residue_mask
        argInvolved = argInvolved[first_mask]
        RCMask = RCMask[first_mask]
        distmat = distmat[first_mask]
        atomPairs = atomPairs[first_mask]
        NcpMask = NcpMask[first_mask]
        CcpMask = CcpMask[first_mask]
        atName = torch.cat([
            atName1[is_charged_mask][first_mask].unsqueeze(-1),
            atName2[is_charged_mask][first_mask].unsqueeze(-1)], dim=-1)
        charge_1 = hashings.atom_Properties[(atName[:, 0], hashings.property_hashings['hbond_params']['charge'])]
        charge_2 = hashings.atom_Properties[(atName[:, 1], hashings.property_hashings['hbond_params']['charge'])]
        virtual_1 = hashings.atom_Properties[(atName[:, 0], hashings.property_hashings['other_params']['virtual'])].eq(1)
        virtual_2 = hashings.atom_Properties[(atName[:, 1], hashings.property_hashings['other_params']['virtual'])].eq(1)
        hDipoleInvolved = (NcpMask | CcpMask).sum(-1).bool()
        charged_charged_mask = (charge_1 != 0) & (charge_2 != 0) & (charge_1 != PADDING_INDEX) & (charge_2 != PADDING_INDEX) & ~virtual_1 & ~virtual_2 & ~hDipoleInvolved
        full_mask = is_charged_mask
        full_mask[full_mask == True] = first_mask
        virtual_energy = torch.zeros(charge_1.shape, dtype=self.float_type, device=self.dev)
        charge_1Effective = charge_1[hDipoleInvolved].clone()
        charge_2Effective = charge_2[hDipoleInvolved].clone()
        NcpMask = NcpMask[hDipoleInvolved]
        CcpMask = CcpMask[hDipoleInvolved]
        charge_1Effective[NcpMask[:, 0]] = self.Ncp_charge
        charge_1Effective[CcpMask[:, 0]] = self.Ccp_charge
        charge_2Effective[NcpMask[:, 1]] = self.Ncp_charge
        charge_2Effective[CcpMask[:, 1]] = self.Ccp_charge
        dist_28 = distmat[hDipoleInvolved].clamp(min=2.8)
        distance_dipole = dist_28.clone()
        onlyOneIsDipole = ~((NcpMask[:, 0] | CcpMask[:, 0]) & (NcpMask[:, 1] | CcpMask[:, 1]))
        distance_dipole[onlyOneIsDipole] = distance_dipole[onlyOneIsDipole].clamp(min=4)
        distance_dipole[~onlyOneIsDipole] = distance_dipole[~onlyOneIsDipole] ** 2
        virtualK = torch.sqrt(200 * torch.abs(charge_1Effective * charge_2Effective) * Ionic_corrected / TEMPERATURE)
        dielec_value = dielec * dist_28
        helixDipoleEpss = 332 / (dielec_value * constant)
        helixDipoleEnergy = charge_1Effective * charge_2Effective * helixDipoleEpss * torch.exp(-dist_28 * virtualK) / distance_dipole
        virtual_energy[hDipoleInvolved] += helixDipoleEnergy
        if True in argInvolved:
            (ang, m) = math_utils.plane_angle(coords[atomPairs[argInvolved][:, 0]], partners[(atomPairs[argInvolved][:, 0], 0)], partners[(atomPairs[argInvolved][:, 0], 1)], coords[atomPairs[argInvolved][:, 1]], partners[(atomPairs[argInvolved][:, 1], 0)], partners[(atomPairs[argInvolved][:, 1], 1)])
            ang = ang.squeeze(-1)
            m = m.squeeze(-1)
            ang_mask = ~((ang > np.radians(30)) & (ang < np.radians(150)) | (distmat[argInvolved] < 3.9) | (distmat[argInvolved] > 4.3) | ~m)
            argInvolved[argInvolved == True] = ang_mask
            dielec_arg = distmat[argInvolved] * dielec
            argEpss = 332 / (dielec_arg * constant)
            temp_distance = 3 + abs(distmat[argInvolved] - 4.1)
            corr_ang = torch.ones(distmat[argInvolved].shape, dtype=torch.float, device=self.dev)
            angCorrMask = distmat[argInvolved] > np.radians(150)
            rads_const = np.radians(10)
            corr_ang[angCorrMask] = ((np.radians(180) - ang[ang_mask][angCorrMask]) / rads_const).clamp(min=1)
            corr_ang[~angCorrMask] = (distmat[argInvolved][~angCorrMask] / rads_const).clamp(1, **('min',))
            argNames = atName[argInvolved]
            arg_arg_mask = (argNames[:, 0] == hashings.atom_hash['ARG']['ARG']) & (argNames[:, 1] == hashings.atom_hash['ARG']['ARG'])
            charge_1Effective = torch.zeros(argNames.shape[0], dtype=torch.float, device=self.dev)
            charge_2Effective = torch.zeros(argNames.shape[0], dtype=torch.float, device=self.dev)
            charge_1Effective[arg_arg_mask] = 1
            charge_2Effective[arg_arg_mask] = -1
            charge_1Effective[~arg_arg_mask] = 1
            charge_2Effective[~arg_arg_mask] = -0.5
            virtualK = torch.sqrt(200 * torch.abs(charge_1Effective * charge_2Effective) * Ionic_corrected / TEMPERATURE)
            argStackEnergy = (1 / corr_ang) * charge_1Effective * charge_2Effective * argEpss * torch.exp(-temp_distance * virtualK) / (temp_distance * temp_distance)
            virtual_energy[argInvolved] += argStackEnergy
        kon_approach = 6
        solv_dielec = 88
        kon_maks = charged_charged_mask.clone()
        initial_kon_maks = (hashings.atom_Properties[(atName[:, 0], hashings.property_hashings['hbond_params']['dipole'])] == 0) & (hashings.atom_Properties[(atName[:, 1], hashings.property_hashings['hbond_params']['dipole'])] == 0) & (atom_description[(atomPairs[:, 0], hashings.atom_description_hash['chain'])] != atom_description[(atomPairs[:, 1], hashings.atom_description_hash['chain'])])
        kon_maks = kon_maks & initial_kon_maks
        konNames = atName[kon_maks]
        sum_radii = hashings.atom_Properties[(konNames[:, 0], hashings.property_hashings['solvenergy_props']['radius'])] + hashings.atom_Properties[(konNames[:, 1], hashings.property_hashings['solvenergy_props']['radius'])] - 0.09
        dists = distmat[kon_maks]
        dists[dists < sum_radii] = sum_radii[dists < sum_radii]
        temp_distance = (dists + dists ** 2 / 30).clamp(min=kon_approach)
        IONSTRENGTH = 0.05
        k_on = torch.sqrt(200 * charge_1[kon_maks].abs() * charge_2[kon_maks].abs() * IONSTRENGTH / TEMPERATURE)
        firstNumber = 332 * charge_1[kon_maks] * charge_2[kon_maks] / (solv_dielec * temp_distance * constant)
        secondNumber = torch.exp(-k_on * (temp_distance - kon_approach))
        thirdSecond = 1 + k_on * kon_approach
        energy_kon = 0.5 * firstNumber * secondNumber / thirdSecond
        virtual_energy[kon_maks] += energy_kon
        atName = atName[charged_charged_mask]
        atomPairs = atomPairs[charged_charged_mask]
        distmat = distmat[charged_charged_mask]
        charge_1 = charge_1[charged_charged_mask]
        charge_2 = charge_2[charged_charged_mask]
        sum_radii = hashings.atom_Properties[(atName[:, 0], hashings.property_hashings['solvenergy_props']['radius'])] + hashings.atom_Properties[(atName[:, 1], hashings.property_hashings['solvenergy_props']['radius'])] - 0.09
        lower_than_sum_radii = sum_radii > distmat
        lower_than_minHbondLen = (distmat < self.MIN_LEN_HBOND) & (charge_1 * charge_2).le(0)
        distmat[lower_than_sum_radii] = sum_radii[lower_than_sum_radii]
        distmat[lower_than_minHbondLen] = self.MIN_LEN_HBOND
        K = torch.sqrt(200 * torch.abs(charge_1) * torch.abs(charge_2) * Ionic_corrected / TEMPERATURE)
        dielec_value = dielec * distmat
        Epss = 332 / (dielec_value * constant).squeeze(-1)
        finalNONVirtual = (charge_1 * charge_2 * Epss * torch.exp(-distmat.squeeze(-1) * K) / distmat.squeeze(-1) - random_coil) * (1 - torch.tanh(-(self.weightNonVirt)))
        virtual_energy[charged_charged_mask] += finalNONVirtual
        return (finalNONVirtual, atomPairs)

    
    def forward(self, coords, partners, atom_description, atom_number, atomPairs, alternativeMask, hbondNet, facc):
        (networkEnergy, networkPairs) = self.net(coords, partners, atom_description, atomPairs, hbondNet, alternativeMask)
        atomEnergy = self.bindToAtoms(networkEnergy, networkPairs, alternativeMask)
        (residueEnergyMC, residueEnergySC) = self.bindToResi(atomEnergy, atom_description, facc, alternativeMask)
        return (residueEnergyMC, residueEnergySC, atomEnergy, networkEnergy)

    
    def bindToResi(self, atomEnergy, atom_description, facc, alternativeMask, minSaCoefficient = (0.3,)):
        batch_ind = atom_description[:, hashings.atom_description_hash['batch']].long()
        resnum = atom_description[:, hashings.atom_description_hash['resnum']].long()
        chain_ind = atom_description[:, hashings.atom_description_hash['chain']].long()
        alt_index = torch.arange(0, alternativeMask.shape[-1], dtype=torch.long, device=self.dev).unsqueeze(0).expand(atomEnergy.shape[0], -1)
        atName = atom_description[:, hashings.atom_description_hash['at_name']]
        for i, bb_atom in enumerate(self.backbone_atoms):
            if i == 0:
                is_backbone_mask = atName.eq(bb_atom)
                continue
            is_backbone_mask += atName.eq(bb_atom)
        batch = batch_ind.max() + 1
        nres = torch.max(resnum) + 1
        nchains = chain_ind.max() + 1
        naltern = alternativeMask.shape[-1]
        saCoefficient = torch.max(facc, torch.tensor(minSaCoefficient,  dtype=torch.float, device=self.dev))
        atomEnergy = atomEnergy * saCoefficient * (1 - torch.tanh(-(self.weight)))
        batch_ind = batch_ind.unsqueeze(-1).expand(-1, naltern)
        chain_ind = chain_ind.unsqueeze(-1).expand(-1, naltern)
        resnum = resnum.unsqueeze(-1).expand(-1, naltern)
        mask_padding = ~resnum.eq(PADDING_INDEX)
        is_backbone_mask = is_backbone_mask.unsqueeze(-1).expand(-1, naltern) & mask_padding
        finalMC = torch.zeros((batch, nchains, nres, naltern), dtype=torch.float, device=self.dev)
        indices = (batch_ind[is_backbone_mask], chain_ind[is_backbone_mask], resnum[is_backbone_mask], alt_index[is_backbone_mask])
        finalMC.index_put_(indices, atomEnergy[is_backbone_mask], accumulate=True)
        nd1 = atName.eq(hashings.atom_hash['HIS']['ND1'])
        ne2 = atName.eq(hashings.atom_hash['HIS']['NE2'])
        his = (nd1 | ne2).unsqueeze(-1).expand(-1, naltern)
        nd1H1S = atName.eq(hashings.atom_hash['HIS']['ND1H1S'])
        ne2H1S = atName.eq(hashings.atom_hash['HIS']['NE2H1S'])
        h1s = (nd1H1S | ne2H1S).unsqueeze(-1).expand(-1, naltern)
        nd1H2S = atName.eq(hashings.atom_hash['HIS']['ND1H2S'])
        ne2H2S = atName.eq(hashings.atom_hash['HIS']['NE2H2S'])
        h2s = (nd1H2S | ne2H2S).unsqueeze(-1).expand(-1, naltern)
        padding_mask = ~resnum.eq(PADDING_INDEX) & ~is_backbone_mask
        final_his = torch.zeros((batch, nchains, nres, naltern),  dtype=torch.float, device=self.dev)
        his_mask = ~h1s & ~h2s
        indices = (batch_ind[padding_mask & his_mask], chain_ind[padding_mask & his_mask], resnum[padding_mask & his_mask].long(), alt_index[padding_mask & his_mask])
        final_his = final_his.index_put(indices, atomEnergy[padding_mask & his_mask], accumulate=True).unsqueeze(-1)
        final_h1s = torch.zeros((batch, nchains, nres, naltern),  dtype=torch.float, device=self.dev)
        his_mask = ~his & ~h2s
        indices = (batch_ind[padding_mask & his_mask], chain_ind[padding_mask & his_mask], resnum[padding_mask & his_mask].long(), alt_index[padding_mask & his_mask])
        final_h1s = final_h1s.index_put(indices, atomEnergy[padding_mask & his_mask], accumulate=True).unsqueeze(-1)
        final_h2s = torch.zeros((batch, nchains, nres, naltern), dtype=torch.float, device=self.dev)
        his_mask = ~h1s & ~his
        indices = (batch_ind[padding_mask & his_mask], chain_ind[padding_mask & his_mask], resnum[padding_mask & his_mask].long(), alt_index[padding_mask & his_mask])
        final_h2s = final_h2s.index_put(indices, atomEnergy[padding_mask & his_mask], accumulate=True).unsqueeze(-1)
        finalSC = torch.cat([
            final_his,
            final_h1s,
            final_h2s], dim=-1)
        return (finalMC, finalSC)

    
    def bindToAtoms(self, networkEnergy, networkPairs, alternMask):
        n_atoms = alternMask.shape[0]
        n_alter = alternMask.shape[-1]
        energy_atoms = torch.zeros((n_atoms, n_alter), dtype=torch.float, device=self.dev)
        for alt in range(alternMask.shape[-1]):
            mask = alternMask[(networkPairs[:, 0], alt)] & alternMask[(networkPairs[:, 1], alt)]
            atom_number_pw1Alt = networkPairs[:, 0][mask]
            atom_number_pw2Alt = networkPairs[:, 1][mask]
            alt_index = torch.full(mask.shape, alt, device=self.dev, dtype=torch.long)[mask]
            indices1 = (atom_number_pw1Alt, alt_index)
            energy_atoms = energy_atoms.index_put(indices1, networkEnergy[mask] * 0.5, accumulate=True)
            indices2 = (atom_number_pw2Alt, alt_index)
            energy_atoms = energy_atoms.index_put(indices2, networkEnergy[mask] * 0.5, accumulate=True)
        return energy_atoms

    __classcell__ = None

