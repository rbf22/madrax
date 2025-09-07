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
"""This module contains the HBondNet class, which is responsible for calculating hydrogen bond energies."""
import torch
import math
import numpy as np
from torch import nn
from vitra.sources import hashings
from vitra.sources import math_utils
from vitra.sources import select_best_virtualAtoms
from vitra.sources.globalVariables import PADDING_INDEX, MISSING_ATOM, NON_ACCEPTABLE_ENERGY, TEMPERATURE
IonStrength = 0.05
temperature = 298
dielec = 8.8
constant = math.exp(-0.004314 * temperature)
random_coil = 0

class HBondNet(torch.nn.Module):
    """
    This class calculates the hydrogen bond energy of a protein.
    """

    def __init__(self, name='BackHbondEnergy', dev='cpu', backbone_atoms=None, donor=None, acceptor=None, hbond_ar=None):
        if hbond_ar is None:
            hbond_ar = []
        if acceptor is None:
            acceptor = []
        if donor is None:
            donor = []
        if backbone_atoms is None:
            backbone_atoms = []
        self.name = name
        self.backbone_atoms = backbone_atoms
        self.dev = dev
        self.donor = donor
        self.acceptor = acceptor
        self.hbond_ar = hbond_ar
        self.float_type = torch.float
        unique_hbond_atoms = list(set(self.acceptor + self.donor))
        self.rehashing = {}
        for i in range(len(unique_hbond_atoms)):
            self.rehashing[unique_hbond_atoms[i]] = i

        virtual_proton_holders = {'CYS':[
          'SG'], 
         'LYS':[
          'NZ'], 
         'SER':[
          'OG'], 
         'THR':[
          'OG1'], 
         'TYR':[
          'OH']}
        self.virtual_proton_holders = []
        for i in virtual_proton_holders.keys():
            for k in virtual_proton_holders[i]:
                self.virtual_proton_holders += [hashings.atom_hash[i][k]]

        self.virtual_proton_holders = list(set(self.virtual_proton_holders))
        has_proton_states = [
         'HIS']
        self.has_proton_states = []
        for i in has_proton_states:
            self.has_proton_states += [hashings.resi_hash[i]]

        self.has_proton_states = list(set(self.has_proton_states))
        has_moving_protons = [
         'CYS', 'LYS', 'SER', 'THR', 'TYR']
        self.has_moving_protons = []
        for i in has_moving_protons:
            self.has_moving_protons += [hashings.resi_hash[i]]

        self.has_moving_protons = list(set(self.has_moving_protons))
        alcholic_groups = [
         'SER', 'TYR', 'THR', 'CYS']
        self.alcholic_groups = []
        for i in alcholic_groups:
            self.alcholic_groups += [hashings.resi_hash[i]]

        self.alcholic_groups = list(set(self.alcholic_groups))
        super(HBondNet, self).__init__()
        self.relu = nn.ReLU()
        self.MAX_H_DIST = 4.5
        self.MAX_DON_ACC_DIST = 3.5
        self.weight = torch.zeros(1, device=dev).float()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.sulfurAt = hashings.atom_hash['CYS']['SG']
        self.value_salt_bridge = -0.88
        self.value_neutral_hbond = -1.32
        self.value_charged_hbond = -1.87
        self.value_double_hbond = -0.825
        self.value_sulfur_hbond = -0.33

    def get_angle_penaltyKDE(self, interaction_angles, masks):
        penalties = torch.zeros((interaction_angles.shape[:-1]), dtype=(self.float_type), device=(self.dev))
        pad = (interaction_angles != PADDING_INDEX).sum(-1).bool()
        for i, m in enumerate(masks):
            if True not in m:
                continue
            m = m.unsqueeze(-1).repeat(1, 36) & pad
            angles_penalities = 1 - self.kde[i].score_samples(interaction_angles[m])
            penalties[m] = angles_penalities

        return penalties

    def get_angle_penalty(self, interaction_angles, minAng, optMin, optMax, maxAng):
        protFreeAccPenality = torch.zeros(interaction_angles.shape[:-1]).type_as(interaction_angles)
        minThanOpt = interaction_angles[:, :, 1] < optMin[:, :, 1]
        protFreeAccPenality[minThanOpt] = (optMin[minThanOpt][:, 1] - interaction_angles[:, :, 1][minThanOpt]) / (optMin[minThanOpt][:, 1] - minAng[minThanOpt][:, 1])
        maxThanOpt = interaction_angles[:, :, 1] > optMax[:, :, 1]
        protFreeAccPenality[maxThanOpt] = (interaction_angles[:, :, 1][maxThanOpt] - optMax[maxThanOpt][:, 1]) / (maxAng[maxThanOpt][:, 1] - optMax[maxThanOpt][:, 1])
        FreeProtDonPenality = torch.zeros(interaction_angles.shape[:-1]).type_as(interaction_angles)
        minThanOpt = interaction_angles[:, :, 0] < optMin[:, :, 0]
        FreeProtDonPenality[minThanOpt] = (optMin[minThanOpt][:, 0] - interaction_angles[:, :, 0][minThanOpt]) / (optMin[minThanOpt][:, 0] - minAng[minThanOpt][:, 0])
        maxThanOpt = interaction_angles[:, :, 0] > optMax[:, :, 0]
        FreeProtDonPenality[maxThanOpt] = (interaction_angles[:, :, 0][maxThanOpt] - optMax[maxThanOpt][:, 0]) / (maxAng[maxThanOpt][:, 0] - optMax[maxThanOpt][:, 0])
        DihedPenality = torch.zeros(interaction_angles.shape[:-1]).type_as(interaction_angles)
        maxThanOpt = torch.abs(interaction_angles[:, :, 2]) < optMin[:, :, 2]
        DihedPenality[maxThanOpt] = (optMin[maxThanOpt][:, 2] - torch.abs(interaction_angles[:, :, 2][maxThanOpt])) / (optMin[maxThanOpt][:, 2] - minAng[maxThanOpt][:, 2])
        angles_penalities = protFreeAccPenality / 3.0 + FreeProtDonPenality / 5.0 + DihedPenality / 4.0
        return angles_penalities

    def get_angles_for_HbondTrain(self, distmat, pairwise_coords_total, fake_atoms, pairwise_atom_description, pairwise_atom_props, fakeatom_rot, disul_atom, batch_pw, atom_number_pw):
        n_H = 12
        n_FO = 3
        disul_threshold = -0.1
        orbitals_pw = fake_atoms[:, n_H:, 0, :]
        hidrogen_pw = fake_atoms[:, :n_H, 1, :]
        pairwise_atom_name = pairwise_atom_description[:, (self.atname1_ha, self.atname2_ha)]
        residue_number = pairwise_atom_description[:, (self.resnum1_ha, self.resnum2_ha)]
        for i, bb_atom in enumerate(self.backbone_atoms):
            if i == 0:
                is_backbone_mask = pairwise_atom_name.eq(bb_atom)
            else:
                is_backbone_mask += pairwise_atom_name.eq(bb_atom)

        for i, donor_atom in enumerate(self.donor):
            if i == 0:
                donor_mask = pairwise_atom_name.eq(donor_atom)
            else:
                donor_mask += pairwise_atom_name.eq(donor_atom)

        for i, acceptor_atom in enumerate(self.acceptor):
            if i == 0:
                acceptor_mask = pairwise_atom_name.eq(acceptor_atom)
            else:
                acceptor_mask += pairwise_atom_name.eq(acceptor_atom)

        for i, aromatic_atom in enumerate(self.hbond_ar):
            if i == 0:
                aromatic_mask = pairwise_atom_name.eq(aromatic_atom)
            else:
                aromatic_mask += pairwise_atom_name.eq(aromatic_atom)

        donor_acceptor_mask = donor_mask[:, 1] & acceptor_mask[:, 0]
        mask_long = distmat.le(self.MAX_DON_ACC_DIST)
        is_disul = disul_atom.le(disul_threshold)
        same_residue_mask = torch.abs(residue_number[:, 0] - residue_number[:, 1]).ge(1) | ~pairwise_atom_description[:, self.chain1_ha].eq(pairwise_atom_description[:, self.chain2_ha]) | pairwise_atom_description[:, self.resname1_ha].eq(hashings.resi_hash['GLU']) | pairwise_atom_description[:, self.resname2_ha].eq(hashings.resi_hash['GLN'])
        peptide_and_bb_mask = ~(is_backbone_mask[:, 0] & is_backbone_mask[:, 1] & torch.abs(residue_number[:, 0] - residue_number[:, 1]).le(1))
        first_mask = mask_long & same_residue_mask & donor_acceptor_mask & peptide_and_bb_mask & ~is_disul
        masked_tot_coords = pairwise_coords_total[first_mask.squeeze(-1)]
        masked_hidrogen_pw = hidrogen_pw[first_mask.squeeze(-1)]
        masked_freeorb_pw = orbitals_pw[first_mask.squeeze(-1)]
        atom_number_pw = atom_number_pw[first_mask.squeeze(-1)]
        masked_atom_props = pairwise_atom_props[first_mask.squeeze(-1)]
        masked_atom_description = pairwise_atom_description[first_mask.squeeze(-1)]
        orbitals_pwRot = fakeatom_rot[:, n_H:, 0][first_mask]
        hidrogen_pwRot = fakeatom_rot[:, :n_H, 1][first_mask]
        minAng, maxAng, optMin, optMax = self.build_angle_boundaries(atom_number_pw, masked_atom_description, is_backbone_mask[first_mask.squeeze(-1)])
        hydrogens_pairwise_coords, freeOrbs_pairwise_coords, interaction_angles_pairwise, interaction_angles_fullmask, planeNanmask, test = select_best_virtualAtoms.do_pairwise_hbonds(masked_tot_coords[:, self.coords1_ha, :], masked_tot_coords[:, self.coords2_ha, :], masked_hidrogen_pw, masked_freeorb_pw, masked_tot_coords[:, self.part11_ha, :], masked_tot_coords[:, self.part12_ha, :], masked_tot_coords[:, self.part21_ha, :], masked_tot_coords[:, self.part22_ha, :], minAng, maxAng)
        orbitals_pwRot = orbitals_pwRot.unsqueeze(2).expand(orbitals_pwRot.shape[0], n_FO, n_H).reshape(orbitals_pwRot.shape[0], n_H * n_FO)
        hidrogen_pwRot = hidrogen_pwRot.unsqueeze(1).expand(hidrogen_pwRot.shape[0], n_FO, n_H).reshape(hidrogen_pwRot.shape[0], n_H * n_FO)
        nonexisting_HFO = (~(hydrogens_pairwise_coords[:, :, 0].eq(MISSING_ATOM) | freeOrbs_pairwise_coords[:, :, 0].eq(MISSING_ATOM))).view(-1, n_FO * n_H)
        second_mask = (interaction_angles_fullmask & nonexisting_HFO).sum(dim=1).type(first_mask.type())
        donor_multiH = masked_tot_coords[:, self.coords2_ha, :].unsqueeze(1).repeat(1, n_H * n_FO, 1)[second_mask]
        acceptor_multiH = masked_tot_coords[:, self.coords1_ha, :].unsqueeze(1).repeat(1, n_H * n_FO, 1)[second_mask]
        accP1_multiH = masked_tot_coords[:, self.part11_ha, :].unsqueeze(1).repeat(1, n_H * n_FO, 1)[second_mask]
        accP2_multiH = masked_tot_coords[:, self.part21_ha, :].unsqueeze(1).repeat(1, n_H * n_FO, 1)[second_mask]
        hydrogens_pairwise_coords = hydrogens_pairwise_coords[second_mask]
        freeOrbs_pairwise_coords = freeOrbs_pairwise_coords[second_mask]
        interaction_angles = interaction_angles_pairwise[second_mask]
        bad_angles_mask = interaction_angles_fullmask[second_mask]
        atom_number_pw = atom_number_pw[second_mask]
        masked_atom_description = masked_atom_description[second_mask]
        masked_atom_props = masked_atom_props[second_mask]
        hidrogen_pwRot = hidrogen_pwRot[second_mask]
        orbitals_pwRot = orbitals_pwRot[second_mask]
        minAng = minAng[second_mask]
        maxAng = maxAng[second_mask]
        optMin = optMin[second_mask]
        optMax = optMax[second_mask]
        acc_flat = acceptor_multiH.permute(2, 0, 1).reshape(3, -1).permute(1, 0)
        acc_pa1 = accP1_multiH.permute(2, 0, 1).reshape(3, -1).permute(1, 0)
        acc_pa2 = accP2_multiH.permute(2, 0, 1).reshape(3, -1).permute(1, 0)
        hpw_flat = hydrogens_pairwise_coords.permute(2, 0, 1).reshape(3, -1).permute(1, 0)
        distance_plane2H = math_utils.point_to_plane_dist(acc_flat, acc_pa1, acc_pa2, hpw_flat).view(-1, n_H * n_FO)
        dists_hyd = torch.norm((acceptor_multiH - hydrogens_pairwise_coords), dim=(-1))
        hybridation_mask = ~masked_atom_description[:, self.hybrid1_ha].eq(hashings.hashing_hybrid['SP2_O_ORB2']) & ~masked_atom_description[:, self.hybrid1_ha].eq(hashings.hashing_hybrid['SP2_O_ORB2'])
        plane_distance_mask = (distance_plane2H / dists_hyd - masked_atom_props[:, self.hbond_plane_dist1_ha].unsqueeze(1).expand(distance_plane2H.shape)).le(0)
        plane_Hybrid_Hmask = hybridation_mask.unsqueeze(-1).expand(plane_distance_mask.shape) | plane_distance_mask
        bad_angles_mask = bad_angles_mask & plane_Hybrid_Hmask
        third_mask = hybridization_mask | plane_distance_mask.sum(dim=(-1)).type(first_mask.type()) & bad_angles_mask.sum(dim=(-1)).type(first_mask.type())
        donor_multiH = donor_multiH[third_mask]
        acceptor_multiH = acceptor_multiH[third_mask]
        accP1_multiH = accP1_multiH[third_mask]
        accP2_multiH = accP2_multiH[third_mask]
        hydrogens_pairwise_coords = hydrogens_pairwise_coords[third_mask]
        freeOrbs_pairwise_coords = freeOrbs_pairwise_coords[third_mask]
        interaction_angles = interaction_angles[third_mask]
        hidrogen_pwRot = hidrogen_pwRot[third_mask]
        orbitals_pwRot = orbitals_pwRot[third_mask]
        bad_angles_mask = bad_angles_mask[third_mask]
        atom_number_pw = atom_number_pw[third_mask]
        masked_atom_description = masked_atom_description[third_mask]
        masked_atom_props = masked_atom_props[third_mask]
        minAng = minAng[third_mask]
        maxAng = maxAng[third_mask]
        optMin = optMin[third_mask]
        optMax = optMax[third_mask]
        full_mask = first_mask.clone()
        full_mask_clone = full_mask.clone()
        full_mask[full_mask_clone] = second_mask
        full_mask_clone = full_mask.clone()
        full_mask[full_mask_clone] = third_mask
        full_mask = full_mask.squeeze(-1)
        batch_pw = batch_pw[full_mask]
        minAng = minAng.unsqueeze(1).repeat(1, n_FO * n_H, 1)
        maxAng = maxAng.unsqueeze(1).repeat(1, n_FO * n_H, 1)
        optMin = optMin.unsqueeze(1).repeat(1, n_FO * n_H, 1)
        optMax = optMax.unsqueeze(1).repeat(1, n_FO * n_H, 1)
        angles_penalities = self.get_angle_penalty(interaction_angles, minAng, optMin, optMax, maxAng)
        masks = self.buildAngleMasks(masked_atom_description, is_backbone_mask[first_mask][second_mask][third_mask])
        angles_groups = []
        for i, m in enumerate(masks):
            if m.sum() == 0:
                angles_groups += []
            indicesH = angles_penalities[m].min(dim=1)[1]
            mask2 = torch.zeros((indicesH.shape[0], 36), dtype=(torch.bool), device=(self.dev))
            mask2.scatter_(1, indicesH.unsqueeze(1), 1)
            pad = (interaction_angles[m] != PADDING_INDEX).sum(-1).bool() & mask2
            angles_groups += [interaction_angles[m][pad]]

        return angles_groups

    def buildAngleMasks(self, masked_atom_description, is_backbone_mask):
        first = True
        for atomName in self.virtual_proton_holders:
            if first:
                donorVirtualHholders = masked_atom_description[:, self.atname2_ha].eq(atomName)
                first = False
            else:
                donorVirtualHholders += masked_atom_description[:, self.atname2_ha].eq(atomName)

        first = True
        for resName in self.has_moving_protons:
            if first:
                donorHasMovingProtons = masked_atom_description[:, self.resname2_ha].eq(resName)
                first = False
            else:
                donorHasMovingProtons += masked_atom_description[:, self.resname2_ha].eq(resName)

        first = True
        for residueName in self.alcholic_groups:
            if first:
                alcholic_groups = masked_atom_description[:, self.resname2_ha].eq(residueName)
                first = False
            else:
                alcholic_groups += masked_atom_description[:, self.resname2_ha].eq(residueName)

        firstIf = donorVirtualHholders & donorHasMovingProtons
        first = True
        for resName in self.has_proton_states:
            if first:
                donorHasprotonStates = masked_atom_description[:, self.resname2_ha].eq(resName)
                first = False
            else:
                donorHasprotonStates += masked_atom_description[:, self.resname2_ha].eq(resName)

        secondIf = ~is_backbone_mask[:, 1] & ~firstIf & donorHasprotonStates
        first = True
        for atomName in self.virtual_proton_holders:
            if first:
                acceptorVirtualHholders = masked_atom_description[:, self.atname1_ha].eq(atomName)
                first = False
            else:
                acceptorVirtualHholders += masked_atom_description[:, self.atname1_ha].eq(atomName)

        first = True
        for resName in self.has_moving_protons:
            if first:
                acceptorHasMovingProtons = masked_atom_description[:, self.resname1_ha].eq(resName)
                first = False
            else:
                acceptorHasMovingProtons += masked_atom_description[:, self.resname1_ha].eq(resName)

        acceptorFirstIf = acceptorVirtualHholders & acceptorHasMovingProtons
        first = True
        for resName in self.has_proton_states:
            if first:
                AcceptorHas_protons = masked_atom_description[:, self.resname1_ha].eq(resName)
                first = False
            else:
                AcceptorHas_protons += masked_atom_description[:, self.resname1_ha].eq(resName)

        acceptorIsNotBB = ~is_backbone_mask[:, 0] & ~acceptorFirstIf & AcceptorHas_protons
        NOTacceptorFirstIf = ~acceptorFirstIf
        NOTfirstIf = ~firstIf
        NOTacceptorIsNotBB = ~acceptorIsNotBB
        NOTsecondIf = ~secondIf
        m = []
        m += [NOTacceptorFirstIf & NOTfirstIf & acceptorIsNotBB & secondIf]
        m += [NOTacceptorFirstIf & firstIf & acceptorIsNotBB & NOTsecondIf]
        m += [acceptorFirstIf & NOTfirstIf & NOTacceptorIsNotBB & secondIf]
        m += [acceptorFirstIf & firstIf & NOTacceptorIsNotBB & NOTsecondIf]
        m += [NOTacceptorFirstIf & NOTfirstIf & NOTacceptorIsNotBB & secondIf]
        m += [NOTacceptorFirstIf & NOTfirstIf & acceptorIsNotBB & NOTsecondIf]
        m += [NOTacceptorFirstIf & firstIf & NOTacceptorIsNotBB & NOTsecondIf]
        m += [acceptorFirstIf & NOTfirstIf & NOTacceptorIsNotBB & NOTsecondIf]
        m += [NOTacceptorFirstIf & NOTfirstIf & NOTacceptorIsNotBB & NOTsecondIf]
        return m

    def net(self, coords, atomPairs, atom_description, fakeAtoms, disulfide, partners):
        n_H = 12
        n_FO = 3
        distmat = torch.pairwise_distance(coords[atomPairs[:, 0]], coords[atomPairs[:, 1]])
        long_mask = distmat.le(self.MAX_DON_ACC_DIST)
        atomPairs = atomPairs[long_mask]
        atName1 = atom_description[(atomPairs[:, 0], hashings.atom_description_hash['at_name'])]
        atName2 = atom_description[(atomPairs[:, 1], hashings.atom_description_hash['at_name'])]
        atName = torch.cat([atName1.unsqueeze(-1), atName2.unsqueeze(-1)], dim=(-1))
        del atName1
        del atName2
        same_residue_mask = (atom_description[(atomPairs[:, 0], hashings.atom_description_hash['resnum'])] != atom_description[(atomPairs[:, 1], hashings.atom_description_hash['resnum'])]) | (atom_description[(atomPairs[:, 0], hashings.atom_description_hash['chain'])] != atom_description[(atomPairs[:, 1], hashings.atom_description_hash['chain'])])
        selfHbondMask = (atom_description[(atomPairs[:, 0], hashings.atom_description_hash['resname'])] == hashings.resi_hash['GLU']) | (atom_description[(atomPairs[:, 1], hashings.atom_description_hash['resname'])] == hashings.resi_hash['GLN'])
        for i, bb_atom in enumerate(self.backbone_atoms):
            if i == 0:
                is_backbone_mask = atName.eq(bb_atom)
            else:
                is_backbone_mask += atName.eq(bb_atom)

        peptide_and_bb_mask = ~(is_backbone_mask[:, 0] & is_backbone_mask[:, 1] & ~same_residue_mask)
        final_general_mask = (same_residue_mask | selfHbondMask) & peptide_and_bb_mask
        atomPairs = atomPairs[final_general_mask]
        is_backbone_mask = is_backbone_mask[final_general_mask]
        disulfide = disulfide[long_mask][final_general_mask]
        atName = atName[final_general_mask].long()
        full_mask = long_mask.clone()
        full_mask_clone = full_mask.clone()
        full_mask[full_mask_clone] = final_general_mask
        del peptide_and_bb_mask
        del selfHbondMask
        del long_mask
        for i, donor_atom in enumerate(self.donor):
            if i == 0:
                donor_mask = atName.eq(donor_atom)
            else:
                donor_mask += atName.eq(donor_atom)

        for i, acceptor_atom in enumerate(self.acceptor):
            if i == 0:
                acceptor_mask = atName.eq(acceptor_atom)
            else:
                acceptor_mask += atName.eq(acceptor_atom)

        for i, aromatic_atom in enumerate(self.hbond_ar):
            if i == 0:
                aromatic_mask = atName.eq(aromatic_atom)
            else:
                aromatic_mask += atName.eq(aromatic_atom)

        donor_acceptor_mask = donor_mask[:, 1] & acceptor_mask[:, 0]
        donor_aromatic_mask = donor_mask[:, 1] & aromatic_mask[:, 0]
        first_mask = donor_acceptor_mask & ~disulfide
        del same_residue_mask
        del disulfide
        del donor_aromatic_mask
        del donor_acceptor_mask
        minAng, maxAng, optMin, optMax = self.build_angle_boundaries(atomPairs[first_mask], atom_description, is_backbone_mask[first_mask])
        hydrogens_pairwise_coords, freeOrbs_pairwise_coords, interaction_angles_pairwise, interaction_angles_fullmask, planeNanmask = select_best_virtualAtoms.do_pairwise_hbonds(coords[atomPairs[:, 0][first_mask]], coords[atomPairs[:, 1][first_mask]], fakeAtoms[atomPairs[:, 1][first_mask]][:, :n_H], fakeAtoms[atomPairs[:, 0][first_mask]][:, n_H:], partners[(atomPairs[:, 0][first_mask], 0)], partners[(atomPairs[:, 1][first_mask], 0)], partners[(atomPairs[:, 0][first_mask], 1)], partners[(atomPairs[:, 1][first_mask], 1)], minAng, maxAng)
        orbitals_pwRot = hashings.fake_atom_Properties[atName[first_mask][:, 0], n_H:, hashings.property_hashingsFake['rotation']]
        hidrogen_pwRot = hashings.fake_atom_Properties[atName[first_mask][:, 1], :n_H, hashings.property_hashingsFake['rotation']]
        orbitals_pwRot = orbitals_pwRot.unsqueeze(2).expand(orbitals_pwRot.shape[0], n_FO, n_H).reshape(orbitals_pwRot.shape[0], n_H * n_FO)
        hidrogen_pwRot = hidrogen_pwRot.unsqueeze(1).expand(hidrogen_pwRot.shape[0], n_FO, n_H).reshape(hidrogen_pwRot.shape[0], n_H * n_FO)
        nonexisting_HFO = (~(hydrogens_pairwise_coords[:, :, 0].eq(PADDING_INDEX) | freeOrbs_pairwise_coords[:, :, 0].eq(PADDING_INDEX))).view(-1, n_FO * n_H)
        second_mask = (interaction_angles_fullmask & nonexisting_HFO).sum(dim=1).type(first_mask.type())
        donor_multiH = coords[atomPairs[first_mask][:, 1]].unsqueeze(1).expand(-1, n_H * n_FO, -1)[second_mask]
        acceptor_multiH = coords[atomPairs[first_mask][:, 0]].unsqueeze(1).expand(-1, n_H * n_FO, -1)[second_mask]
        accP1_multiH = partners[(atomPairs[first_mask][:, 0], 0)].unsqueeze(1).expand(-1, n_H * n_FO, -1)[second_mask]
        accP2_multiH = partners[(atomPairs[first_mask][:, 0], 1)].unsqueeze(1).expand(-1, n_H * n_FO, -1)[second_mask]
        hydrogens_pairwise_coords = hydrogens_pairwise_coords[second_mask]
        freeOrbs_pairwise_coords = freeOrbs_pairwise_coords[second_mask]
        interaction_angles = interaction_angles_pairwise[second_mask]
        bad_angles_mask = interaction_angles_fullmask[second_mask]
        hidrogen_pwRot = hidrogen_pwRot[second_mask]
        orbitals_pwRot = orbitals_pwRot[second_mask]
        minAng = minAng[second_mask]
        maxAng = maxAng[second_mask]
        optMin = optMin[second_mask]
        optMax = optMax[second_mask]
        acc_flat = acceptor_multiH.permute(2, 0, 1).reshape(3, -1).permute(1, 0)
        acc_pa1 = accP1_multiH.permute(2, 0, 1).reshape(3, -1).permute(1, 0)
        acc_pa2 = accP2_multiH.permute(2, 0, 1).reshape(3, -1).permute(1, 0)
        hpw_flat = hydrogens_pairwise_coords.permute(2, 0, 1).reshape(3, -1).permute(1, 0)
        distance_plane2H = math_utils.point_to_plane_dist(acc_flat, acc_pa1, acc_pa2, hpw_flat).view(-1, n_H * n_FO)
        dists_hyd = torch.norm((acceptor_multiH - hydrogens_pairwise_coords), dim=(-1))
        hybridization_mask = ~((hashings.atom_Properties[(atName[first_mask][second_mask][:, 0], hashings.property_hashings['hbond_params']['hybridation'])] == hashings.hashing_hybrid['SP2_O_ORB2']) | (hashings.atom_Properties[(atName[first_mask][second_mask][:, 1], hashings.property_hashings['hbond_params']['hybridation'])] == hashings.hashing_hybrid['SP2_O_ORB2']))
        hbond_plane_dist = hashings.atom_Properties[(atName[first_mask][second_mask][:, 0], hashings.property_hashings['hbond_params']['hbond_plane_dist'])]
        plane_distance_mask = (distance_plane2H / dists_hyd - hbond_plane_dist.unsqueeze(1).expand(distance_plane2H.shape)).le(0)
        plane_Hybrid_Hmask = hybridization_mask.unsqueeze(-1).expand(plane_distance_mask.shape) | plane_distance_mask
        bad_angles_mask = bad_angles_mask & plane_Hybrid_Hmask
        third_mask = hybridization_mask | plane_distance_mask.sum(dim=(-1)).type(first_mask.type()) & bad_angles_mask.sum(dim=(-1)).type(first_mask.type())
        donor_multiH = donor_multiH[third_mask]
        acceptor_multiH = acceptor_multiH[third_mask]
        interaction_angles = interaction_angles[third_mask]
        hidrogen_pwRot = hidrogen_pwRot[third_mask]
        orbitals_pwRot = orbitals_pwRot[third_mask]
        bad_angles_mask = bad_angles_mask[third_mask]
        minAng = minAng[third_mask]
        maxAng = maxAng[third_mask]
        optMin = optMin[third_mask]
        optMax = optMax[third_mask]
        full_mask_clone = full_mask.clone()
        full_mask[full_mask_clone] = first_mask
        full_mask_clone = full_mask.clone()
        full_mask[full_mask_clone] = second_mask
        full_mask_clone = full_mask.clone()
        full_mask[full_mask_clone] = third_mask
        full_mask = full_mask.squeeze(-1)
        minAng = minAng.unsqueeze(1).repeat(1, n_FO * n_H, 1)
        maxAng = maxAng.unsqueeze(1).repeat(1, n_FO * n_H, 1)
        optMin = optMin.unsqueeze(1).repeat(1, n_FO * n_H, 1)
        optMax = optMax.unsqueeze(1).repeat(1, n_FO * n_H, 1)
        angles_penalities = self.get_angle_penalty(interaction_angles, minAng, optMin, optMax, maxAng)
        cd1 = 0.2
        cd1 = torch.full([angles_penalities.shape[0]], cd1, device=(self.dev))
        donorIsNotBB = ~is_backbone_mask[first_mask][second_mask][third_mask][:, 1]
        cd1[donorIsNotBB] = 0.1
        dists = torch.norm((donor_multiH - acceptor_multiH), dim=(-1))
        radius_distances = hashings.atom_Properties[(atName[first_mask][second_mask][third_mask][:, 0], hashings.property_hashings['solvenergy_props']['minradius'])] + hashings.atom_Properties[(atName[first_mask][second_mask][third_mask][:, 1], hashings.property_hashings['solvenergy_props']['minradius'])] + cd1
        distance_penality = self.relu(dists - radius_distances.unsqueeze(-1).expand(dists.shape))
        final_penality = distance_penality + angles_penalities
        out_energies = torch.masked_fill(final_penality, ~bad_angles_mask, NON_ACCEPTABLE_ENERGY)
        acceptable_energies = bad_angles_mask
        charged1 = hashings.atom_Properties[(atName[first_mask][second_mask][third_mask][:, 0], hashings.property_hashings['hbond_params']['charged'])]
        charged2 = hashings.atom_Properties[(atName[first_mask][second_mask][third_mask][:, 1], hashings.property_hashings['hbond_params']['charged'])]
        dipole1 = hashings.atom_Properties[(atName[first_mask][second_mask][third_mask][:, 0], hashings.property_hashings['hbond_params']['dipole'])]
        dipole2 = hashings.atom_Properties[(atName[first_mask][second_mask][third_mask][:, 1], hashings.property_hashings['hbond_params']['dipole'])]
        salt_bonds_mask = (charged1 == 1) & (charged2 == 1) & (dipole1 == 0) & (dipole2 == 0)
        Ion_corr = 0.02 + IonStrength / 1.4
        K = math.sqrt(50 * Ion_corr / temperature)
        constant = math.exp(-0.004314 * (temperature - 273.0))
        corr = 0.2 * (332.0 / (dielec * dists[salt_bonds_mask] * constant) * torch.exp(-dists[salt_bonds_mask] * K)) / dists[salt_bonds_mask]
        final_corr_salt = (1 - out_energies[salt_bonds_mask]) * corr
        one_charged_mask = charged1 != charged2
        dipoled_mask = charged1.eq(1) & dipole1.eq(1) & charged2.eq(0) | charged2.eq(1) & dipole2.eq(1) & charged1.eq(0)
        charged_hbond_mask = one_charged_mask & ~dipoled_mask
        final_corr_charged = 1 - out_energies[charged_hbond_mask]
        sulfur_mask = atName[first_mask][second_mask][third_mask][:, 0].eq(self.sulfurAt) | atName[first_mask][second_mask][third_mask][:, 1].eq(self.sulfurAt)
        final_corr_sulfur = 1 - out_energies[sulfur_mask]
        neutral_mask = ~(charged_hbond_mask | salt_bonds_mask | sulfur_mask)
        final_corr_Neutral = 1 - out_energies[neutral_mask]
        final_hbond_score = []
        final_indices = []
        atomPairsNoArom = atomPairs[first_mask][second_mask][third_mask]
        if final_corr_sulfur.shape[0] > 0:
            sulfurFakeatomsMask = acceptable_energies[sulfur_mask]
            out_energiesSulfur = final_corr_sulfur[sulfurFakeatomsMask]
            fakeIndexing = sulfurFakeatomsMask.view(-1, n_FO, n_H).nonzero()
            fo_index = fakeIndexing[:, 1].unsqueeze(-1)
            h_index = fakeIndexing[:, 2].unsqueeze(-1)
            hrot = hidrogen_pwRot[sulfur_mask][sulfurFakeatomsMask].unsqueeze(-1)
            forot = orbitals_pwRot[sulfur_mask][sulfurFakeatomsMask].unsqueeze(-1)
            r1 = atomPairsNoArom[:, 0][sulfur_mask].unsqueeze(-1).expand(sulfurFakeatomsMask.shape)[sulfurFakeatomsMask].unsqueeze(-1)
            r2 = atomPairsNoArom[:, 1][sulfur_mask].unsqueeze(-1).expand(sulfurFakeatomsMask.shape)[sulfurFakeatomsMask].unsqueeze(-1)
            indicesSulfur = torch.cat([r1, r2, fo_index, h_index, forot, hrot], dim=1)
            en = out_energiesSulfur * self.value_sulfur_hbond
            final_hbond_score += [en]
            final_indices += [indicesSulfur]
        if final_corr_salt.shape[0] > 0:
            saltFakeatomsMask = acceptable_energies[salt_bonds_mask]
            out_energiesSalt = final_corr_salt[saltFakeatomsMask]
            fakeIndexing = saltFakeatomsMask.view(-1, n_FO, n_H).nonzero()
            fo_index = fakeIndexing[:, 1].unsqueeze(-1)
            h_index = fakeIndexing[:, 2].unsqueeze(-1)
            hrot = hidrogen_pwRot[salt_bonds_mask][saltFakeatomsMask].unsqueeze(-1)
            forot = orbitals_pwRot[salt_bonds_mask][saltFakeatomsMask].unsqueeze(-1)
            r1 = atomPairsNoArom[:, 0][salt_bonds_mask].unsqueeze(-1).expand(saltFakeatomsMask.shape)[saltFakeatomsMask].unsqueeze(-1)
            r2 = atomPairsNoArom[:, 1][salt_bonds_mask].unsqueeze(-1).expand(saltFakeatomsMask.shape)[saltFakeatomsMask].unsqueeze(-1)
            indicesSalt = torch.cat([r1, r2, fo_index, h_index, forot, hrot], dim=1)
            final_hbond_score += [out_energiesSalt * self.value_salt_bridge]
            final_indices += [indicesSalt]
        if final_corr_charged.shape[0] > 0:
            chargedFakeatomsMask = acceptable_energies[charged_hbond_mask]
            out_energiesCharged = final_corr_charged[chargedFakeatomsMask]
            fakeIndexing = chargedFakeatomsMask.view(-1, n_FO, n_H).nonzero()
            fo_index = fakeIndexing[:, 1].unsqueeze(-1)
            h_index = fakeIndexing[:, 2].unsqueeze(-1)
            r1 = atomPairsNoArom[:, 0][charged_hbond_mask].unsqueeze(-1).expand(chargedFakeatomsMask.shape)[chargedFakeatomsMask].unsqueeze(-1)
            r2 = atomPairsNoArom[:, 1][charged_hbond_mask].unsqueeze(-1).expand(chargedFakeatomsMask.shape)[chargedFakeatomsMask].unsqueeze(-1)
            hrot = hidrogen_pwRot[charged_hbond_mask][chargedFakeatomsMask].unsqueeze(-1)
            forot = orbitals_pwRot[charged_hbond_mask][chargedFakeatomsMask].unsqueeze(-1)
            indicesCharged = torch.cat([r1, r2, fo_index, h_index, forot, hrot], dim=1)
            final_hbond_score += [out_energiesCharged * self.value_charged_hbond]
            final_indices += [indicesCharged]
        if final_corr_Neutral.shape[0] > 0:
            neutralFakeatomsMask = acceptable_energies[neutral_mask]
            out_energiesNeutrals = final_corr_Neutral[neutralFakeatomsMask]
            fakeIndexing = neutralFakeatomsMask.view(-1, n_FO, n_H).nonzero()
            fo_index = fakeIndexing[:, 1].unsqueeze(-1)
            h_index = fakeIndexing[:, 2].unsqueeze(-1)
            r1 = atomPairsNoArom[:, 0][neutral_mask].unsqueeze(-1).expand(neutralFakeatomsMask.shape)[neutralFakeatomsMask].unsqueeze(-1)
            r2 = atomPairsNoArom[:, 1][neutral_mask].unsqueeze(-1).expand(neutralFakeatomsMask.shape)[neutralFakeatomsMask].unsqueeze(-1)
            hrot = hidrogen_pwRot[neutral_mask][neutralFakeatomsMask].unsqueeze(-1)
            forot = orbitals_pwRot[neutral_mask][neutralFakeatomsMask].unsqueeze(-1)
            indicesNeutral = torch.cat([r1, r2, fo_index, h_index, forot, hrot], dim=1)
            en = out_energiesNeutrals * self.value_neutral_hbond
            final_hbond_score += [en]
            final_indices += [indicesNeutral]
        if len(final_indices) > 0:
            final_indices = torch.cat(final_indices).long()
            final_hbond_score = torch.cat(final_hbond_score)
            used = {}
            for i in range(len(final_indices)):
                key = tuple(final_indices[(i, (0, 1, 4, 5))].cpu().tolist())
                if key not in used:
                    used[key] = [
                     (
                      float(final_hbond_score[i].cpu().data), i)]
                else:
                    used[key] += [(float(final_hbond_score[i].cpu().data), i)]

            m = torch.ones((final_indices.shape[0]), dtype=(torch.bool), device=(self.dev))
            for i in used.keys():
                if len(used[i]) > 1:
                    s = sorted(used[i])
                    for k in range(1, len(used[i])):
                        m[s[k][1]] = False

            final_indices = final_indices[m]
            final_hbond_score = final_hbond_score[m]
        else:
            final_hbond_score = []
            final_indices = []
            return (
             final_hbond_score, final_indices, full_mask)

    def build_angle_boundariesNew(self, masked_atom_description, is_backbone_mask):
        dev = is_backbone_mask.device
        float_type = torch.float
        n_elements = masked_atom_description.shape[0]
        maxFreeProtDon = torch.full([n_elements], (np.radians(180.0)), device=dev, dtype=float_type)
        minFreeProtDon = torch.full([n_elements], (np.radians(115.0)), device=dev, dtype=float_type)
        maxProtFreeAcc = torch.full([n_elements], (np.radians(155.0)), device=dev, dtype=float_type)
        minProtFreeAcc = torch.full([n_elements], (np.radians(80.0)), device=dev, dtype=float_type)
        minDihed = torch.full([n_elements], (np.radians(80.0)), device=dev, dtype=float_type)
        optDihed = torch.full([n_elements], (np.radians(130.0)), device=dev, dtype=float_type)
        optMinFreeProtDon = torch.full([n_elements], (np.radians(145.0)), device=dev, dtype=float_type)
        optMaxFreeProtDon = torch.full([n_elements], (np.radians(160.0)), device=dev, dtype=float_type)
        optMinProtFreeAcc = torch.full([n_elements], (np.radians(90.0)), device=dev, dtype=float_type)
        optMaxProtFreeAcc = torch.full([n_elements], (np.radians(110.0)), device=dev, dtype=float_type)
        if len(self.has_proton_states) == 0:
            pass
        first = True
        for atomName in self.virtual_proton_holders:
            if first:
                donorVirtualHholders = masked_atom_description[:, self.atname2_ha].eq(atomName)
                first = False
            else:
                donorVirtualHholders += masked_atom_description[:, self.atname2_ha].eq(atomName)

        first = True
        for resName in self.has_moving_protons:
            if first:
                donorHasMovingProtons = masked_atom_description[:, self.resname2_ha].eq(resName)
                first = False
            else:
                donorHasMovingProtons += masked_atom_description[:, self.resname2_ha].eq(resName)

        first = True
        for residueName in self.alcholic_groups:
            if first:
                alcholic_groups = masked_atom_description[:, self.resname2_ha].eq(residueName)
                first = False
            else:
                alcholic_groups += masked_atom_description[:, self.resname2_ha].eq(residueName)

        maxProtFreeAccNewval = np.radians(180.0)
        minFreeProtDonNewval = np.radians(90.0)
        optMinFreeProtDonNewval = np.radians(120.0)
        optMaxFreeProtDonNewval = np.radians(165.0)
        optMinProtFreeAccNewval = np.radians(85.0)
        optMaxProtFreeAccNewval = np.radians(130.0)
        optDihedNewval = np.radians(120.0)
        firstIf = donorVirtualHholders & donorHasMovingProtons
        maxProtFreeAcc = torch.masked_fill(maxProtFreeAcc, firstIf, maxProtFreeAccNewval)
        minFreeProtDon = torch.masked_fill(minFreeProtDon, firstIf, minFreeProtDonNewval)
        optMinFreeProtDon = torch.masked_fill(optMinFreeProtDon, firstIf, optMinFreeProtDonNewval)
        optMaxFreeProtDon = torch.masked_fill(optMaxFreeProtDon, firstIf, optMaxFreeProtDonNewval)
        optMinProtFreeAcc = torch.masked_fill(optMinProtFreeAcc, firstIf, optMinProtFreeAccNewval)
        optMaxProtFreeAcc = torch.masked_fill(optMaxProtFreeAcc, firstIf, optMaxProtFreeAccNewval)
        optDihed = torch.masked_fill(optDihed, firstIf & ~alcholic_groups, optDihedNewval)
        optDihed = torch.masked_fill(optDihed, firstIf & alcholic_groups, np.radians(140.0))
        first = True
        for resName in self.has_proton_states:
            if first:
                donorHasprotonStates = masked_atom_description[:, self.resname2_ha].eq(resName)
                first = False
            else:
                donorHasprotonStates += masked_atom_description[:, self.resname2_ha].eq(resName)

        secondIf = ~is_backbone_mask[:, 1] & ~firstIf & donorHasprotonStates
        maxProtFreeAccNewval = np.radians(180.0)
        minFreeProtDonNewval = np.radians(110.0)
        optMinFreeProtDonNewval = np.radians(140.0)
        optMaxFreeProtDonNewval = np.radians(165.0)
        optMaxProtFreeAccNewval = np.radians(135.0)
        optDihedNewval = np.radians(140.0)
        maxProtFreeAcc = torch.masked_fill(maxProtFreeAcc, secondIf, maxProtFreeAccNewval)
        minFreeProtDon = torch.masked_fill(minFreeProtDon, secondIf, minFreeProtDonNewval)
        optMinFreeProtDon = torch.masked_fill(optMinFreeProtDon, secondIf, optMinFreeProtDonNewval)
        optMaxFreeProtDon = torch.masked_fill(optMaxFreeProtDon, secondIf, optMaxFreeProtDonNewval)
        optMaxProtFreeAcc = torch.masked_fill(optMaxProtFreeAcc, secondIf, optMaxProtFreeAccNewval)
        optDihed = torch.masked_fill(optDihed, secondIf, optDihedNewval)
        first = True
        for atomName in self.virtual_proton_holders:
            if first:
                acceptorVirtualHholders = masked_atom_description[:, self.atname1_ha].eq(atomName)
                first = False
            else:
                acceptorVirtualHholders += masked_atom_description[:, self.atname1_ha].eq(atomName)

        first = True
        for resName in self.has_moving_protons:
            if first:
                acceptorHasMovingProtons = masked_atom_description[:, self.resname1_ha].eq(resName)
                first = False
            else:
                acceptorHasMovingProtons += masked_atom_description[:, self.resname1_ha].eq(resName)

        maxProtFreeAccNewval = np.radians(180.0)
        optMinProtFreeAccNewval = np.radians(120.0)
        optMaxProtFreeAccNewval = np.radians(155.0)
        acceptorFirstIf = acceptorVirtualHholders & acceptorHasMovingProtons
        maxProtFreeAcc = torch.masked_fill(maxProtFreeAcc, acceptorFirstIf, maxProtFreeAccNewval)
        optMinProtFreeAcc = torch.masked_fill(optMinProtFreeAcc, acceptorFirstIf, optMinProtFreeAccNewval)
        optMaxProtFreeAcc = torch.masked_fill(optMaxProtFreeAcc, acceptorFirstIf, optMaxProtFreeAccNewval)
        first = True
        for resName in self.has_proton_states:
            if first:
                AcceptorHas_protons = masked_atom_description[:, self.resname1_ha].eq(resName)
                first = False
            else:
                AcceptorHas_protons += masked_atom_description[:, self.resname1_ha].eq(resName)

        acceptorIsNotBB = ~is_backbone_mask[:, 0] & ~acceptorFirstIf & AcceptorHas_protons
        maxProtFreeAccNewval = np.radians(180.0)
        optMinProtFreeAccNewval = np.radians(130.0)
        optMaxProtFreeAccNewval = np.radians(165.0)
        optDihedNewval = np.radians(155.0)
        maxProtFreeAcc = torch.masked_fill(maxProtFreeAcc, acceptorIsNotBB, maxProtFreeAccNewval)
        optMinProtFreeAcc = torch.masked_fill(optMinProtFreeAcc, acceptorIsNotBB, optMinProtFreeAccNewval)
        optMaxProtFreeAcc = torch.masked_fill(optMaxProtFreeAcc, acceptorIsNotBB, optMaxProtFreeAccNewval)
        optDihed = torch.masked_fill(optDihed, acceptorIsNotBB, optDihedNewval)
        optMax = torch.cat([optMaxFreeProtDon.unsqueeze(-1), optMaxProtFreeAcc.unsqueeze(-1)], dim=(-1))
        optMin = torch.cat([optMinFreeProtDon.unsqueeze(-1), optMinProtFreeAcc.unsqueeze(-1), optDihed.unsqueeze(-1)], dim=(-1))
        maxAng = torch.cat([maxFreeProtDon.unsqueeze(-1), maxProtFreeAcc.unsqueeze(-1)], dim=(-1))
        minAng = torch.cat([minFreeProtDon.unsqueeze(-1), minProtFreeAcc.unsqueeze(-1), minDihed.unsqueeze(-1)], dim=(-1))
        return (
         minAng, maxAng, optMin, optMax)

    def build_angle_boundaries(self, atom_pairs, atom_description, is_backbone_mask):
        atName1 = atom_description[(atom_pairs[:, 0], hashings.atom_description_hash['at_name'])]
        atName2 = atom_description[(atom_pairs[:, 1], hashings.atom_description_hash['at_name'])]
        resname1 = atom_description[(atom_pairs[:, 0], hashings.atom_description_hash['resname'])]
        resname2 = atom_description[(atom_pairs[:, 1], hashings.atom_description_hash['resname'])]
        dev = is_backbone_mask.device
        n_elements = atom_pairs.shape[0]
        float_type = torch.float
        maxFreeProtDon = torch.full([n_elements], (np.radians(180.0)), device=dev, dtype=float_type)
        minFreeProtDon = torch.full([n_elements], (np.radians(115.0)), device=dev, dtype=float_type)
        maxProtFreeAcc = torch.full([n_elements], (np.radians(155.0)), device=dev, dtype=float_type)
        minProtFreeAcc = torch.full([n_elements], (np.radians(80.0)), device=dev, dtype=float_type)
        minDihed = torch.full([n_elements], (np.radians(80.0)), device=dev, dtype=float_type)
        optDihed = torch.full([n_elements], (np.radians(130.0)), device=dev, dtype=float_type)
        optMinFreeProtDon = torch.full([n_elements], (np.radians(145.0)), device=dev, dtype=float_type)
        optMaxFreeProtDon = torch.full([n_elements], (np.radians(160.0)), device=dev, dtype=float_type)
        optMinProtFreeAcc = torch.full([n_elements], (np.radians(90.0)), device=dev, dtype=float_type)
        optMaxProtFreeAcc = torch.full([n_elements], (np.radians(110.0)), device=dev, dtype=float_type)
        if len(self.has_proton_states) == 0:
            pass
        first = True
        for atomName in self.virtual_proton_holders:
            if first:
                donorVirtualHholders = atName2.eq(atomName)
                first = False
            else:
                donorVirtualHholders += atName2.eq(atomName)

        first = True
        for resName in self.has_moving_protons:
            if first:
                donorHasMovingProtons = resname2.eq(resName)
                first = False
            else:
                donorHasMovingProtons += resname2.eq(resName)

        first = True
        for residueName in self.alcholic_groups:
            if first:
                alcholic_groups = resname2.eq(residueName)
                first = False
            else:
                alcholic_groups += resname2.eq(residueName)

        maxProtFreeAccNewval = np.radians(180.0)
        minFreeProtDonNewval = np.radians(90.0)
        optMinFreeProtDonNewval = np.radians(120.0)
        optMaxFreeProtDonNewval = np.radians(165.0)
        optMinProtFreeAccNewval = np.radians(85.0)
        optMaxProtFreeAccNewval = np.radians(130.0)
        optDihedNewval = np.radians(120.0)
        firstIf = donorVirtualHholders & donorHasMovingProtons
        maxProtFreeAcc = torch.masked_fill(maxProtFreeAcc, firstIf, maxProtFreeAccNewval)
        minFreeProtDon = torch.masked_fill(minFreeProtDon, firstIf, minFreeProtDonNewval)
        optMinFreeProtDon = torch.masked_fill(optMinFreeProtDon, firstIf, optMinFreeProtDonNewval)
        optMaxFreeProtDon = torch.masked_fill(optMaxFreeProtDon, firstIf, optMaxFreeProtDonNewval)
        optMinProtFreeAcc = torch.masked_fill(optMinProtFreeAcc, firstIf, optMinProtFreeAccNewval)
        optMaxProtFreeAcc = torch.masked_fill(optMaxProtFreeAcc, firstIf, optMaxProtFreeAccNewval)
        optDihed = torch.masked_fill(optDihed, firstIf & ~alcholic_groups, optDihedNewval)
        optDihed = torch.masked_fill(optDihed, firstIf & alcholic_groups, np.radians(140.0))
        first = True
        for resName in self.has_proton_states:
            if first:
                donorHasprotonStates = resname2.eq(resName)
                first = False
            else:
                donorHasprotonStates += resname2.eq(resName)

        secondIf = ~is_backbone_mask[:, 1] & ~firstIf & donorHasprotonStates
        maxProtFreeAccNewval = np.radians(180.0)
        minFreeProtDonNewval = np.radians(110.0)
        optMinFreeProtDonNewval = np.radians(140.0)
        optMaxFreeProtDonNewval = np.radians(165.0)
        optMaxProtFreeAccNewval = np.radians(135.0)
        optDihedNewval = np.radians(140.0)
        maxProtFreeAcc = torch.masked_fill(maxProtFreeAcc, secondIf, maxProtFreeAccNewval)
        minFreeProtDon = torch.masked_fill(minFreeProtDon, secondIf, minFreeProtDonNewval)
        optMinFreeProtDon = torch.masked_fill(optMinFreeProtDon, secondIf, optMinFreeProtDonNewval)
        optMaxFreeProtDon = torch.masked_fill(optMaxFreeProtDon, secondIf, optMaxFreeProtDonNewval)
        optMaxProtFreeAcc = torch.masked_fill(optMaxProtFreeAcc, secondIf, optMaxProtFreeAccNewval)
        optDihed = torch.masked_fill(optDihed, secondIf, optDihedNewval)
        first = True
        for atomName in self.virtual_proton_holders:
            if first:
                acceptorVirtualHholders = atName1.eq(atomName)
                first = False
            else:
                acceptorVirtualHholders += atName1.eq(atomName)

        first = True
        for resName in self.has_moving_protons:
            if first:
                acceptorHasMovingProtons = resname1.eq(resName)
                first = False
            else:
                acceptorHasMovingProtons += resname1.eq(resName)

        maxProtFreeAccNewval = np.radians(180.0)
        optMinProtFreeAccNewval = np.radians(120.0)
        optMaxProtFreeAccNewval = np.radians(155.0)
        acceptorFirstIf = acceptorVirtualHholders & acceptorHasMovingProtons
        maxProtFreeAcc = torch.masked_fill(maxProtFreeAcc, acceptorFirstIf, maxProtFreeAccNewval)
        optMinProtFreeAcc = torch.masked_fill(optMinProtFreeAcc, acceptorFirstIf, optMinProtFreeAccNewval)
        optMaxProtFreeAcc = torch.masked_fill(optMaxProtFreeAcc, acceptorFirstIf, optMaxProtFreeAccNewval)
        first = True
        for resName in self.has_proton_states:
            if first:
                AcceptorHas_protons = resname1.eq(resName)
                first = False
            else:
                AcceptorHas_protons += resname1.eq(resName)

        acceptorIsNotBB = ~is_backbone_mask[:, 0] & ~acceptorFirstIf & AcceptorHas_protons
        maxProtFreeAccNewval = np.radians(180.0)
        optMinProtFreeAccNewval = np.radians(130.0)
        optMaxProtFreeAccNewval = np.radians(165.0)
        optDihedNewval = np.radians(155.0)
        maxProtFreeAcc = torch.masked_fill(maxProtFreeAcc, acceptorIsNotBB, maxProtFreeAccNewval)
        optMinProtFreeAcc = torch.masked_fill(optMinProtFreeAcc, acceptorIsNotBB, optMinProtFreeAccNewval)
        optMaxProtFreeAcc = torch.masked_fill(optMaxProtFreeAcc, acceptorIsNotBB, optMaxProtFreeAccNewval)
        optDihed = torch.masked_fill(optDihed, acceptorIsNotBB, optDihedNewval)
        optMax = torch.cat([optMaxFreeProtDon.unsqueeze(-1), optMaxProtFreeAcc.unsqueeze(-1)], dim=(-1))
        optMin = torch.cat([optMinFreeProtDon.unsqueeze(-1), optMinProtFreeAcc.unsqueeze(-1), optDihed.unsqueeze(-1)], dim=(-1))
        maxAng = torch.cat([maxFreeProtDon.unsqueeze(-1), maxProtFreeAcc.unsqueeze(-1)], dim=(-1))
        minAng = torch.cat([minFreeProtDon.unsqueeze(-1), minProtFreeAcc.unsqueeze(-1), minDihed.unsqueeze(-1)], dim=(-1))
        return (
         minAng, maxAng, optMin, optMax)

    def forward(self, coords, atom_description, atom_number, atomPairs, fakeAtoms, alternativeMask, disulfide, partners, facc):
        final_hbond_score, final_indices, hbond_net = self.net(coords, atomPairs, atom_description, fakeAtoms, disulfide, partners)
        atomEnergy, sharedBonds = self.bindToAtoms(final_hbond_score, final_indices, atom_description, alternativeMask)
        residueEnergyMC, residueEnergySC = self.bindToResi(atomEnergy, sharedBonds, atom_description, alternativeMask, facc)
        return (
         residueEnergyMC, residueEnergySC, atomEnergy, hbond_net)

    def bindToAtomsNew(self, final_hbond_score, final_indices, atom_number_pw, hbond_net, batch_ind, rotationNumb, pairwise_fake_atoms, pairwise_fake_atomsRot, alternMask, contribution, prot_dims, isrotatingMask, fake_atom_rot, minSaCoefficient=0.3, maximumEnergy=-2.39):
        n_FO = 3
        n_H = 12
        n_rotFakeat = 5
        batch, L, altern, nrots = prot_dims
        hbondEnergy = torch.zeros((batch, L, altern, nrots, n_rotFakeat), dtype=(self.float_type), device=(self.dev))
        shared_bonds = []
        for alt in range(alternMask.shape[-1]):
            mask = alternMask[:, alt]
            alt_index = torch.full((mask.shape), alt, device=(self.dev), dtype=(torch.long))[mask]
            contrib = contribution[:, alt][mask]
            noneIsRotating = ~isrotatingMask[:, 0][mask] & ~isrotatingMask[:, 1][mask]
            allrotations = torch.arange(0, nrots, device=(self.dev)).unsqueeze(0).expand(noneIsRotating.shape[0], nrots)[noneIsRotating]
            final_indicesAlt = final_indices[mask]
            real_rot_index1 = final_indicesAlt[:, 7].clone()
            real_rot_index2 = final_indicesAlt[:, 8].clone()
            rm1 = ~isrotatingMask[:, 0][mask] & isrotatingMask[:, 1][mask]
            real_rot_index1[rm1] = real_rot_index2[rm1]
            rm2 = ~isrotatingMask[:, 1][mask] & isrotatingMask[:, 0][mask]
            real_rot_index2[rm2] = real_rot_index1[rm2]
            usedFO = torch.zeros(batch, L, nrots, n_FO, dtype=(torch.long), device=(self.dev))
            usedH = torch.zeros(batch, L, nrots, n_H, dtype=(torch.long), device=(self.dev))
            usedHrot = torch.zeros(batch, L, nrots, n_H, dtype=(torch.long), device=(self.dev))
            usedFOrot = torch.zeros(batch, L, nrots, n_FO, dtype=(torch.long), device=(self.dev))
            usedFOrot.index_put_((final_indicesAlt[:, 0],
             final_indicesAlt[:, 1],
             real_rot_index1,
             final_indicesAlt[:, 3]),
              (final_indicesAlt[:, 5]),
              accumulate=False)
            indices = (
             final_indicesAlt[:, 0][noneIsRotating].unsqueeze(-1).repeat(1, nrots)[:, 1:].long(),
             final_indicesAlt[:, 1].long()[noneIsRotating].unsqueeze(-1).repeat(1, nrots)[:, 1:],
             allrotations[:, 1:],
             final_indicesAlt[:, 3][noneIsRotating].unsqueeze(-1).repeat(1, nrots)[:, 1:])
            usedFOrot.index_put_(indices,
              (final_indicesAlt[:, 5][noneIsRotating].unsqueeze(-1).repeat(1, nrots)[:, 1:]),
              accumulate=False)
            usedHrot.index_put_((final_indicesAlt[:, 0], final_indicesAlt[:, 2], real_rot_index2, final_indicesAlt[:, 4]), (final_indicesAlt[:, 6]),
              accumulate=False)
            indices = (
             final_indicesAlt[:, 0][noneIsRotating].unsqueeze(-1).repeat(1, nrots)[:, 1:].long(),
             final_indicesAlt[:, 2].long()[noneIsRotating].unsqueeze(-1).repeat(1, nrots)[:, 1:],
             allrotations[:, 1:],
             final_indicesAlt[:, 4][noneIsRotating].unsqueeze(-1).repeat(1, nrots)[:, 1:])
            usedHrot.index_put_(indices,
              (final_indicesAlt[:, 6][noneIsRotating].unsqueeze(-1).repeat(1, nrots)[:, 1:]),
              accumulate=False)
            usedFO = usedFO.index_put_((final_indicesAlt[:, 0], final_indicesAlt[:, 1], real_rot_index1, final_indicesAlt[:, 3]), torch.ones((final_indicesAlt[:, 0].shape), dtype=(torch.long), device=(self.dev)),
              accumulate=True)
            indices = (
             final_indicesAlt[:, 0][noneIsRotating].unsqueeze(-1).repeat(1, nrots)[:, 1:].long(),
             final_indicesAlt[:, 1].long()[noneIsRotating].unsqueeze(-1).repeat(1, nrots)[:, 1:],
             allrotations[:, 1:],
             final_indicesAlt[:, 3][noneIsRotating].unsqueeze(-1).repeat(1, nrots)[:, 1:])
            usedFO.index_put_(indices,
              (torch.ones((final_indicesAlt[:, 0].shape), dtype=(torch.long), device=(self.dev))[noneIsRotating].unsqueeze(-1).repeat(1, nrots)[:, 1:]),
              accumulate=False)
            usedH = usedH.index_put_((final_indicesAlt[:, 0], final_indicesAlt[:, 2], real_rot_index2, final_indicesAlt[:, 4]), torch.ones((final_indicesAlt[:, 0].shape), dtype=(torch.long), device=(self.dev)),
              accumulate=True)
            indices = (
             final_indicesAlt[:, 0][noneIsRotating].unsqueeze(-1).repeat(1, nrots)[:, 1:].long(),
             final_indicesAlt[:, 2].long()[noneIsRotating].unsqueeze(-1).repeat(1, nrots)[:, 1:],
             allrotations[:, 1:],
             final_indicesAlt[:, 4][noneIsRotating].unsqueeze(-1).repeat(1, nrots)[:, 1:])
            usedH.index_put_(indices,
              (torch.ones((final_indicesAlt[:, 0].shape), dtype=(torch.long), device=(self.dev))[noneIsRotating].unsqueeze(-1).repeat(1, nrots)[:, 1:]),
              accumulate=False)
            hbondEnergy.index_put_((final_indicesAlt[:, 0],
             final_indicesAlt[:, 1],
             alt_index,
             real_rot_index1,
             final_indicesAlt[:, 5]),
              (final_hbond_score[mask] * 0.5 * contrib), accumulate=True)
            hbondEnergy.index_put_((final_indicesAlt[:, 0],
             final_indicesAlt[:, 2],
             alt_index,
             real_rot_index2,
             final_indicesAlt[:, 6]),
              (final_hbond_score[mask] * 0.5 * contrib), accumulate=True)
            if nrots != 1:
                indices = (
                 final_indicesAlt[:, 0][noneIsRotating].unsqueeze(-1).repeat(1, nrots)[:, 1:].long(),
                 final_indicesAlt[:, 1].long()[noneIsRotating].unsqueeze(-1).repeat(1, nrots)[:, 1:],
                 alt_index[noneIsRotating].unsqueeze(-1).repeat(1, nrots)[:, 1:],
                 allrotations[:, 1:],
                 final_indicesAlt[:, 5][noneIsRotating].unsqueeze(-1).repeat(1, nrots)[:, 1:])
                hbondEnergy.index_put_(indices, (final_hbond_score[mask][noneIsRotating].unsqueeze(-1).repeat(1, nrots)[:, 1:] * contrib[noneIsRotating].unsqueeze(-1).repeat(1, nrots)[:, 1:] * 0.5), accumulate=True)
                indices = (
                 final_indicesAlt[:, 0][noneIsRotating].unsqueeze(-1).repeat(1, nrots)[:, 1:].long(),
                 final_indicesAlt[:, 2].long()[noneIsRotating].unsqueeze(-1).repeat(1, nrots)[:, 1:],
                 alt_index[noneIsRotating].unsqueeze(-1).repeat(1, nrots)[:, 1:],
                 allrotations[:, 1:],
                 final_indicesAlt[:, 6][noneIsRotating].unsqueeze(-1).repeat(1, nrots)[:, 1:])
                hbondEnergy.index_put_(indices, (final_hbond_score[mask][noneIsRotating].unsqueeze(-1).repeat(1, nrots)[:, 1:] * contrib[noneIsRotating].unsqueeze(-1).repeat(1, nrots)[:, 1:] * 0.5), accumulate=True)

        energy_atoms, rotation = hbondEnergy.min(-1)
        availableH = torch.full((pairwise_fake_atomsRot.shape[0], alternMask.shape[-1], n_H + n_FO, 2), False, device=(self.dev))
        existing_H = ~pairwise_fake_atomsRot[:, :].eq(PADDING_INDEX)
        zero_energy = energy_atoms.eq(0.0)
        avaibleRots = torch.zeros(batch, L, (alternMask.shape[-1]), nrots, n_rotFakeat, dtype=(torch.bool), device=(self.dev))
        avaibleRotsNozer = avaibleRots[~zero_energy]
        avaibleRotsNozer.index_put_([torch.arange((rotation[~zero_energy].shape[0]), device=(self.dev)),
         rotation[~zero_energy]], torch.ones((rotation[~zero_energy].shape[0]), dtype=(torch.bool), device=(self.dev)))
        avaibleRots[~zero_energy] = avaibleRotsNozer
        maxRotFakeAtom = fake_atom_rot.max(-1)[0].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(batch, L, alternMask.shape[-1], nrots, n_rotFakeat)
        fakeRotInd = torch.arange(n_rotFakeat, device=(self.dev)).unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(batch, L, alternMask.shape[-1], nrots, n_rotFakeat)
        existing_rot = fakeRotInd.le(maxRotFakeAtom) & ~maxRotFakeAtom.eq(PADDING_INDEX)
        avaibleRots[zero_energy.unsqueeze(-1) & existing_rot] = True
        for alt in range(alternMask.shape[-1]):
            alt_index = torch.full((atom_number_pw.shape), alt, device=(self.dev), dtype=(torch.long))
            totalStuff = torch.cat([
             batch_ind.unsqueeze(1),
             atom_number_pw[:, 0].unsqueeze(1),
             alt_index[:, 0].unsqueeze(1),
             rotationNumb[:, 0].unsqueeze(1)],
              dim=1)
            goodRotA1 = rotation[(totalStuff[:, 0], totalStuff[:, 1], totalStuff[:, 2], totalStuff[:, 3])].unsqueeze(1).expand(-1, n_H + n_FO)
            zeroEnergyA1 = energy_atoms[(totalStuff[:, 0], totalStuff[:, 1], totalStuff[:, 2], totalStuff[:, 3])].unsqueeze(1).expand(-1, n_H + n_FO).eq(0) & existing_H[:, :, 0]
            totalStuff = torch.cat([
             batch_ind.unsqueeze(1),
             atom_number_pw[:, 1].unsqueeze(1),
             alt_index[:, 1].unsqueeze(1),
             rotationNumb[:, 1].unsqueeze(1)],
              dim=1)
            goodRotA2 = rotation[(totalStuff[:, 0], totalStuff[:, 1], totalStuff[:, 2], totalStuff[:, 3])].unsqueeze(1).expand(-1, n_H + n_FO)
            zeroEnergyA2 = energy_atoms[(totalStuff[:, 0], totalStuff[:, 1], totalStuff[:, 2], totalStuff[:, 3])].unsqueeze(1).expand(-1, n_H + n_FO).eq(0) & existing_H[:, :, 1]
            isGoodRotMask1 = (pairwise_fake_atomsRot[:, :, 0] == goodRotA1) | zeroEnergyA1
            isGoodRotMask2 = (pairwise_fake_atomsRot[:, :, 1] == goodRotA2) | zeroEnergyA2
            availableH[:, alt, :, 0][isGoodRotMask1] = True
            availableH[:, alt, :, 1][isGoodRotMask2] = True

        for alt in range(alternMask.shape[-1]):
            rotmaskFO = usedFOrot == rotation[:, :, alt].unsqueeze(-1).repeat(1, 1, 1, n_FO)
            usedFO = usedFO.masked_fill_(~rotmaskFO, 0)
            rotmaskH = usedHrot == rotation[:, :, alt].unsqueeze(-1).repeat(1, 1, 1, n_H)
            usedH = usedH.masked_fill_(~rotmaskH, 0)
            shared_bonds += [((usedFO - usedFO.clamp(max=1)).sum(-1) + (usedH - usedH.clamp(max=1)).sum(-1)).unsqueeze(2)]

        shared_bonds = torch.cat(shared_bonds, dim=2)
        return (energy_atoms / (shared_bonds + 1), shared_bonds, availableH, existing_rot)

    def bindToAtomsNew2(self, final_hbond_score, final_indices, pairwise_fake_atomsRot, alternMask, contribution, prot_dims, minSaCoefficient=0.3, maximumEnergy=-2.39):
        n_FO = 3
        n_H = 12
        n_rotFakeat = 5
        batch, L, altern, nrots = prot_dims
        hbondEnergy = torch.zeros((batch, L, altern, nrots, n_rotFakeat), dtype=(self.float_type), device=(self.dev))
        shared_bonds = []
        for alt in range(alternMask.shape[-1]):
            mask = alternMask[:, alt]
            alt_index = torch.full((mask.shape), alt, device=(self.dev), dtype=(torch.long))[mask]
            contrib = contribution[:, alt][mask]
            final_indicesAlt = final_indices[mask]
            usedFO = torch.zeros(batch, L, nrots, n_FO, dtype=(torch.long), device=(self.dev))
            usedH = torch.zeros(batch, L, nrots, n_H, dtype=(torch.long), device=(self.dev))
            usedHrot = torch.zeros(batch, L, nrots, n_H, dtype=(torch.long), device=(self.dev))
            usedFOrot = torch.zeros(batch, L, nrots, n_FO, dtype=(torch.long), device=(self.dev))
            usedFOrot.index_put_((final_indicesAlt[:, 0], final_indicesAlt[:, 1], final_indicesAlt[:, 7], final_indicesAlt[:, 3]), (final_indicesAlt[:, 5]),
              accumulate=False)
            usedHrot.index_put_((final_indicesAlt[:, 0], final_indicesAlt[:, 2], final_indicesAlt[:, 8], final_indicesAlt[:, 4]), (final_indicesAlt[:, 6]),
              accumulate=False)
            usedFO = usedFO.index_put_((final_indicesAlt[:, 0], final_indicesAlt[:, 1], final_indicesAlt[:, 7], final_indicesAlt[:, 3]), torch.ones((final_indicesAlt[:, 0].shape), dtype=(torch.long), device=(self.dev)),
              accumulate=True)
            usedH = usedH.index_put_((final_indicesAlt[:, 0], final_indicesAlt[:, 2], final_indicesAlt[:, 8], final_indicesAlt[:, 4]), torch.ones((final_indicesAlt[:, 0].shape), dtype=(torch.long), device=(self.dev)),
              accumulate=True)
            hbondEnergy.index_put_((final_indicesAlt[:, 0], final_indicesAlt[:, 1], alt_index, final_indicesAlt[:, 7], final_indicesAlt[:, 5]), (final_hbond_score[mask] * 0.5 * contrib), accumulate=True)
            hbondEnergy.index_put_((final_indicesAlt[:, 0], final_indicesAlt[:, 2], alt_index, final_indicesAlt[:, 8], final_indicesAlt[:, 6]), (final_hbond_score[mask] * 0.5 * contrib), accumulate=True)

        _, rotation = hbondEnergy.min(-1)
        shared_bonds = []
        hbondEnergy = torch.zeros((batch, L, altern, nrots, n_rotFakeat), dtype=(self.float_type), device=(self.dev))
        for alt in range(alternMask.shape[-1]):
            mask = alternMask[:, alt]
            contrib = contribution[:, alt][mask]
            final_indicesAlt = final_indices[mask]
            alt_index = torch.full((mask.shape), alt, device=(self.dev), dtype=(torch.long))[mask]
            rotMask = rotation[(final_indicesAlt[:, 0], final_indicesAlt[:, 1], alt_index, final_indicesAlt[:, 7])] == final_indicesAlt[:, 5]
            final_indicesAlt = final_indicesAlt[rotMask]
            alt_index = alt_index[rotMask]
            mask = mask & rotMask
            contrib = contrib[rotMask]
            usedFO = torch.zeros(batch, L, nrots, n_FO, dtype=(torch.long), device=(self.dev))
            usedH = torch.zeros(batch, L, nrots, n_H, dtype=(torch.long), device=(self.dev))
            usedHrot = torch.zeros(batch, L, nrots, n_H, dtype=(torch.long), device=(self.dev))
            usedFOrot = torch.zeros(batch, L, nrots, n_FO, dtype=(torch.long), device=(self.dev))
            usedFOrot.index_put_((final_indicesAlt[:, 0], final_indicesAlt[:, 1], final_indicesAlt[:, 7], final_indicesAlt[:, 3]), (final_indicesAlt[:, 5]),
              accumulate=False)
            usedHrot.index_put_((final_indicesAlt[:, 0], final_indicesAlt[:, 2], final_indicesAlt[:, 8], final_indicesAlt[:, 4]), (final_indicesAlt[:, 6]),
              accumulate=False)
            usedFO = usedFO.index_put_((final_indicesAlt[:, 0], final_indicesAlt[:, 1], final_indicesAlt[:, 7], final_indicesAlt[:, 3]), torch.ones((final_indicesAlt[:, 0].shape), dtype=(torch.long), device=(self.dev)),
              accumulate=True)
            usedH = usedH.index_put_((final_indicesAlt[:, 0], final_indicesAlt[:, 2], final_indicesAlt[:, 8], final_indicesAlt[:, 4]), torch.ones((final_indicesAlt[:, 0].shape), dtype=(torch.long), device=(self.dev)),
              accumulate=True)
            hbondEnergy.index_put_((final_indicesAlt[:, 0], final_indicesAlt[:, 1], alt_index, final_indicesAlt[:, 7], final_indicesAlt[:, 5]), (final_hbond_score[mask] * 0.5 * contrib), accumulate=True)
            hbondEnergy.index_put_((final_indicesAlt[:, 0], final_indicesAlt[:, 2], alt_index, final_indicesAlt[:, 8], final_indicesAlt[:, 6]), (final_hbond_score[mask] * 0.5 * contrib), accumulate=True)

        for alt in range(alternMask.shape[-1]):
            rotmaskFO = usedFOrot == rotation[:, :, alt].unsqueeze(-1).repeat(1, 1, 1, n_FO)
            usedFO = usedFO.masked_fill_(~rotmaskFO, 0)
            rotmaskH = usedHrot == rotation[:, :, alt].unsqueeze(-1).repeat(1, 1, 1, n_H)
            usedH = usedH.masked_fill_(~rotmaskH, 0)
            shared_bonds += [((usedFO - usedFO.clamp(max=1)).sum(-1) + (usedH - usedH.clamp(max=1)).sum(-1)).unsqueeze(2)]

        shared_bonds = torch.cat(shared_bonds, dim=2)
        return (
         hbondEnergy.sum(-1) / (shared_bonds + 1), shared_bonds)

    def bindToAtoms(self, final_hbond_score, final_indices, atom_description, alternMask, minSaCoefficient=0.3, maximumEnergy=-2.39):
        n_FO = 3
        n_H = 12
        n_rotFakeat = 5
        n_atoms = atom_description.shape[0]
        n_alter = alternMask.shape[-1]
        hbondEnergy = torch.zeros((n_atoms, n_alter, n_rotFakeat), dtype=(self.float_type), device=(self.dev))
        if len(final_hbond_score) == 0:
            return (
             hbondEnergy[:, :, 0], hbondEnergy[:, :, 0])
        for alt in range(alternMask.shape[-1]):
            mask = alternMask[(final_indices[:, 0], alt)] & alternMask[(final_indices[:, 1], alt)]
            alt_index = torch.full((mask.shape), alt, device=(self.dev), dtype=(torch.long))[mask]
            final_indicesAlt = final_indices[mask]
            usedFO = torch.zeros(n_atoms, n_FO, dtype=(torch.long), device=(self.dev))
            usedH = torch.zeros(n_atoms, n_H, dtype=(torch.long), device=(self.dev))
            usedHrot = torch.zeros(n_atoms, n_H, dtype=(torch.long), device=(self.dev))
            usedFOrot = torch.zeros(n_atoms, n_FO, dtype=(torch.long), device=(self.dev))
            usedFOrot.index_put_((final_indicesAlt[:, 0],
             final_indicesAlt[:, 2]),
              (final_indicesAlt[:, 4]),
              accumulate=False)
            usedHrot.index_put_((final_indicesAlt[:, 1],
             final_indicesAlt[:, 3]),
              (final_indicesAlt[:, 5]),
              accumulate=False)
            usedFO = usedFO.index_put_((
             final_indicesAlt[:, 0],
             final_indicesAlt[:, 2]),
              torch.ones((final_indicesAlt[:, 0].shape), dtype=(torch.long), device=(self.dev)),
              accumulate=True)
            usedH = usedH.index_put_((
             final_indicesAlt[:, 1],
             final_indicesAlt[:, 3]),
              torch.ones((final_indicesAlt[:, 0].shape), dtype=(torch.long), device=(self.dev)),
              accumulate=True)
            hbondEnergy.index_put_((final_indicesAlt[:, 0],
             alt_index,
             final_indicesAlt[:, 4]),
              (final_hbond_score[mask] * 0.5), accumulate=True)
            hbondEnergy.index_put_((final_indicesAlt[:, 1],
             alt_index,
             final_indicesAlt[:, 5]),
              (final_hbond_score[mask] * 0.5), accumulate=True)

        _, rotation = hbondEnergy.min(-1)
        hbondEnergy = torch.zeros((n_atoms, n_alter), dtype=(self.float_type), device=(self.dev))
        multiple_bonds = torch.zeros((n_atoms, n_alter), dtype=(torch.long), device=(self.dev))
        for alt in range(alternMask.shape[-1]):
            mask = alternMask[(final_indices[:, 0], alt)] & alternMask[(final_indices[:, 1], alt)]
            alt_index = torch.full((mask.shape), alt, device=(self.dev), dtype=(torch.long))[mask]
            final_indicesAlt = final_indices[mask]
            goodFakeAtomRotIndex1 = rotation[(final_indicesAlt[:, 0], alt_index)]
            goodFakeAtomRotIndex2 = rotation[(final_indicesAlt[:, 1], alt_index)]
            best_fake_atom_rot_mask = (goodFakeAtomRotIndex1 == final_indicesAlt[:, 4]) & (goodFakeAtomRotIndex2 == final_indicesAlt[:, 5])
            hbondEnergy.index_put_((final_indicesAlt[:, 0][best_fake_atom_rot_mask],
             alt_index[best_fake_atom_rot_mask]),
              (final_hbond_score[mask][best_fake_atom_rot_mask] * 0.5),
              accumulate=True)
            hbondEnergy.index_put_((final_indicesAlt[:, 1][best_fake_atom_rot_mask],
             alt_index[best_fake_atom_rot_mask]),
              (final_hbond_score[mask][best_fake_atom_rot_mask] * 0.5),
              accumulate=True)
            multiple_bonds.index_put_((final_indicesAlt[:, 0][best_fake_atom_rot_mask],
             alt_index[best_fake_atom_rot_mask]),
              torch.ones((final_indicesAlt[:, 0][best_fake_atom_rot_mask].shape), dtype=(torch.long), device=(self.dev)),
              accumulate=True)
            multiple_bonds.index_put_((final_indicesAlt[:, 1][best_fake_atom_rot_mask],
             alt_index[best_fake_atom_rot_mask]),
              torch.ones((final_indicesAlt[:, 0][best_fake_atom_rot_mask].shape), dtype=(torch.long), device=(self.dev)),
              accumulate=True)

        energy_atoms = hbondEnergy
        return (energy_atoms, multiple_bonds)

    def bindToResi(self, atomEnergy, sharedBonds, atom_description, alternativeMask, facc, minSaCoefficient=0.9):
        corr_temp = 0.0015 * (TEMPERATURE - 273.0)
        for i, bb_atom in enumerate(self.backbone_atoms):
            if i == 0:
                is_backbone_mask = atom_description[:, hashings.atom_description_hash['at_name']].eq(bb_atom)
            else:
                is_backbone_mask += atom_description[:, hashings.atom_description_hash['at_name']].eq(bb_atom)

        resnum = atom_description[:, hashings.atom_description_hash['resnum']]
        chain_ind = atom_description[:, hashings.atom_description_hash['chain']]
        batch_ind = atom_description[:, hashings.atom_description_hash['batch']]
        atom_name = atom_description[:, hashings.atom_description_hash['at_name']]
        mask_padding = ~resnum.eq(PADDING_INDEX)
        saCoefficient = facc.clamp(min=minSaCoefficient)
        atomEnergy[~atomEnergy.le(corr_temp)] += corr_temp
        energy = atomEnergy * saCoefficient * (1 - torch.tanh(self.weight))
        not_bb_mask = ~is_backbone_mask & mask_padding
        bb_mask = is_backbone_mask & mask_padding
        batch = batch_ind.max() + 1
        nres = torch.max(resnum) + 1
        nchains = chain_ind.max() + 1
        naltern = alternativeMask.shape[-1]
        finalMC = torch.zeros((batch, nchains, nres, naltern), dtype=(torch.float), device=(self.dev))
        if atomEnergy.sum() == 0:
            finalMC_min, _ = finalMC.min(dim=-1, keepdim=True)
            finalSC_components = torch.zeros((batch, nchains, nres, 1), dtype=(torch.float), device=(self.dev))
            return (finalMC_min, finalSC_components)
        batch_ind = atom_description[:, hashings.atom_description_hash['batch']].unsqueeze(-1).expand(-1, naltern).long()
        chain_ind = atom_description[:, hashings.atom_description_hash['chain']].unsqueeze(-1).expand(-1, naltern).long()
        resIndex = atom_description[:, hashings.atom_description_hash['resnum']].unsqueeze(-1).expand(-1, naltern).long()
        alt_index = torch.arange(0, naltern, dtype=(torch.long), device=(self.dev)).unsqueeze(0).expand(atom_description.shape[0], -1)
        bbmask = bb_mask.unsqueeze(-1).expand(-1, naltern)
        indices = tuple([batch_ind[bbmask], chain_ind[bbmask], resIndex[bbmask], alt_index[bbmask]])
        finalMC.index_put_(indices, (energy[bbmask]), accumulate=True).unsqueeze(-1)
        bbmask = not_bb_mask.unsqueeze(-1).expand(-1, naltern)
        nd1 = atom_name.eq(hashings.atom_hash['HIS']['ND1'])
        ne2 = atom_name.eq(hashings.atom_hash['HIS']['NE2'])
        his = (nd1 | ne2).unsqueeze(-1).expand(-1, naltern)
        nd1H1S = atom_name.eq(hashings.atom_hash['HIS']['ND1H1S'])
        ne2H1S = atom_name.eq(hashings.atom_hash['HIS']['NE2H1S'])
        h1s = (nd1H1S | ne2H1S).unsqueeze(-1).expand(-1, naltern)
        nd1H2S = atom_name.eq(hashings.atom_hash['HIS']['ND1H2S'])
        ne2H2S = atom_name.eq(hashings.atom_hash['HIS']['NE2H2S'])
        h2s = (nd1H2S | ne2H2S).unsqueeze(-1).expand(-1, naltern)
        final_h1s = torch.zeros((batch, nchains, nres, naltern), dtype=(torch.float), device=(self.dev))
        final_his = torch.zeros((batch, nchains, nres, naltern), dtype=(torch.float), device=(self.dev))
        final_h2s = torch.zeros((batch, nchains, nres, naltern), dtype=(torch.float), device=(self.dev))
        his_mask = ~h1s & ~h2s
        indices = tuple([batch_ind[bbmask & his_mask], chain_ind[bbmask & his_mask], resIndex[bbmask & his_mask], alt_index[bbmask & his_mask]])
        final_his = final_his.index_put(indices, (energy[bbmask & his_mask]), accumulate=True).unsqueeze(-1)
        his_mask = ~his & ~h2s
        indices = tuple([batch_ind[bbmask & his_mask], chain_ind[bbmask & his_mask], resIndex[bbmask & his_mask], alt_index[bbmask & his_mask]])
        final_h1s = final_h1s.index_put(indices, (energy[bbmask & his_mask]), accumulate=True).unsqueeze(-1)
        his_mask = ~h1s & ~his
        indices = tuple([batch_ind[bbmask & his_mask], chain_ind[bbmask & his_mask], resIndex[bbmask & his_mask], alt_index[bbmask & his_mask]])
        final_h2s = final_h2s.index_put(indices, (energy[bbmask & his_mask]), accumulate=True).unsqueeze(-1)
        finalSC = torch.cat([final_his, final_h1s, final_h2s], dim=(-1))

        # Collapse naltern with min (or choose mean/sum depending on semantics)
        # Use keepdim=True so result shape is (..., 1)
        finalMC_min, _ = finalMC.min(dim=-1, keepdim=True)   # shape: (batch, nchains, nres, 1)

        # For sidechain hbonds: ensure we produce the same trailing dim
        # Your original returned finalSC.sum(dim=-1) — we'll retain that but keep a trailing dim:
        finalSC_sum_components = finalSC.sum(dim=-1)  # sum over components
        finalSC_components, _ = finalSC_sum_components.min(dim=-1, keepdim=True)  # min over alternatives

        # Return 5-D tensors consistent with electro_net (which uses unsqueeze(-1))
        return finalMC_min, finalSC_components

    def getWeights(self):
        pass

    def getNumParams(self):
        p = []
        for i in self.parameters():
            p += list(i.data.cpu().numpy().flat)

        print('Number of parameters=', len(p))
