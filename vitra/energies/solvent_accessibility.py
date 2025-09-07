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
import numpy as np

from vitra.sources.globalVariables import PADDING_INDEX, EPS
from vitra.sources.math_utils import angle2dVectors
from vitra.sources import hashings

inverse_letters = {
    'C': 'CYS', 'D': 'ASP', 'S': 'SER', 'Q': 'GLN', 'K': 'LYS', 'N': 'ASN',
    'P': 'PRO', 'T': 'THR', 'F': 'PHE', 'A': 'ALA', 'H': 'HIS', 'G': 'GLY',
    'I': 'ILE', 'L': 'LEU', 'R': 'ARG', 'W': 'TRP', 'V': 'VAL', 'E': 'GLU',
    'Y': 'TYR', 'M': 'MET'}


def size(tensor, to='m', bsize=1024):
    byte_size = tensor.element_size() * tensor.nelement()
    a = {'k': 1, 'm': 2, 'g': 3, 't': 4, 'p': 5, 'e': 6}
    r = float(byte_size)
    for i in range(a[to]):
        r = r / bsize
    return round(r)


class SolventAccessibility(torch.nn.Module):

    def __init__(self, donor={}, acceptor={}, dev='cpu', float_type=torch.float, backbone_atoms=[]):
        super(SolventAccessibility, self).__init__()
        self.nFO = 3
        self.nH = 12
        self.backbone_atoms = backbone_atoms

        # moving protons
        has_moving_protons = ['CYS', 'LYS', 'SER', 'THR', 'TYR']
        self.has_moving_protons = []
        for i in has_moving_protons:
            self.has_moving_protons += [hashings.resi_hash[i]]
        self.has_moving_protons = list(set(self.has_moving_protons))

        # OH groups
        alcholic_groups = ['SER', 'TYR', 'THR', 'CYS']
        self.alcholic_groups = []
        for i in alcholic_groups:
            self.alcholic_groups += [hashings.resi_hash[i]]
        self.alcholic_groups = list(set(self.alcholic_groups))

        # protonation states
        has_proton_states = ['HIS']
        self.has_proton_states = []
        for i in has_proton_states:
            self.has_proton_states += [hashings.resi_hash[i]]
        self.has_proton_states = list(set(self.has_proton_states))

        nitrogen = {
            'ASN': ['ND2'],
            'GLN': ['NE2'],
            'HIS': ['ND1', 'NE1'],
            'TRP': ['NE1'],
            'ARG': ['NE', 'NH1', 'NH2'],
            'LYS': ['NZ']}

        for res in hashings.resi_hash.keys():
            if res not in nitrogen:
                nitrogen[res] = []
            nitrogen[res] += ['N']

        self.nitrogen = []
        for i in nitrogen.keys():
            self.nitrogen += [hashings.resi_hash[i]]
        self.nitrogen = list(set(self.nitrogen))

        # backbone donors
        self.Oatom = []
        self.Natom = []
        for res in hashings.atom_hash.keys():
            self.Natom += [hashings.atom_hash[res]['tN']]
            self.Natom += [hashings.atom_hash[res]['N']]
            self.Oatom += [hashings.atom_hash[res]['O']]
            self.Oatom += [hashings.atom_hash[res]['OXT']]

        self.Natom = set(self.Natom)
        self.Oatom = set(self.Oatom)

        # Virtual sites
        virtual_proton_holders = {
            'CYS': ['SG'],
            'LYS': ['NZ'],
            'SER': ['OG'],
            'THR': ['OG1'],
            'TYR': ['OH']}

        self.virtual_proton_holders = []
        for i in virtual_proton_holders.keys():
            for k in virtual_proton_holders[i]:
                self.virtual_proton_holders += [hashings.atom_hash[i][k]]
        self.virtual_proton_holders = list(set(self.virtual_proton_holders))

        self.donor = donor
        self.acceptor = acceptor
        self.dev = dev
        self.float_type = float_type

    def bindToResiSideAndMC(self, contRat, atom_description, alternatives):
        atname = atom_description[:, hashings.atom_description_hash['at_name']].long()
        resnum = atom_description[:, hashings.atom_description_hash['resnum']].long()
        chainInd = atom_description[:, hashings.atom_description_hash['chain']].long()
        batchInd = atom_description[:, hashings.atom_description_hash['batch']].long()
        resname = atom_description[:, hashings.atom_description_hash['resname']].long()
        for i, bb_atom in enumerate(self.backbone_atoms):
            if i == 0:
                is_backbone_mask = atname.eq(bb_atom)
            else:
                is_backbone_mask += atname.eq(bb_atom)

        mask_padding = ~atname.eq(PADDING_INDEX)
        seqSize = resnum.max() + 1
        chain_size = chainInd.max() + 1
        naltern = alternatives.shape[-1]
        batch = batchInd.max() + 1
        natoms = atname.shape[0]
        alt_index = torch.arange(0, naltern, dtype=torch.long, device=self.dev).unsqueeze(0).expand(natoms, -1)
        rsaSC = torch.zeros((batch, chain_size, seqSize, naltern), dtype=self.float_type, device=self.dev)
        rsaMC = torch.zeros((batch, chain_size, seqSize, naltern), dtype=self.float_type, device=self.dev)
        sequence = torch.full((batch, chain_size, seqSize, naltern), (int(PADDING_INDEX)), device=self.dev)
        for alt in range(naltern):
            mask = alternatives[:, alt] & is_backbone_mask & mask_padding
            index = tuple([batchInd[mask],
                           chainInd[mask].long(),
                           resnum[mask].long(),
                           alt_index[(mask, alt)]])
            rsaMC.index_put_(index, (contRat[mask][:, alt]), accumulate=True)
            sequence.index_put_(index, (resname[mask]), accumulate=False)
            mask = alternatives[:, alt] & ~is_backbone_mask & mask_padding
            index = tuple([batchInd[mask],
                           chainInd[mask].long(),
                           resnum[mask].long(),
                           alt_index[(mask, alt)]])
            rsaSC.index_put_(index, (contRat[mask][:, alt]), accumulate=True)

        for res in hashings.resi_hash.keys():
            resInt = hashings.resi_hash[res]
            m = sequence.eq(resInt)
            if not res == 'GLY':
                rsaMC[m] = (rsaMC[m] - hashings.res_acid_pointMC[res][1]) / (
                        hashings.res_acid_pointMC[res][0] - hashings.res_acid_pointMC[res][1])
                rsaSC[m] = (rsaSC[m] - hashings.res_acid_point[res][1]) / (
                        hashings.res_acid_point[res][0] - hashings.res_acid_point[res][1])
            else:
                rsaMC[m] = (rsaMC[m] - hashings.res_acid_pointMC[res][1]) / (
                        hashings.res_acid_pointMC[res][0] - hashings.res_acid_pointMC[res][1])
                rsaSC[m] = 0.0

        rsaSC = rsaSC.clamp(0, 1)
        rsaMC = rsaMC.clamp(0, 1)
        return rsaMC, rsaSC

    def build_angle_boundaries(self, atName1, resName1, is_backbone_mask, acceptor_mask, donor_mask):
        dev = self.dev
        float_type = self.float_type
        n_elements = is_backbone_mask.shape[0]
        useThetaMin = torch.full([n_elements], float(np.radians(125.0)), device=dev, dtype=float_type)
        useThetaMax = torch.full([n_elements], float(np.radians(180.0)), device=dev, dtype=float_type)
        useOptMinTheta = torch.full([n_elements], float(np.radians(153.0)), device=dev, dtype=float_type)
        useOptMaxTheta = torch.full([n_elements], float(np.radians(163.0)), device=dev, dtype=float_type)
        useOptMinThetaLow = torch.full([n_elements], float(np.radians(145.0)), device=dev, dtype=float_type)
        useOptMaxThetaHigh = torch.full([n_elements], float(np.radians(180.0)), device=dev, dtype=float_type)
        optMinThetaProtAlc = np.radians(148)
        optMinThetaProtLys = np.radians(135.0)
        # optMinThetaProtMov = np.radians(100.0)
        thetaMinprot = np.radians(125)
        optMinThetaProtAlcLow = np.radians(130.0)
        optMinThetaProtLysLow = np.radians(120.0)
        thetaMinProtMov = np.radians(100.0)
        # ThetaMaxOrbAlc = np.radians(180.0)
        # optMaxThetaOrbAlc = np.radians(140.0)
        # optMinThetaOrbAlc = np.radians(130.0)
        # optMaxThetaOrbAlcLow = np.radians(100.0)
        # optMinThetaOrbAlcHigh = np.radians(180.0)
        # optMinThetaOrbAlcLow = np.radians(100.0)
        # optMaxThetaOrbAlcHigh = np.radians(180.0)
        optMinThetaProtHis = np.radians(160.0)
        optMaxThetaProtHis = np.radians(170.0)
        thetaMaxOrbHis = np.radians(180.0)
        # optMinThetaOrbHis = np.radians(155.0)
        optMaxThetaOrbHis = np.radians(165.0)
        optMinThetaProtHisLow = np.radians(140.0)
        # optMaxThetaProtHisHigh = np.radians(180.0)
        # limHigh = 1.0
        # limLow = 0.1
        # factorLimit = limHigh - limLow
        first = True
        for resName in self.has_moving_protons:
            if first:
                hasMovingProtons = resName1.eq(resName)
                first = False
            else:
                hasMovingProtons += resName1.eq(resName)

        first = True
        for residueName in self.alcholic_groups:
            if first:
                alcholic_groups = resName1.eq(residueName)
                first = False
            else:
                alcholic_groups += resName1.eq(residueName)

        first = True
        for resName in self.has_proton_states:
            if first:
                hasprotonStates = resName1.eq(resName)
                first = False
            else:
                hasprotonStates += resName1.eq(resName)

        first = True
        for atomName in self.virtual_proton_holders:
            if first:
                virtualHholders = atName1.eq(atomName)
                first = False
            else:
                virtualHholders += atName1.eq(atomName)

        explicitH = [hashings.atom_hash['CYS']['SG'],
                     hashings.atom_hash['THR']['OG1'],
                     hashings.atom_hash['TYR']['OH'],
                     hashings.atom_hash['SER']['OG']]
        first = True
        for atomName in explicitH:
            if first:
                is_explicitH = atName1.eq(atomName)
                first = False
            else:
                is_explicitH += atName1.eq(atomName)

        if explicitH == []:
            is_explicitH = torch.full((hasprotonStates.shape), False, dtype=(torch.bool), device=(self.dev))
        alchol = donor_mask & hasMovingProtons & virtualHholders & alcholic_groups
        # extra_alcol = donor_mask & alcholic_groups & ~is_explicitH
        lys = donor_mask & hasMovingProtons & virtualHholders & ~alcholic_groups
        moving = donor_mask & hasMovingProtons & virtualHholders
        protStates = donor_mask & ~moving & hasprotonStates & ~is_backbone_mask
        # nonmoving = donor_mask & ~moving & ~hasprotonStates
        useOptMinTheta[alchol] = optMinThetaProtAlc
        useOptMinThetaLow[alchol] = optMinThetaProtAlcLow
        useOptMinTheta[lys] = optMinThetaProtLys
        useOptMinThetaLow[lys] = optMinThetaProtLysLow
        useThetaMin[moving] = thetaMinProtMov
        useOptMinTheta[protStates] = optMinThetaProtHis
        useOptMinThetaLow[protStates] = optMinThetaProtHisLow
        acceptor_mask = acceptor_mask & ~donor_mask
        useThetaMin[acceptor_mask] = np.radians(90)
        useThetaMax[acceptor_mask] = np.radians(155)
        useOptMinTheta[acceptor_mask] = np.radians(105)
        useOptMaxTheta[acceptor_mask] = np.radians(110)
        useOptMinThetaLow[acceptor_mask] = np.radians(100)
        useOptMaxThetaHigh[acceptor_mask] = np.radians(115)
        protStates = acceptor_mask & ~moving & hasprotonStates & ~is_backbone_mask
        useThetaMax[protStates] = thetaMaxOrbHis
        useOptMinTheta[protStates] = optMinThetaProtHis
        useOptMaxTheta[protStates] = optMaxThetaOrbHis
        useOptMinThetaLow[protStates] = np.radians(140)
        useOptMaxThetaHigh[protStates] = np.radians(180)
        maskhis = ~hasMovingProtons & hasprotonStates & ~is_backbone_mask
        useThetaMax[maskhis] = thetaMinprot
        useOptMinTheta[maskhis] = optMinThetaProtHis
        useOptMaxTheta[maskhis] = optMaxThetaProtHis
        useOptMinThetaLow[maskhis] = np.radians(145)
        return useThetaMin, useThetaMax, useOptMinTheta, useOptMaxTheta, useOptMinThetaLow, useOptMaxThetaHigh

    def forward(self, coords, atom_description, atom_number, atomPairs, fakeAtoms, alternativeMask):
        contRat, facc, contRatPol = self.contRat(coords, atom_description, atom_number, atomPairs, fakeAtoms,
                                                 alternativeMask)
        contRatMC, contRatSC = self.bindToResiSideAndMC(contRat, atom_description, alternatives=alternativeMask)
        return contRat, facc, contRatPol, contRatMC, contRatSC

    def directed_solvation(self, atomPairs1, atomPairs2, coor_dist, temp1, fake_atoms, atom_description, coords,
                           alternativeMask):
        atName1 = atom_description[(atomPairs1, hashings.atom_description_hash['at_name'])].long()
        resName1 = atom_description[(atomPairs1, hashings.atom_description_hash['resname'])].long()
        # atName2 = atom_description[(atomPairs2, hashings.atom_description_hash['at_name'])].long()
        for i, bb_atom in enumerate(self.backbone_atoms):
            if i == 0:
                is_backbone_mask = atName1.eq(bb_atom)
            else:
                is_backbone_mask += atName1.eq(bb_atom)

        for i, donor_atom in enumerate(self.donor):
            if i == 0:
                donor_mask = atName1.eq(donor_atom)
            else:
                donor_mask += atName1.eq(donor_atom)

        for i, acceptor_atom in enumerate(self.acceptor):
            if i == 0:
                acceptor_mask = atName1.eq(acceptor_atom)
                flat_acceptor_mask = atName1.eq(acceptor_atom)
            else:
                acceptor_mask += atName1.eq(acceptor_atom)
                flat_acceptor_mask += atName1.eq(acceptor_atom)

        nfakeAtoms = fake_atoms.shape[1]
        naltern = alternativeMask.shape[-1]
        useThetaMinOrig, useThetaMaxOrig, useOptMinThetaOrig, useOptMaxThetaOrig, \
            useOptMinThetaLowOrig, useOptMaxThetaHighOrig = self.build_angle_boundaries(
            atName1, resName1, is_backbone_mask, acceptor_mask, donor_mask)

        orbsize = torch.ones([atomPairs1.shape[0]], device=self.dev, dtype=torch.long)
        orbsize[acceptor_mask] = (~fake_atoms[atomPairs1, -self.nFO:, 0][acceptor_mask].eq(PADDING_INDEX)).sum(1)
        orbsize[donor_mask] = (~fake_atoms[atomPairs1, :self.nH, 0][donor_mask].eq(PADDING_INDEX)).sum(1).clamp(min=1)
        factor_red = 1.0 / (orbsize * 2)
        correction_dist = coor_dist * factor_red
        existing_fakeatomsA1 = ~fake_atoms[atomPairs1, :, 0].eq(PADDING_INDEX)
        existing_fakeatomsA1[acceptor_mask][:, :-self.nFO] = False
        existing_fakeatomsA1[donor_mask][:, self.nH:] = False
        fakeAtomsPW1 = fake_atoms[atomPairs1][existing_fakeatomsA1]
        coordsPW = coords[atomPairs1].unsqueeze(1).expand(-1, nfakeAtoms, -1)[existing_fakeatomsA1]
        coordsOtherPW = coords[atomPairs2].unsqueeze(1).expand(-1, nfakeAtoms, -1)[existing_fakeatomsA1]
        carrierOrb = coordsPW - fakeAtomsPW1
        otherOrb = coordsOtherPW - fakeAtomsPW1
        theta, nanmask, _ = angle2dVectors(carrierOrb, otherOrb)
        useThetaMin = useThetaMinOrig.unsqueeze(1).expand(-1, existing_fakeatomsA1.shape[1])[existing_fakeatomsA1]
        useThetaMax = useThetaMaxOrig.unsqueeze(1).expand(-1, existing_fakeatomsA1.shape[1])[existing_fakeatomsA1]
        useOptMinTheta = useOptMinThetaOrig.unsqueeze(1).expand(-1, existing_fakeatomsA1.shape[1])[existing_fakeatomsA1]
        useOptMaxTheta = useOptMaxThetaOrig.unsqueeze(1).expand(-1, existing_fakeatomsA1.shape[1])[existing_fakeatomsA1]
        useOptMinThetaLow = useOptMinThetaLowOrig.unsqueeze(1).expand(-1, existing_fakeatomsA1.shape[1])[
            existing_fakeatomsA1]
        useOptMaxThetaHigh = useOptMaxThetaHighOrig.unsqueeze(1).expand(-1, existing_fakeatomsA1.shape[1])[
            existing_fakeatomsA1]
        factor_red = factor_red.unsqueeze(1).expand(-1, existing_fakeatomsA1.shape[1])[existing_fakeatomsA1]
        coor_dist = coor_dist.unsqueeze(1).expand(-1, existing_fakeatomsA1.shape[1])[existing_fakeatomsA1]
        temp1 = temp1.unsqueeze(1).expand(-1, existing_fakeatomsA1.shape[1])[existing_fakeatomsA1]
        fake_atoms_indexing = \
            torch.arange(nfakeAtoms, device=self.dev, dtype=torch.long).expand(atomPairs1.shape[0], nfakeAtoms)[
                existing_fakeatomsA1]
        correction_angle, zero_vals_mask = self.calculateCorrection(theta.squeeze(-1), factor_red, coor_dist,
                                                                    useThetaMin, useThetaMax, useOptMinTheta,
                                                                    useOptMaxTheta, useOptMinThetaLow,
                                                                    useOptMaxThetaHigh)
        final_dist_correction = correction_dist.unsqueeze(1).expand(-1, existing_fakeatomsA1.shape[1])[
                                    existing_fakeatomsA1] + correction_angle
        final_dist_correction[zero_vals_mask] = 0.0
        atomPairs1_expanded = atomPairs1.unsqueeze(1).expand(existing_fakeatomsA1.shape)[existing_fakeatomsA1]
        atomPairs2_expanded = atomPairs2.unsqueeze(1).expand(existing_fakeatomsA1.shape)[existing_fakeatomsA1]

        contRat_Pol = torch.zeros((atom_description.shape[0], naltern, nfakeAtoms), dtype=self.float_type,
                                  device=self.dev)

        for alt in range(naltern):
            mask = alternativeMask[(atomPairs1_expanded, alt)] & alternativeMask[(atomPairs2_expanded, alt)]
            altPairs = atomPairs1_expanded[mask]
            atom_number_pwAlt = altPairs
            altern_ind = torch.full(altPairs.shape, alt, dtype=torch.long, device=self.dev)
            fake_atoms_indexing_a1Alt = fake_atoms_indexing[mask]

            indices = (atom_number_pwAlt.long(), altern_ind, fake_atoms_indexing_a1Alt)
            fac_pol_values = temp1[mask] * final_dist_correction[mask]

            contRat_Pol = contRat_Pol.index_put(indices, fac_pol_values, accumulate=True)
        return contRat_Pol

    def contRat(self, coords, atom_description, atom_number, atomPairs, fakeAtoms, alternativeMask):
        vdw_limit = 6
        # disulfDist = 3.5
        constant_distPol = 2.8
        constant_distAPol = 2.8
        distmat = torch.pairwise_distance(coords[atomPairs[:, 0]], coords[atomPairs[:, 1]])
        atName1 = atom_description[(atomPairs[:, 0], hashings.atom_description_hash['at_name'])].long()
        atName2 = atom_description[(atomPairs[:, 1], hashings.atom_description_hash['at_name'])].long()
        no_his_iso_maks = (atName1 != hashings.atom_hash['HIS']['ND1H1S']) & (
                atName1 != hashings.atom_hash['HIS']['NE2H1S']) & (
                                  atName1 != hashings.atom_hash['HIS']['ND1H2S']) & (
                                  atName1 != hashings.atom_hash['HIS']['NE2H2S']) & (
                                  atName2 != hashings.atom_hash['HIS']['ND1H1S']) & (
                                  atName2 != hashings.atom_hash['HIS']['NE2H1S']) & (
                                  atName2 != hashings.atom_hash['HIS']['ND1H2S']) & (
                                  atName2 != hashings.atom_hash['HIS']['NE2H2S'])
        nonvirtual_mask = (
                hashings.atom_Properties[(atName1, hashings.property_hashings['other_params']['virtual'])].eq(0) &
                hashings.atom_Properties[(atName2, hashings.property_hashings['other_params']['virtual'])].eq(
                    0)).to(self.dev)
        long_mask = distmat.le(vdw_limit)
        full_maskContRat = long_mask & nonvirtual_mask & no_his_iso_maks
        atomPairs = atomPairs[full_maskContRat]
        distmat = distmat[full_maskContRat]
        atName = torch.cat([atName1[full_maskContRat].unsqueeze(-1), atName2[full_maskContRat].unsqueeze(-1)], dim=(-1))
        del atName2
        del atName1
        # triangular_mask = atomPairs[:, 0] < atomPairs[:, 1]

        for i, bb_atom in enumerate(self.backbone_atoms):
            if i == 0:
                is_backbone_mask1 = atName.eq(bb_atom)
            else:
                is_backbone_mask1 += atName.eq(bb_atom)

        for i, donor_atom in enumerate(self.donor):
            if i == 0:
                donor_mask = atName.eq(donor_atom)
            else:
                donor_mask += atName.eq(donor_atom)

        for i, acceptor_atom in enumerate(self.acceptor):
            if i == 0:
                acceptor_mask = atName.eq(acceptor_atom)
                flat_acceptor_mask = atName.eq(acceptor_atom)
            else:
                acceptor_mask += atName.eq(acceptor_atom)
                flat_acceptor_mask += atName.eq(acceptor_atom)

        # NoFake = ~acceptor_mask & ~donor_mask
        # del NoFake
        same_chain_mask = atom_description[(atomPairs[:, 0], hashings.atom_description_hash['chain'])] == \
                          atom_description[(atomPairs[:, 1], hashings.atom_description_hash['chain'])]
        plus1_residues_mask = (atom_description[(atomPairs[:, 0], hashings.atom_description_hash['resnum'])] -
                               atom_description[(atomPairs[:, 1], hashings.atom_description_hash['resnum'])]).eq(
            -1) & same_chain_mask

        Oatoms = torch.any(torch.stack(
            [torch.eq(atName[:, 0], aelem).logical_or_(torch.eq(atName[:, 0],
                                                                aelem)) for aelem in self.Oatom], dim=0), dim=0)

        Natoms = torch.any(torch.stack(
            [torch.eq(atName[:, 1], aelem).logical_or_(torch.eq(atName[:, 1],
                                                                aelem)) for aelem in self.Natom], dim=0), dim=0)

        ON_correction = Oatoms & Natoms & plus1_residues_mask
        distmat[ON_correction] = 3.5
        factor = torch.exp(-distmat ** 2 / 24.5).squeeze(-1)

        interaction1 = factor * hashings.atom_Properties[
            (atName[:, 1], hashings.property_hashings['solvenergy_props']['volume'])].to(self.dev)
        interaction2 = factor * hashings.atom_Properties[
            (atName[:, 0], hashings.property_hashings['solvenergy_props']['volume'])].to(self.dev)
        interaction1Direct = factor * hashings.atom_Properties[
            (atName[:, 1], hashings.property_hashings['solvenergy_props']['volume'])].to(self.dev)
        interaction2Direct = factor * hashings.atom_Properties[
            (atName[:, 0], hashings.property_hashings['solvenergy_props']['volume'])].to(self.dev)

        naltern = alternativeMask.shape[-1]
        natoms = atom_number.shape[0]
        occ_atoms = torch.zeros((natoms, naltern), device=self.dev, dtype=self.float_type)
        for alt in range(naltern):
            mask = alternativeMask[(atomPairs[:, 0], alt)] & alternativeMask[(atomPairs[:, 1], alt)]
            atom_number_pwAlt = atomPairs[mask]
            altern_ind = torch.full([atom_number_pwAlt.shape[0]], alt, device=self.dev)
            indices = (
                atom_number_pwAlt[:, 0].long(), altern_ind)
            occ_atoms = occ_atoms.index_put(indices, (interaction1[mask]), accumulate=True)
            indices = (
                atom_number_pwAlt[:, 1].long(), altern_ind)
            occ_atoms = occ_atoms.index_put(indices, (interaction2[mask]), accumulate=True)

        occ_atomsFacc = occ_atoms.clone()
        m = occ_atomsFacc != 0.0
        atomName = atom_description[(atom_number, hashings.atom_description_hash['at_name'])].long()
        occ = hashings.atom_Properties[(atomName, hashings.property_hashings['solvenergy_props']['Occ'])].unsqueeze(
            -1).expand(occ_atomsFacc.shape)
        occMax = hashings.atom_Properties[
            (atomName, hashings.property_hashings['solvenergy_props']['Occmax'])].unsqueeze(-1).expand(
            occ_atomsFacc.shape)
        occ_atomsFacc[m & (occ_atomsFacc < occ)] = occ[m & (occ_atomsFacc < occ)]
        occ_atomsFacc[m & (occ_atomsFacc < occMax)] = occMax[m & (occ_atomsFacc < occMax)]
        denom = occMax - occ
        bad_denom = denom.eq(0)
        denom[bad_denom] = 1

        facc = torch.zeros(natoms, naltern, device=self.dev, dtype=torch.float)

        for alt in range(naltern):
            mask = alternativeMask[(atom_number, alt)] & (~bad_denom)[:, alt] & m[:, alt]
            facc[:, alt][mask] = (occ_atomsFacc[:, alt][mask] - occ[:, alt][mask]) / denom[:, alt][mask]

        at_names1 = atom_description[(atomPairs[:, 0], hashings.atom_description_hash['at_name'])].long()
        at_names2 = atom_description[(atomPairs[:, 1], hashings.atom_description_hash['at_name'])].long()
        same_residue_mask = atom_description[(atomPairs[:, 0], hashings.atom_description_hash['resnum'])] == \
                            atom_description[(atomPairs[:, 1], hashings.atom_description_hash['resnum'])]

        consec_residues = (atom_description[(atomPairs[:, 0], hashings.atom_description_hash['resnum'])] -
                           atom_description[(atomPairs[:, 1], hashings.atom_description_hash['resnum'])]).eq(
            -1) & same_chain_mask

        for i, res in enumerate(hashings.resi_hash.keys()):
            if i == 0:
                caMask1 = at_names1.eq(hashings.atom_hash[res]['CA'])
                caMask2 = at_names2.eq(hashings.atom_hash[res]['CA'])
                nMask = at_names2.eq(hashings.atom_hash[res]['N'])
                oMask = at_names1.eq(hashings.atom_hash[res]['O'])
            else:
                caMask1 += at_names1.eq(hashings.atom_hash[res]['CA'])
                caMask2 += at_names2.eq(hashings.atom_hash[res]['CA'])
                nMask += at_names2.eq(hashings.atom_hash[res]['N'])
                oMask += at_names1.eq(hashings.atom_hash[res]['O'])

        directFirstMask = ~(same_residue_mask | consec_residues & ((caMask2 | nMask) & oMask | caMask1 & nMask))
        at_names1 = at_names1[directFirstMask]
        at_names2 = at_names2[directFirstMask]

        acceptor_mask = acceptor_mask[directFirstMask]
        donor_mask = donor_mask[directFirstMask]
        hbondAbleMask = acceptor_mask | donor_mask
        oneHbondAble = hbondAbleMask[:, 0] | hbondAbleMask[:, 1]

        minradius1 = hashings.atom_Properties[(at_names1, hashings.property_hashings['solvenergy_props']['minradius'])]
        minradius2 = hashings.atom_Properties[(at_names2, hashings.property_hashings['solvenergy_props']['minradius'])]
        dif = distmat[directFirstMask][oneHbondAble] - (minradius1[oneHbondAble] + minradius2[oneHbondAble])
        bothHbondAbleMask = hbondAbleMask[:, 0] & hbondAbleMask[:, 1]
        m1 = bothHbondAbleMask[oneHbondAble] & (dif <= constant_distPol)
        m2 = ~bothHbondAbleMask[oneHbondAble] & (dif <= constant_distAPol)
        directSecondMask = m1 | m2
        dif = dif[directSecondMask]
        m1 = m1[directSecondMask]
        m2 = m2[directSecondMask]
        coor_dist = dif
        coor_dist[m1] = 1 - coor_dist[m1] / constant_distPol
        coor_dist[m2] = 1 - coor_dist[m2] / constant_distAPol
        accMaskDirect = acceptor_mask[oneHbondAble][directSecondMask][:, 0] | \
                        donor_mask[oneHbondAble][directSecondMask][:, 0]

        atomPairsAcceptor = atomPairs[directFirstMask][oneHbondAble][directSecondMask][accMaskDirect]
        temp1 = interaction1Direct[directFirstMask][oneHbondAble][directSecondMask][accMaskDirect]
        contRat_PolAcceptor = self.directed_solvation(atomPairsAcceptor[:, 0], atomPairsAcceptor[:, 1],
                                                      coor_dist[accMaskDirect], temp1, fakeAtoms,
                                                      atom_description, coords, alternativeMask)

        accMaskDirect = donor_mask[oneHbondAble][directSecondMask][:, 1] | \
                        acceptor_mask[oneHbondAble][directSecondMask][:, 1]

        atomPairsDonor = atomPairs[directFirstMask][oneHbondAble][directSecondMask][accMaskDirect]
        temp1 = interaction2Direct[directFirstMask][oneHbondAble][directSecondMask][accMaskDirect]
        contRat_PolDonor = self.directed_solvation(atomPairsDonor[:, 1], atomPairsDonor[:, 0], coor_dist[accMaskDirect],
                                                   temp1, fakeAtoms, atom_description, coords, alternativeMask)

        contRat_Pol = contRat_PolAcceptor + contRat_PolDonor
        return occ_atoms, facc, contRat_Pol

    def calculateCorrection(self, theta, factor_red, coor_dist, useThetaMin, useThetaMax, useOptMinTheta,
                            useOptMaxTheta, useOptMinThetaLow, useOptMaxThetaHigh):
        limit_high = 1.0
        limit_low = 0.1
        factor_limit = limit_high - limit_low
        zero_vals_mask = (theta < useThetaMin) | (theta > useThetaMax)

        difftheta = torch.zeros(theta.shape, device=self.dev, dtype=self.float_type)
        deltatheta = torch.zeros(theta.shape, device=self.dev, dtype=self.float_type)
        factor = torch.full(theta.shape, factor_limit, device=self.dev, dtype=self.float_type)
        limit = torch.full(theta.shape, limit_high, device=self.dev, dtype=self.float_type)

        initialThresh = (theta > useThetaMin) & (theta < useThetaMax)
        IfIf = (theta < useOptMinTheta) & (theta < useOptMinThetaLow) & initialThresh
        difftheta[IfIf] = useOptMinThetaLow[IfIf] - theta[IfIf]
        deltatheta[IfIf] = useOptMinThetaLow[IfIf] - useThetaMin[IfIf]
        factor[IfIf] = limit[IfIf] = limit_low

        IfElse = (theta < useOptMinTheta) & (theta >= useOptMinThetaLow) & initialThresh
        difftheta[IfElse] = useOptMinTheta[IfElse] - theta[IfElse]
        deltatheta[IfElse] = useOptMinTheta[IfElse] - useOptMinThetaLow[IfElse]

        ElseIf = (theta >= useOptMinTheta) & (theta >= useOptMaxTheta) & (theta >= useOptMaxThetaHigh) & initialThresh
        difftheta[ElseIf] = theta[ElseIf] - useOptMaxThetaHigh[ElseIf]
        deltatheta[ElseIf] = useThetaMax[ElseIf] - useOptMaxThetaHigh[ElseIf]
        factor[ElseIf] = limit[ElseIf] = limit_low

        elseElse = (theta >= useOptMinTheta) & (theta >= useOptMaxTheta) & (theta < useOptMaxThetaHigh) & initialThresh
        difftheta[elseElse] = theta[elseElse] - useOptMaxTheta[elseElse]
        deltatheta[elseElse] = useOptMaxThetaHigh[elseElse] - useOptMaxTheta[elseElse]
        factor[elseElse & (useOptMaxThetaHigh == useThetaMax)] = limit_high

        correction = (limit - factor * difftheta / deltatheta.clamp(min=EPS)) * factor_red
        correction[deltatheta == 0] = factor_red[deltatheta == 0]
        # cc = theta.clone()
        correction[zero_vals_mask] = 0.0
        return correction, zero_vals_mask
