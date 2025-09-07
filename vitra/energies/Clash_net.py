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

import torch, math, time, numpy as np
from vitra.sources.globalVariables import *
from vitra.sources import hashings
cov = {}
IonStrength = 0.05
temp = 25
dielec = 4
constant = math.exp(-0.004314 * temp)
random_coil = 0
g = 0

class Clash_net(torch.nn.Module):

    def __init__(self, name='clashes', dev='cpu', embedding_dimension=5, command='SequenceDetail', properties_hashing={}, backbone_atoms=[], donor=[], acceptor=[], hbond_ar=[], hashing={}):
        super(Clash_net, self).__init__()
        self.name = name
        self.backbone_atoms = backbone_atoms
        self.donor = donor
        self.acceptor = acceptor
        self.dev = dev
        self.int_type = torch.short
        self.self_covalent_bonds = []
        covalent_bonds = [
         ('O', 'N'),
         ('C', 'N'),
         ('C', 'CA'),
         ('O', 'CA'),
         ('CA', 'N')]
        pro_covalent_bonds = [
         ('C', 'CD'),
         ('CA', 'CD')]
        for pair in covalent_bonds:
            for res1 in hashings.resi_hash.keys():
                for res2 in hashings.resi_hash.keys():
                    self.self_covalent_bonds += [(hashings.atom_hash[res1][pair[0]], hashings.atom_hash[res2][pair[1]])]

        for pair in pro_covalent_bonds:
            for res1 in hashings.resi_hash.keys():
                res2 = 'PRO'
                self.self_covalent_bonds += [(hashings.atom_hash[res1][pair[0]], hashings.atom_hash[res2][pair[1]])]

        self.excluded_heavy_OCA = []
        for pair in (('O', 'CA'), ):
            for res1 in hashings.resi_hash.keys():
                for res2 in hashings.resi_hash.keys():
                    self.excluded_heavy_OCA += [(hashings.atom_hash[res1][pair[0]], hashings.atom_hash[res2][pair[1]])]

        self.self_covalent_bonds = list(set(self.self_covalent_bonds))
        self.Natoms = []
        self.Catoms = []
        self.Oatoms = []
        self.CAatoms = []
        for res1 in hashings.resi_hash.keys():
            self.Natoms += [hashings.atom_hash[res1]['N']]
            self.Natoms += [hashings.atom_hash[res1]['tN']]
            self.Catoms += [hashings.atom_hash[res1]['C']]
            self.Oatoms += [hashings.atom_hash[res1]['O']]
            self.CAatoms += [hashings.atom_hash[res1]['CA']]

        self.Natoms = list(set(self.Natoms))
        self.Catoms = list(set(self.Catoms))
        self.Oatoms = list(set(self.Oatoms))
        self.CAatoms = list(set(self.CAatoms))
        self.cbDisul = hashings.atom_hash['CYS']['CB']
        self.sgDisul = hashings.atom_hash['CYS']['SG']
        self.terminals = []
        for res1 in hashings.resi_hash.keys():
            self.terminals += [hashings.atom_hash[res1]['tN']]
            self.terminals += [hashings.atom_hash[res1]['OXT']]

        self.terminals = list(set(self.terminals))
        new_hash = hashings.atom_hashTPL
        max_hash = -1
        for r in hashings.atom_hash.keys():
            max_hash = max(max_hash, max(hashings.atom_hash[r].values()))

        max_hash += 1
        self.hashShort = torch.zeros(max_hash, dtype=(torch.long), device=(self.dev))
        for r in hashings.atom_hash.keys():
            for at in hashings.atom_hash[r].keys():
                if at in new_hash[r]:
                    self.hashShort[hashings.atom_hash[r][at]] = new_hash[r][at]
                else:
                    self.hashShort[hashings.atom_hash[r][at]] = PADDING_INDEX

        self.tollerances = torch.tensor([-0.03150951862335205,
         0.18205130100250244,
         0.060976290702819826,
         -0.03601369857788085,
         0.0007129192352294959,
         -1.3129888772964478],
          device=(self.dev), dtype=(torch.float))
        self.weight = torch.nn.Parameter(torch.tensor([0.0], device=(self.dev)))
        self.weight.requires_grad = True

    def load(self):
        self.tollerances = torch.load('marshalled/tollerancesNew.weights').to(self.dev)

    def fit(self, pdb_dir='../datasets/hiresPDB/'):
        from ForceField import Container
        from sources import utils, dataStructures, math_utils, fakeAtomsGeneration
        device = 'cpu'
        pdb_dir = 'dataset/toy_big_small/'
        coo, atnames = utils.parsePDB(pdb_dir)
        train_diz = {'bb':{},  'omega':{}}
        split_numb = 50
        container = Container(dev=device)
        vals = [[], [], [], [], [], []]
        for split in range(split_numb, len(atnames), split_numb):
            info_tensors = dataStructures.create_info_tensors((atnames[split - split_numb:split]), device=device)
            atom_number, atom_description, coordsIndexingAtom, partnersIndexingAtom, angle_indices = info_tensors
            coordinates = coo[split - split_numb:split]
            partnersFinal1 = torch.full((atom_description.shape[0], 3), (float(PADDING_INDEX)), device=(self.dev))
            partnersFinal2 = torch.full((atom_description.shape[0], 3), (float(PADDING_INDEX)), device=(self.dev))
            padPart1 = partnersIndexingAtom[(Ellipsis, 0)] != PADDING_INDEX
            coordsP1 = coordinates[(atom_description[:, 0].long()[padPart1], partnersIndexingAtom[(Ellipsis, 0)][padPart1])]
            partnersFinal1[padPart1] = coordsP1
            padPart2 = partnersIndexingAtom[(Ellipsis, 1)] != PADDING_INDEX
            coordsP2 = coordinates[(atom_description[:, 0].long()[padPart2], partnersIndexingAtom[(Ellipsis, 1)][padPart2])]
            partnersFinal2[padPart2] = coordsP2
            del coordsP1
            del coordsP2
            del padPart1
            del padPart2
            coords = coordinates[(atom_description[:, 0].long(), coordsIndexingAtom)]
            partnersFinal = torch.cat([partnersFinal1.unsqueeze(1), partnersFinal2.unsqueeze(1)], dim=1)
            del coordsIndexingAtom
            fakeAtoms = fakeAtomsGeneration.generateFakeAtomTensor(coords, partnersFinal, atom_description, hashings.fake_atom_Properties.to(self.dev))
            independent_groups = atom_description[:, hashings.atom_description_hash['batch']]
            atomPairs = []
            for batch in range(independent_groups.max() + 1):
                independentMask = independent_groups.eq(batch)
                atomPairs += [container.getPairwiseRepresentation(coords, atom_number[independentMask], atom_description)]
                torch.cuda.empty_cache()

            atomPairs = torch.cat(atomPairs, dim=0)
            alternativeMask = torch.ones((atom_number.shape), device=(self.dev), dtype=(torch.bool)).unsqueeze(-1)
            contRat, facc, contRatPol, contRatMC, contRatSC = container.solventAccessibiliry(coords, atom_description, atom_number, atomPairs, fakeAtoms, alternativeMask)
            torch.cuda.empty_cache()
            residueEnergyDisulf, atomDISULF, disulfideNetwork = container.disulfide_net(coords, atom_description, atom_number, atomPairs, alternativeMask, partnersFinal, facc)
            torch.cuda.empty_cache()
            timeOld = time.time()
            hbondMC, hbondSC, atomHB, hbondNetwork = container.hbond_net(coords, atom_description, atom_number, atomPairs, fakeAtoms, alternativeMask, disulfideNetwork, partnersFinal, facc)
            clash_masks, first_mask = self.getMasks(coords, atom_description, atomPairs, hbondNetwork, disulfideNetwork)
            atName1 = atom_description[(atomPairs[first_mask][:, 0], hashings.atom_description_hash['at_name'])].long()
            atName2 = atom_description[(atomPairs[first_mask][:, 1], hashings.atom_description_hash['at_name'])].long()
            resname1 = atom_description[(atomPairs[first_mask][:, 0], hashings.atom_description_hash['resname'])].long()
            resname2 = atom_description[(atomPairs[first_mask][:, 1], hashings.atom_description_hash['resname'])].long()
            resnum1 = atom_description[(atomPairs[first_mask][:, 0], hashings.atom_description_hash['resnum'])].long()
            resnum2 = atom_description[(atomPairs[first_mask][:, 1], hashings.atom_description_hash['resnum'])].long()
            radius1 = hashings.atom_Properties[(atName1, hashings.property_hashings['solvenergy_props']['radius'])]
            radius2 = hashings.atom_Properties[(atName2, hashings.property_hashings['solvenergy_props']['radius'])]
            distmat = torch.pairwise_distance(coords[atomPairs[:, 0]], coords[atomPairs[:, 1]])
            dists = distmat[first_mask]
            for e, mask in enumerate(clash_masks):
                vals[e] += [dists[mask] - (radius1 + radius2)[mask]]

        for e in range(len(vals)):
            vals[e] = torch.cat(vals[e])

        max_hash = 18
        self.tollerances = torch.full(size=(len(vals), max_hash, max_hash), fill_value=PADDING_INDEX, device=(self.dev))
        for i in vals:
            print(np.percentile(i.cpu().numpy(), 10))

        for i in vals:
            plt.hist((i.cpu().numpy()), bins=30, alpha=0.2)
            plt.show()

        fd
        for i in range(len(vals)):
            for pair in vals[i].keys():
                value = np.percentile(vals[i][pair].cpu().numpy(), 1)
                self.tollerances[(i, pair[0], pair[1])] = value
                self.tollerances[(i, pair[1], pair[0])] = value

        torch.save(self.tollerances, 'marshalled/tollerancesNew.weights')

    def net(self, coords, atom_description, atomPairs, hbondNetwork, disulfideNetwork):
        clash_masks, first_mask = self.getMasks(coords, atom_description, atomPairs, hbondNetwork, disulfideNetwork)
        distmat = torch.pairwise_distance(coords[atomPairs[first_mask][:, 0]], coords[atomPairs[first_mask][:, 1]])
        assert len(clash_masks) == len(self.tollerances)
        atName1 = atom_description[(atomPairs[first_mask][:, 0], hashings.atom_description_hash['at_name'])].long()
        atName2 = atom_description[(atomPairs[first_mask][:, 1], hashings.atom_description_hash['at_name'])].long()
        radius1 = hashings.atom_Properties[(atName1, hashings.property_hashings['solvenergy_props']['radius'])]
        radius2 = hashings.atom_Properties[(atName2, hashings.property_hashings['solvenergy_props']['radius'])]
        pairs = []
        vals = []
        for i, m in enumerate(clash_masks):
            if True in m:
                clashing_distance = radius1[m] + radius2[m] - (distmat[m] - self.tollerances[i])
                actual_clash = clashing_distance.ge(0)
                todo = m.clone()
                todo[m == True] = actual_clash
                pairs += [atomPairs[first_mask][m][actual_clash]]
                vals += [self.clashPenalty((clashing_distance[actual_clash]), usesoft=False)]

        clash_net = torch.cat(vals, dim=0)
        pairs = torch.cat(pairs, dim=0)
        return (
         pairs, clash_net)

    def getMasks(self, coords, atom_description, atomPairs, hbondNetwork, disulfideNetwork):
        min_dist_look_H = 0.6
        tol_clash_polar = 0.2
        tolerance_hbonds = 0.5
        tolerance_scTobbClash = 0.5
        minDistDisulp = 1.9
        minDistDisulpCBSG = 2.9
        atom_distance_threshold = 5.0
        distmat = torch.pairwise_distance(coords[atomPairs[:, 0]], coords[atomPairs[:, 1]])
        pre_longMask = distmat.le(atom_distance_threshold)
        atomPairs = atomPairs[pre_longMask]
        distmat = distmat[pre_longMask]
        atName1 = atom_description[(atomPairs[:, 0], hashings.atom_description_hash['at_name'])].long()
        atName2 = atom_description[(atomPairs[:, 1], hashings.atom_description_hash['at_name'])].long()
        radius1 = hashings.atom_Properties[(atName1, hashings.property_hashings['solvenergy_props']['radius'])]
        radius2 = hashings.atom_Properties[(atName2, hashings.property_hashings['solvenergy_props']['radius'])]
        virtual1 = hashings.atom_Properties[(atName1, hashings.property_hashings['other_params']['virtual'])] == 0
        virtual2 = hashings.atom_Properties[(atName2, hashings.property_hashings['other_params']['virtual'])] == 0
        non_virtualmask = virtual1 & virtual2
        first_mask = (distmat - (radius1 + radius2 + min_dist_look_H)).le(0) & non_virtualmask
        hbond_mask = hbondNetwork[pre_longMask][first_mask]
        disul_mask = disulfideNetwork[pre_longMask][first_mask]
        atomPairs = atomPairs[first_mask]
        atName = torch.cat([atName1[first_mask].unsqueeze(-1), atName2[first_mask].unsqueeze(-1)], dim=(-1))
        for i, bb_atom in enumerate(self.backbone_atoms):
            if i == 0:
                is_backbone_mask = atName.eq(bb_atom)
            else:
                is_backbone_mask += atName.eq(bb_atom)

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

        same_chain_mask = atom_description[(atomPairs[:, 0], hashings.atom_description_hash['chain'])] == atom_description[(atomPairs[:, 1], hashings.atom_description_hash['chain'])]
        same_residue_mask = (atom_description[(atomPairs[:, 0], hashings.atom_description_hash['resnum'])] == atom_description[(atomPairs[:, 1], hashings.atom_description_hash['resnum'])]) & same_chain_mask
        minus1_residues_mask = (atom_description[(atomPairs[:, 0], hashings.atom_description_hash['resnum'])] - atom_description[(atomPairs[:, 1], hashings.atom_description_hash['resnum'])]).eq(1) & same_chain_mask
        plus1_residues_mask = (atom_description[(atomPairs[:, 0], hashings.atom_description_hash['resnum'])] - atom_description[(atomPairs[:, 1], hashings.atom_description_hash['resnum'])]).eq(-1) & same_chain_mask
        consecutive_residues_mask = minus1_residues_mask | plus1_residues_mask
        for i, pair in enumerate(self.self_covalent_bonds):
            if i == 0:
                covalent_bonds_mask = atName[:, 0].eq(pair[0]) & atName[:, 1].eq(pair[1])
                covalent_bonds_maskReverse = atName[:, 0].eq(pair[1]) & atName[:, 1].eq(pair[0])
            else:
                covalent_bonds_mask += atName[:, 0].eq(pair[0]) & atName[:, 1].eq(pair[1])
                covalent_bonds_maskReverse += atName[:, 0].eq(pair[1]) & atName[:, 1].eq(pair[0])

        for i, pair in enumerate(self.excluded_heavy_OCA):
            if i == 0:
                excluded_heavy_OCA_mask = atName[:, 0].eq(pair[0]) & atName[:, 1].eq(pair[1])
                excluded_heavy_OCA_mask += atName[:, 0].eq(pair[1]) & atName[:, 1].eq(pair[0])
            else:
                excluded_heavy_OCA_mask += atName[:, 0].eq(pair[0]) & atName[:, 1].eq(pair[1])
                excluded_heavy_OCA_mask += atName[:, 0].eq(pair[1]) & atName[:, 1].eq(pair[0])

        bb_of_consecutive_res_mask = (plus1_residues_mask | minus1_residues_mask) & (is_backbone_mask[:, 0] & is_backbone_mask[:, 1])
        excluded = torch.zeros((first_mask.shape), dtype=(torch.bool), device=(self.dev))[first_mask]
        excluded_heavy = torch.zeros((first_mask.shape), dtype=(torch.bool), device=(self.dev))[first_mask]
        first_excluded = plus1_residues_mask & covalent_bonds_mask | minus1_residues_mask & covalent_bonds_maskReverse
        excluded[first_excluded] = True
        excluded_heavy[first_excluded & excluded_heavy_OCA_mask] = True
        connectivity = torch.zeros((atomPairs.shape[0]), dtype=(self.int_type), device=(self.dev))
        for i, natom in enumerate(self.Natoms):
            if i == 0:
                natom_mask = atName.eq(natom)
            else:
                natom_mask += atName.eq(natom)

        for i, catom in enumerate(self.Catoms):
            if i == 0:
                catom_mask = atName.eq(catom)
            else:
                catom_mask += atName.eq(catom)

        for i, terminal in enumerate(self.terminals):
            if i == 0:
                terminal_mask = atName.eq(terminal)
            else:
                terminal_mask += atName.eq(terminal)

        condition1 = (natom_mask[:, 0] | (hashings.atom_Properties[(atName[:, 0], hashings.property_hashings['solvenergy_props']['level'])] == hashings.level_hashing['LEVEL_B'])) & natom_mask[:, 1]
        condition2 = catom_mask[:, 0] & ((hashings.atom_Properties[(
         atName[:, 1], hashings.property_hashings['solvenergy_props']['level'])] == hashings.level_hashing['LEVEL_A']) | catom_mask[:, 1])
        condition3 = ((hashings.atom_Properties[(
         atName[:, 0], hashings.property_hashings['solvenergy_props']['level'])] == hashings.level_hashing['LEVEL_A']) | (hashings.atom_Properties[(
         atName[:, 0], hashings.property_hashings['solvenergy_props']['level'])] == hashings.level_hashing['LEVEL_O'])) & (hashings.atom_Properties[(atName[:, 1], hashings.property_hashings['solvenergy_props']['level'])] == hashings.level_hashing['LEVEL_A'])
        connectivity_peptide_bond_maskP1 = plus1_residues_mask & (condition1 | condition2 | condition3)
        condition1 = (natom_mask[:, 1] | (hashings.atom_Properties[(atName[:, 1], hashings.property_hashings['solvenergy_props']['level'])] == hashings.level_hashing['LEVEL_B'])) & natom_mask[:, 0]
        condition2 = catom_mask[:, 1] & ((hashings.atom_Properties[(
         atName[:, 0], hashings.property_hashings['solvenergy_props']['level'])] == hashings.level_hashing['LEVEL_A']) | catom_mask[:, 0])
        condition3 = ((hashings.atom_Properties[(
         atName[:, 1], hashings.property_hashings['solvenergy_props']['level'])] == hashings.level_hashing['LEVEL_A']) | (hashings.atom_Properties[(
         atName[:, 1], hashings.property_hashings['solvenergy_props']['level'])] == hashings.level_hashing['LEVEL_O'])) & (hashings.atom_Properties[(atName[:, 0], hashings.property_hashings['solvenergy_props']['level'])] == hashings.level_hashing['LEVEL_A'])
        connectivity_peptide_bond_maskP2 = minus1_residues_mask & (condition1 | condition2 | condition3)
        connectivity_peptide_bond_mask = connectivity_peptide_bond_maskP2 | connectivity_peptide_bond_maskP1
        connectivity[connectivity_peptide_bond_mask] = 3
        connectivity[same_residue_mask] = torch.abs(hashings.atom_Properties[(atName[same_residue_mask][:, 0],
         hashings.property_hashings['solvenergy_props']['level'])] - hashings.atom_Properties[(atName[same_residue_mask][:, 1],
         hashings.property_hashings['solvenergy_props']['level'])]).type_as(connectivity)
        ile_connectivity_mask = (atName[:, 0] == hashings.atom_hash['ILE']['CG2']) & (atName[:, 1] == hashings.atom_hash['ILE']['CD1']) | (atName[:, 1] == hashings.atom_hash['ILE']['CG2']) & (atName[:, 0] == hashings.atom_hash['ILE']['CD1'])
        self_pro_mask = (atName[:, 0] == hashings.atom_hash['PRO']['C']) & (atName[:, 1] == hashings.atom_hash['PRO']['CD']) | (atName[:, 1] == hashings.atom_hash['PRO']['C']) & (atName[:, 0] == hashings.atom_hash['PRO']['CD'])
        excluded[same_residue_mask & ~ile_connectivity_mask & (connectivity < 3)] = True
        excluded[same_residue_mask & ile_connectivity_mask] = True
        excluded[self_pro_mask] = True
        cycle1_mask = hashings.atom_Properties[(
         atName[:, 0], hashings.property_hashings['solvenergy_props']['cycle'])].eq(1)
        cycle2_mask = hashings.atom_Properties[(
         atName[:, 1], hashings.property_hashings['solvenergy_props']['cycle'])].eq(1)
        excluded[same_residue_mask & (cycle1_mask & cycle2_mask | terminal_mask[:, 0] | terminal_mask[:, 1])] = True
        ile_pro_correction_mask = same_residue_mask & (ile_connectivity_mask | self_pro_mask | connectivity.le(3))
        connectivity[same_residue_mask] = torch.clamp((connectivity[same_residue_mask]), min=3)
        connectivity[ile_pro_correction_mask] = 3
        low_connectivity_mask = same_residue_mask & connectivity.lt(4)
        disul_CB_SG = (atName[:, 0].eq(self.cbDisul) & atName[:, 1].eq(self.sgDisul) | atName[:, 1].eq(self.cbDisul) & atName[:, 0].eq(self.sgDisul)) & disul_mask
        excluded[disul_CB_SG] = True
        polar_mask = ~disul_mask & (donor_mask[:, 0] | acceptor_mask[:, 0]) & (donor_mask[:, 1] | acceptor_mask[:, 1])
        charge1 = hashings.atom_Properties[(atName[:, 0], hashings.property_hashings['hbond_params']['charge'])]
        charge2 = hashings.atom_Properties[(atName[:, 1], hashings.property_hashings['hbond_params']['charge'])]
        omocharge_mask = ~hbond_mask & (charge1 * charge2).gt(0) & ((atom_description[(atomPairs[:, 0], hashings.atom_description_hash['resname'])] != atom_description[(atomPairs[:, 1], hashings.atom_description_hash['resname'])]) | (atom_description[(atomPairs[:, 0], hashings.atom_description_hash['resname'])] != hashings.resi_hash['HIS']))
        backbone_of_same_residue = same_residue_mask & (is_backbone_mask[:, 0] & is_backbone_mask[:, 1])
        excluded[backbone_of_same_residue] = True
        potentialHbond_mask = donor_mask[:, 1] & acceptor_mask[:, 0]
        resname1 = atom_description[(atomPairs[:, 0], hashings.atom_description_hash['resname'])]
        resname2 = atom_description[(atomPairs[:, 1], hashings.atom_description_hash['resname'])]
        nd1nd2ToBB_mask = same_residue_mask & ((resname1 == hashings.resi_hash['HIS']) & ((atName[:, 0] == hashings.atom_hash['HIS']['ND1']) | atName[:, 0].eq(hashings.atom_hash['HIS']['NE2'])) & is_backbone_mask[:, 1] | resname2.eq(hashings.resi_hash['HIS']) & ((atName[:, 1].eq(hashings.atom_hash['HIS']['ND1']) | atName[:, 1].eq(hashings.atom_hash['HIS']['NE2'])) & is_backbone_mask[:, 0]))
        hbondM1 = is_backbone_mask[:, 0] & donor_mask[:, 0] & ~is_backbone_mask[:, 1] | is_backbone_mask[:, 1] & donor_mask[:, 1] & ~is_backbone_mask[:, 0]
        hbondM2 = is_backbone_mask[:, 0] & acceptor_mask[:, 0] & ~is_backbone_mask[:, 1] | is_backbone_mask[:, 1] & acceptor_mask[:, 1] & ~is_backbone_mask[:, 0]
        hbondClashMask = polar_mask & ~omocharge_mask & potentialHbond_mask & ~nd1nd2ToBB_mask & hbond_mask
        isCG1A1 = (atName[:, 0] == hashings.atom_hash['ILE']['CG1']) | (atName[:, 0] == hashings.atom_hash['VAL']['CG1'])
        isCG2A1 = (atName[:, 0] == hashings.atom_hash['ILE']['CG2']) | (atName[:, 0] == hashings.atom_hash['VAL']['CG2']) | (atName[:, 0] == hashings.atom_hash['THR']['CG2'])
        NCG_Masko = plus1_residues_mask & (natom_mask[:, 1] & (isCG1A1 | isCG2A1))
        isCG1A1r = (atName[:, 1] == hashings.atom_hash['ILE']['CG1']) | (atName[:, 1] == hashings.atom_hash['VAL']['CG1'])
        isCG2A1r = (atName[:, 1] == hashings.atom_hash['ILE']['CG2']) | (atName[:, 1] == hashings.atom_hash['VAL']['CG2']) | (atName[:, 1] == hashings.atom_hash['THR']['CG2'])
        NCG_Maskr = minus1_residues_mask & (natom_mask[:, 0] & (isCG1A1r | isCG2A1r))
        NCG_Mask = NCG_Masko | NCG_Maskr
        donor_acceptor = donor_mask[:, 1] & acceptor_mask[:, 0]
        selfClash = same_residue_mask & ~excluded
        otherClash = ~same_residue_mask & ~excluded
        actual_group_masks = [
         selfClash,
         otherClash & (~hbond_mask & ~disul_mask & ~bb_of_consecutive_res_mask) & ~donor_acceptor,
         otherClash & (~hbond_mask & ~disul_mask & ~bb_of_consecutive_res_mask) & donor_acceptor,
         otherClash & bb_of_consecutive_res_mask,
         otherClash & hbond_mask,
         otherClash & disul_mask]
        pre_longMask[pre_longMask == True] = first_mask
        return (
         actual_group_masks, pre_longMask)

    def hardClash(self, clashDistance, distance, usesoft=False):
        dif = clashDistance - distance
        fin = torch.zeros((dif.shape), dtype=(self.float_type), device=(self.dev))
        clash_grad_limit = 0
        reluObj = torch.nn.ReLU()
        return self.clashPenalty(reluObj(dif), usesoft)

    def clashPenalty(self, dist, usesoft):
        threshold = 0.5
        linear_coeff = 30
        linear = dist.gt(0.5)
        ret = torch.zeros(dist.shape).type_as(dist)
        if usesoft:
            ret[~linear] = torch.exp(dist[~linear] * 5 - 5)
            ret[linear] = np.exp(threshold * 5 - 5) + linear_coeff * (dist[linear] - threshold)
            return ret
        ret[~linear] = torch.exp(dist[~linear] * 10 - 2)
        ret[linear] = np.exp(threshold * 10 - 2) + linear_coeff * (dist[linear] - threshold)
        return ret

    def forward(self, coords, atom_description, atom_number, atomPairs, alternativeMask, facc, hbondNetwork, disulfideNetwork):
        atomPairs, clashNet = self.net(coords, atom_description, atomPairs, hbondNetwork, disulfideNetwork)
        atomEnergy = self.bindToAtoms(clashNet, atomPairs, alternativeMask)
        residueEnergy = self.bindToResi(atomEnergy, atom_description, facc, alternativeMask)
        clash_mask = torch.zeros(atomPairs.shape[0], dtype=torch.bool, device=self.dev)
        clashing_pairs_indices = atomPairs.cpu().numpy()
        original_indices = atomPairs.cpu().numpy()
        clashing_rows = np.where((original_indices[:, None] == clashing_pairs_indices).all(-1).any(-1))[0]
        clash_mask[clashing_rows] = True
        return (
         residueEnergy, atomEnergy, clash_mask)

    def bindToAtoms(self, atomAtom, atomPairs, alternMask):
        value = atomAtom
        capping = 999
        capping_mask = value.ge(capping)
        value[capping_mask] = capping + (value[capping_mask] - capping) * 0.00012001
        netEnergy = value * 0.5
        energy_atomAtom = torch.zeros((alternMask.shape[0]), (alternMask.shape[-1]), dtype=(torch.float), device=(self.dev))
        for alt in range(alternMask.shape[-1]):
            mask = alternMask[(atomPairs[:, 0], alt)] & alternMask[(atomPairs[:, 1], alt)]
            alt_index = torch.full((mask.shape), alt, device=(self.dev), dtype=(torch.long))[mask]
            atomPairAlter = atomPairs[mask]
            energy_atomAtom.index_put_((atomPairAlter[:, 0], alt_index), (netEnergy[mask]), accumulate=True)
            energy_atomAtom.index_put_((atomPairAlter[:, 1], alt_index), (netEnergy[mask]), accumulate=True)

        return energy_atomAtom

    def bindToResi(self, atomEnergy, atom_description, facc, alternativeMask, minSaCoefficient=0.5):
        batch_ind = atom_description[:, hashings.atom_description_hash['batch']].long().unsqueeze(-1).expand(-1, alternativeMask.shape[-1])
        resnum = atom_description[:, hashings.atom_description_hash['resnum']].long().unsqueeze(-1).expand(-1, alternativeMask.shape[-1])
        chain_ind = atom_description[:, hashings.atom_description_hash['chain']].long().unsqueeze(-1).expand(-1, alternativeMask.shape[-1])
        atName = atom_description[:, hashings.atom_description_hash['at_name']].long().unsqueeze(-1).expand(-1, alternativeMask.shape[-1])
        batch = batch_ind.max() + 1
        nres = resnum.max() + 1
        nchains = chain_ind.max() + 1
        naltern = alternativeMask.shape[-1]
        alt_index = torch.arange(0, naltern, dtype=(torch.long), device=(self.dev)).unsqueeze(0).expand(atomEnergy.shape[0], naltern)
        mask_padding = ~resnum.eq(PADDING_INDEX)
        final_h2s = torch.zeros((batch, nchains, nres, naltern), dtype=(torch.float), device=(self.dev))
        final_h1s = torch.zeros((batch, nchains, nres, naltern), dtype=(torch.float), device=(self.dev))
        final_his = torch.zeros((batch, nchains, nres, naltern), dtype=(torch.float), device=(self.dev))
        saCoefficient = torch.max(facc, torch.tensor(minSaCoefficient, device=(self.dev), dtype=(torch.float)))
        atomEnergy = atomEnergy * saCoefficient * (1 - torch.tanh(self.weight)) * 0.27
        highMask = atomEnergy.gt(5)
        nd1 = atName.eq(hashings.atom_hash['HIS']['ND1'])
        ne2 = atName.eq(hashings.atom_hash['HIS']['NE2'])
        his = (nd1 | ne2) & mask_padding
        nd1H1S = atName.eq(hashings.atom_hash['HIS']['ND1H1S'])
        ne2H1S = atName.eq(hashings.atom_hash['HIS']['NE2H1S'])
        h1s = (nd1H1S | ne2H1S) & mask_padding
        nd1H2S = atName.eq(hashings.atom_hash['HIS']['ND1H2S'])
        ne2H2S = atName.eq(hashings.atom_hash['HIS']['NE2H2S'])
        h2s = (nd1H2S | ne2H2S) & mask_padding
        his_mask = ~h1s & ~h2s & mask_padding
        indices = (batch_ind[his_mask], chain_ind[his_mask], resnum[his_mask].long(), alt_index[his_mask].long())
        final_his = final_his.index_put(indices, (atomEnergy[his_mask]), accumulate=True)
        his_mask = ~his & ~h2s & mask_padding
        indices = (batch_ind[his_mask], chain_ind[his_mask], resnum[his_mask].long(), alt_index[his_mask].long())
        final_h1s = final_h1s.index_put(indices, (atomEnergy[his_mask]), accumulate=True)
        his_mask = ~h1s & ~his & mask_padding
        indices = (batch_ind[his_mask], chain_ind[his_mask], resnum[his_mask].long(), alt_index[his_mask].long())
        final_h2s = final_h2s.index_put(indices, (atomEnergy[his_mask]), accumulate=True)
        final = torch.cat([final_his.unsqueeze(-1), final_h1s.unsqueeze(-1), final_h2s.unsqueeze(-1)], dim=(-1))
        return final

    def getWeights(self):
        pass

    def getNumParams(self):
        p = []
        for i in self.parameters():
            p += list(i.data.cpu().numpy().flat)

        print('Number of parameters=', len(p))

# okay decompiling Clash_net37.pyc
