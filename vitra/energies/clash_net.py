#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import numpy as np
from vitra.sources.globalVariables import PADDING_INDEX
from vitra.sources import hashings

COVALENT_BONDS = [('O', 'N'), ('C', 'N'), ('C', 'CA'), ('O', 'CA'), ('CA', 'N')]
PRO_COVALENT_BONDS = [('C', 'CD'), ('CA', 'CD')]
EXCLUDED_HEAVY_OCA = [('O', 'CA')]


class ClashEnergy(torch.nn.Module):
    """
    Calculates the clashing energy between atoms.
    """

    def __init__(self, name='clashes', dev='cpu', backbone_atoms=None, donor=None, acceptor=None, **_kwargs):
        super().__init__()
        self.name = name
        self.dev = dev
        self.int_type = torch.short
        self.weight = torch.nn.Parameter(torch.tensor([0.0], device=self.dev))
        self.backbone_atoms = backbone_atoms if backbone_atoms is not None else []
        self.donor = donor if donor is not None else []
        self.acceptor = acceptor if acceptor is not None else []
        self._initialize_bond_data()
        self._initialize_atom_lists()
        self._initialize_hash_short()
        self.tollerances = torch.tensor([-0.0315, 0.182, 0.061, -0.036, 0.0007, -1.313],
                                        device=self.dev, dtype=torch.float)

    def _initialize_bond_data(self):
        """Initializes covalent bond data."""
        self.self_covalent_bonds = []
        for pair in COVALENT_BONDS:
            for res1 in hashings.resi_hash:
                for res2 in hashings.resi_hash:
                    self.self_covalent_bonds.append(
                        (hashings.atom_hash[res1][pair[0]], hashings.atom_hash[res2][pair[1]]))
        for pair in PRO_COVALENT_BONDS:
            for res1 in hashings.resi_hash:
                self.self_covalent_bonds.append(
                    (hashings.atom_hash[res1][pair[0]], hashings.atom_hash['PRO'][pair[1]]))

        self.excluded_heavy_oca = []
        for pair in EXCLUDED_HEAVY_OCA:
            for res1 in hashings.resi_hash:
                for res2 in hashings.resi_hash:
                    self.excluded_heavy_oca.append(
                        (hashings.atom_hash[res1][pair[0]], hashings.atom_hash[res2][pair[1]]))

        self.self_covalent_bonds = list(set(self.self_covalent_bonds))

    def _initialize_atom_lists(self):
        """Initializes lists of atom types."""
        self.natoms, self.catoms, self.oatoms, self.caatoms = [], [], [], []
        for res1 in hashings.resi_hash:
            self.natoms.extend([hashings.atom_hash[res1]['N'], hashings.atom_hash[res1]['tN']])
            self.catoms.append(hashings.atom_hash[res1]['C'])
            self.oatoms.append(hashings.atom_hash[res1]['O'])
            self.caatoms.append(hashings.atom_hash[res1]['CA'])

        self.natoms = list(set(self.natoms))
        self.catoms = list(set(self.catoms))
        self.oatoms = list(set(self.oatoms))
        self.caatoms = list(set(self.caatoms))
        self.cb_disul = hashings.atom_hash['CYS']['CB']
        self.sg_disul = hashings.atom_hash['CYS']['SG']
        self.terminals = []
        for res1 in hashings.resi_hash:
            self.terminals.extend([hashings.atom_hash[res1]['tN'], hashings.atom_hash[res1]['OXT']])
        self.terminals = list(set(self.terminals))

    def _initialize_hash_short(self):
        """Initializes the short hash tensor."""
        new_hash = hashings.atom_hashTPL
        max_hash = -1
        for r in hashings.atom_hash:
            max_hash = max(max_hash, max(hashings.atom_hash[r].values()))
        max_hash += 1
        self.hash_short = torch.zeros(max_hash, dtype=torch.long, device=self.dev)
        for r, atoms in hashings.atom_hash.items():
            for at, at_hash in atoms.items():
                if at in new_hash.get(r, {}):
                    self.hash_short[at_hash] = new_hash[r][at]
                else:
                    self.hash_short[at_hash] = PADDING_INDEX

    def load(self):
        """Loads the tolerances from a file."""
        self.tollerances = torch.load('marshalled/tollerancesNew.weights').to(self.dev)

    def net(self, coords, atom_description, atom_pairs, hbond_network, disulfide_network):
        """Calculates the clashing energy."""
        clash_masks, first_mask = self.get_masks(
            coords, atom_description, atom_pairs, hbond_network, disulfide_network)
        distmat = torch.pairwise_distance(
            coords[atom_pairs[first_mask][:, 0]], coords[atom_pairs[first_mask][:, 1]])
        assert len(clash_masks) == len(self.tollerances)
        at_name1 = atom_description[
            (atom_pairs[first_mask][:, 0], hashings.atom_description_hash['at_name'])].long()
        at_name2 = atom_description[
            (atom_pairs[first_mask][:, 1], hashings.atom_description_hash['at_name'])].long()
        radius1 = hashings.atom_Properties[
            (at_name1, hashings.property_hashings['solvenergy_props']['radius'])]
        radius2 = hashings.atom_Properties[
            (at_name2, hashings.property_hashings['solvenergy_props']['radius'])]
        pairs = []
        vals = []
        for i, m in enumerate(clash_masks):
            if m.any():
                clashing_distance = radius1[m] + radius2[m] - (distmat[m] - self.tollerances[i])
                actual_clash = clashing_distance.ge(0)
                pairs.append(atom_pairs[first_mask][m][actual_clash])
                vals.append(self.clash_penalty(clashing_distance[actual_clash], usesoft=False))

        if not vals:
            return torch.empty(0, 2, dtype=torch.long, device=self.dev), torch.empty(0, device=self.dev)

        clash_net = torch.cat(vals, dim=0)
        pairs = torch.cat(pairs, dim=0)
        return pairs, clash_net

    def get_masks(self, coords, atom_description, atom_pairs, hbond_network, disulfide_network):
        """
        Calculates masks for different types of clashes.
        """
        min_dist_look_H = 0.6
        atom_distance_threshold = 5.0
        distmat = torch.pairwise_distance(coords[atom_pairs[:, 0]], coords[atom_pairs[:, 1]])
        pre_long_mask = distmat.le(atom_distance_threshold)
        atom_pairs = atom_pairs[pre_long_mask]
        distmat = distmat[pre_long_mask]
        at_name1 = atom_description[(atom_pairs[:, 0], hashings.atom_description_hash['at_name'])].long()
        at_name2 = atom_description[(atom_pairs[:, 1], hashings.atom_description_hash['at_name'])].long()
        radius1 = hashings.atom_Properties[
            (at_name1, hashings.property_hashings['solvenergy_props']['radius'])]
        radius2 = hashings.atom_Properties[
            (at_name2, hashings.property_hashings['solvenergy_props']['radius'])]
        virtual1 = hashings.atom_Properties[
                       (at_name1, hashings.property_hashings['other_params']['virtual'])] == 0
        virtual2 = hashings.atom_Properties[
                       (at_name2, hashings.property_hashings['other_params']['virtual'])] == 0
        non_virtualmask = virtual1 & virtual2
        first_mask = (distmat - (radius1 + radius2 + min_dist_look_H)).le(0) & non_virtualmask
        atom_pairs = atom_pairs[first_mask]
        at_name = torch.cat([at_name1[first_mask].unsqueeze(-1), at_name2[first_mask].unsqueeze(-1)], dim=-1)

        is_backbone_mask = torch.zeros_like(at_name, dtype=torch.bool)
        for bb_atom in self.backbone_atoms:
            is_backbone_mask |= at_name.eq(bb_atom)

        same_chain_mask = atom_description[(atom_pairs[:, 0], hashings.atom_description_hash['chain'])] == \
                          atom_description[(atom_pairs[:, 1], hashings.atom_description_hash['chain'])]
        minus1_residues_mask = (atom_description[(atom_pairs[:, 0], hashings.atom_description_hash['resnum'])] -
                                atom_description[(atom_pairs[:, 1], hashings.atom_description_hash['resnum'])]).eq(1) & same_chain_mask
        plus1_residues_mask = (atom_description[(atom_pairs[:, 0], hashings.atom_description_hash['resnum'])] -
                               atom_description[(atom_pairs[:, 1], hashings.atom_description_hash['resnum'])]).eq(-1) & same_chain_mask

        covalent_bonds_mask = torch.zeros_like(at_name[:, 0], dtype=torch.bool)
        covalent_bonds_mask_reverse = torch.zeros_like(at_name[:, 0], dtype=torch.bool)
        for pair in self.self_covalent_bonds:
            covalent_bonds_mask |= at_name[:, 0].eq(pair[0]) & at_name[:, 1].eq(pair[1])
            covalent_bonds_mask_reverse |= at_name[:, 0].eq(pair[1]) & at_name[:, 1].eq(pair[0])

        excluded_heavy_oca_mask = torch.zeros_like(at_name[:, 0], dtype=torch.bool)
        for pair in self.excluded_heavy_oca:
            excluded_heavy_oca_mask |= (at_name[:, 0].eq(pair[0]) & at_name[:, 1].eq(pair[1])) | \
                                       (at_name[:, 0].eq(pair[1]) & at_name[:, 1].eq(pair[0]))

        excluded = torch.zeros(first_mask.shape, dtype=torch.bool, device=self.dev)[first_mask]
        first_excluded = (plus1_residues_mask & covalent_bonds_mask) | (
                    minus1_residues_mask & covalent_bonds_mask_reverse)
        excluded[first_excluded] = True
        excluded[first_excluded & excluded_heavy_oca_mask] = True

        pre_long_mask[pre_long_mask.clone()] = first_mask

        at_name1 = atom_description[(atom_pairs[:, 0], hashings.atom_description_hash['at_name'])].long()
        at_name2 = atom_description[(atom_pairs[:, 1], hashings.atom_description_hash['at_name'])].long()

        type1 = self.hash_short[at_name1]
        type2 = self.hash_short[at_name2]

        clash_masks = []
        for i in range(6):
            mask = (type1 == i) | (type2 == i)
            clash_masks.append(mask)

        return clash_masks, pre_long_mask

    def hard_clash(self, clash_distance, distance, usesoft=False):
        """Calculates the hard clash penalty."""
        dif = clash_distance - distance
        relu_obj = torch.nn.ReLU()
        return self.clash_penalty(relu_obj(dif), usesoft)

    def clash_penalty(self, dist, usesoft):
        """Calculates the clash penalty."""
        threshold = 0.5
        linear_coeff = 30
        linear = dist.gt(0.5)
        ret = torch.zeros(dist.shape).type_as(dist)
        if usesoft:
            ret[~linear] = torch.exp(dist[~linear] * 5 - 5)
            ret[linear] = np.exp(threshold * 5 - 5) + linear_coeff * (dist[linear] - threshold)
        else:
            ret[~linear] = torch.exp(dist[~linear] * 10 - 2)
            ret[linear] = np.exp(threshold * 10 - 2) + linear_coeff * (dist[linear] - threshold)
        return ret

    def forward(self, coords, atom_description, atom_number, atom_pairs,
                alternative_mask, facc, hbond_network, disulfide_network):
        """Forward pass for the ClashEnergy module."""
        atom_pairs, clash_net = self.net(
            coords, atom_description, atom_pairs, hbond_network, disulfide_network)
        atom_energy = self.bind_to_atoms(clash_net, atom_pairs, alternative_mask)
        residue_energy = self.bind_to_resi(atom_energy, atom_description, facc, alternative_mask)
        clash_mask = torch.zeros(atom_pairs.shape[0], dtype=torch.bool, device=self.dev)
        if atom_pairs.numel() > 0:
            clashing_pairs_indices = atom_pairs.cpu().numpy()
            original_indices = atom_pairs.cpu().numpy()
            clashing_rows = np.where(
                (original_indices[:, None] == clashing_pairs_indices).all(-1).any(-1))[0]
            clash_mask[clashing_rows] = True
        return (residue_energy, atom_energy, clash_mask)

    def bind_to_atoms(self, atom_atom, atom_pairs, altern_mask):
        """Binds the clashing energy to atoms."""
        value = atom_atom
        capping = 999
        capping_mask = value.ge(capping)
        value[capping_mask] = capping + (value[capping_mask] - capping) * 0.00012001
        net_energy = value * 0.5
        energy_atom_atom = torch.zeros((altern_mask.shape[0]), (altern_mask.shape[-1]),
                                       dtype=torch.float, device=self.dev)
        for alt in range(altern_mask.shape[-1]):
            if atom_pairs.numel() > 0:
                mask = altern_mask[(atom_pairs[:, 0], alt)] & altern_mask[(atom_pairs[:, 1], alt)]
                alt_index = torch.full(mask.shape, alt, device=self.dev, dtype=torch.long)[mask]
                atom_pair_alter = atom_pairs[mask]
                energy_atom_atom.index_put_((atom_pair_alter[:, 0], alt_index), (net_energy[mask]), accumulate=True)
                energy_atom_atom.index_put_((atom_pair_alter[:, 1], alt_index), (net_energy[mask]), accumulate=True)
        return energy_atom_atom

    def bind_to_resi(self, atom_energy, atom_description, facc, alternative_mask, min_sa_coefficient=0.5):
        """Binds the clashing energy to residues."""
        batch_ind = atom_description[:, hashings.atom_description_hash['batch']].long().unsqueeze(-1).expand(
            -1, alternative_mask.shape[-1])
        resnum = atom_description[:, hashings.atom_description_hash['resnum']].long().unsqueeze(-1).expand(
            -1, alternative_mask.shape[-1])
        chain_ind = atom_description[:, hashings.atom_description_hash['chain']].long().unsqueeze(-1).expand(
            -1, alternative_mask.shape[-1])
        at_name = atom_description[:, hashings.atom_description_hash['at_name']].long().unsqueeze(-1).expand(
            -1, alternative_mask.shape[-1])
        batch = batch_ind.max() + 1
        nres = resnum.max() + 1
        nchains = chain_ind.max() + 1
        naltern = alternative_mask.shape[-1]
        alt_index = torch.arange(
            0, naltern, dtype=torch.long, device=self.dev).unsqueeze(0).expand(atom_energy.shape[0], naltern)
        mask_padding = ~resnum.eq(PADDING_INDEX)
        final_h2s = torch.zeros((batch, nchains, nres, naltern), dtype=torch.float, device=self.dev)
        final_h1s = torch.zeros((batch, nchains, nres, naltern), dtype=torch.float, device=self.dev)
        final_his = torch.zeros((batch, nchains, nres, naltern), dtype=torch.float, device=self.dev)
        sa_coefficient = torch.max(facc, torch.tensor(min_sa_coefficient, device=self.dev, dtype=torch.float))
        atom_energy = atom_energy * sa_coefficient * (1 - torch.tanh(self.weight)) * 0.27
        nd1 = at_name.eq(hashings.atom_hash['HIS']['ND1'])
        ne2 = at_name.eq(hashings.atom_hash['HIS']['NE2'])
        his = (nd1 | ne2) & mask_padding
        nd1h1s = at_name.eq(hashings.atom_hash['HIS']['ND1H1S'])
        ne2h1s = at_name.eq(hashings.atom_hash['HIS']['NE2H1S'])
        h1s = (nd1h1s | ne2h1s) & mask_padding
        nd1h2s = at_name.eq(hashings.atom_hash['HIS']['ND1H2S'])
        ne2h2s = at_name.eq(hashings.atom_hash['HIS']['NE2H2S'])
        h2s = (nd1h2s | ne2h2s) & mask_padding
        his_mask = ~h1s & ~h2s & mask_padding
        indices = (batch_ind[his_mask], chain_ind[his_mask], resnum[his_mask].long(), alt_index[his_mask].long())
        final_his = final_his.index_put(indices, (atom_energy[his_mask]), accumulate=True)
        his_mask = ~his & ~h2s & mask_padding
        indices = (batch_ind[his_mask], chain_ind[his_mask], resnum[his_mask].long(), alt_index[his_mask].long())
        final_h1s = final_h1s.index_put(indices, (atom_energy[his_mask]), accumulate=True)
        his_mask = ~h1s & ~his & mask_padding
        indices = (batch_ind[his_mask], chain_ind[his_mask], resnum[his_mask].long(), alt_index[his_mask].long())
        final_h2s = final_h2s.index_put(indices, (atom_energy[his_mask]), accumulate=True)
        final = torch.cat([final_his.unsqueeze(-1), final_h1s.unsqueeze(-1), final_h2s.unsqueeze(-1)], dim=-1)
        return final

    def get_weights(self):
        """Returns the weights of the module."""

    def get_num_params(self):
        """Returns the number of parameters in the module."""
        p = []
        for i in self.parameters():
            p += list(i.data.cpu().numpy().flat)
        print(f'Number of parameters= {len(p)}')
