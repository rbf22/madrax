#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from vitra.sources import hashings
from vitra.sources.globalVariables import PADDING_INDEX, TEMPERATURE

LETTERS = {
    'CYS': 'C', 'ASP': 'D', 'SER': 'S', 'GLN': 'Q', 'LYS': 'K', 'ASN': 'N',
    'PRO': 'P', 'THR': 'T', 'PHE': 'F', 'ALA': 'A', 'HIS': 'H', 'GLY': 'G',
    'ILE': 'I', 'LEU': 'L', 'ARG': 'R', 'TRP': 'W', 'VAL': 'V', 'GLU': 'E',
    'TYR': 'Y', 'MET': 'M'
}
INVERSE_LETTERS = {v: k for k, v in LETTERS.items()}


class SideChainEntropyEnergy(torch.nn.Module):

    def __init__(self, name='EntropySC', dev='cpu', **_kwargs):
        super().__init__()
        self.name = name
        self.dev = dev
        self.entropy_abayan = {
            'ALA': 0, 'CYS': 0.00375, 'ASP': 0.0020333, 'GLU': 0.0055, 'PHE': 0.0029333,
            'GLY': 0, 'HIS': 0.0033, 'ILE': 0.00375, 'LYS': 0.0073667, 'LEU': 0.00375,
            'MET': 0.0051, 'ASN': 0.0027, 'PRO': 0.001, 'GLN': 0.0067333, 'ARG': 0.0071,
            'SER': 0.0020667, 'THR': 0.0020333, 'VAL': 0.0016667, 'TRP': 0.0032333,
            'TYR': 0.0033
        }
        self.ca_atoms = []
        for res1 in hashings.resi_hash:
            self.ca_atoms.append(hashings.atom_hash[res1]['CA'])
        self.ca_atoms = list(set(self.ca_atoms))
        self.weight = torch.nn.Parameter(torch.tensor([0.0], device=self.dev))

    def _get_ca_mask(self, atom_description):
        """Gets the mask for C-alpha atoms."""
        at_name = atom_description[:, hashings.atom_description_hash['at_name']]
        ca_mask = torch.zeros_like(at_name, dtype=torch.bool)
        for ca_atom in self.ca_atoms:
            ca_mask |= at_name.eq(ca_atom)
        return ca_mask

    def _calculate_lookup_entropy(self, seq):
        """Calculates the lookup entropy for each residue."""
        lookup_entropy_sc = torch.zeros(seq.shape, dtype=torch.float, device=self.dev)
        for res_name, res_hash in hashings.resi_hash.items():
            mask = seq.eq(res_hash)
            if mask.any():
                lookup_entropy_sc[mask] = (1 - torch.tanh(-self.weight)) * \
                                          TEMPERATURE * self.entropy_abayan[res_name]
        return lookup_entropy_sc

    def forward(self, atom_description, sa_sc, hbond, vdw, electro, clash, alternatives):
        """
        Forward pass for the SideChainEntropyEnergy module.
        """
        ca_mask = self._get_ca_mask(atom_description)
        seq = atom_description[ca_mask][:, hashings.atom_description_hash['resname']]

        batch_size = sa_sc.shape[0]
        n_chains = sa_sc.shape[1]
        n_resi = sa_sc.shape[2]
        n_altern = sa_sc.shape[-1]

        lookup_entropy_sc = self._calculate_lookup_entropy(seq)

        batch_ind = atom_description[:, hashings.atom_description_hash['batch']][ca_mask].long()
        resnum = atom_description[:, hashings.atom_description_hash['resnum']][ca_mask].long()
        chain_ind = atom_description[:, hashings.atom_description_hash['chain']][ca_mask].long()

        final_entropy_sc = torch.zeros((batch_size, n_chains, n_resi, n_altern),
                                       device=self.dev, dtype=torch.float)
        mask_padding = ~seq.eq(PADDING_INDEX)

        for alt in range(n_altern):
            mask_alt = alternatives[ca_mask][:, alt] & mask_padding
            alt_index = torch.full(mask_alt.shape, alt, device=self.dev, dtype=torch.long)
            index = (batch_ind[mask_alt], chain_ind[mask_alt], resnum[mask_alt], alt_index[mask_alt])
            final_entropy_sc.index_put_(index, lookup_entropy_sc[mask_alt])

        lookup_entropy_sc = final_entropy_sc.clone()
        final_entropy_sc = final_entropy_sc * torch.nn.functional.relu(sa_sc)

        corr = torch.ones(electro.shape).type_as(electro)
        corr[electro > 0] = 0.2
        residue_energy = torch.abs(
            hbond.sum(-1) + vdw.squeeze(-1) + electro.squeeze(-1) + clash.sum(-1)
        )

        if residue_energy.shape != lookup_entropy_sc.shape:
            residue_energy, _ = residue_energy.max(dim=-1, keepdim=True)

        mask1 = lookup_entropy_sc < residue_energy
        mask2 = ~mask1 & (final_entropy_sc < residue_energy)
        final_entropy_sc[mask1] = lookup_entropy_sc[mask1]
        final_entropy_sc[mask2] = residue_energy[mask2]

        return final_entropy_sc

    def get_weights(self):
        """Returns the weights of the module."""
        pass

    def get_num_params(self):
        """Returns the number of parameters in the module."""
        p = []
        for i in self.parameters():
            p += list(i.data.cpu().numpy().flat)
        print(f'Number of parameters= {len(p)}')
