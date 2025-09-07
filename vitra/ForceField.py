#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import torch

from vitra.sources import hashings
from vitra.sources import math_utils
from vitra.energies import AngleScorerEnergy, BondLengthConstraintEnergy, ClashEnergy, DisulfideEnergy, ElectrostaticsEnergy, \
    SideChainEntropyEnergy, HBondNet, Solvatation, SolventAccessibility, Vdw
from vitra.energies.disulfide_net import DisulfideData
from vitra.energies.electro_net import ElectrostaticsData
from vitra.sources import fakeAtomsGeneration
from vitra.sources.globalVariables import PADDING_INDEX

HBOND_DONORS_SIDECHAIN = {
    "ARG": ["NE", "NH1", "NH2"], "ASN": ["ND2"], "GLN": ["NE2"],
    "HIS": ["ND1", "NE2", "ND1H1S", "NE2H2S"], "LYS": ["NZ"], "SER": ["OG"],
    "THR": ["OG1"], "TRP": ["NE1"], "TYR": ["OH"], "CYS": ["SG"]
}
HBOND_ACCEPTOR_SIDECHAIN = {
    "ASN": ["OD1"], "ASP": ["OD1", "OD2"], "GLN": ["OE1"], "GLU": ["OE1", "OE2"],
    "HIS": ["ND1H2S", "NE2H1S"], "SER": ["OG"], "THR": ["OG1"], "TYR": ["OH"], "CYS": ["SG"]
}
HBOND_AROMATIC = {
    "HIS": ["ND1", "NE2"], "PHE": ["RC"], "TYR": ["RC"]
}
BACKBONE_ATOMS = ["N", "C", "O", "CA", "tN", "OXT"]


def score_min(x, dim, score):
    """Gives you the tensor with the dim dimension as min."""
    _tmp = [1] * len(x.size())
    _tmp[dim] = x.size(dim)
    return torch.gather(x, dim, score.min(dim)[1].unsqueeze(dim).repeat(tuple(_tmp))).select(dim, 0)


class ForceField(torch.nn.Module):
    """
    The main ForceField class that combines all energy terms.
    """

    def __init__(self, device='cpu'):
        """
        It initializes the force field object
        """

        super().__init__()
        hashings.atom_Properties = hashings.atom_Properties.to(device)
        hashings.fake_atom_Properties = hashings.fake_atom_Properties.to(device)
        self.device = device
        self._initialize_atom_sets()
        weights_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                    "parameters", "final_model.weights")
        self.load_state_dict(torch.load(weights_path, map_location=device), strict=False)

    def _initialize_atom_sets(self):
        """Initializes sets of atoms for different purposes."""
        hbond_donor = set()
        for res, atoms in HBOND_DONORS_SIDECHAIN.items():
            for at in atoms:
                hbond_donor.add(hashings.atom_hash[res][at])
        for res in hashings.resi_hash:
            if res != "PRO":
                hbond_donor.add(hashings.atom_hash[res]["N"])
                hbond_donor.add(hashings.atom_hash[res]["tN"])
        self.donor = list(hbond_donor)

        hbond_acceptor = set()
        for res, atoms in HBOND_ACCEPTOR_SIDECHAIN.items():
            for at in atoms:
                hbond_acceptor.add(hashings.atom_hash[res][at])
        for res in hashings.resi_hash:
            hbond_acceptor.add(hashings.atom_hash[res]["O"])
        self.acceptor = list(hbond_acceptor)

        hbond_ar = set()
        for res, atoms in HBOND_AROMATIC.items():
            for at in atoms:
                hbond_ar.add(hashings.atom_hash[res][at])
        self.aromatic = list(hbond_ar)

        bb_atoms = set()
        for res in hashings.resi_hash:
            for at in BACKBONE_ATOMS:
                bb_atoms.add(hashings.atom_hash[res][at])
        self.backbone = list(bb_atoms)

    def get_pairwise_representation(self, coords, atom_number, distance_threshold=7):
        """
        Calculates the pairwise representation of atoms.
        """
        num_atoms = atom_number.shape[0]
        atom_number_pw1 = atom_number.unsqueeze(1).expand(num_atoms, num_atoms).reshape(-1)
        atom_number_pw2 = atom_number.unsqueeze(0).expand(num_atoms, num_atoms).reshape(-1)

        padding_mask = ~(coords[atom_number_pw1, 0].eq(PADDING_INDEX) |
                         coords[atom_number_pw2, 0].eq(PADDING_INDEX))

        atom_number_pw1 = atom_number_pw1[padding_mask]
        atom_number_pw2 = atom_number_pw2[padding_mask]

        long_mask = torch.pairwise_distance(coords[atom_number_pw1.long()],
                                          coords[atom_number_pw2.long()]).le(distance_threshold)

        atom_number_pw1 = atom_number_pw1[long_mask]
        atom_number_pw2 = atom_number_pw2[long_mask]

        return torch.cat([atom_number_pw1.unsqueeze(-1), atom_number_pw2.unsqueeze(-1)], dim=-1)

    def _prepare_tensors(self, coordinates, info_tensors):
        """Prepares the initial tensors for the energy calculation."""
        atom_number, atom_description, coords_indexing_atom, partners_indexing_atom, _, _ = info_tensors

        partners_final1 = torch.full((atom_description.shape[0], 3), float(PADDING_INDEX), device=self.device)
        partners_final2 = torch.full((atom_description.shape[0], 3), float(PADDING_INDEX), device=self.device)

        pad_part1 = partners_indexing_atom[..., 0] != PADDING_INDEX
        coords_p1 = coordinates[atom_description[:, 0].long()[pad_part1],
                                partners_indexing_atom[..., 0][pad_part1]]
        partners_final1[pad_part1] = coords_p1

        pad_part2 = partners_indexing_atom[..., 1] != PADDING_INDEX
        coords_p2 = coordinates[atom_description[:, 0].long()[pad_part2],
                                partners_indexing_atom[..., 1][pad_part2]]
        partners_final2[pad_part2] = coords_p2

        coords = coordinates[atom_description[:, 0].long(), coords_indexing_atom]
        partners_final = torch.cat([partners_final1.unsqueeze(1), partners_final2.unsqueeze(1)], dim=1)

        fake_atoms = fakeAtomsGeneration.generateFakeAtomTensor(coords, partners_final, atom_description,
                                                               hashings.fake_atom_Properties.to(self.device))

        independent_groups = atom_description[:, hashings.atom_description_hash["batch"]]
        atom_pairs = []
        for batch in range(independent_groups.max() + 1):
            independent_mask = independent_groups.eq(batch)
            atom_pairs.append(self.get_pairwise_representation(
                coords, atom_number[independent_mask]))
        atom_pairs = torch.cat(atom_pairs, dim=0)

        return coords, partners_final, fake_atoms, atom_pairs

    def _calculate_angles(self, coords, angle_indices):
        """Calculates torsion angles."""
        existent_angles_mask = (angle_indices != PADDING_INDEX).prod(-1).bool()
        flat_angle_indices = angle_indices[existent_angles_mask]
        flat_angles, _ = math_utils.dihedral2dVectors(coords[flat_angle_indices[:, 0]],
                                                      coords[flat_angle_indices[:, 1]],
                                                      coords[flat_angle_indices[:, 2]],
                                                      coords[flat_angle_indices[:, 3]])
        angles = torch.full(existent_angles_mask.shape, float(PADDING_INDEX), device=self.device)
        angles[existent_angles_mask] = flat_angles.squeeze(-1)
        return angles

    def _calculate_energies(self, coords, partners_final, fake_atoms, atom_pairs, info_tensors):
        """Calculates all energy terms."""
        atom_number, atom_description, _, _, angle_indices, alternative_mask = info_tensors
        angles = self._calculate_angles(coords, angle_indices)

        solvent_accessibility = SolventAccessibility(
            dev=self.device, donor=self.donor, acceptor=self.acceptor, backbone_atoms=self.backbone)
        cont_rat, facc, cont_rat_pol, _, cont_rat_sc = solvent_accessibility(
            coords, atom_description, atom_number, atom_pairs, fake_atoms, alternative_mask)

        disulfide_net = DisulfideEnergy(dev=self.device)
        disulfide_data = DisulfideData(
            coords=coords, atom_description=atom_description, atom_pairs=atom_pairs,
            partners=partners_final, alternative_mask=alternative_mask, facc=facc
        )
        residue_disulfide, atom_disulfide, disulfide_network = disulfide_net(disulfide_data)

        hbond_net = HBondNet(
            dev=self.device, donor=self.donor, acceptor=self.acceptor,
            backbone_atoms=self.backbone, hbond_ar=self.aromatic)
        hbond_mc, hbond_sc, atom_hb, hbond_network = hbond_net(
            coords, atom_description, atom_number, atom_pairs, fake_atoms,
            alternative_mask, disulfide_network, partners_final, facc)

        electro_net = ElectrostaticsEnergy(dev=self.device, backbone_atoms=self.backbone)
        electro_data = ElectrostaticsData(
            coords=coords, atom_description=atom_description, atom_pairs=atom_pairs,
            hbond_net=hbond_network, alternative_mask=alternative_mask, facc=facc
        )
        residue_electro_mc, residue_electro_sc, _, _ = electro_net(electro_data)

        clash_net = ClashEnergy(
            dev=self.device, backbone_atoms=self.backbone, donor=self.donor, acceptor=self.acceptor)
        residue_clash, _, _ = clash_net(
            coords, atom_description, atom_number, atom_pairs,
            alternative_mask, facc, hbond_network, disulfide_network)

        vdw = Vdw(dev=self.device, backbone_atoms=self.backbone)
        vdw_mc, vdw_sc = vdw(coords, atom_description, alternative_mask, facc)
        solvatation = Solvatation(
            dev=self.device, backbone_atoms=self.backbone, acceptor=self.acceptor, donor=self.donor)
        solv_p, solv_h = solvatation(
            atom_description, facc, cont_rat, cont_rat_pol, atom_hb, atom_disulfide)
        entropy_sc_module = SideChainEntropyEnergy(dev=self.device)
        entropy_sc = entropy_sc_module(
            atom_description,
            cont_rat_sc,
            hbond_sc,
            vdw_sc,
            residue_electro_sc,
            residue_clash,
            alternative_mask
        )
        bond_len_constraint = BondLengthConstraintEnergy(dev=self.device)
        peptide_bond_constraints = bond_len_constraint(atom_description, coords, alternative_mask)
        angle_scorer = AngleScorerEnergy(dev=self.device)
        entropy_mc, rotamer_violation = angle_scorer(atom_description, angles, alternative_mask)

        return {
            "disulfide": residue_disulfide, "hbond_mc": hbond_mc, "hbond_sc": hbond_sc,
            "electro_mc": residue_electro_mc, "electro_sc": residue_electro_sc,
            "clash": residue_clash, "vdw_mc": vdw_mc, "vdw_sc": vdw_sc, "solv_p": solv_p,
            "solv_h": solv_h, "entropy_sc": entropy_sc, "entropy_mc": entropy_mc,
            "peptide_bond_constraints": peptide_bond_constraints, "rotamer_violation": rotamer_violation
        }

    def forward(self, coordinates, info_tensors, verbose=False):
        """
        Calculates the energy of the protein(s) or complex(es).
        """
        if verbose:
            print("Verbose mode is on. Energy calculation started.")

        coords, partners_final, fake_atoms, atom_pairs = self._prepare_tensors(coordinates, info_tensors)
        energies = self._calculate_energies(coords, partners_final, fake_atoms, atom_pairs, info_tensors)

        hbonds = energies["hbond_sc"] + energies["hbond_mc"]
        electro = energies["electro_sc"] + energies["electro_mc"]
        vdw_energy = energies["vdw_mc"] + energies["vdw_sc"]
        clash = energies["clash"]
        clash_reduced, _ = clash.min(dim=-1, keepdim=True)

        def _ensure_alt_dim(tensor):
            # if tensor has 4 dims, add a trailing 1 so cat is consistent
            if tensor.dim() == 4:
                return tensor.unsqueeze(-1)
            return tensor

        terms = [
            _ensure_alt_dim(energies["disulfide"]),
            _ensure_alt_dim(hbonds),
            _ensure_alt_dim(electro),
            _ensure_alt_dim(clash_reduced),
            _ensure_alt_dim(energies["solv_p"]),
            _ensure_alt_dim(energies["solv_h"]),
            _ensure_alt_dim(vdw_energy),
            _ensure_alt_dim(energies["entropy_mc"]),
            _ensure_alt_dim(energies["entropy_sc"]),
            _ensure_alt_dim(energies["peptide_bond_constraints"]),
            _ensure_alt_dim(energies["rotamer_violation"])
        ]

        # optional sanity assert
        for t in terms:
            assert t.shape[-1] == 1, f"expected trailing alt dim==1 but got {t.shape} for a term"

        final_energy = torch.cat(terms, dim=-1)

        return final_energy
