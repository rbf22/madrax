"""
This module provides functions for creating data structures used in the vitra forcefield.
These data structures are pre-calculated from the protein structure and are used to
speed up the energy calculations.
"""
import time
from dataclasses import dataclass
import torch

from vitra.sources import hashings
from vitra.sources.globalVariables import PADDING_INDEX

TORSION_ANGLE_HASHINGS = {
    "phi": 0, "psi": 1, "omega": 2, "chi1": 3, "chi2": 4,
    "chi3": 5, "chi4": 6, "chi5": 7
}


@dataclass
class AtomData:
    """A data class to hold atom information for angle index calculation."""
    prot_n: int
    chain_ind: int
    res_index: int
    alt: int
    atom_idx: int
    atom_name: str
    res_name: str
    max_res_idx: int


def _create_chain_hashing(atom_names):
    """Creates a hashing for chain identifiers."""
    chain_ids = set(atom.split("_")[3] for prot in atom_names for atom in prot)
    return {chain_id: i for i, chain_id in enumerate(sorted(list(chain_ids)))}


def _initialize_atom_data():
    """Initializes a dictionary to store atom data."""
    return {
        "terminals": {}, "alternative": [], "batch_index": [], "atom_number": [],
        "resname": [], "chain": [], "at_name": [], "res_num": [],
        "alternative_conf": [], "coords_indexing_atom": [], "partners_tens": [],
        "atom_counter": 0, "max_chain": 0, "max_res": 0,
        "atn_to_at_index": {},
    }


def _update_terminals(prot_data, chain_id, res_index, atom_name, atom_number):
    """Updates the terminal residues information."""
    if chain_id not in prot_data["terminals"]:
        prot_data["terminals"][chain_id] = [atom_number, atom_number, res_index, res_index]
    else:
        terminals = prot_data["terminals"][chain_id]
        if res_index >= terminals[3] and atom_name == "O":
            terminals[3] = res_index
            terminals[1] = atom_number
        elif res_index <= terminals[2] and atom_name == "N":
            terminals[2] = res_index
            terminals[0] = atom_number


def _process_protein(prot_data, prot_n, prot, chain_hashing):
    """Processes a single protein's atoms to populate data lists."""
    prot_data["atn_to_at_index"][prot_n] = {}
    indices_dict = {}

    for atn, atom_str in enumerate(prot):
        r, res_index_str, a, chain_ind, mut, alt_conf = atom_str.split("_")
        res_index = int(res_index_str)
        chain_id = chain_hashing[chain_ind]

        if r not in hashings.resi_hash or a not in hashings.atom_hash.get(r, {}):
            continue

        prot_data["at_name"].append(hashings.atom_hash[r][a])
        prot_data["batch_index"].append(prot_n)
        prot_data["res_num"].append(res_index)
        prot_data["alternative"].append(int(mut))
        prot_data["alternative_conf"].append(int(alt_conf))
        prot_data["chain"].append(chain_id)
        prot_data["resname"].append(hashings.resi_hash[r])
        prot_data["atom_number"].append(prot_data["atom_counter"])
        prot_data["coords_indexing_atom"].append(atn)

        current_atom_number = prot_data["atom_counter"]
        prot_data["atn_to_at_index"][prot_n][atn] = current_atom_number

        if (chain_id, res_index) not in indices_dict:
            indices_dict[(chain_id, res_index)] = {}
        indices_dict[(chain_id, res_index)][a] = atn

        _update_terminals(prot_data, chain_id, res_index, a, current_atom_number)

        prot_data["max_chain"] = max(prot_data["max_chain"], chain_id)
        prot_data["max_res"] = max(prot_data["max_res"], res_index)
        prot_data["atom_counter"] += 1

    return indices_dict


def _calculate_partners(prot_data, prot, chain_hashing, indices_dict):
    """Calculates atom partners for a protein."""
    partners = hashings.partners

    for _, term_data in prot_data["terminals"].items():
        tn_atom_idx, oxt_atom_idx = term_data[0], term_data[1]
        resname_tn = hashings.resi_hash_inverse[prot_data["resname"][tn_atom_idx]]
        prot_data["at_name"][tn_atom_idx] = hashings.atom_hash[resname_tn]["tN"]
        resname_oxt = hashings.resi_hash_inverse[prot_data["resname"][oxt_atom_idx]]
        prot_data["at_name"][oxt_atom_idx] = hashings.atom_hash[resname_oxt]["OXT"]

    for atom_str in prot:
        r, res_index_str, a, chain_ind, _, _ = atom_str.split("_")
        res_index = int(res_index_str)

        if r not in hashings.resi_hash or a not in hashings.atom_hash.get(r, {}):
            continue

        if a not in partners.get(r, {}):
            prot_data["partners_tens"].append([PADDING_INDEX, PADDING_INDEX])
            continue

        p1_res, p1_atom = partners[r][a][0][1], partners[r][a][0][0]
        p2_res, p2_atom = partners[r][a][1][1], partners[r][a][1][0]

        p1_ind = indices_dict.get(
            (chain_hashing[chain_ind], res_index + p1_res), {}).get(p1_atom, PADDING_INDEX)
        p2_ind = indices_dict.get(
            (chain_hashing[chain_ind], res_index + p2_res), {}).get(p2_atom, PADDING_INDEX)
        prot_data["partners_tens"].append([p1_ind, p2_ind])


def _update_angle_indices(angle_indices, atom_data):
    """Updates the angle indices tensor for a single atom."""
    ad = atom_data
    if ad.atom_name == "N":
        angle_indices[ad.prot_n, ad.chain_ind, ad.res_index, ad.alt, TORSION_ANGLE_HASHINGS["phi"], 1] = ad.atom_idx
        angle_indices[ad.prot_n, ad.chain_ind, ad.res_index, ad.alt, TORSION_ANGLE_HASHINGS["psi"], 0] = ad.atom_idx
        if ad.res_index > 0:
            angle_indices[ad.prot_n, ad.chain_ind, ad.res_index - 1, ad.alt, TORSION_ANGLE_HASHINGS["psi"], 3] = ad.atom_idx
            angle_indices[ad.prot_n, ad.chain_ind, ad.res_index - 1, ad.alt, TORSION_ANGLE_HASHINGS["omega"], 2] = ad.atom_idx
    elif ad.atom_name == "C":
        if ad.res_index + 1 < ad.max_res_idx:
            angle_indices[ad.prot_n, ad.chain_ind, ad.res_index + 1, ad.alt, TORSION_ANGLE_HASHINGS["phi"], 0] = ad.atom_idx
        angle_indices[ad.prot_n, ad.chain_ind, ad.res_index, ad.alt, TORSION_ANGLE_HASHINGS["phi"], 3] = ad.atom_idx
        angle_indices[ad.prot_n, ad.chain_ind, ad.res_index, ad.alt, TORSION_ANGLE_HASHINGS["psi"], 2] = ad.atom_idx
        angle_indices[ad.prot_n, ad.chain_ind, ad.res_index, ad.alt, TORSION_ANGLE_HASHINGS["omega"], 1] = ad.atom_idx
    elif ad.atom_name == "CA":
        angle_indices[ad.prot_n, ad.chain_ind, ad.res_index, ad.alt, TORSION_ANGLE_HASHINGS["phi"], 2] = ad.atom_idx
        angle_indices[ad.prot_n, ad.chain_ind, ad.res_index, ad.alt, TORSION_ANGLE_HASHINGS["psi"], 1] = ad.atom_idx
        angle_indices[ad.prot_n, ad.chain_ind, ad.res_index, ad.alt, TORSION_ANGLE_HASHINGS["omega"], 0] = ad.atom_idx
        if ad.res_index > 0:
            angle_indices[ad.prot_n, ad.chain_ind, ad.res_index - 1, ad.alt, TORSION_ANGLE_HASHINGS["omega"], 3] = ad.atom_idx


def _calculate_angle_indices(prot_data, atom_names, alternative_mask, device, chain_hashing):
    """Calculates torsion angle indices."""
    angle_indices = torch.full(
        (len(atom_names), prot_data["max_chain"] + 1, prot_data["max_res"] + 1,
         alternative_mask.shape[-1], 8, 4),
        int(PADDING_INDEX), device=device
    )
    max_res_idx = angle_indices.shape[2]

    for prot_n, prot in enumerate(atom_names):
        for atn, atom_str in enumerate(prot):
            r, res_index_str, a, chain_ind_str, _, _ = atom_str.split("_")
            if r not in hashings.resi_hash or a not in hashings.atom_hash.get(r, {}):
                continue

            atom_idx = prot_data["atn_to_at_index"][prot_n][atn]
            for alt in range(alternative_mask.shape[-1]):
                if not alternative_mask[atom_idx, alt]:
                    continue

                atom_data = AtomData(prot_n, chain_hashing[chain_ind_str], int(res_index_str),
                                     alt, atom_idx, a, r, max_res_idx)
                _update_angle_indices(angle_indices, atom_data)

                if r in hashings.chi_angle_builder_hashing and a in hashings.chi_angle_builder_hashing[r]:
                    for side_angle, angle_idx in hashings.chi_angle_builder_hashing[r][a]:
                        angle_indices[
                            prot_n, atom_data.chain_ind, atom_data.res_index, alt,
                            TORSION_ANGLE_HASHINGS[side_angle], angle_idx] = atom_idx
    return angle_indices


def create_info_tensors(atom_names, int_dtype=torch.short, device="cpu", verbose=False):
    """Calculates pre-computed tensors for vitra energy calculations."""
    if verbose:
        start_time = time.time()

    chain_hashing = _create_chain_hashing(atom_names)
    prot_data = _initialize_atom_data()

    for prot_n, prot in enumerate(atom_names):
        indices_dict = _process_protein(prot_data, prot_n, prot, chain_hashing)
        _calculate_partners(prot_data, prot, chain_hashing, indices_dict)

    atom_desc_keys = ["batch_index", "chain", "res_num", "resname",
                      "at_name", "alternative", "alternative_conf"]
    atom_desc_tensors = [torch.tensor(prot_data[k], dtype=int_dtype).unsqueeze(-1)
                         for k in atom_desc_keys]
    atom_description = torch.cat(atom_desc_tensors, dim=-1).to(device)

    alternative_mask = build_altern_mask(atom_description)

    coords_indexing_atom = torch.tensor(prot_data["coords_indexing_atom"],
                                        dtype=torch.long, device=device)
    partners_indexing_atom = torch.tensor(prot_data["partners_tens"],
                                          dtype=torch.long, device=device)
    atom_number_tensor = torch.tensor(prot_data["atom_number"],
                                      dtype=torch.long, device=device)

    angle_indices = _calculate_angle_indices(prot_data, atom_names,
                                             alternative_mask, device, chain_hashing)

    if verbose:
        print(f"Info tensor creation time: {time.time() - start_time:.4f}s")

    return (atom_number_tensor, atom_description, coords_indexing_atom,
            partners_indexing_atom, angle_indices, alternative_mask)


def get_helical_dipoles(atom_pairs, hbond_net, atom_description, altern_mask):
    """
    Identifies helical regions and returns masks for atoms at the helix termini
    to apply dipole corrections.

    Warning: This feature is experimental.
    """
    print("WARNING: helix dipole is under development. Dipole is set at the C and N "
          "of the helix extremities, but it should be in the middle of the last turn")

    initial_atom_pairs_shape = atom_pairs.shape

    o_atoms_list, n_atoms_list, c_atoms_list = [], [], []
    for _, atoms in hashings.atom_hash.items():
        if "O" in atoms:
            o_atoms_list.append(atoms["O"])
        if "OXT" in atoms:
            o_atoms_list.append(atoms["OXT"])
        if "N" in atoms:
            n_atoms_list.append(atoms["N"])
        if "tN" in atoms:
            n_atoms_list.append(atoms["tN"])
        if "C" in atoms:
            c_atoms_list.append(atoms["C"])

    a = atom_description[:, hashings.atom_description_hash["at_name"]]
    oatom_mask = torch.zeros_like(a, dtype=torch.bool)
    for o_atom in o_atoms_list:
        oatom_mask |= a.eq(o_atom)

    natom_mask = torch.zeros_like(a, dtype=torch.bool)
    for n_atom in n_atoms_list:
        natom_mask |= a.eq(n_atom)

    catom_mask = torch.zeros_like(a, dtype=torch.bool)
    for c_atom in c_atoms_list:
        catom_mask |= a.eq(c_atom)

    max_res = atom_description[:, hashings.atom_description_hash["resnum"]].max() + 1
    max_batch = atom_description[:, hashings.atom_description_hash["batch"]].max() + 1
    max_chain = atom_description[:, hashings.atom_description_hash["chain"]].max() + 1
    n_alter = altern_mask.shape[-1]

    o_n_atoms = natom_mask[atom_pairs[:, 1]] & oatom_mask[atom_pairs[:, 0]]
    atom_pairs_on = atom_pairs[o_n_atoms]

    same_chain_mask = (atom_description[atom_pairs_on[:, 0], hashings.atom_description_hash["chain"]] ==
                       atom_description[atom_pairs_on[:, 1], hashings.atom_description_hash["chain"]])

    res_dist = torch.abs(atom_description[atom_pairs_on[:, 0], hashings.atom_description_hash["resnum"]] -
                         atom_description[atom_pairs_on[:, 1], hashings.atom_description_hash["resnum"]])

    hbond_net_on = hbond_net[o_n_atoms]
    helix3 = (res_dist == 3) & same_chain_mask & hbond_net_on
    helix4 = (res_dist == 4) & same_chain_mask & hbond_net_on
    helix5 = (res_dist == 5) & same_chain_mask & hbond_net_on

    helix = torch.zeros((max_batch, max_chain, max_res, n_alter), dtype=torch.bool,
                        device=atom_description.device)

    for alt in range(altern_mask.shape[1]):
        for helix_mask in [helix3, helix4, helix5]:
            atom_pairs_h = atom_pairs_on[helix_mask]
            mask = altern_mask[atom_pairs_h[:, 0], alt] & altern_mask[atom_pairs_h[:, 1], alt]

            batch_alt = atom_description[atom_pairs_h[:, 0], hashings.atom_description_hash["batch"]][mask].long()
            chain_alt = atom_description[atom_pairs_h[:, 0], hashings.atom_description_hash["chain"]][mask].long()
            res1 = atom_description[atom_pairs_h[:, 0], hashings.atom_description_hash["resnum"]][mask].long()
            res2 = atom_description[atom_pairs_h[:, 1], hashings.atom_description_hash["resnum"]][mask].long()
            alt_index = torch.full_like(res1, int(alt))

            helix.index_put_((batch_alt, chain_alt, res1, alt_index), torch.ones_like(res1, dtype=torch.bool))
            helix.index_put_((batch_alt, chain_alt, res2, alt_index), torch.ones_like(res2, dtype=torch.bool))

    resnum = torch.arange(0, helix.shape[2] - 3, device=helix.device)
    bad_two_helices = (~helix[:, :, resnum]) & (helix[:, :, resnum + 1]) & \
                      (helix[:, :, resnum + 2]) & (~helix[:, :, resnum + 3])
    helix[:, :, resnum + 1] *= ~bad_two_helices
    helix[:, :, resnum + 2] *= ~bad_two_helices

    resnum = torch.arange(0, helix.shape[2] - 2, device=helix.device)
    bad_single_helix = (~helix[:, :, resnum]) & (helix[:, :, resnum + 1]) & (~helix[:, :, resnum + 2])
    helix[:, :, resnum + 1] *= ~bad_single_helix

    resnum = torch.arange(0, helix.shape[2] - 1, device=helix.device)
    helix_dipoled_c = torch.cat(
        [((helix[:, :, resnum]) & (~helix[:, :, resnum + 1])), (helix[:, :, [helix.shape[2] - 1]])],
        dim=2)

    resnum = torch.arange(1, helix.shape[2], device=helix.device)
    helix_dipoled_n = torch.cat(
        [(helix[:, :, [0]]), ((~helix[:, :, resnum - 1]) & (helix[:, :, resnum]))],
        dim=2)

    n_cp_mask = torch.zeros(initial_atom_pairs_shape, dtype=torch.bool, device=atom_pairs.device)
    c_cp_mask = torch.zeros(initial_atom_pairs_shape, dtype=torch.bool, device=atom_pairs.device)

    resnum = atom_description[:, hashings.atom_description_hash["resnum"]].long()
    batch = atom_description[:, hashings.atom_description_hash["batch"]].long()
    chain = atom_description[:, hashings.atom_description_hash["chain"]].long()

    for alt in range(altern_mask.shape[1]):
        mask = altern_mask[atom_pairs[:, 0], alt] & altern_mask[atom_pairs[:, 1], alt]
        resnum_alt = resnum[atom_pairs[mask]]
        batch_alt = batch[atom_pairs[mask]]
        chain_alt = chain[atom_pairs[mask]]

        n_mask1 = mask & (natom_mask[atom_pairs[:, 0]]) & (
            helix_dipoled_n[batch_alt[:, 0], chain_alt[:, 0], resnum_alt[:, 0], alt])
        n_mask2 = mask & (natom_mask[atom_pairs[:, 1]]) & (
            helix_dipoled_n[batch_alt[:, 0], chain_alt[:, 1], resnum_alt[:, 1], alt])

        n_cp_mask[n_mask1, 0] = True
        n_cp_mask[n_mask2, 1] = True

        c_mask1 = mask & (catom_mask[atom_pairs[:, 0]]) & (
            helix_dipoled_c[batch_alt[:, 0], chain_alt[:, 0], resnum_alt[:, 0], alt])
        c_mask2 = mask & (catom_mask[atom_pairs[:, 1]]) & (
            helix_dipoled_c[batch_alt[:, 0], chain_alt[:, 1], resnum_alt[:, 1], alt])

        c_cp_mask[c_mask1, 0] = True
        c_cp_mask[c_mask2, 1] = True

    return n_cp_mask, c_cp_mask


def build_altern_mask(atom_description):
    """Builds a mask for alternative conformations."""
    altern_indices = atom_description[:, hashings.atom_description_hash["alternative"]].long()

    if altern_indices.max() == 0:
        return torch.ones(
            atom_description.shape[0],
            device=atom_description.device,
            dtype=torch.bool
        ).unsqueeze(-1)

    batch = atom_description[:, hashings.atom_description_hash["batch"]].long()
    chain = atom_description[:, hashings.atom_description_hash["chain"]].long()
    resnum = atom_description[:, hashings.atom_description_hash["resnum"]].long()

    max_altern = altern_indices.max() + 1
    mask_wild_type = altern_indices.eq(0)
    altern_mask_list = [mask_wild_type.unsqueeze(-1)]

    for alt in range(1, max_altern):
        mask = altern_indices.eq(alt)
        alt_mask = mask_wild_type.clone()

        remove_indices = torch.stack(
            [batch[mask], chain[mask], resnum[mask]], dim=-1).unique(dim=0)

        for mut in remove_indices:
            m = (batch == mut[0]) & (chain == mut[1]) & \
                (resnum == mut[2]) & (altern_indices != alt)
            alt_mask[m] = False

        alt_mask[mask] = True
        altern_mask_list.append(alt_mask.unsqueeze(-1))

    return torch.cat(altern_mask_list, dim=-1)
