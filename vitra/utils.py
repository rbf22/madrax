"""
This module provides utility functions for handling PDB files and other data manipulations.
"""
import os
from dataclasses import dataclass
import torch

from vitra import data_structures
from vitra.sources import hashings
from vitra.sources.globalVariables import PADDING_INDEX


@dataclass
class MutantNameData:
    """Data class for mutant name generation."""
    pdb_name: str
    alt: int
    wt_mask: torch.Tensor
    batch_mask: torch.Tensor
    ca_mask: torch.Tensor
    res_num: torch.Tensor
    chain_ind: torch.Tensor
    resname: torch.Tensor
    altern_mask: torch.Tensor
    chain_hashing: dict


def _parse_single_pdb(pdb_file_path, keep_only_chains=None, bb_only=False):
    """Parses a single PDB file."""
    bb_atoms = ["N", "CA", "C"]
    ring_residues = ["PHE", "TYR"]
    r_c_atoms = ["CD1", "CE2"]

    atom_names_tmp = []
    coords_tmp = []
    rc_atoms_dict = {}

    with open(pdb_file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("ENDMDL"):
                break
            if not line.startswith("ATOM"):
                continue

            if keep_only_chains is not None and (line[21] not in keep_only_chains):
                continue
            if bb_only and line[12:16].strip() not in bb_atoms:
                continue
            if line[12:16].strip().startswith("H"):
                continue

            resnum = int(line[22:26])
            resname = line[17:20].strip()
            atom_name = line[12:16].strip()
            chain_id = line[21]

            atom_names_tmp.append(f"{resname}_{resnum}_{atom_name}_{chain_id}_0_0")

            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
            coords_tmp.append([x, y, z])

            if resname in ring_residues:
                if resnum not in rc_atoms_dict:
                    rc_name = f"{resname}_{resnum}_RC_{chain_id}_0_0"
                    rc_atoms_dict[resnum] = ({}, rc_name)
                if atom_name in r_c_atoms:
                    rc_atoms_dict[resnum][0][atom_name] = [x, y, z]

    for _, (rc_atoms, rc_name) in rc_atoms_dict.items():
        if "CD1" in rc_atoms and "CE2" in rc_atoms:
            cd1 = rc_atoms["CD1"]
            ce2 = rc_atoms["CE2"]
            new_coords = [(cd1[0] + ce2[0]) / 2.0, (cd1[1] + ce2[1]) / 2.0, (cd1[2] + ce2[2]) / 2.0]
            coords_tmp.append(new_coords)
            atom_names_tmp.append(rc_name)

    return coords_tmp, atom_names_tmp


def parse_pdb(pdb_path, keep_only_chains=None, bb_only=False):
    """
    Parses PDB files. It can parse a single file or all PDB files in a folder.
    """
    if not os.path.isdir(pdb_path):
        coords, atom_names = _parse_single_pdb(pdb_path, keep_only_chains, bb_only)
        return torch.tensor(coords).unsqueeze(0), [atom_names], [os.path.basename(pdb_path).replace(".pdb", "")]

    all_coords = []
    all_atom_names = []
    pdb_names = []

    for filename in sorted(os.listdir(pdb_path)):
        if not filename.endswith(".pdb"):
            continue

        pdb_names.append(filename.replace(".pdb", ""))
        file_path = os.path.join(pdb_path, filename)
        coords, atom_names = _parse_single_pdb(file_path, keep_only_chains, bb_only)
        all_coords.append(torch.tensor(coords))
        all_atom_names.append(atom_names)

    return torch.nn.utils.rnn.pad_sequence(all_coords, batch_first=True,
                                           padding_value=PADDING_INDEX), all_atom_names, pdb_names


def _get_mutant_name(data: MutantNameData):
    """Generates a name for the mutant PDB file."""
    if ((data.altern_mask[:, data.alt] != data.wt_mask) & data.batch_mask).sum() > 0:
        changing = (data.altern_mask[:, data.alt] != data.wt_mask) & data.batch_mask & data.ca_mask
        pos_wt = data.res_num[changing & data.wt_mask].tolist()
        chain_wt = data.chain_ind[changing & data.wt_mask].tolist()
        resname_wt = data.resname[changing & data.wt_mask].tolist()
        resname_mu = data.resname[changing & data.altern_mask[:, data.alt]].tolist()

        name = data.pdb_name
        for k, res_mu in enumerate(resname_mu):
            n_mut = (f"{hashings.resi_hash_inverse[resname_wt[k]]}"
                     f"{data.chain_hashing[chain_wt[k]]}{pos_wt[k]}"
                     f"{hashings.resi_hash_inverse[res_mu]}")
            name += "_" + n_mut
        if len(name) > 50:
            name = data.pdb_name + "_mutant" + str(data.alt)
        return name
    if data.alt == 0:
        return data.pdb_name
    return None


def _format_pdb_line(i, atom_name, res_name, chain_id, res_num, x, y, z):
    """Formats a single line of a PDB file."""
    atom_name_map = {"tN": "N", "OXT": "O"}
    atom_name = atom_name_map.get(atom_name, atom_name)

    line = (f"ATOM  {i + 1: >5}  {atom_name: <4}{res_name}{chain_id: >2}{res_num: >4}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00 64.10           {atom_name[0]}\n")
    return line


def write_pdb(coords, atnames, pdb_names=None, output_folder="outpdb/"):
    """
    Writes a PDB file from coordinates and atom names.
    """
    os.makedirs(output_folder, exist_ok=True)

    chain_ids = sorted(list(set(atom.split("_")[3] for prot in atnames for atom in prot)))
    chain_hashing = dict(enumerate(chain_ids))

    if pdb_names is None:
        pdb_names = [str(i) for i in range(len(atnames))]

    info_tensors = data_structures.create_info_tensors(atnames, device="cpu", verbose=False)
    _, atom_description, coords_indexing_atom, _, _, altern_mask = info_tensors
    coords = coords[atom_description[:, 0].long(), coords_indexing_atom]

    atname = atom_description[:, hashings.atom_description_hash["at_name"]]
    resname = atom_description[:, hashings.atom_description_hash["resname"]]
    batch_ind = atom_description[:, hashings.atom_description_hash["batch"]]
    chain_ind = atom_description[:, hashings.atom_description_hash["chain"]]
    res_num = atom_description[:, hashings.atom_description_hash["resnum"]]
    wt_mask = altern_mask[:, 0]

    ca_mask = torch.zeros(coords.shape[0], dtype=torch.bool)
    for res_hash in hashings.atom_hash.values():
        if "CA" in res_hash:
            ca_mask |= (atname == res_hash["CA"])

    for batch, pdb_name in enumerate(pdb_names):
        for alt in range(altern_mask.shape[-1]):
            mutant_data = MutantNameData(pdb_name, alt, wt_mask, batch_ind.eq(batch), ca_mask,
                                         res_num, chain_ind, resname, altern_mask, chain_hashing)
            name = _get_mutant_name(mutant_data)
            if name is None:
                continue

            with open(os.path.join(output_folder, name + ".pdb"), "w", encoding="utf-8") as f:
                skip_atoms = {"RC", "RE"}
                skip_mask = torch.ones(coords.shape[0], dtype=torch.bool)
                for res_hash in hashings.atom_hash.values():
                    for s in skip_atoms:
                        if s in res_hash:
                            skip_mask &= (atname != res_hash[s])

                coords_mask = batch_ind.eq(batch) & altern_mask[:, alt] & skip_mask

                atname_loc = atname[coords_mask].tolist()
                resname_loc = resname[coords_mask].tolist()
                chain_ind_loc = chain_ind[coords_mask].tolist()
                res_num_loc = res_num[coords_mask].tolist()
                coords_loc = coords[coords_mask].tolist()

                for i, res_num_val in enumerate(res_num_loc):
                    line = _format_pdb_line(i, hashings.atom_hash_inverse[atname_loc[i]],
                                            hashings.resi_hash_inverse[resname_loc[i]],
                                            chain_hashing[chain_ind_loc[i]], res_num_val,
                                            coords_loc[i][0], coords_loc[i][1], coords_loc[i][2])
                    f.write(line)


def _get_sequence_from_batch(batch):
    """Extracts the sequence from a single batch of atoms."""
    letters = {'CYS': 'C', 'ASP': 'D', 'SER': 'S', 'GLN': 'Q', 'LYS': 'K', 'ASN': 'N', 'PRO': 'P', 'THR': 'T',
               'PHE': 'F', 'ALA': 'A', 'HIS': 'H', 'GLY': 'G', 'ILE': 'I', 'LEU': 'L', 'ARG': 'R', 'TRP': 'W',
               'VAL': 'V', 'GLU': 'E', 'TYR': 'Y', 'MET': 'M'}

    ch = None
    minres = 999
    maxres = -1
    rname = ""
    rnum = {}
    cont = 0
    for atom in batch:
        resname, resind, a, chains, _, _ = atom.split("_")
        resind = int(resind)
        if not (ch == chains or ch is None):
            raise ValueError("This function only supports single chain PDBs for sequence extraction.")
        if a == "CA":
            maxres = max(maxres, resind)
            minres = min(minres, resind)
            rname += letters[resname]
            rnum[resind] = cont
            cont += 1

    s = ""
    if maxres != -1:
        for k in range(1, maxres + 1):
            if k in rnum:
                s += rname[rnum[k]]
            else:
                s += "X"
    return s


def atom_name_to_seq(atom_names):
    """
    Function that returns the sequence of proteins given the atom names
    """
    return [_get_sequence_from_batch(batch) for batch in atom_names]
