import os
import torch
import pytest
from vitra import utils
import tempfile
import shutil

@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    d = tempfile.mkdtemp()
    yield d
    shutil.rmtree(d)

def test_atom_name_to_seq_single_chain():
    atName = [['ALA_1_CA_A_0_0', 'CYS_2_CA_A_0_0']]
    seqs = utils.atom_name_to_seq(atName)
    assert seqs == ['AC']

def test_atom_name_to_seq_multiple_chains():
    atName = [
        ['ALA_1_CA_A_0_0', 'CYS_2_CA_A_0_0'],
        ['GLY_1_CA_B_0_0', 'PHE_3_CA_B_0_0']
    ]
    seqs = utils.atom_name_to_seq(atName)
    assert seqs == ['AC', 'GXF']

def test_parse_pdb_single_file():
    pdb_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vitra", "exampleStructures", "alanine.pdb")
    coords, atom_names, _ = utils.parse_pdb(pdb_file)
    assert coords.shape[0] == 1
    assert len(atom_names) == 1

def test_parse_pdb_directory():
    pdb_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vitra", "exampleStructures")
    coords, atom_names, pdb_names = utils.parse_pdb(pdb_dir)
    assert coords.shape[0] == 1
    assert len(atom_names) == 1
    assert pdb_names == ['alanine']

def test_parse_pdb_bb_only():
    pdb_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vitra", "exampleStructures", "alanine.pdb")
    coords, atom_names, _ = utils.parse_pdb(pdb_file, bb_only=True)
    assert len(atom_names[0]) == 27 # 9 residues * 3 backbone atoms
    for atom in atom_names[0]:
        assert atom.split('_')[2] in ['N', 'CA', 'C']

def test_parse_pdb_keep_only_chains():
    pdb_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vitra", "exampleStructures", "alanine.pdb")
    # This pdb only has chain A, so testing with a different chain should result in 0 atoms
    coords, atom_names, _ = utils.parse_pdb(pdb_file, keep_only_chains="B")
    assert len(atom_names[0]) == 0

def test_write_pdb(temp_dir):
    coords = torch.tensor([[
        [1.0, 2.0, 3.0], # N
        [2.0, 3.0, 4.0], # CA
        [3.0, 4.0, 5.0], # C
        [4.0, 5.0, 6.0], # O
        [5.0, 6.0, 7.0]  # CB
    ]])
    atnames = [[
        'ALA_1_N_A_0_0',
        'ALA_1_CA_A_0_0',
        'ALA_1_C_A_0_0',
        'ALA_1_O_A_0_0',
        'ALA_1_CB_A_0_0'
    ]]
    pdb_names = ['test_protein']
    output_folder = temp_dir
    utils.write_pdb(coords, atnames, pdb_names, output_folder=output_folder)

    output_file = os.path.join(output_folder, "test_protein.pdb")
    assert os.path.exists(output_file)

    with open(output_file, 'r') as f:
        lines = f.readlines()
        assert len(lines) == 5
        # Note: write_pdb might reorder atoms, so we check for presence, not order
        file_content = "".join(lines)
        assert "ATOM      1  N   ALA A   1       1.000   2.000   3.000" in file_content
        assert "ATOM      2  CA  ALA A   1       2.000   3.000   4.000" in file_content
        assert "ATOM      3  C   ALA A   1       3.000   4.000   5.000" in file_content
        assert "ATOM      4  O   ALA A   1       4.000   5.000   6.000" in file_content
        assert "ATOM      5  CB  ALA A   1       5.000   6.000   7.000" in file_content
