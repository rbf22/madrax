import torch
from vitra.energies.vdw import Vdw
from vitra.sources import hashings

def test_vdw_energy():
    """
    Tests the Vdw energy module with a simple case.
    This test verifies the current behavior of the Vdw module, which calculates a
    SASA-scaled per-atom energy, not a pairwise Lennard-Jones potential.
    """
    dev = 'cpu'
    # Mock backbone atoms
    backbone_atoms = [hashings.atom_hash['ALA']['N'], hashings.atom_hash['ALA']['CA'], hashings.atom_hash['ALA']['C']]
    vdw_module = Vdw(dev=dev, backbone_atoms=backbone_atoms)

    # Create dummy inputs for a single alanine residue (3 atoms)
    # Let's say we have one protein (batch 0), one chain (chain 0), one residue (resnum 0)
    atom_description = torch.tensor([
        # batch, chain, resnum, at_name
        [0, 0, 0, hashings.atom_hash['ALA']['N']],
        [0, 0, 0, hashings.atom_hash['ALA']['CA']],
        [0, 0, 0, hashings.atom_hash['ALA']['CB']],
    ], device=dev)

    # Mock atom_description hashing to only have 'at_name' for simplicity
    original_ad_hash = hashings.atom_description_hash
    try:
        hashings.atom_description_hash = {'batch': 0, 'chain': 1, 'resnum': 2, 'at_name': 3}


        alternativeMask = torch.tensor([[True], [True], [True]], device=dev)
        facc = torch.tensor([[10.0], [20.0], [30.0]], device=dev)

        # Dummy coordinates (not used by this module, but required by forward)
        coords = torch.zeros((1, 3, 3), device=dev)

        # Get the expected VdW values from the hashings table
        vdw_n = hashings.atom_Properties[hashings.atom_hash['ALA']['N'], hashings.property_hashings['solvenergy_props']['VdW']]
        vdw_ca = hashings.atom_Properties[hashings.atom_hash['ALA']['CA'], hashings.property_hashings['solvenergy_props']['VdW']]
        vdw_cb = hashings.atom_Properties[hashings.atom_hash['ALA']['CB'], hashings.property_hashings['solvenergy_props']['VdW']]

        # Calculate expected atom energies
        # Energy = vdw_value * facc * (1 - tanh(weight)) * 0.3
        # Since weight is initialized to 0, tanh(0) = 0, so factor is 0.3
        expected_energy_n = vdw_n * 10.0 * 0.3
        expected_energy_ca = vdw_ca * 20.0 * 0.3
        expected_energy_cb = vdw_cb * 30.0 * 0.3

        # The module separates MC and SC energies. N and CA are backbone. CB is sidechain.
        expected_mc_energy = expected_energy_n + expected_energy_ca
        expected_sc_energy = expected_energy_cb

        residueEnergyMC, residueEnergySC = vdw_module(coords, atom_description, alternativeMask, facc)

        # Check that the output tensors have the correct shape (batch, chain, res, altern, 1)
        assert residueEnergyMC.shape == (1, 1, 1, 1, 1)
        assert residueEnergySC.shape == (1, 1, 1, 1, 1)

        # Check the calculated energy values
        assert torch.allclose(residueEnergyMC[0, 0, 0, 0, 0], expected_mc_energy)
        assert torch.allclose(residueEnergySC[0, 0, 0, 0, 0], expected_sc_energy)
    finally:
        # Restore original hashings
        hashings.atom_description_hash = original_ad_hash
