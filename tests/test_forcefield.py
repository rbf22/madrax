import os
import torch
import vitra.dataStructures as dataStructures
import vitra.utils as utils
from vitra.ForceField import ForceField

def test_forcefield_regression():
    """
    Tests that the ForceField output matches the reference data.
    """
    # Load reference data
    reference_data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reference_data.pt")
    reference_energies = torch.load(reference_data_path)

    # Load PDB and compute energies
    pdb_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vitra", "exampleStructures", "alanine.pdb")
    device = "cpu"

    coordinates, atom_names = utils.parsePDB(pdb_file)
    info_tensors = dataStructures.create_info_tensors(atom_names, device=device)

    container = ForceField(device=device)
    weights_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vitra", "parameters", "final_model.weights")
    container.load_state_dict(torch.load(weights_path, map_location=torch.device(device)), strict=False)

    energies = container(coordinates.to(device), info_tensors).data

    # Compare current output with reference data
    assert torch.allclose(energies, reference_energies), "ForceField output does not match reference data."
