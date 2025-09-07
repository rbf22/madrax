import os
import torch
import vitra.data_structures as data_structures
import vitra.utils as utils
from vitra.ForceField import ForceField

def generate_reference_data():
    """
    Generates reference energy data from the alanine.pdb file and saves it to tests/reference_data.pt.
    """
    pdb_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vitra", "exampleStructures", "alanine.pdb")
    device = "cpu"

    coordinates, atom_names, _ = utils.parse_pdb(pdb_file)

    info_tensors = data_structures.create_info_tensors(atom_names, device=device)

    container = ForceField(device=device)

    # Load the model with map_location=torch.device('cpu')
    weights_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vitra", "parameters", "final_model.weights")
    container.load_state_dict(torch.load(weights_path, map_location=torch.device(device)), strict=False)

    energies = container(coordinates.to(device), info_tensors).data

    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tests", "reference_data.pt")
    torch.save(energies, output_path)
    print(f"Reference data saved to {output_path}")

if __name__ == '__main__':
    generate_reference_data()
