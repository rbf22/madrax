import torch
import unittest
import os
import vitra.data_structures as data_structures
import vitra.utils as utils
from vitra.energies.hbond_net import HBondNet
from vitra.sources import hashings
from vitra.ForceField import ForceField

class TestHbondNetShapes(unittest.TestCase):
    def test_forward_output_shapes(self):
        """
        Tests that the HbondNet forward pass returns tensors with the correct shape,
        specifically with the 'naltern' dimension collapsed to 1.
        """
        device = 'cpu'

        # 1. Load data and create info tensors
        pdb_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vitra", "exampleStructures", "alanine.pdb")
        coordinates, atom_names, _ = utils.parse_pdb(pdb_file)
        coordinates = coordinates.to(device)

        (atom_number, atom_description, coords_indexing_atom,
         partners_indexing_atom, angle_indices, alternative_mask) = data_structures.create_info_tensors(atom_names, device=device)

        # 2. Manually create an alternative mask with naltern=10
        naltern = 10
        n_atoms = atom_description.shape[0]

        # The original alternative_mask has shape (n_atoms, 1). We expand it.
        alternative_mask_expanded = alternative_mask.expand(-1, naltern)

        # 3. Instantiate HbondNet
        # We can get the atom sets from a ForceField instance
        ff = ForceField(device=device)
        hb_net = HBondNet(
            dev=device,
            donor=ff.donor,
            acceptor=ff.acceptor,
            backbone_atoms=ff.backbone,
            hbond_ar=ff.aromatic
        )

        # 4. Create other dummy inputs for forward pass
        # These tensors are not directly available from create_info_tensors, so we create them.
        # We need to get some other tensors that are calculated inside ForceField
        coords, partners_final, fake_atoms, atom_pairs = ff._prepare_tensors(coordinates,
            (atom_number, atom_description, coords_indexing_atom, partners_indexing_atom, angle_indices, alternative_mask))

        # The disulfide network and facc are also needed.
        # Let's create dummy versions for them.
        disulfide_network = torch.zeros(atom_pairs.shape[0], dtype=torch.bool, device=device)
        facc = torch.rand(n_atoms, naltern, device=device)

        # 5. Call forward
        res_mc, res_sc, _, _ = hb_net.forward(
            coords,
            atom_description,
            atom_number,
            atom_pairs,
            fake_atoms,
            alternative_mask_expanded,
            disulfide_network,
            partners_final,
            facc
        )

        # 6. Assert shapes
        self.assertEqual(res_mc.shape[-1], 1, f"Expected trailing dim of 1 for hbond_mc, but got {res_mc.shape}")
        self.assertEqual(res_sc.shape[-1], 1, f"Expected trailing dim of 1 for hbond_sc, but got {res_sc.shape}")
