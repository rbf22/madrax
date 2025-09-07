import torch
import numpy as np
from vitra.energies.hbond_net import HBondNet

def test_hbond_angle_penalty():
    """
    Tests the get_angle_penalty method in the HBondNet module.
    This test verifies that the angular penalty is calculated correctly for a simple case.
    """
    dev = 'cpu'
    hbond_module = HBondNet(dev=dev)

    # All angles are in radians
    # Case 1: All angles are optimal, penalty should be 0
    interaction_angles = torch.tensor([[[np.radians(150), np.radians(100), np.radians(130)]]], device=dev, dtype=torch.float)
    minAng = torch.tensor([[[np.radians(115), np.radians(80), np.radians(80)]]], device=dev, dtype=torch.float)
    optMin = torch.tensor([[[np.radians(145), np.radians(90), np.radians(130)]]], device=dev, dtype=torch.float)
    optMax = torch.tensor([[[np.radians(160), np.radians(110), np.radians(180)]]], device=dev, dtype=torch.float)
    maxAng = torch.tensor([[[np.radians(180), np.radians(155), np.radians(180)]]], device=dev, dtype=torch.float)

    penalty = hbond_module.get_angle_penalty(interaction_angles, minAng, optMin, optMax, maxAng)
    assert torch.allclose(penalty, torch.tensor([[0.0]], device=dev, dtype=torch.float))

    # Case 2: One angle is outside the optimal range, but within the min/max range
    # Let's make the first angle 140, which is less than optMin (145)
    interaction_angles = torch.tensor([[[np.radians(140), np.radians(100), np.radians(130)]]], device=dev, dtype=torch.float)

    # Expected penalty for the first angle: (145 - 140) / (145 - 115) = 5 / 30 = 0.16666
    # The penalty is divided by 5 for this angle component.
    expected_penalty = (5.0 / 30.0) / 5.0

    penalty = hbond_module.get_angle_penalty(interaction_angles, minAng, optMin, optMax, maxAng)
    assert torch.allclose(penalty, torch.tensor([[expected_penalty]], device=dev, dtype=torch.float))
