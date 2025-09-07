import torch
import numpy as np
from vitra.energies.Clash_net import Clash_net

def test_clash_penalty():
    """
    Tests the clashPenalty method in the Clash_net module.
    """
    dev = 'cpu'
    clash_module = Clash_net(dev=dev)

    # Case 1: No clash (distance is negative)
    dist = torch.tensor([-0.1], device=dev)
    penalty = clash_module.clashPenalty(dist, usesoft=False)
    assert torch.allclose(penalty, torch.exp(torch.tensor([-0.1 * 10 - 2])))

    # Case 2: Clash (distance is positive)
    dist = torch.tensor([0.1], device=dev)
    penalty = clash_module.clashPenalty(dist, usesoft=False)
    expected_penalty = torch.exp(dist * 10 - 2)
    assert torch.allclose(penalty, expected_penalty)

    # Case 3: Clash with linear penalty
    dist = torch.tensor([0.6], device=dev)
    penalty = clash_module.clashPenalty(dist, usesoft=False)
    threshold = 0.5
    linear_coeff = 30
    expected_penalty = np.exp(threshold * 10 - 2) + linear_coeff * (dist - threshold)
    assert torch.allclose(penalty, expected_penalty)
