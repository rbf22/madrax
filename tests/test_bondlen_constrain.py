import torch
import math
from vitra.energies.bond_len_constrain import BondLengthConstraintEnergy

def test_bondlen_constrain_score_distro():
    """
    Tests the scoreDistro method in the BondLenConstrain module.
    """
    dev = 'cpu'
    bondlen_module = BondLengthConstraintEnergy(dev=dev)

    # Mock mean and std tensors
    bondlen_module.mean = torch.tensor([[1.3, 2.1, 2.0]], device=dev)
    bondlen_module.std = torch.tensor([[0.1, 0.2, 0.3]], device=dev)

    # Case 1: Input is exactly the mean, score should be close to 0
    inputs = torch.tensor([[1.3, 2.1, 2.0]], device=dev)
    seq = torch.tensor([0], device=dev, dtype=torch.long)

    score = bondlen_module._score_distribution(inputs, seq)
    # The score is -log(P), where P is the probability density. At the mean, P is maximal.
    # The formula is constructed such that the score at the mean is 0.
    assert torch.allclose(score, torch.tensor([[0.0, 0.0, 0.0]], device=dev), atol=1e-6)

    # We can also check that the sum of scores is close to the sum of individual scores
    expected_score_sum = 0
    for i in range(3):
        var = bondlen_module.std[0, i] ** 2
        denom = (2 * math.pi * var) ** 0.5
        num = torch.exp(-(inputs[0, i] - bondlen_module.mean[0, i]) ** 2 / (2 * var))
        norm_factor = 1.0 / denom
        expected_score_sum += -((num / denom).log() - torch.log(norm_factor))

    assert torch.allclose(score.sum(), expected_score_sum)

    # Case 2: Input is far from the mean, score should be large
    inputs = torch.tensor([[2.0, 3.0, 3.0]], device=dev)
    score_large = bondlen_module._score_distribution(inputs, seq)
    assert (score_large > score).all()
