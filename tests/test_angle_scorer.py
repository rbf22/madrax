import torch
from vitra.energies.AngleScorer import AngleScorer

def test_angle_scorer_instantiation():
    """
    Tests that the AngleScorer module can be instantiated without errors.
    This also tests that the pre-trained KDE models can be loaded correctly.
    """
    dev = 'cpu'
    angle_scorer_module = AngleScorer(dev=dev)
    assert angle_scorer_module is not None
    assert len(angle_scorer_module.kdeBB) > 0
    assert len(angle_scorer_module.kdeOmega) > 0
    assert len(angle_scorer_module.kdeSC) > 0
