from vitra.energies.angle_scorer import AngleScorerEnergy

def test_angle_scorer_instantiation():
    """
    Tests that the AngleScorer module can be instantiated without errors.
    This also tests that the pre-trained KDE models can be loaded correctly.
    """
    dev = 'cpu'
    angle_scorer_module = AngleScorerEnergy(dev=dev)
    assert angle_scorer_module is not None
    assert len(angle_scorer_module.kde_bb) > 0
    assert len(angle_scorer_module.kde_omega) > 0
    assert len(angle_scorer_module.kde_sc) > 0
