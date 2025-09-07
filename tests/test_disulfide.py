from vitra.energies.disulfide_net import DisulfideEnergy

def test_disulfide_instantiation():
    """
    Tests that the DisulfideEnergy module can be instantiated without errors.
    """
    dev = 'cpu'
    disulfide_module = DisulfideEnergy(dev=dev)
    assert disulfide_module is not None
