from vitra.energies.electro_net import ElectrostaticsEnergy

def test_electrostatic_energy():
    """
    Tests that the ElectroNet module can be instantiated without errors.
    A more detailed test would require refactoring the net() method.
    """
    dev = 'cpu'
    electro_module = ElectrostaticsEnergy(dev=dev)
    assert electro_module is not None
