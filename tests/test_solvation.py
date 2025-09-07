from vitra.energies.solvatation import Solvatation

def test_solvation_instantiation():
    """
    Tests that the Solvatation module can be instantiated without errors.
    """
    dev = 'cpu'
    solvation_module = Solvatation(dev=dev)
    assert solvation_module is not None
