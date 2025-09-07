import torch
from vitra.energies.Disulfide_net import Disulfide_net

def test_disulfide_instantiation():
    """
    Tests that the Disulfide_net module can be instantiated without errors.
    """
    dev = 'cpu'
    disulfide_module = Disulfide_net(dev=dev)
    assert disulfide_module is not None
