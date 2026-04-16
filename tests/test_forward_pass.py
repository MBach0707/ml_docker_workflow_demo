from model import IrisClassificationNetwork
import torch

def test_forward_pass():
    ICN = IrisClassificationNetwork()

    test_input = torch.randn(1,4)

    test_output = ICN(test_input)

    assert test_output.shape == (1,3)