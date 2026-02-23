"""
Uses [hypothesis]() and [hypothesis-torch](https://hypothesis-torch.readthedocs.io/en/stable/compatability/) to generate test cases to compare our normalized MSE implementation to a reference MSE implementation.
"""

import hypothesis
import hypothesis_torch
import pytest
import torch
from jaxtyping import Float
from torch import Tensor

from . import nn


def test_safe_mse_same():
    x = torch.ones((45, 12), dtype=torch.float)
    x_hat = torch.ones((45, 12), dtype=torch.float)
    expected = torch.zeros((45, 12), dtype=torch.float)
    actual = nn.safe_mse(x_hat, x)
    torch.testing.assert_close(actual, expected)


def test_safe_mse_zero_x_hat():
    x = torch.ones((3, 2), dtype=torch.float)
    x_hat = torch.zeros((3, 2), dtype=torch.float)
    expected = torch.ones((3, 2), dtype=torch.float)
    actual = nn.safe_mse(x_hat, x, norm=False)
    torch.testing.assert_close(actual, expected)


def test_safe_mse_nonzero():
    x = torch.full((3, 2), 3, dtype=torch.float)
    x_hat = torch.ones((3, 2), dtype=torch.float)
    expected = nn.ref_mse(x_hat, x)
    actual = nn.safe_mse(x_hat, x)
    torch.testing.assert_close(actual, expected)


def test_safe_mse_large_x():
    x = torch.full((3, 2), 3e28, dtype=torch.float)
    x_hat = torch.ones((3, 2), dtype=torch.float)

    ref = nn.ref_mse(x_hat, x)
    assert ref.isnan().any()

    safe = nn.safe_mse(x_hat, x)
    assert not safe.isnan().any()


@pytest.mark.slow
@hypothesis.settings(suppress_health_check=[hypothesis.HealthCheck.too_slow])
@hypothesis.given(
    x_hat=hypothesis_torch.tensor_strategy(dtype=torch.float32, shape=(1, 2, 3)),
    x=hypothesis_torch.tensor_strategy(dtype=torch.float32, shape=(1, 2, 3)),
)
def test_safe_mse_hypothesis(x_hat: Float[Tensor, "1 2 3"], x: Float[Tensor, "1 2 3"]):
    hypothesis.assume((torch.linalg.norm(x, axis=-1) > 1e-15).all())
    hypothesis.assume(not x.isinf().all())

    expected = nn.ref_mse(x_hat, x)
    actual = nn.safe_mse(x_hat, x)
    torch.testing.assert_close(actual, expected)
