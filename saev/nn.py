"""
Neural network architectures for sparse autoencoders.
"""

import io
import json
import logging
import os
import typing

import beartype
import einops
import torch
from jaxtyping import Float, jaxtyped
from torch import Tensor

from . import config


class Loss(typing.NamedTuple):
    """The composite loss terms for an autoencoder training batch."""

    mse: Float[Tensor, ""]
    """Reconstruction loss (mean squared error)."""
    sparsity: Float[Tensor, ""]
    """Sparsity loss, typically lambda * L1."""
    ghost_grad: Float[Tensor, ""]
    """Ghost gradient loss, if any."""
    l0: Float[Tensor, ""]
    """L0 magnitude of hidden activations."""
    l1: Float[Tensor, ""]
    """L1 magnitude of hidden activations."""

    @property
    def loss(self) -> Float[Tensor, ""]:
        """Total loss."""
        return self.mse + self.sparsity + self.ghost_grad


@jaxtyped(typechecker=beartype.beartype)
class SparseAutoencoder(torch.nn.Module):
    """
    Sparse auto-encoder (SAE) using L1 sparsity penalty.
    """

    cfg: config.SparseAutoencoder

    def __init__(self, cfg: config.SparseAutoencoder):
        super().__init__()

        self.cfg = cfg

        self.W_enc = torch.nn.Parameter(
            torch.nn.init.kaiming_uniform_(torch.empty(cfg.d_vit, cfg.d_sae))
        )
        self.b_enc = torch.nn.Parameter(torch.zeros(cfg.d_sae))

        self.W_dec = torch.nn.Parameter(
            torch.nn.init.kaiming_uniform_(torch.empty(cfg.d_sae, cfg.d_vit))
        )
        self.b_dec = torch.nn.Parameter(torch.zeros(cfg.d_vit))

        self.logger = logging.getLogger(f"sae(seed={cfg.seed})")

    def forward(
        self, x: Float[Tensor, "batch d_model"]
    ) -> tuple[Float[Tensor, "batch d_model"], Float[Tensor, "batch d_sae"], Loss]:
        """
        Given x, calculates the reconstructed x_hat, the intermediate activations f_x, and the loss.

        Arguments:
            x: a batch of ViT activations.
        """

        # Remove encoder bias as per Anthropic
        h_pre = (
            einops.einsum(
                x - self.b_dec, self.W_enc, "... d_vit, d_vit d_sae -> ... d_sae"
            )
            + self.b_enc
        )
        f_x = torch.nn.functional.relu(h_pre)
        x_hat = self.decode(f_x)

        # Some values of x and x_hat can be very large. We can calculate a safe MSE
        mse_loss = safe_mse(x_hat, x)

        mse_loss = mse_loss.mean()
        l0 = (f_x > 0).float().sum(axis=1).mean(axis=0)
        l1 = f_x.sum(axis=1).mean(axis=0)
        sparsity_loss = self.cfg.sparsity_coeff * l1
        # Ghost loss is included for backwards compatibility.
        ghost_loss = torch.zeros_like(mse_loss)

        return x_hat, f_x, Loss(mse_loss, sparsity_loss, ghost_loss, l0, l1)

    def decode(
        self, f_x: Float[Tensor, "batch d_sae"]
    ) -> Float[Tensor, "batch d_model"]:
        x_hat = (
            einops.einsum(f_x, self.W_dec, "... d_sae, d_sae d_vit -> ... d_vit")
            + self.b_dec
        )
        return x_hat

    @torch.no_grad()
    def init_b_dec(self, vit_acts: Float[Tensor, "n d_vit"]):
        if self.cfg.n_reinit_samples <= 0:
            self.logger.info("Skipping init_b_dec.")
            return
        previous_b_dec = self.b_dec.clone().cpu()
        vit_acts = vit_acts[: self.cfg.n_reinit_samples]
        assert len(vit_acts) == self.cfg.n_reinit_samples
        mean = vit_acts.mean(axis=0)
        previous_distances = torch.norm(vit_acts - previous_b_dec, dim=-1)
        distances = torch.norm(vit_acts - mean, dim=-1)
        self.logger.info(
            "Prev dist: %.3f; new dist: %.3f",
            previous_distances.median(axis=0).values.mean().item(),
            distances.median(axis=0).values.mean().item(),
        )
        self.b_dec.data = mean.to(self.b_dec.dtype).to(self.b_dec.device)

    @torch.no_grad()
    def normalize_w_dec(self):
        """
        Set W_dec to unit-norm columns.
        """
        if self.cfg.normalize_w_dec:
            self.W_dec.data /= torch.norm(self.W_dec.data, dim=1, keepdim=True)

    @torch.no_grad()
    def remove_parallel_grads(self):
        """
        Update grads so that they remove the parallel component
            (d_sae, d_vit) shape
        """
        if not self.cfg.remove_parallel_grads:
            return

        parallel_component = einops.einsum(
            self.W_dec.grad,
            self.W_dec.data,
            "d_sae d_vit, d_sae d_vit -> d_sae",
        )

        self.W_dec.grad -= einops.einsum(
            parallel_component,
            self.W_dec.data,
            "d_sae, d_sae d_vit -> d_sae d_vit",
        )


@jaxtyped(typechecker=beartype.beartype)
def ref_mse(
    x_hat: Float[Tensor, "*d"], x: Float[Tensor, "*d"], norm: bool = True
) -> Float[Tensor, "*d"]:
    mse_loss = torch.pow((x_hat - x.float()), 2)

    if norm:
        mse_loss /= (x**2).sum(dim=-1, keepdim=True).sqrt()
    return mse_loss


@jaxtyped(typechecker=beartype.beartype)
def safe_mse(
    x_hat: Float[Tensor, "*batch d"], x: Float[Tensor, "*batch d"], norm: bool = False
) -> Float[Tensor, "*batch d"]:
    upper = x.abs().max()
    x = x / upper
    x_hat = x_hat / upper

    mse = (x_hat - x) ** 2
    # (sam): I am now realizing that we normalize by the L2 norm of x.
    if norm:
        mse /= torch.linalg.norm(x, axis=-1, keepdim=True) + 1e-12
        return mse * upper

    return mse * upper * upper


@beartype.beartype
def dump(fpath: str, sae: SparseAutoencoder):
    """
    Save an SAE checkpoint to disk along with configuration, using the [trick from equinox](https://docs.kidger.site/equinox/examples/serialisation).

    Arguments:
        fpath: filepath to save checkpoint to.
        sae: sparse autoencoder checkpoint to save.
    """
    kwargs = vars(sae.cfg)

    os.makedirs(os.path.dirname(fpath), exist_ok=True)
    with open(fpath, "wb") as fd:
        kwargs_str = json.dumps(kwargs)
        fd.write((kwargs_str + "\n").encode("utf-8"))
        torch.save(sae.state_dict(), fd)


@beartype.beartype
def load(fpath: str, *, device: str = "cpu") -> SparseAutoencoder:
    """
    Loads a sparse autoencoder from disk.
    """
    with open(fpath, "rb") as fd:
        kwargs = json.loads(fd.readline().decode())
        buffer = io.BytesIO(fd.read())

    cfg = config.SparseAutoencoder(**kwargs)
    model = SparseAutoencoder(cfg)
    state_dict = torch.load(buffer, weights_only=True, map_location=device)
    model.load_state_dict(state_dict)
    return model
