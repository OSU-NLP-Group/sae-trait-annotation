import dataclasses
import pathlib

import beartype

from .. import config


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class Config:
    """Configuration for a Vision Transformer (ViT) and Sparse Autoencoder (SAE) model pair.

    Stores paths and configuration needed to load and run a specific ViT+SAE combination.
    """

    key: str
    """The lookup key."""

    vit_family: str
    """The family of ViT model, e.g. 'clip' for CLIP models."""

    vit_ckpt: str
    """Checkpoint identifier for the ViT model, either as HuggingFace path or model/checkpoint pair."""

    sae_ckpt: str
    """Identifier for the SAE checkpoint to load."""

    tensor_dpath: pathlib.Path
    """Directory containing precomputed tensors for this model combination."""

    dataset_name: str
    """Which dataset to use."""

    acts_cfg: config.DataLoad
    """Which activations to load for normalizing."""

    @property
    def wrapped_cfg(self) -> config.Activations:
        n_patches = 196
        if self.vit_family == "dinov2":
            n_patches = 256

        return config.Activations(
            vit_family=self.vit_family,
            vit_ckpt=self.vit_ckpt,
            vit_layers=[-2],
            n_patches_per_img=n_patches,
        )


def get_model_lookup() -> dict[str, Config]:
    cfgs = [
        Config(
            "bioclip/inat21",
            "clip",
            "hf-hub:imageomics/bioclip",
            "gpnn7x3p",
            pathlib.Path(
                "/research/nfs_su_809/workspace/stevens.994/saev/features/gpnn7x3p-high-freq/sort_by_patch/"
            ),
            "inat21__train_mini",
            config.DataLoad(
                shard_root="/local/scratch/stevens.994/cache/saev/50149a5a12c70d378dc38f1976d676239839b591cadbfc9af5c84268ac30a868",
                n_random_samples=2**16,
            ),
        ),
        Config(
            "clip/inat21",
            "clip",
            "ViT-B-16/openai",
            "rscsjxgd",
            pathlib.Path(
                "/research/nfs_su_809/workspace/stevens.994/saev/features/rscsjxgd-high-freq/sort_by_patch/"
            ),
            "inat21__train_mini",
            config.DataLoad(
                shard_root="/local/scratch/stevens.994/cache/saev/07aed612e3f70b93ecff46e5a3beea7b8a779f0376dcd3bddf1d5a6ffb4c8f76",
                n_random_samples=2**16,
            ),
        ),
        Config(
            "clip/imagenet",
            "clip",
            "ViT-B-16/openai",
            "usvhngx4",
            pathlib.Path(
                "/research/nfs_su_809/workspace/stevens.994/saev/features/usvhngx4-high-freq/sort_by_patch/"
            ),
            "imagenet__train",
            config.DataLoad(
                shard_root="/local/scratch/stevens.994/cache/saev/ac89246f1934b45e2f0487298aebe36ad998b6bd252d880c0c9ec5de78d793c8",
                n_random_samples=2**16,
            ),
        ),
        Config(
            "dinov2/imagenet",
            "dinov2",
            "dinov2_vitb14_reg",
            "oebd6e6i",
            pathlib.Path(
                "/research/nfs_su_809/workspace/stevens.994/saev/features/oebd6e6i/sort_by_patch/"
            ),
            "imagenet__train",
            config.DataLoad(
                shard_root="/local/scratch/stevens.994/cache/saev/724b1b7be995ef7212d64640fec2885737a706a33b8e5a18f7f323223bd43af1",
                n_random_samples=2**16,
            ),
        ),
    ]
    # TODO: figure out how to normalize the activations from the ViT using the same mean/scalar as in the sorted data.
    return {cfg.key: cfg for cfg in cfgs}
