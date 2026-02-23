"""
To save lots of activations, we want to do things in parallel, with lots of slurm jobs, and save multiple files, rather than just one.

This module handles that additional complexity.

Conceptually, activations are either thought of as

1. A single [n_imgs x n_layers x (n_patches + 1), d_vit] tensor. This is a *dataset*
2. Multiple [n_imgs_per_shard, n_layers, (n_patches + 1), d_vit] tensors. This is a set of sharded activations.
"""

import dataclasses
import hashlib
import json
import logging
import math
import os
import typing
from collections.abc import Callable

import beartype
import numpy as np
import torch
import torchvision.datasets
from jaxtyping import Float, jaxtyped
from PIL import Image
from torch import Tensor

from . import config, helpers

log_format = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=log_format)
logger = logging.getLogger(__name__)


#######################
# VISION TRANSFORMERS #
#######################


@jaxtyped(typechecker=beartype.beartype)
class RecordedVisionTransformer(torch.nn.Module):
    _storage: Float[Tensor, "batch n_layers all_patches dim"] | None
    _i: int

    def __init__(
        self,
        vit: torch.nn.Module,
        n_patches_per_img: int,
        cls_token: bool,
        layers: list[int],
    ):
        super().__init__()

        self.vit = vit

        self.n_patches_per_img = n_patches_per_img
        self.cls_token = cls_token
        self.layers = layers

        self.patches = vit.get_patches(n_patches_per_img)

        self._storage = None
        self._i = 0

        self.logger = logging.getLogger(f"recorder({vit.name})")

        for i in self.layers:
            self.vit.get_residuals()[i].register_forward_hook(self.hook)

    def hook(
        self, module, args: tuple, output: Float[Tensor, "batch n_layers dim"]
    ) -> None:
        if self._storage is None:
            batch, _, dim = output.shape
            self._storage = self._empty_storage(batch, dim, output.device)

        if self._storage[:, self._i, 0, :].shape != output[:, 0, :].shape:
            batch, _, dim = output.shape

            old_batch, _, _, old_dim = self._storage.shape
            msg = "Output shape does not match storage shape: (batch) %d != %d or (dim) %d != %d"
            self.logger.warning(msg, old_batch, batch, old_dim, dim)

            self._storage = self._empty_storage(batch, dim, output.device)

        self._storage[:, self._i] = output[:, self.patches, :].detach()
        self._i += 1

    def _empty_storage(self, batch: int, dim: int, device: torch.device):
        n_patches_per_img = self.n_patches_per_img
        if self.cls_token:
            n_patches_per_img += 1

        return torch.zeros(
            (batch, len(self.layers), n_patches_per_img, dim), device=device
        )

    def reset(self):
        self._i = 0

    @property
    def activations(self) -> Float[Tensor, "batch n_layers all_patches dim"]:
        if self._storage is None:
            raise RuntimeError("First call forward()")
        return self._storage.cpu()

    def forward(
        self, batch: Float[Tensor, "batch 3 width height"]
    ) -> tuple[
        Float[Tensor, "batch patches dim"],
        Float[Tensor, "batch n_layers all_patches dim"],
    ]:
        self.reset()
        result = self.vit(batch)
        return result, self.activations


@jaxtyped(typechecker=beartype.beartype)
class Clip(torch.nn.Module):
    def __init__(self, vit_ckpt: str):
        super().__init__()

        import open_clip

        if vit_ckpt.startswith("hf-hub:"):
            clip, _ = open_clip.create_model_from_pretrained(
                vit_ckpt, cache_dir=helpers.get_cache_dir()
            )
        else:
            arch, ckpt = vit_ckpt.split("/")
            clip, _ = open_clip.create_model_from_pretrained(
                arch, pretrained=ckpt, cache_dir=helpers.get_cache_dir()
            )

        model = clip.visual
        model.proj = None
        model.output_tokens = True  # type: ignore
        self.model = model.eval()

        assert not isinstance(self.model, open_clip.timm_model.TimmModel)

        self.name = f"clip/{vit_ckpt}"

    def get_residuals(self) -> list[torch.nn.Module]:
        return self.model.transformer.resblocks

    def get_patches(self, cfg: config.Activations) -> slice:
        return slice(None, None, None)

    def forward(
        self, batch: Float[Tensor, "batch 3 width height"]
    ) -> Float[Tensor, "batch patches dim"]:
        cls, patches = self.model(batch)
        return {"cls": cls, "patches": patches}


@jaxtyped(typechecker=beartype.beartype)
class Siglip(torch.nn.Module):
    def __init__(self, vit_ckpt: str):
        super().__init__()

        import open_clip

        if vit_ckpt.startswith("hf-hub:"):
            clip, _ = open_clip.create_model_from_pretrained(
                vit_ckpt, cache_dir=helpers.get_cache_dir()
            )
        else:
            arch, ckpt = vit_ckpt.split("/")
            clip, _ = open_clip.create_model_from_pretrained(
                arch, pretrained=ckpt, cache_dir=helpers.get_cache_dir()
            )

        model = clip.visual
        model.proj = None
        model.output_tokens = True  # type: ignore
        self.model = model

        assert isinstance(self.model, open_clip.timm_model.TimmModel)

    def get_residuals(self) -> list[torch.nn.Module]:
        return self.model.trunk.blocks

    def get_patches(self, cfg: config.Activations) -> slice:
        return slice(None, None, None)

    def forward(
        self, batch: Float[Tensor, "batch 3 width height"]
    ) -> Float[Tensor, "batch patches dim"]:
        result = self.model(batch)
        return result


@jaxtyped(typechecker=beartype.beartype)
class DinoV2(torch.nn.Module):
    def __init__(self, vit_ckpt: str):
        super().__init__()

        self.model = torch.hub.load("facebookresearch/dinov2", vit_ckpt)
        self.name = f"dinov2/{vit_ckpt}"

    def get_residuals(self) -> list[torch.nn.Module]:
        return self.model.blocks

    def get_patches(self, n_patches_per_img: int) -> slice:
        n_reg = self.model.num_register_tokens
        patches = torch.cat((
            torch.tensor([0]),  # CLS token
            torch.arange(n_reg + 1, n_reg + 1 + n_patches_per_img),  # patches
        ))
        return patches

    def forward(
        self, batch: Float[Tensor, "batch 3 width height"]
    ) -> Float[Tensor, "batch patches dim"]:
        dct = self.model.forward_features(batch)

        features = torch.cat(
            (dct["x_norm_clstoken"][:, None, :], dct["x_norm_patchtokens"]), axis=1
        )
        return features


@jaxtyped(typechecker=beartype.beartype)
class Moondream2(torch.nn.Module):
    """
    Moondream2 has 14x14 pixel patches. For a 378x378 image (as we use here), this is 27x27 patches for a total of 729, with no [CLS] token.
    """

    def __init__(self, vit_ckpt: str):
        super().__init__()

        import transformers

        vit_id, revision = vit_ckpt.split(":")

        mllm = transformers.AutoModelForCausalLM.from_pretrained(
            vit_id, revision=revision, trust_remote_code=True
        )
        self.model = mllm.vision_encoder.encoder.model.visual

    def get_patches(self, cfg: config.Activations) -> slice:
        return slice(None, None, None)

    def get_residuals(self) -> list[torch.nn.Module]:
        return self.model.blocks

    def forward(
        self, batch: Float[Tensor, "batch 3 width height"]
    ) -> Float[Tensor, "batch patches dim"]:
        features = self.model(batch)
        return features


@beartype.beartype
def make_vit(vit_family: str, vit_ckpt: str):
    if vit_family == "clip":
        return Clip(vit_ckpt)
    elif vit_family == "siglip":
        return Siglip(vit_ckpt)
    elif vit_family == "dinov2":
        return DinoV2(vit_ckpt)
    elif vit_family == "moondream2":
        return Moondream2(vit_ckpt)
    else:
        typing.assert_never(vit_family)


@beartype.beartype
def make_img_transform(vit_family: str, vit_ckpt: str) -> Callable:
    if vit_family == "clip" or vit_family == "siglip":
        import open_clip

        if vit_ckpt.startswith("hf-hub:"):
            _, img_transform = open_clip.create_model_from_pretrained(
                vit_ckpt, cache_dir=helpers.get_cache_dir()
            )
        else:
            arch, ckpt = vit_ckpt.split("/")
            _, img_transform = open_clip.create_model_from_pretrained(
                arch, pretrained=ckpt, cache_dir=helpers.get_cache_dir()
            )
        return img_transform

    elif vit_family == "dinov2":
        from torchvision.transforms import v2

        return v2.Compose([
            # TODO: I bet this should be 256, 256, which is causing localization issues in non-square images.
            v2.Resize(size=256),
            v2.CenterCrop(size=(224, 224)),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.4850, 0.4560, 0.4060], std=[0.2290, 0.2240, 0.2250]),
        ])

    elif vit_family == "moondream2":
        from torchvision.transforms import v2

        # Assume fixed image ratio, 378x378
        return v2.Compose([
            v2.Resize(size=(378, 378), interpolation=v2.InterpolationMode.BICUBIC),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
    else:
        typing.assert_never(vit_family)


###############
# ACTIVATIONS #
###############


@jaxtyped(typechecker=beartype.beartype)
class Dataset(torch.utils.data.Dataset):
    """
    Dataset of activations from disk.
    """

    class Example(typing.TypedDict):
        """Individual example."""

        act: Float[Tensor, " d_vit"]
        image_i: int
        patch_i: int

    cfg: config.DataLoad
    """Configuration; set via CLI args."""
    metadata: "Metadata"
    """Activations metadata; automatically loaded from disk."""
    layer_index: int
    """Layer index into the shards if we are choosing a specific layer."""
    scalar: float
    """Normalizing scalar such that ||x / scalar ||_2 ~= sqrt(d_vit)."""
    act_mean: Float[Tensor, " d_vit"]
    """Mean activation."""

    def __init__(self, cfg: config.DataLoad):
        self.cfg = cfg
        if not os.path.isdir(self.cfg.shard_root):
            raise RuntimeError(f"Activations are not saved at '{self.cfg.shard_root}'.")

        metadata_fpath = os.path.join(self.cfg.shard_root, "metadata.json")
        self.metadata = Metadata.load(metadata_fpath)

        # Pick a really big number so that if you accidentally use this when you shouldn't, you get an out of bounds IndexError.
        self.layer_index = 1_000_000
        if isinstance(self.cfg.layer, int):
            err_msg = f"Non-exact matches for .layer field not supported; {self.cfg.layer} not in {self.metadata.layers}."
            assert self.cfg.layer in self.metadata.layers, err_msg
            self.layer_index = self.metadata.layers.index(self.cfg.layer)

        # Premptively set these values so that preprocess() doesn't freak out.
        self.scalar = 1.0
        self.act_mean = torch.zeros(self.d_vit)

        # If either of these are true, we must do this work.
        if self.cfg.scale_mean is True or self.cfg.scale_norm is True:
            # Load a random subset of samples to calculate the mean activation and mean L2 norm.
            perm = np.random.default_rng(seed=42).permutation(len(self))
            perm = perm[: cfg.n_random_samples]

            samples = [
                self[p.item()]
                for p in helpers.progress(
                    perm, every=25_000, desc="examples to calc means"
                )
            ]
            samples = torch.stack([sample["act"] for sample in samples])
            if samples.abs().max() > 1e3:
                raise ValueError(
                    "You found an abnormally large activation {example.abs().max().item():.5f} that will mess up your L2 mean."
                )

            # Activation mean
            if self.cfg.scale_mean:
                self.act_mean = samples.mean(axis=0)
                if (self.act_mean > 1e3).any():
                    raise ValueError(
                        "You found an abnormally large activation that is messing up your activation mean."
                    )

            # Norm
            if self.cfg.scale_norm:
                l2_mean = torch.linalg.norm(samples - self.act_mean, axis=1).mean()
                if l2_mean > 1e3:
                    raise ValueError(
                        "You found an abnormally large activation that is messing up your L2 mean."
                    )

                self.scalar = l2_mean / math.sqrt(self.d_vit)
        elif isinstance(self.cfg.scale_mean, str):
            # Load mean activations from disk
            self.act_mean = torch.load(self.cfg.scale_mean)
        elif isinstance(self.cfg.scale_norm, str):
            # Load scalar normalization from disk
            self.scalar = torch.load(self.cfg.scale_norm).item()

    def transform(self, act: Float[np.ndarray, " d_vit"]) -> Float[Tensor, " d_vit"]:
        """
        Apply a scalar normalization so the mean squared L2 norm is same as d_vit. This is from 'Scaling Monosemanticity':

        > As a preprocessing step we apply a scalar normalization to the model activations so their average squared L2 norm is the residual stream dimension

        So we divide by self.scalar which is the datasets (approximate) L2 mean before normalization divided by sqrt(d_vit).
        """
        act = torch.from_numpy(act.copy())
        act = act.clamp(-self.cfg.clamp, self.cfg.clamp)
        return (act - self.act_mean) / self.scalar

    @property
    def d_vit(self) -> int:
        """Dimension of the underlying vision transformer's embedding space."""
        return self.metadata.d_vit

    @jaxtyped(typechecker=beartype.beartype)
    def __getitem__(self, i: int) -> Example:
        match (self.cfg.patches, self.cfg.layer):
            case ("cls", int()):
                img_act = self.get_img_patches(i)
                # Select layer's cls token.
                act = img_act[self.layer_index, 0, :]
                return self.Example(act=self.transform(act), image_i=i, patch_i=-1)
            case ("cls", "meanpool"):
                img_act = self.get_img_patches(i)
                # Select cls tokens from across all layers
                cls_act = img_act[:, 0, :]
                # Meanpool over the layers
                act = cls_act.mean(axis=0)
                return self.Example(act=self.transform(act), image_i=i, patch_i=-1)
            case ("meanpool", int()):
                img_act = self.get_img_patches(i)
                # Select layer's patches.
                layer_act = img_act[self.layer_index, 1:, :]
                # Meanpool over the patches
                act = layer_act.mean(axis=0)
                return self.Example(act=self.transform(act), image_i=i, patch_i=-1)
            case ("meanpool", "meanpool"):
                img_act = self.get_img_patches(i)
                # Select all layer's patches.
                act = img_act[:, 1:, :]
                # Meanpool over the layers and patches
                act = act.mean(axis=(0, 1))
                return self.Example(act=self.transform(act), image_i=i, patch_i=-1)
            case ("patches", int()):
                n_imgs_per_shard = (
                    self.metadata.n_patches_per_shard
                    // len(self.metadata.layers)
                    // (self.metadata.n_patches_per_img + 1)
                )
                n_examples_per_shard = (
                    n_imgs_per_shard * self.metadata.n_patches_per_img
                )

                shard = i // n_examples_per_shard
                pos = i % n_examples_per_shard

                acts_fpath = os.path.join(self.cfg.shard_root, f"acts{shard:06}.bin")
                shape = (
                    n_imgs_per_shard,
                    len(self.metadata.layers),
                    self.metadata.n_patches_per_img + 1,
                    self.metadata.d_vit,
                )
                acts = np.memmap(acts_fpath, mode="c", dtype=np.float32, shape=shape)
                # Choose the layer and the non-CLS tokens.
                acts = acts[:, self.layer_index, 1:]

                # Choose a patch among n and the patches.
                act = acts[
                    pos // self.metadata.n_patches_per_img,
                    pos % self.metadata.n_patches_per_img,
                ]
                return self.Example(
                    act=self.transform(act),
                    # What image is this?
                    image_i=i // self.metadata.n_patches_per_img,
                    patch_i=i % self.metadata.n_patches_per_img,
                )
            case _:
                print((self.cfg.patches, self.cfg.layer))
                typing.assert_never((self.cfg.patches, self.cfg.layer))

    def get_shard_patches(self):
        raise NotImplementedError()

    def get_img_patches(
        self, i: int
    ) -> Float[np.ndarray, "n_layers all_patches d_vit"]:
        n_imgs_per_shard = (
            self.metadata.n_patches_per_shard
            // len(self.metadata.layers)
            // (self.metadata.n_patches_per_img + 1)
        )
        shard = i // n_imgs_per_shard
        pos = i % n_imgs_per_shard
        acts_fpath = os.path.join(self.cfg.shard_root, f"acts{shard:06}.bin")
        shape = (
            n_imgs_per_shard,
            len(self.metadata.layers),
            self.metadata.n_patches_per_img + 1,
            self.metadata.d_vit,
        )
        acts = np.memmap(acts_fpath, mode="c", dtype=np.float32, shape=shape)
        # Note that this is not yet copied!
        return acts[pos]

    def __len__(self) -> int:
        """
        Dataset length depends on `patches` and `layer`.
        """
        match (self.cfg.patches, self.cfg.layer):
            case ("cls", "all"):
                # Return a CLS token from a random image and random layer.
                return self.metadata.n_imgs * len(self.metadata.layers)
            case ("cls", int()):
                # Return a CLS token from a random image and fixed layer.
                return self.metadata.n_imgs
            case ("cls", "meanpool"):
                # Return a CLS token from a random image and meanpool over all layers.
                return self.metadata.n_imgs
            case ("meanpool", "all"):
                # Return the meanpool of all patches from a random image and random layer.
                return self.metadata.n_imgs * len(self.metadata.layers)
            case ("meanpool", int()):
                # Return the meanpool of all patches from a random image and fixed layer.
                return self.metadata.n_imgs
            case ("meanpool", "meanpool"):
                # Return the meanpool of all patches from a random image and meanpool over all layers.
                return self.metadata.n_imgs
            case ("patches", int()):
                # Return a patch from a random image, fixed layer, and random patch.
                return self.metadata.n_imgs * (self.metadata.n_patches_per_img)
            case ("patches", "meanpool"):
                # Return a patch from a random image, meanpooled over all layers, and a random patch.
                return self.metadata.n_imgs * (self.metadata.n_patches_per_img)
            case ("patches", "all"):
                # Return a patch from a random image, random layer and random patch.
                return (
                    self.metadata.n_imgs
                    * len(self.metadata.layers)
                    * self.metadata.n_patches_per_img
                )
            case _:
                typing.assert_never((self.cfg.patches, self.cfg.layer))


##########
# IMAGES #
##########


@beartype.beartype
def setup(cfg: config.Activations):
    """
    Run dataset-specific setup. These setup functions can assume they are the only job running, but they should be idempotent; they should be safe (and ideally cheap) to run multiple times in a row.
    """
    if isinstance(cfg.data, config.ImagenetDataset):
        setup_imagenet(cfg)
    elif isinstance(cfg.data, config.ImageFolderDataset):
        setup_imagefolder(cfg)
    elif isinstance(cfg.data, config.Ade20kDataset):
        setup_ade20k(cfg)
    else:
        typing.assert_never(cfg.data)


@beartype.beartype
def setup_imagenet(cfg: config.Activations):
    assert isinstance(cfg.data, config.ImagenetDataset)


@beartype.beartype
def setup_imagefolder(cfg: config.Activations):
    assert isinstance(cfg.data, config.ImageFolderDataset)
    logger.info("No dataset-specific setup for ImageFolder.")


@beartype.beartype
def setup_ade20k(cfg: config.Activations):
    assert isinstance(cfg.data, config.Ade20kDataset)

    # url = "http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip"
    # breakpoint()

    # 1. Check


@beartype.beartype
def get_dataset(cfg: config.DatasetConfig, *, img_transform):
    """
    Gets the dataset for the current experiment; delegates construction to dataset-specific functions.

    Args:
        cfg: Experiment config.
        img_transform: Image transform to be applied to each image.

    Returns:
        A dataset that has dictionaries with `'image'`, `'index'`, `'target'`, and `'label'` keys containing examples.
    """
    if isinstance(cfg, config.ImagenetDataset):
        return Imagenet(cfg, img_transform=img_transform)
    elif isinstance(cfg, config.Ade20kDataset):
        return Ade20k(cfg, img_transform=img_transform)
    elif isinstance(cfg, config.ImageFolderDataset):
        return ImageFolder(cfg.root, transform=img_transform)
    else:
        typing.assert_never(cfg)


@beartype.beartype
def get_dataloader(cfg: config.Activations, *, img_transform=None):
    """
    Gets the dataloader for the current experiment; delegates dataloader construction to dataset-specific functions.

    Args:
        cfg: Experiment config.
        img_transform: Image transform to be applied to each image.

    Returns:
        A PyTorch Dataloader that yields dictionaries with `'image'` keys containing image batches.
    """
    if isinstance(
        cfg.data,
        (config.ImagenetDataset, config.ImageFolderDataset, config.Ade20kDataset),
    ):
        dataloader = get_default_dataloader(cfg, img_transform=img_transform)
    else:
        typing.assert_never(cfg.data)

    return dataloader


@beartype.beartype
def get_default_dataloader(
    cfg: config.Activations, *, img_transform: Callable
) -> torch.utils.data.DataLoader:
    """
    Get a dataloader for a default map-style dataset.

    Args:
        cfg: Config.
        img_transform: Image transform to be applied to each image.

    Returns:
        A PyTorch Dataloader that yields dictionaries with `'image'` keys containing image batches, `'index'` keys containing original dataset indices and `'label'` keys containing label batches.
    """
    dataset = get_dataset(cfg.data, img_transform=img_transform)

    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=cfg.vit_batch_size,
        drop_last=False,
        num_workers=cfg.n_workers,
        persistent_workers=cfg.n_workers > 0,
        shuffle=False,
        pin_memory=False,
    )
    return dataloader


@beartype.beartype
class Imagenet(torch.utils.data.Dataset):
    def __init__(self, cfg: config.ImagenetDataset, *, img_transform=None):
        import datasets

        self.hf_dataset = datasets.load_dataset(
            cfg.name, split=cfg.split, trust_remote_code=True
        )

        self.img_transform = img_transform
        self.labels = self.hf_dataset.info.features["label"].names

    def __getitem__(self, i):
        example = self.hf_dataset[i]
        example["index"] = i

        example["image"] = example["image"].convert("RGB")
        if self.img_transform:
            example["image"] = self.img_transform(example["image"])
        example["target"] = example.pop("label")
        example["label"] = self.labels[example["target"]]

        return example

    def __len__(self) -> int:
        return len(self.hf_dataset)


@beartype.beartype
class ImageFolder(torchvision.datasets.ImageFolder):
    def __getitem__(self, index: int) -> dict[str, object]:
        """
        Args:
            index: Index

        Returns:
            dict with keys 'image', 'index', 'target' and 'label'.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return {
            "image": sample,
            "target": target,
            "label": self.classes[target],
            "index": index,
        }


@beartype.beartype
class Ade20k(torch.utils.data.Dataset):
    @beartype.beartype
    @dataclasses.dataclass(frozen=True)
    class Sample:
        img_path: str
        seg_path: str
        label: str
        target: int

    samples: list[Sample]

    def __init__(
        self,
        cfg: config.Ade20kDataset,
        *,
        img_transform: Callable | None = None,
        seg_transform: Callable | None = lambda x: None,
    ):
        self.logger = logging.getLogger("ade20k")
        self.cfg = cfg
        self.img_dir = os.path.join(cfg.root, "images")
        self.seg_dir = os.path.join(cfg.root, "annotations")
        self.img_transform = img_transform
        self.seg_transform = seg_transform

        # Check that we have the right path.
        for subdir in ("images", "annotations"):
            if not os.path.isdir(os.path.join(cfg.root, subdir)):
                # Something is missing.
                if os.path.realpath(cfg.root).endswith(subdir):
                    self.logger.warning(
                        "The ADE20K root should contain 'images/' and 'annotations/' directories."
                    )
                raise ValueError(f"Can't find path '{os.path.join(cfg.root, subdir)}'.")

        _, split_mapping = torchvision.datasets.folder.find_classes(self.img_dir)
        split_lookup: dict[int, str] = {
            value: key for key, value in split_mapping.items()
        }
        self.loader = torchvision.datasets.folder.default_loader

        assert cfg.split in set(split_lookup.values())

        # Load all the image paths.
        imgs: list[str] = [
            path
            for path, s in torchvision.datasets.folder.make_dataset(
                self.img_dir,
                split_mapping,
                extensions=torchvision.datasets.folder.IMG_EXTENSIONS,
            )
            if split_lookup[s] == cfg.split
        ]

        segs: list[str] = [
            path
            for path, s in torchvision.datasets.folder.make_dataset(
                self.seg_dir,
                split_mapping,
                extensions=torchvision.datasets.folder.IMG_EXTENSIONS,
            )
            if split_lookup[s] == cfg.split
        ]

        # Load all the targets, classes and mappings
        with open(os.path.join(cfg.root, "sceneCategories.txt")) as fd:
            img_labels: list[str] = [line.split()[1] for line in fd.readlines()]

        label_set = sorted(set(img_labels))
        label_to_idx = {label: i for i, label in enumerate(label_set)}

        self.samples = [
            self.Sample(img_path, seg_path, label, label_to_idx[label])
            for img_path, seg_path, label in zip(imgs, segs, img_labels)
        ]

    def __getitem__(self, index: int) -> dict[str, object]:
        # Convert to dict.
        sample = dataclasses.asdict(self.samples[index])

        sample["image"] = self.loader(sample.pop("img_path"))
        if self.img_transform is not None:
            image = self.img_transform(sample.pop("image"))
            if image is not None:
                sample["image"] = image

        sample["segmentation"] = Image.open(sample.pop("seg_path")).convert("L")
        if self.seg_transform is not None:
            segmentation = self.seg_transform(sample.pop("segmentation"))
            if segmentation is not None:
                sample["segmentation"] = segmentation

        sample["index"] = index

        return sample

    def __len__(self) -> int:
        return len(self.samples)


########
# MAIN #
########


@beartype.beartype
def main(cfg: config.Activations):
    """
    Args:
        cfg: Config for activations.
    """
    logger = logging.getLogger("dump")

    if not cfg.ssl:
        logger.warning("Ignoring SSL certs. Try not to do this!")
        # https://github.com/openai/whisper/discussions/734#discussioncomment-4491761
        # Ideally we don't have to disable SSL but we are only downloading weights.
        import ssl

        ssl._create_default_https_context = ssl._create_unverified_context

    # Run any setup steps.
    setup(cfg)

    # Actually record activations.
    if cfg.slurm:
        import submitit

        executor = submitit.SlurmExecutor(folder=cfg.log_to)
        executor.update_parameters(
            time=24 * 60,
            partition="gpu",
            gpus_per_node=1,
            cpus_per_task=cfg.n_workers + 4,
            stderr_to_stdout=True,
            account=cfg.slurm_acct,
        )

        job = executor.submit(worker_fn, cfg)
        logger.info("Running job '%s'.", job.job_id)
        job.result()

    else:
        worker_fn(cfg)


@beartype.beartype
def worker_fn(cfg: config.Activations):
    """
    Args:
        cfg: Config for activations.
    """

    if torch.cuda.is_available():
        # This enables tf32 on Ampere GPUs which is only 8% slower than
        # float16 and almost as accurate as float32
        # This was a default in pytorch until 1.12
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    logger = logging.getLogger("dump")

    vit = make_vit(cfg.vit_family, cfg.vit_ckpt).to(cfg.device)
    vit = RecordedVisionTransformer(
        vit, cfg.n_patches_per_img, cfg.cls_token, cfg.vit_layers
    )
    img_transform = make_img_transform(cfg.vit_family, cfg.vit_ckpt)
    dataloader = get_dataloader(cfg, img_transform=img_transform)

    writer = ShardWriter(cfg)

    n_batches = cfg.data.n_imgs // cfg.vit_batch_size + 1
    logger.info("Dumping %d batches of %d examples.", n_batches, cfg.vit_batch_size)

    if cfg.device == "cuda" and not torch.cuda.is_available():
        logger.warning("No CUDA device available, using CPU.")
        cfg = dataclasses.replace(cfg, device="cpu")

    vit = vit.to(cfg.device)
    # vit = torch.compile(vit)

    i = 0
    # Calculate and write ViT activations.
    with torch.inference_mode():
        for batch in helpers.progress(dataloader, total=n_batches):
            images = batch.pop("image").to(cfg.device)
            # cache has shape [batch size, n layers, n patches + 1, d vit]
            out, cache = vit(images)
            del out

            writer[i : i + len(cache)] = cache
            i += len(cache)

    writer.flush()


@beartype.beartype
class ShardWriter:
    """
    ShardWriter is a stateful object that handles sharded activation writing to disk.
    """

    root: str
    shape: tuple[int, int, int, int]
    shard: int
    acts_path: str
    acts: Float[np.ndarray, "n_imgs_per_shard n_layers all_patches d_vit"] | None
    filled: int

    def __init__(self, cfg: config.Activations):
        self.logger = logging.getLogger("shard-writer")

        self.root = get_acts_dir(cfg)

        n_patches_per_img = cfg.n_patches_per_img
        if cfg.cls_token:
            n_patches_per_img += 1
        self.n_imgs_per_shard = (
            cfg.n_patches_per_shard // len(cfg.vit_layers) // n_patches_per_img
        )
        self.shape = (
            self.n_imgs_per_shard,
            len(cfg.vit_layers),
            n_patches_per_img,
            cfg.d_vit,
        )

        self.shard = -1
        self.acts = None
        self.next_shard()

    @jaxtyped(typechecker=beartype.beartype)
    def __setitem__(
        self, i: slice, val: Float[Tensor, "_ n_layers all_patches d_vit"]
    ) -> None:
        assert i.step is None
        a, b = i.start, i.stop
        assert len(val) == b - a

        offset = self.n_imgs_per_shard * self.shard

        if b >= offset + self.n_imgs_per_shard:
            # We have run out of space in this mmap'ed file. Let's fill it as much as we can.
            n_fit = offset + self.n_imgs_per_shard - a
            self.acts[a - offset : a - offset + n_fit] = val[:n_fit]
            self.filled = a - offset + n_fit

            self.next_shard()

            # Recursively call __setitem__ in case we need *another* shard
            self[a + n_fit : b] = val[n_fit:]
        else:
            msg = f"0 <= {a} - {offset} <= {offset} + {self.n_imgs_per_shard}"
            assert 0 <= a - offset <= offset + self.n_imgs_per_shard, msg
            msg = f"0 <= {b} - {offset} <= {offset} + {self.n_imgs_per_shard}"
            assert 0 <= b - offset <= offset + self.n_imgs_per_shard, msg
            self.acts[a - offset : b - offset] = val
            self.filled = b - offset

    def flush(self) -> None:
        if self.acts is not None:
            self.acts.flush()

        self.acts = None

    def next_shard(self) -> None:
        self.flush()

        self.shard += 1
        self._count = 0
        self.acts_path = os.path.join(self.root, f"acts{self.shard:06}.bin")
        self.acts = np.memmap(
            self.acts_path, mode="w+", dtype=np.float32, shape=self.shape
        )
        self.filled = 0

        self.logger.info("Opened shard '%s'.", self.acts_path)


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class Metadata:
    vit_family: str
    vit_ckpt: str
    layers: tuple[int, ...]
    n_patches_per_img: int
    cls_token: bool
    d_vit: int
    seed: int
    n_imgs: int
    n_patches_per_shard: int
    data: str

    @classmethod
    def from_cfg(cls, cfg: config.Activations) -> "Metadata":
        return cls(
            cfg.vit_family,
            cfg.vit_ckpt,
            tuple(cfg.vit_layers),
            cfg.n_patches_per_img,
            cfg.cls_token,
            cfg.d_vit,
            cfg.seed,
            cfg.data.n_imgs,
            cfg.n_patches_per_shard,
            str(cfg.data),
        )

    @classmethod
    def load(cls, fpath) -> "Metadata":
        with open(fpath) as fd:
            dct = json.load(fd)
        dct["layers"] = tuple(dct.pop("layers"))
        return cls(**dct)

    def dump(self, fpath):
        with open(fpath, "w") as fd:
            json.dump(dataclasses.asdict(self), fd, indent=4)

    @property
    def hash(self) -> str:
        cfg_str = json.dumps(dataclasses.asdict(self), sort_keys=True)
        return hashlib.sha256(cfg_str.encode("utf-8")).hexdigest()


@beartype.beartype
def get_acts_dir(cfg: config.Activations) -> str:
    """
    Return the activations directory based on the relevant values of a config.
    Also saves a metadata.json file to that directory for human reference.

    Args:
        cfg: Config for experiment.

    Returns:
        Directory to where activations should be dumped/loaded from.
    """
    metadata = Metadata.from_cfg(cfg)

    acts_dir = os.path.join(cfg.dump_to, metadata.hash)
    os.makedirs(acts_dir, exist_ok=True)

    metadata.dump(os.path.join(acts_dir, "metadata.json"))

    return acts_dir
