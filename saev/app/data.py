import base64
import functools
import logging
import typing

import beartype
import pyvips
import torch
import torchvision.datasets
from PIL import Image

from .. import activations, config

logger = logging.getLogger("app.data")


class VipsImageFolder(torchvision.datasets.ImageFolder):
    """
    Clone of ImageFolder that returns pyvips.Image instead of PIL.Image.Image.
    """

    def __init__(
        self,
        root: str,
        transform: typing.Callable | None = None,
        target_transform: typing.Callable | None = None,
    ):
        super().__init__(
            root,
            transform=transform,
            target_transform=target_transform,
            loader=self._vips_loader,
        )

    @staticmethod
    def _vips_loader(path: str) -> torch.Tensor:
        """Load and convert image to tensor using pyvips."""
        image = pyvips.Image.new_from_file(path, access="random")
        return image

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
class VipsImagenet(activations.Imagenet):
    def __getitem__(self, i):
        example = self.hf_dataset[i]
        example["index"] = i

        example["image"] = example["image"].convert("RGB")
        # Convert to pyvips
        example["image"] = pyvips.Image.new_from_memory(
            example["image"].tobytes(),
            example["image"].width,
            example["image"].height,
            3,  # bands (RGB)
            "uchar",
        )
        if self.img_transform:
            example["image"] = self.img_transform(example["image"])
        example["target"] = example.pop("label")
        example["label"] = self.labels[example["target"]]

        return example


@functools.cache
def get_datasets():
    datasets = {
        "inat21__train_mini": VipsImageFolder(
            root="/research/nfs_su_809/workspace/stevens.994/datasets/inat21/train_mini/"
        ),
        "imagenet__train": VipsImagenet(config.ImagenetDataset()),
    }
    logger.info("Loaded datasets.")
    return datasets


@beartype.beartype
def get_img_v_raw(key: str, i: int) -> tuple[pyvips.Image, str]:
    """
    Get raw image and processed label from dataset.

    Returns:
        Tuple of pyvips.Image and classname.
    """
    dataset = get_datasets()[key]
    sample = dataset[i]
    # iNat21 specific: Remove taxonomy prefix
    label = " ".join(sample["label"].split("_")[1:])
    return sample["image"], label


def to_sized(
    img_v_raw: pyvips.Image, min_px: int, crop_px: tuple[int, int]
) -> pyvips.Image:
    """Convert raw vips image to standard model input size (resize + crop)."""
    # Calculate scale factor to make smallest dimension = min_px
    scale = min_px / min(img_v_raw.width, img_v_raw.height)

    # Resize maintaining aspect ratio
    img_v_raw = img_v_raw.resize(scale)
    assert min(img_v_raw.width, img_v_raw.height) == min_px

    # Calculate crop coordinates to center crop
    left = (img_v_raw.width - crop_px[0]) // 2
    top = (img_v_raw.height - crop_px[1]) // 2

    # Crop to final size
    return img_v_raw.crop(left, top, crop_px[0], crop_px[1])


@beartype.beartype
def pil_to_vips(img_p: Image.Image) -> pyvips.Image:
    """Convert a PIL Image to a pyvips Image."""
    # Convert PIL image to bytes
    img_bytes = img_p.tobytes()
    # Create new pyvips image from memory buffer
    return pyvips.Image.new_from_memory(
        img_bytes,
        img_p.width,
        img_p.height,
        len(img_p.getbands()),  # Number of bands (channels)
        "uchar",  # 8-bit unsigned data
    )


@beartype.beartype
def vips_to_base64(img_v: pyvips.Image) -> str:
    buf = img_v.write_to_buffer(".webp")
    b64 = base64.b64encode(buf)
    s64 = b64.decode("utf8")
    return "data:image/webp;base64," + s64
