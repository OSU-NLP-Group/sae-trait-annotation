import base64
import concurrent.futures
import functools
import json
import logging
import math
import pathlib
import time
import typing

import beartype
import einops.layers.torch
import gradio as gr
import matplotlib
import numpy as np
import PIL.Image
import pyvips
import torch
from jaxtyping import Float, Int, jaxtyped
from torch import Tensor

from .. import activations, nn
from . import data, modeling

log_format = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=log_format)
logger = logging.getLogger("app")
# Disable pyvips info logging
logging.getLogger("pyvips").setLevel(logging.WARNING)


###########
# Globals #
###########


RESIZE_SIZE = 512
"""Resize shorter size to this size in pixels."""

CROP_SIZE = (448, 448)
"""Crop size in pixels."""

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
"""Hardware accelerator, if any."""

CWD = pathlib.Path(".")

MODEL_LOOKUP = modeling.get_model_lookup()

COLORMAP = matplotlib.colormaps.get_cmap("plasma")

logger.info("Set global constants.")


##########
# Models #
##########


@functools.cache
def load_vit(
    model_cfg: modeling.Config,
) -> tuple[
    activations.WrappedVisionTransformer,
    typing.Callable,
    float,
    Float[Tensor, " d_vit"],
]:
    """
    Returns the wrapped ViT, the vit transform, the activation scalar and the activation mean to normalize the activations.
    """
    vit = activations.WrappedVisionTransformer(model_cfg.wrapped_cfg).to(DEVICE).eval()
    vit_transform = activations.make_img_transform(
        model_cfg.vit_family, model_cfg.vit_ckpt
    )
    logger.info("Loaded ViT: %s.", model_cfg.key)

    try:
        # Normalizing constants
        acts_dataset = activations.Dataset(model_cfg.acts_cfg)
        logger.info("Loaded dataset norms: %s.", model_cfg.key)
    except RuntimeError as err:
        logger.warning("Error loading ViT: %s", err)
        return None, None, None, None

    return vit, vit_transform, acts_dataset.scalar.item(), acts_dataset.act_mean


@beartype.beartype
@functools.cache
def load_sae(model_cfg: modeling.Config) -> nn.SparseAutoencoder:
    sae_ckpt_fpath = CWD / "checkpoints" / model_cfg.sae_ckpt / "sae.pt"
    sae = nn.load(sae_ckpt_fpath.as_posix())
    sae.to(DEVICE).eval()
    logger.info("Loaded SAE: %s.", model_cfg.sae_ckpt)
    return sae


############
# Datasets #
############


@beartype.beartype
def load_tensor(path: str | pathlib.Path) -> Tensor:
    return torch.load(path, weights_only=True, map_location="cpu")


@beartype.beartype
@functools.cache
def load_tensors(
    model_cfg: modeling.Config,
) -> tuple[Int[Tensor, "d_sae top_k"], Float[Tensor, "d_sae top_k n_patches"]]:
    top_img_i = load_tensor(model_cfg.tensor_dpath / "top_img_i.pt")
    # TODO: For some reason, the top_values are about 4 times larger.
    top_values = load_tensor(model_cfg.tensor_dpath / "top_values.pt") / 4
    return top_img_i, top_values


@beartype.beartype
def get_image(example_id: str) -> list[str]:
    dataset, split, i_str = example_id.split("__")
    i = int(i_str)
    img_v_raw, label = data.get_img_v_raw(f"{dataset}__{split}", i)
    img_v_sized = data.to_sized(img_v_raw, RESIZE_SIZE, CROP_SIZE)

    return [data.vips_to_base64(img_v_sized), label]


@jaxtyped(typechecker=beartype.beartype)
def add_highlights(
    img_v_sized: pyvips.Image,
    patches: np.ndarray,
    *,
    upper: float | None = None,
    opacity: float = 0.9,
) -> pyvips.Image:
    """Add colored highlights to an image based on patch activation values.

    Overlays a colored highlight on each patch of the image, with intensity proportional
    to the activation value for that patch. Used to visualize which parts of an image
    most strongly activated a particular SAE latent.

    Args:
        img: The base image to highlight
        patches: Array of activation values, one per patch
        upper: Optional maximum value to normalize activations against
        opacity: Opacity of the highlight overlay (0-1)

    Returns:
        A new image with colored highlights overlaid on the original
    """
    if not len(patches):
        return img_v_sized

    # Calculate patch grid dimensions
    grid_w = grid_h = int(math.sqrt(len(patches)))
    assert grid_w * grid_h == len(patches)

    patch_w = img_v_sized.width // grid_w
    patch_h = img_v_sized.height // grid_h
    assert patch_w == patch_h

    patches = np.clip(patches, a_min=0.0, a_max=upper + 1e-9)

    assert upper is not None
    colors = (COLORMAP(patches / (upper + 1e-9))[:, :3] * 256).astype(np.uint8)

    # Create overlay by processing each patch
    overlay = np.zeros((img_v_sized.width, img_v_sized.height, 4), dtype=np.uint8)
    for idx, (val, color) in enumerate(zip(patches, colors)):
        val = val / (upper + 1e-9)

        x = (idx % grid_w) * patch_w
        y = (idx // grid_w) * patch_h

        # Create patch overlay
        patch = np.zeros((patch_w, patch_h, 4), dtype=np.uint8)
        patch[:, :, 0:3] = color
        patch[:, :, 3] = int(256 * val * opacity)
        overlay[y : y + patch_h, x : x + patch_w, :] = patch
    overlay = pyvips.Image.new_from_array(overlay).copy(interpretation="srgb")
    return img_v_sized.addalpha().composite(overlay, "over")


@beartype.beartype
class Example(typing.TypedDict):
    """Represents an example image and its associated label.

    Used to store examples of SAE latent activations for visualization.
    """

    orig_url: str
    """The URL or path to access the original example image."""
    highlighted_url: str
    """The URL or path to access the SAE-highlighted image."""
    label: str
    """The class label or description associated with this example."""
    example_id: str
    """Unique ID to idenfify the original dataset instance."""


@beartype.beartype
class SaeActivation(typing.TypedDict):
    """Represents the activation pattern of a single SAE latent across patches.

    This captures how strongly a particular SAE latent fires on different patches of an input image.
    """

    model_cfg: modeling.Config
    """The model config."""

    latent: int
    """The index of the SAE latent being measured."""

    activations: list[float]
    """The activation values of this latent across different patches. Each value represents how strongly this latent fired on a particular patch."""

    highlighted_url: str
    """The image with the colormaps applied."""

    examples: list[Example]
    """Top examples for this latent."""


@beartype.beartype
def pil_to_vips(pil_img: PIL.Image.Image) -> pyvips.Image:
    # Convert to numpy array
    np_array = np.asarray(pil_img)
    # Handle different formats
    if np_array.ndim == 2:  # Grayscale
        return pyvips.Image.new_from_memory(
            np_array.tobytes(),
            np_array.shape[1],  # width
            np_array.shape[0],  # height
            1,  # bands
            "uchar",
        )
    else:  # RGB/RGBA
        return pyvips.Image.new_from_memory(
            np_array.tobytes(),
            np_array.shape[1],  # width
            np_array.shape[0],  # height
            np_array.shape[2],  # bands
            "uchar",
        )


@beartype.beartype
def vips_to_pil(vips_img: PIL.Image.Image) -> PIL.Image.Image:
    # Convert to numpy array
    np_array = vips_img.numpy()
    # Convert numpy array to PIL Image
    return PIL.Image.fromarray(np_array)


@beartype.beartype
class BufferInfo(typing.NamedTuple):
    buffer: bytes
    width: int
    height: int
    bands: int
    format: object

    @classmethod
    def from_img_v(cls, img_v: pyvips.Image) -> "BufferInfo":
        return cls(
            img_v.write_to_memory(),
            img_v.width,
            img_v.height,
            img_v.bands,
            img_v.format,
        )


@beartype.beartype
def bufferinfo_to_base64(bufferinfo: BufferInfo) -> str:
    img_v = pyvips.Image.new_from_memory(*bufferinfo)
    buf = img_v.write_to_buffer(".webp")
    b64 = base64.b64encode(buf)
    s64 = b64.decode("utf8")
    return "data:image/webp;base64," + s64


@jaxtyped(typechecker=beartype.beartype)
def make_sae_activation(
    model_cfg: modeling.Config,
    latent: int,
    acts: Float[np.ndarray, " n_patches"],
    img_v: pyvips.Image,
    top_img_i: list[int],
    top_values: Float[Tensor, "top_k n_patches"],
    pool: concurrent.futures.Executor,
) -> SaeActivation:
    raw_examples: list[tuple[int, pyvips.Image, Float[np.ndarray, "..."], str]] = []
    seen_i_im = set()
    for i_im, values_p in zip(top_img_i, top_values):
        if i_im in seen_i_im:
            continue

        ex_img_v_raw, ex_label = data.get_img_v_raw(model_cfg.dataset_name, i_im)
        ex_img_v_sized = data.to_sized(ex_img_v_raw, RESIZE_SIZE, CROP_SIZE)
        raw_examples.append((i_im, ex_img_v_sized, values_p.numpy(), ex_label))

        seen_i_im.add(i_im)

        # Only need 4 example images per latent.
        if len(seen_i_im) >= 4:
            break

    upper = top_values.max().item()

    futures = []
    for i_im, ex_img, values_p, ex_label in raw_examples:
        highlighted_img = add_highlights(ex_img, values_p, upper=upper)
        # Submit both conversions to the thread pool
        orig_future = pool.submit(data.vips_to_base64, ex_img)
        highlight_future = pool.submit(data.vips_to_base64, highlighted_img)
        futures.append((i_im, orig_future, highlight_future, ex_label))

    # Wait for all conversions to complete and build examples
    examples = []
    for i_im, orig_future, highlight_future, ex_label in futures:
        example = Example(
            orig_url=orig_future.result(),
            highlighted_url=highlight_future.result(),
            label=ex_label,
            example_id=f"{model_cfg.dataset_name}__{i_im}",
        )
        examples.append(example)

    print(model_cfg.key, latent, top_values.max(), acts.max())

    # Highlight the original image.
    img_sized_v = data.to_sized(img_v, RESIZE_SIZE, CROP_SIZE)
    highlighted_img = add_highlights(img_sized_v, acts, upper=upper)
    highlighted_url = data.vips_to_base64(highlighted_img)

    return SaeActivation(
        model_cfg=model_cfg,
        latent=latent,
        activations=acts.tolist(),
        highlighted_url=highlighted_url,
        examples=examples,
    )


@beartype.beartype
@torch.inference_mode
def get_sae_activations(
    img_p: PIL.Image.Image, latents: dict[str, list[int]]
) -> dict[str, list[SaeActivation]]:
    """
    Args:
        image: Image to get SAE activations for.
        latents: A lookup from model name (string) to a list of latents to report latents for (integers).

    Returns:
        A lookup from model name (string) to a list of SaeActivations, one for each latent in the `latents` argument.
    """
    logger.info("latents: %s", json.dumps(latents))

    response = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
        for model_name, requested_latents in latents.items():
            sae_activations = []
            if not requested_latents:
                logger.warning(
                    "Skipping ViT '%s' with no requested latents.", model_name
                )
                response[model_name] = sae_activations
                continue

            model_cfg = MODEL_LOOKUP[model_name]
            vit, vit_transform, scalar, mean = load_vit(model_cfg)
            if vit is None:
                logger.warning("Skipping ViT '%s'", model_name)
                continue
            sae = load_sae(model_cfg)

            mean = mean.to(DEVICE)
            x = vit_transform(img_p)[None, ...].to(DEVICE)

            _, vit_acts_BLPD = vit(x)
            vit_acts_PD = (
                vit_acts_BLPD[0, 0, 1:].to(DEVICE).clamp(-1e-5, 1e5) - mean
            ) / scalar

            _, f_x_PS, _ = sae(vit_acts_PD)
            # Ignore [CLS] token and get just the requested latents.
            acts_SP = einops.rearrange(f_x_PS, "patches n_latents -> n_latents patches")
            logger.info("Got SAE activations for '%s'.", model_name)
            top_img_i, top_values = load_tensors(model_cfg)
            logger.info("Loaded top SAE activations for '%s'.", model_name)

            for latent in requested_latents:
                sae_activations.append(
                    make_sae_activation(
                        model_cfg,
                        latent,
                        acts_SP[latent].cpu().numpy(),
                        data.pil_to_vips(img_p),
                        top_img_i[latent].tolist(),
                        top_values[latent],
                        pool,
                    )
                )
            response[model_name] = sae_activations
    return response


@beartype.beartype
class progress:
    def __init__(self, it, *, every: int = 10, desc: str = "progress", total: int = 0):
        """
        Wraps an iterable with a logger like tqdm but doesn't use any control codes to manipulate a progress bar, which doesn't work well when your output is redirected to a file. Instead, simple logging statements are used, but it includes quality-of-life features like iteration speed and predicted time to finish.

        Args:
            it: Iterable to wrap.
            every: How many iterations between logging progress.
            desc: What to name the logger.
            total: If non-zero, how long the iterable is.
        """
        self.it = it
        self.every = every
        self.logger = logging.getLogger(desc)
        self.total = total

    def __iter__(self):
        start = time.time()

        try:
            total = len(self)
        except TypeError:
            total = None

        for i, obj in enumerate(self.it):
            yield obj

            if (i + 1) % self.every == 0:
                now = time.time()
                duration_s = now - start
                per_min = (i + 1) / (duration_s / 60)

                if total is not None:
                    pred_min = (total - (i + 1)) / per_min
                    self.logger.info(
                        "%d/%d (%.1f%%) | %.1f it/m (expected finish in %.1fm)",
                        i + 1,
                        total,
                        (i + 1) / total * 100,
                        per_min,
                        pred_min,
                    )
                else:
                    self.logger.info("%d/? | %.1f it/m", i + 1, per_min)

    def __len__(self) -> int:
        if self.total > 0:
            return self.total

        # Will throw exception.
        return len(self.it)


#############
# Interface #
#############


with gr.Blocks() as demo:
    example_id_text = gr.Text(label="Test Example")

    input_image_base64 = gr.Text(label="Image in Base64")
    input_image_label = gr.Text(label="Image Label")
    get_input_image_btn = gr.Button(value="Get Input Image")
    get_input_image_btn.click(
        get_image,
        inputs=[example_id_text],
        outputs=[input_image_base64, input_image_label],
        api_name="get-image",
        postprocess=False,
    )

    latents_json = gr.JSON(label="Latents", value={})
    activations_json = gr.JSON(label="Activations", value={})

    input_image = gr.Image(
        label="Input Image",
        sources=["upload", "clipboard"],
        type="pil",
        interactive=True,
    )
    get_sae_activations_btn = gr.Button(value="Get SAE Activations")
    get_sae_activations_btn.click(
        get_sae_activations,
        inputs=[input_image, latents_json],
        outputs=[activations_json],
        api_name="get-sae-activations",
    )


if __name__ == "__main__":
    demo.launch()
