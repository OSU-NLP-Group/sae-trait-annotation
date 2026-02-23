"""
Test that the cached activations are actually correct.
These tests are quite slow
"""

import tempfile

import pytest
import torch

from . import activations, config


@pytest.mark.slow
def test_dataloader_batches():
    cfg = config.Activations(
        vit_ckpt="ViT-B-32/openai",
        d_vit=768,
        layers=[-2, -1],
        n_patches_per_img=49,
        vit_batch_size=8,
    )
    vit = activations.make_vit(cfg)
    dataloader = activations.get_dataloader(cfg, vit.make_img_transform())
    batch = next(iter(dataloader))

    assert isinstance(batch, dict)
    assert "image" in batch
    assert "index" in batch

    torch.testing.assert_close(batch["index"], torch.arange(8))
    assert batch["image"].shape == (8, 3, 224, 224)


@pytest.mark.slow
def test_shard_writer_and_dataset_e2e():
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = config.Activations(
            model_org="timm",
            vit_ckpt="hf_hub:timm/test_vit3.r160_in1k",
            d_vit=96,
            n_patches_per_img=100,
            layers=[-2, -1],
            vit_batch_size=8,
            n_workers=8,
            dump_to=tmpdir,
        )
        vit = activations.make_vit(cfg)
        dataloader = activations.get_dataloader(cfg, vit.make_img_transform())
        writer = activations.ShardWriter(cfg)
        dataset = activations.Dataset(
            config.DataLoad(
                shard_root=activations.get_acts_dir(cfg),
                patches="cls",
                layer=-1,
                scale_mean=False,
                scale_norm=False,
            )
        )

        i = 0
        for b, batch in zip(range(100), dataloader):
            # Don't care about the forward pass.
            _, cache = vit(batch["image"])
            writer[i : i + len(cache)] = cache
            i += len(cache)
            assert cache.shape == (cfg.vit_batch_size, len(cfg.layers), 101, 96)

            acts, _, _ = zip(*[dataset[i.item()] for i in batch["index"]])
            from_dataset = torch.stack(acts)
            torch.testing.assert_close(cache[:, -1, 0], from_dataset)
            print(f"Batch {b} matched.")
