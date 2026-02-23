# Reproduce

To reproduce our findings from our preprint, you will need to train a couple SAEs on various datasets, then save visual examples so you can browse them in the notebooks.

## Table of Contents

1. Save activations for ImageNet and iNat2021 for DINOv2, CLIP and BioCLIP.
2. Train SAEs on these activation datasets.
3. Pick the best SAE checkpoints for each combination.
4. Save visualizations for those best checkpoints.

## Save Activations

## Train SAEs

## Choose Best Checkpoints

## Save Visualizations

Get visuals for the iNat-trained SAEs (BioCLIP and CLIP):

```sh
uv run python -m saev visuals \
  --ckpt checkpoints/$CKPT/sae.pt \
  --dump-to /$NFS/$USER/saev-visuals/$CKPT/ \
  --log-freq-range -2.0 -1.0 \
  --log-value-range -0.75 2.0 \
  --data.shard-root /local/scratch/$USER/cache/saev/$SHARDS \
  images:image-folder-dataset \
  --images.root /$NFS/$USER/datasets/inat21/train_mini/
```

Look at these visuals in the interactive notebook.

```sh
uv run marimo edit
```

Then open [localhost:2718](https://localhost:2718) in your browser and open the `saev/interactive/features.py` file.
Choose one of the checkpoints in the dropdown and click through the different neurons to find patterns in the underlying ViT.
