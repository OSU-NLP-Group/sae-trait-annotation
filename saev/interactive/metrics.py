import marimo

__generated_with = "0.9.32"
app = marimo.App(
    width="medium",
    css_file="/home/stevens.994/.config/marimo/custom.css",
)


@app.cell
def __():
    import json
    import os

    import altair as alt
    import beartype
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import polars as pl
    from jaxtyping import Float, jaxtyped

    import wandb

    return Float, alt, beartype, jaxtyped, json, mo, np, os, pl, plt, wandb


@app.cell
def __(mo):
    tag_input = mo.ui.text(value="classification-v1.0", label="Sweep Tag:")
    return (tag_input,)


@app.cell
def __(mo, tag_input):
    mo.vstack([
        mo.md(
            "Look at [WandB](https://wandb.ai/samuelstevens/saev/table) to pick your tag."
        ),
        tag_input,
    ])
    return


@app.cell
def __(alt, df, mo, pl):
    chart = mo.ui.altair_chart(
        alt.Chart(
            df.filter(
                pl.col("config/data/shard_root")
                == "/local/scratch/stevens.994/cache/saev/724b1b7be995ef7212d64640fec2885737a706a33b8e5a18f7f323223bd43af1/"
            ).select(
                "summary/eval/l0",
                "summary/losses/mse",
                "id",
                "config/sae/sparsity_coeff",
                "config/lr",
                "config/sae/d_sae",
                "model_key",
            )
        )
        .mark_point()
        .encode(
            x=alt.X("summary/eval/l0"),
            y=alt.Y("summary/losses/mse"),
            tooltip=["id", "config/lr"],
            color="config/lr:Q",
            # shape="config/sae/sparsity_coeff:N",
            shape="config/sae/d_sae:N",
            # shape="model_key",
        )
    )
    chart
    return (chart,)


@app.cell
def __(chart, df, mo, np, plot_dist, plt):
    mo.stop(
        len(chart.value) < 2,
        mo.md(
            "Select two or more points. Exactly one point is not supported because of a [Polars bug](https://github.com/pola-rs/polars/issues/19855)."
        ),
    )

    sub_df = (
        df.select(
            "id",
            "summary/eval/freqs",
            "summary/eval/mean_values",
            "summary/eval/l0",
        )
        .join(chart.value.select("id"), on="id", how="inner")
        .sort(by="summary/eval/l0")
        .head(4)
    )

    scatter_fig, scatter_axes = plt.subplots(
        ncols=len(sub_df), figsize=(12, 3), squeeze=False, sharey=True, sharex=True
    )

    hist_fig, hist_axes = plt.subplots(
        ncols=len(sub_df),
        nrows=2,
        figsize=(12, 6),
        squeeze=False,
        sharey=True,
        sharex=True,
    )

    # Always one row
    scatter_axes = scatter_axes.reshape(-1)
    hist_axes = hist_axes.T

    for (id, freqs, values, _), scatter_ax, (freq_hist_ax, values_hist_ax) in zip(
        sub_df.iter_rows(), scatter_axes, hist_axes
    ):
        plot_dist(
            freqs.astype(float),
            (-1.0, 1.0),
            values.astype(float),
            (-2.0, 2.0),
            scatter_ax,
        )
        # ax.scatter(freqs, values, marker=".", alpha=0.03)
        # ax.set_yscale("log")
        # ax.set_xscale("log")
        scatter_ax.set_title(id)

        # Plot feature
        bins = np.linspace(-6, 1, 100)
        freq_hist_ax.hist(np.log10(freqs.astype(float)), bins=bins)
        freq_hist_ax.set_title(f"{id} Feat. Freq. Dist.")

        values_hist_ax.hist(np.log10(values.astype(float)), bins=bins)
        values_hist_ax.set_title(f"{id} Mean Val. Distribution")

    scatter_fig.tight_layout()
    hist_fig.tight_layout()
    return (
        bins,
        freq_hist_ax,
        freqs,
        hist_axes,
        hist_fig,
        id,
        scatter_ax,
        scatter_axes,
        scatter_fig,
        sub_df,
        values,
        values_hist_ax,
    )


@app.cell
def __(scatter_fig):
    scatter_fig
    return


@app.cell
def __(hist_fig):
    hist_fig
    return


@app.cell
def __(chart, df, pl):
    df.join(chart.value.select("id"), on="id", how="inner").sort(
        by="summary/eval/l0"
    ).select("id", pl.selectors.starts_with("config/"))
    return


@app.cell
def __(Float, beartype, jaxtyped, np):
    @jaxtyped(typechecker=beartype.beartype)
    def plot_dist(
        freqs: Float[np.ndarray, " d_sae"],
        freqs_log_range: tuple[float, float],
        values: Float[np.ndarray, " d_sae"],
        values_log_range: tuple[float, float],
        ax,
    ):
        log_sparsity = np.log10(freqs + 1e-9)
        log_values = np.log10(values + 1e-9)

        mask = np.ones(len(log_sparsity)).astype(bool)
        min_log_freq, max_log_freq = freqs_log_range
        mask[log_sparsity < min_log_freq] = False
        mask[log_sparsity > max_log_freq] = False
        min_log_value, max_log_value = values_log_range
        mask[log_values < min_log_value] = False
        mask[log_values > max_log_value] = False

        n_shown = mask.sum()
        ax.scatter(
            log_sparsity[mask],
            log_values[mask],
            marker=".",
            alpha=0.1,
            color="tab:blue",
            label=f"Shown ({n_shown})",
        )
        n_filtered = (~mask).sum()
        ax.scatter(
            log_sparsity[~mask],
            log_values[~mask],
            marker=".",
            alpha=0.1,
            color="tab:red",
            label=f"Filtered ({n_filtered})",
        )

        ax.axvline(min_log_freq, linewidth=0.5, color="tab:red")
        ax.axvline(max_log_freq, linewidth=0.5, color="tab:red")
        ax.axhline(min_log_value, linewidth=0.5, color="tab:red")
        ax.axhline(max_log_value, linewidth=0.5, color="tab:red")

        ax.set_xlabel("Feature Frequency (log10)")
        ax.set_ylabel("Mean Activation Value (log10)")

    return (plot_dist,)


@app.cell
def __(
    beartype,
    get_data_key,
    get_model_key,
    json,
    load_freqs,
    load_mean_values,
    mo,
    os,
    pl,
    tag_input,
    wandb,
):
    @beartype.beartype
    def make_df(tag: str):
        runs = wandb.Api().runs(path="samuelstevens/saev", filters={"config.tag": tag})

        rows = []
        for run in mo.status.progress_bar(
            runs,
            remove_on_exit=True,
            title="Loading",
            subtitle="Parsing runs from WandB",
        ):
            row = {}
            row["id"] = run.id

            row.update(**{
                f"summary/{key}": value for key, value in run.summary.items()
            })
            try:
                row["summary/eval/freqs"] = load_freqs(run)
            except ValueError:
                print(f"Run {run.id} did not log eval/freqs.")
                continue
            except RuntimeError:
                print(f"Wandb blew up on run {run.id}.")
                continue
            try:
                row["summary/eval/mean_values"] = load_mean_values(run)
            except ValueError:
                print(f"Run {run.id} did not log eval/mean_values.")
                continue
            except RuntimeError:
                print(f"Wandb blew up on run {run.id}.")
                continue

            # config
            row.update(**{
                f"config/data/{key}": value
                for key, value in run.config.pop("data").items()
            })
            row.update(**{
                f"config/sae/{key}": value
                for key, value in run.config.pop("sae").items()
            })

            row.update(**{f"config/{key}": value for key, value in run.config.items()})
            try:
                with open(
                    os.path.join(row["config/data/shard_root"], "metadata.json")
                ) as fd:
                    metadata = json.load(fd)
            except FileNotFoundError:
                print(f"Bad run {run.id}: missing metadata.json")
                continue

            row["model_key"] = get_model_key(metadata)

            data_key = get_data_key(metadata)
            if data_key is None:
                print(f"Bad run {run.id}: unknown data.")
                continue
            row["data_key"] = data_key

            row["config/d_vit"] = metadata["d_vit"]
            rows.append(row)

        if not rows:
            raise ValueError("No runs found.")

        df = pl.DataFrame(rows).with_columns(
            (pl.col("config/sae/d_vit") * pl.col("config/sae/exp_factor")).alias(
                "config/sae/d_sae"
            )
        )
        return df

    df = make_df(tag_input.value)
    return df, make_df


@app.cell
def __(beartype):
    @beartype.beartype
    def get_model_key(metadata: dict[str, object]) -> str | None:
        family, ckpt = metadata["vit_family"], metadata["vit_ckpt"]
        if family == "dinov2" and ckpt == "dinov2_vitb14_reg":
            return "DINOv2 ViT-B/14"
        if family == "clip" and ckpt == "ViT-B-16/openai":
            return "CLIP ViT-B/16"
        if family == "clip" and ckpt == "hf-hub:imageomics/bioclip":
            return "BioCLIP ViT-B/16"

        print(f"Unknown model: {(family, ckpt)}")
        return None

    @beartype.beartype
    def get_data_key(metadata: dict[str, object]) -> str | None:
        if (
            "train_mini" in metadata["data"]
            and "ImageFolderDataset" in metadata["data"]
        ):
            return "iNat21"

        if "train" in metadata["data"] and "Imagenet" in metadata["data"]:
            return "ImageNet-1K"

        print(f"Unknown data: {metadata['data']}")
        return None

    return get_data_key, get_model_key


@app.cell
def __(Float, json, np, os):
    def load_freqs(run) -> Float[np.ndarray, " d_sae"]:
        try:
            for artifact in run.logged_artifacts():
                if "evalfreqs" not in artifact.name:
                    continue

                dpath = artifact.download()
                fpath = os.path.join(dpath, "eval", "freqs.table.json")
                print(fpath)
                with open(fpath) as fd:
                    raw = json.load(fd)
                return np.array(raw["data"]).reshape(-1)
        except Exception as err:
            raise RuntimeError("Wandb sucks.") from err

        raise ValueError(f"freqs not found in run '{run.id}'")

    def load_mean_values(run) -> Float[np.ndarray, " d_sae"]:
        try:
            for artifact in run.logged_artifacts():
                if "evalmean_values" not in artifact.name:
                    continue

                dpath = artifact.download()
                fpath = os.path.join(dpath, "eval", "mean_values.table.json")
                print(fpath)
                with open(fpath) as fd:
                    raw = json.load(fd)
                return np.array(raw["data"]).reshape(-1)
        except Exception as err:
            raise RuntimeError("Wandb sucks.") from err

        raise ValueError(f"mean_values not found in run '{run.id}'")

    return load_freqs, load_mean_values


@app.cell
def __(df):
    df.drop(
        "config/log_every",
        "config/slurm_acct",
        "config/device",
        "config/n_workers",
        "config/wandb_project",
        "config/track",
        "config/slurm",
        "config/log_to",
        "config/ckpt_path",
        "config/sae/ghost_grads",
    )
    return


if __name__ == "__main__":
    app.run()
