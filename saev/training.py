"""
Trains many SAEs in parallel to amortize the cost of loading a single batch of data over many SAE training runs.
"""

import dataclasses
import json
import logging
import os.path

import beartype
import einops
import numpy as np
import torch
from jaxtyping import Float
from torch import Tensor

import wandb

from . import activations, config, helpers, nn

log_format = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=log_format)
logger = logging.getLogger("train")


@torch.no_grad()
def init_b_dec_batched(saes: torch.nn.ModuleList, dataset: activations.Dataset):
    n_samples = max(sae.cfg.n_reinit_samples for sae in saes)
    if not n_samples:
        return
    # Pick random samples using first SAE's seed.
    perm = np.random.default_rng(seed=saes[0].cfg.seed).permutation(len(dataset))
    perm = perm[:n_samples]
    examples, _, _ = zip(*[
        dataset[p.item()]
        for p in helpers.progress(perm, every=25_000, desc="examples to re-init b_dec")
    ])
    vit_acts = torch.stack(examples)
    for sae in saes:
        sae.init_b_dec(vit_acts[: sae.cfg.n_reinit_samples])


@beartype.beartype
def make_saes(
    cfgs: list[config.SparseAutoencoder],
) -> tuple[torch.nn.ModuleList, list[dict[str, object]]]:
    param_groups = []
    saes = []
    for cfg in cfgs:
        sae = nn.SparseAutoencoder(cfg)
        saes.append(sae)
        # Use an empty LR because our first step is warmup.
        param_groups.append({"params": sae.parameters(), "lr": 0.0})

    return torch.nn.ModuleList(saes), param_groups


##################
# Parallel Wandb #
##################


MetricQueue = list[tuple[int, dict[str, object]]]


class ParallelWandbRun:
    """
    Inspired by https://community.wandb.ai/t/is-it-possible-to-log-to-multiple-runs-simultaneously/4387/3.
    """

    def __init__(
        self, project: str, cfgs: list[config.Train], mode: str, tags: list[str]
    ):
        cfg, *cfgs = cfgs
        self.project = project
        self.cfgs = cfgs
        self.mode = mode
        self.tags = tags

        self.live_run = wandb.init(project=project, config=cfg, mode=mode, tags=tags)

        self.metric_queues: list[MetricQueue] = [[] for _ in self.cfgs]

    def log(self, metrics: list[dict[str, object]], *, step: int):
        metric, *metrics = metrics
        self.live_run.log(metric, step=step)
        for queue, metric in zip(self.metric_queues, metrics):
            queue.append((step, metric))

    def finish(self) -> list[str]:
        ids = [self.live_run.id]
        # Log the rest of the runs.
        self.live_run.finish()

        for queue, cfg in zip(self.metric_queues, self.cfgs):
            run = wandb.init(
                project=self.project,
                config=cfg,
                mode=self.mode,
                tags=self.tags + ["queued"],
            )
            for step, metric in queue:
                run.log(metric, step=step)
            ids.append(run.id)
            run.finish()

        return ids


@beartype.beartype
def main(cfgs: list[config.Train]) -> list[str]:
    saes, run, steps = train(cfgs)
    # Cheap(-ish) evaluation
    eval_metrics = evaluate(cfgs, saes)
    metrics = [metric.for_wandb() for metric in eval_metrics]
    run.log(metrics, step=steps)
    ids = run.finish()

    for cfg, id, metric, sae in zip(cfgs, ids, eval_metrics, saes):
        logger.info(
            "Checkpoint %s has %d dense features (%.1f)",
            id,
            metric.n_dense,
            metric.n_dense / sae.cfg.d_sae * 100,
        )
        logger.info(
            "Checkpoint %s has %d dead features (%.1f%%)",
            id,
            metric.n_dead,
            metric.n_dead / sae.cfg.d_sae * 100,
        )
        logger.info(
            "Checkpoint %s has %d *almost* dead (<1e-7) features (%.1f)",
            id,
            metric.n_almost_dead,
            metric.n_almost_dead / sae.cfg.d_sae * 100,
        )

        ckpt_fpath = os.path.join(cfg.ckpt_path, id, "sae.pt")
        nn.dump(ckpt_fpath, sae)
        logger.info("Dumped checkpoint to '%s'.", ckpt_fpath)
        cfg_fpath = os.path.join(cfg.ckpt_path, id, "config.json")
        with open(cfg_fpath, "w") as fd:
            json.dump(dataclasses.asdict(cfg), fd, indent=4)

    return ids


@beartype.beartype
def train(
    cfgs: list[config.Train],
) -> tuple[torch.nn.ModuleList, ParallelWandbRun, int]:
    """
    Explicitly declare the optimizer, schedulers, dataloader, etc outside of `main` so that all the variables are dropped from scope and can be garbage collected.
    """
    if len(split_cfgs(cfgs)) != 1:
        raise ValueError("Configs are not parallelizeable: {cfgs}.")

    err_msg = "ghost grads are disabled in current codebase."
    assert all(not c.sae.ghost_grads for c in cfgs), err_msg

    logger.info("Parallelizing %d runs.", len(cfgs))

    cfg = cfgs[0]
    if torch.cuda.is_available():
        # This enables tf32 on Ampere GPUs which is only 8% slower than
        # float16 and almost as accurate as float32
        # This was a default in pytorch until 1.12
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    dataset = activations.Dataset(cfg.data)
    saes, param_groups = make_saes([c.sae for c in cfgs])

    mode = "online" if cfg.track else "disabled"
    tags = [cfg.tag] if cfg.tag else []
    run = ParallelWandbRun(cfg.wandb_project, cfgs, mode, tags)

    optimizer = torch.optim.Adam(param_groups, fused=True)
    lr_schedulers = [Warmup(0.0, c.lr, c.n_lr_warmup) for c in cfgs]
    sparsity_schedulers = [
        Warmup(0.0, c.sae.sparsity_coeff, c.n_sparsity_warmup) for c in cfgs
    ]

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=cfg.sae_batch_size, num_workers=cfg.n_workers, shuffle=True
    )

    dataloader = BatchLimiter(dataloader, cfg.n_patches)

    saes.train()
    saes = saes.to(cfg.device)

    global_step, n_patches_seen = 0, 0

    for batch in helpers.progress(dataloader, every=cfg.log_every):
        acts_BD = batch["act"].to(cfg.device, non_blocking=True)
        for sae in saes:
            sae.normalize_w_dec()
        # Forward passes
        _, _, losses = zip(*(sae(acts_BD) for sae in saes))

        n_patches_seen += len(acts_BD)

        with torch.no_grad():
            if (global_step + 1) % cfg.log_every == 0:
                metrics = [
                    {
                        "losses/mse": loss.mse.item(),
                        "losses/l1": loss.l1.item(),
                        "losses/sparsity": loss.sparsity.item(),
                        "losses/ghost_grad": loss.ghost_grad.item(),
                        "losses/loss": loss.loss.item(),
                        "metrics/l0": loss.l0.item(),
                        "metrics/l1": loss.l1.item(),
                        "progress/n_patches_seen": n_patches_seen,
                        "progress/learning_rate": group["lr"],
                        "progress/sparsity_coeff": sae.sparsity_coeff,
                    }
                    for loss, sae, group in zip(losses, saes, optimizer.param_groups)
                ]
                run.log(metrics, step=global_step)

                logger.info(
                    "loss: %.5f, mse loss: %.5f, sparsity loss: %.5f, l0: %.5f, l1: %.5f",
                    losses[0].loss.item(),
                    losses[0].mse.item(),
                    losses[0].sparsity.item(),
                    losses[0].l0.item(),
                    losses[0].l1.item(),
                )

        for loss in losses:
            loss.loss.backward()

        for sae in saes:
            sae.remove_parallel_grads()

        optimizer.step()

        # Update LR and sparsity coefficients.
        for param_group, scheduler in zip(optimizer.param_groups, lr_schedulers):
            param_group["lr"] = scheduler.step()

        for sae, scheduler in zip(saes, sparsity_schedulers):
            sae.sparsity_coeff = scheduler.step()

        # Don't need these anymore.
        optimizer.zero_grad()

        global_step += 1

    return saes, run, global_step


@beartype.beartype
@dataclasses.dataclass(frozen=True)
class EvalMetrics:
    """Results of evaluating a trained SAE on a datset."""

    l0: float
    """Mean L0 across all examples."""
    l1: float
    """Mean L1 across all examples."""
    mse: float
    """Mean MSE across all examples."""
    n_dead: int
    """Number of neurons that never fired on any example."""
    n_almost_dead: int
    """Number of neurons that fired on fewer than `almost_dead_threshold` of examples."""
    n_dense: int
    """Number of neurons that fired on more than `dense_threshold` of examples."""

    freqs: Float[Tensor, " d_sae"]
    """How often each feature fired."""
    mean_values: Float[Tensor, " d_sae"]
    """The mean value for each feature when it did fire."""

    almost_dead_threshold: float
    """Threshold for an "almost dead" neuron."""
    dense_threshold: float
    """Threshold for a dense neuron."""

    def for_wandb(self) -> dict[str, int | float]:
        dct = dataclasses.asdict(self)
        # Store arrays as tables.
        dct["freqs"] = wandb.Table(columns=["freq"], data=dct["freqs"][:, None].numpy())
        dct["mean_values"] = wandb.Table(
            columns=["mean_value"], data=dct["mean_values"][:, None].numpy()
        )
        return {f"eval/{key}": value for key, value in dct.items()}


@beartype.beartype
@torch.no_grad()
def evaluate(cfgs: list[config.Train], saes: torch.nn.ModuleList) -> list[EvalMetrics]:
    """
    Evaluates SAE quality by counting the number of dead features and the number of dense features.
    Also makes histogram plots to help human qualitative comparison.

    .. todo:: Develop automatic methods to use histogram and feature frequencies to evaluate quality with a single number.
    """

    torch.cuda.empty_cache()

    if len(split_cfgs(cfgs)) != 1:
        raise ValueError("Configs are not parallelizeable: {cfgs}.")

    saes.eval()

    cfg = cfgs[0]

    almost_dead_lim = 1e-7
    dense_lim = 1e-2

    dataset = activations.Dataset(cfg.data)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=cfg.sae_batch_size, num_workers=cfg.n_workers, shuffle=False
    )

    n_fired = torch.zeros((len(cfgs), saes[0].cfg.d_sae))
    values = torch.zeros((len(cfgs), saes[0].cfg.d_sae))
    total_l0 = torch.zeros(len(cfgs))
    total_l1 = torch.zeros(len(cfgs))
    total_mse = torch.zeros(len(cfgs))

    for batch in helpers.progress(dataloader, desc="eval", every=cfg.log_every):
        acts_BD = batch["act"].to(cfg.device, non_blocking=True)
        for i, sae in enumerate(saes):
            _, f_x_BS, loss = sae(acts_BD)
            n_fired[i] += einops.reduce(f_x_BS > 0, "batch d_sae -> d_sae", "sum").cpu()
            values[i] += einops.reduce(f_x_BS, "batch d_sae -> d_sae", "sum").cpu()
            total_l0[i] += loss.l0.cpu()
            total_l1[i] += loss.l1.cpu()
            total_mse[i] += loss.mse.cpu()

    mean_values = values / n_fired
    freqs = n_fired / len(dataset)

    l0 = (total_l0 / len(dataloader)).tolist()
    l1 = (total_l1 / len(dataloader)).tolist()
    mse = (total_mse / len(dataloader)).tolist()

    n_dead = einops.reduce(freqs == 0, "n_saes d_sae -> n_saes", "sum").tolist()
    n_almost_dead = einops.reduce(
        freqs < almost_dead_lim, "n_saes d_sae -> n_saes", "sum"
    ).tolist()
    n_dense = einops.reduce(freqs > dense_lim, "n_saes d_sae -> n_saes", "sum").tolist()

    metrics = []
    for row in zip(l0, l1, mse, n_dead, n_almost_dead, n_dense, freqs, mean_values):
        metrics.append(EvalMetrics(*row, almost_dead_lim, dense_lim))

    return metrics


class BatchLimiter:
    """
    Limits the number of batches to only return `n_samples` total samples.
    """

    def __init__(self, dataloader: torch.utils.data.DataLoader, n_samples: int):
        self.dataloader = dataloader
        self.n_samples = n_samples
        self.batch_size = dataloader.batch_size

    def __len__(self) -> int:
        return self.n_samples // self.batch_size

    def __iter__(self):
        self.n_seen = 0
        while True:
            for batch in self.dataloader:
                yield batch

                # Sometimes we underestimate because the final batch in the dataloader might not be a full batch.
                self.n_seen += self.batch_size
                if self.n_seen > self.n_samples:
                    return

            # We try to mitigate the above issue by ignoring the last batch if we don't have drop_last.
            if not self.dataloader.drop_last:
                self.n_seen -= self.batch_size


#####################
# Parallel Training #
#####################


CANNOT_PARALLELIZE = set([
    "data",
    "n_workers",
    "n_patches",
    "sae_batch_size",
    "track",
    "wandb_project",
    "tag",
    "log_every",
    "ckpt_path",
    "device",
    "slurm",
    "slurm_acct",
    "log_to",
    "sae.exp_factor",
    "sae.d_vit",
])


@beartype.beartype
def split_cfgs(cfgs: list[config.Train]) -> list[list[config.Train]]:
    """
    Splits configs into groups that can be parallelized.

    Arguments:
        A list of configs from a sweep file.

    Returns:
        A list of lists, where the configs in each sublist do not differ in any keys that are in `CANNOT_PARALLELIZE`. This means that each sublist is a valid "parallel" set of configs for `train`.
    """
    # Group configs by their values for CANNOT_PARALLELIZE keys
    groups = {}
    for cfg in cfgs:
        dct = dataclasses.asdict(cfg)

        # Create a key tuple from the values of CANNOT_PARALLELIZE keys
        key_values = []
        for key in sorted(CANNOT_PARALLELIZE):
            key_values.append((key, make_hashable(helpers.get(dct, key))))
        group_key = tuple(key_values)

        if group_key not in groups:
            groups[group_key] = []
        groups[group_key].append(cfg)

    # Convert groups dict to list of lists
    return list(groups.values())


def make_hashable(obj):
    return json.dumps(obj, sort_keys=True)


##############
# Schedulers #
##############


@beartype.beartype
class Scheduler:
    def step(self) -> float:
        err_msg = f"{self.__class__.__name__} must implement step()."
        raise NotImplementedError(err_msg)

    def __repr__(self) -> str:
        err_msg = f"{self.__class__.__name__} must implement __repr__()."
        raise NotImplementedError(err_msg)


@beartype.beartype
class Warmup(Scheduler):
    """
    Linearly increases from `init` to `final` over `n_warmup_steps` steps.
    """

    def __init__(self, init: float, final: float, n_steps: int):
        self.final = final
        self.init = init
        self.n_steps = n_steps
        self._step = 0

    def step(self) -> float:
        self._step += 1
        if self._step < self.n_steps:
            return self.init + (self.final - self.init) * (self._step / self.n_steps)

        return self.final

    def __repr__(self) -> str:
        return f"Warmup(init={self.init}, final={self.final}, n_steps={self.n_steps})"


def _plot_example_schedules():
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots()

    n_steps = 1000
    xs = np.arange(n_steps)

    schedule = Warmup(0.1, 0.9, 100)
    ys = [schedule.step() for _ in xs]

    ax.plot(xs, ys, label=str(schedule))

    fig.tight_layout()
    fig.savefig("schedules.png")


if __name__ == "__main__":
    _plot_example_schedules()
