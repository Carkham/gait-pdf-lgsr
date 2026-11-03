"""Export experiment evidence and figures; reads completed runs, never fits models."""

import csv
import gzip
import hashlib
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


out = Path("results")
out.mkdir(exist_ok=True)
figures = Path("figures")
figures.mkdir(exist_ok=True)
seeds = [0, 1, 2]
runs = [json.loads((Path("artifacts") / f"seed-{seed}/results-default.json").read_text()) for seed in seeds]
methods = ["Gray-NN", "Gabor-NN", "Gabor-PDF-NN", "Gabor-PDF-GSR-maxWIRE", "Gabor-PDF-LGSR-maxWIRE"]
summary = {}
all_records = []
solver_statistics = {}
for seed, run in zip(seeds, runs):
    folder = Path("artifacts") / f"seed-{seed}"
    for filename in ["results-default.json", "solver-default.json", "ubm-history.json", "provenance.json"]:
        payload = json.loads((folder / filename).read_text())
        with gzip.GzipFile(filename=str(out / f"seed-{seed}-{filename}.gz"), mode="wb", mtime=0) as handle:
            handle.write(json.dumps(payload, separators=(",", ":"), allow_nan=False).encode())
    all_records.extend({"seed": seed, **r} for r in run["records"])
    diagnostics = json.loads((folder / "solver-default.json").read_text())
    solver_statistics[str(seed)] = {}
    for method in ["Gabor-PDF-GSR", "Gabor-PDF-LGSR"]:
        rows = [d for d in diagnostics if d["method"] == method]
        solver_statistics[str(seed)][method] = {
            "iteration_limit_count": sum(d["stop_reason"] == "iteration_limit" for d in rows),
            "median_iterations": float(np.median([d["iterations"] for d in rows])),
            "median_kkt_residual": float(np.median([d["max_kkt_residual"] for d in rows])),
            "max_kkt_residual": max(d["max_kkt_residual"] for d in rows),
            "stalled_block_updates": sum(d["stalled_block_updates"] for d in rows),
            "empty_active_set_count": sum(d["active_groups"] == 0 for d in rows),
            "objective_increases": sum(sum(t["objective"] > prev["objective"] + 1e-10
                                           for prev, t in zip(d["trace"], d["trace"][1:])) for d in rows),
        }
for method in runs[0]["summary"]:
    summary[method] = {}
    for split in ["A", "B", "C", "overall"]:
        summary[method][split] = {}
        for rank in ["rank1", "rank5"]:
            values = [run["summary"][method][split][rank + "_percent"] for run in runs]
            summary[method][split][rank] = {"mean": float(np.mean(values)),
                                           "sample_std": float(np.std(values, ddof=1)), "seeds": values}
(out / "summary.json").write_text(json.dumps({"scope": runs[0]["scope"], "seeds": seeds,
                                               "summary": summary, "solver": solver_statistics}, indent=2))
with (out / "predictions.csv").open("w") as f:
    fields = ["seed", "split", "method", "subject", "rank", "rank1", "rank5", "active_groups"]
    writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(all_records)
with (out / "metrics.csv").open("w") as f:
    writer = csv.writer(f)
    writer.writerow(["method", "split", "rank", "mean_percent", "sample_std_percent", "seed0", "seed1", "seed2"])
    for method, splits in summary.items():
        for split, ranks in splits.items():
            for rank, data in ranks.items():
                writer.writerow([method, split, rank, data["mean"], data["sample_std"], *data["seeds"]])
shutil.copyfile("data/usf-v1.7-mpca/manifest.json", out / "data-manifest.json")

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 13, "axes.linewidth": 1.8,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "legend.frameon": False, "svg.fonttype": "none"})
colors = ["#CFCECE", "#E9A6A1", "#8BCF8B", "#42949E", "#0F4D92"]
labels = ["Gray\nNN", "Gabor\nNN", "Gabor-PDF\nNN", "Gabor-PDF\nGSR", "Gabor-PDF\nLGSR"]
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)
for ax, rank in zip(axes, ["rank1", "rank5"]):
    values = [summary[m]["overall"][rank]["mean"] for m in methods]
    errors = [summary[m]["overall"][rank]["sample_std"] for m in methods]
    ax.bar(np.arange(len(methods)), values, yerr=errors, capsize=4, color=colors,
           edgecolor="black", linewidth=1.0)
    ax.set_xticks(np.arange(len(methods)), labels)
    ax.set_ylim(0, 105)
    ax.set_title("Rank-1" if rank == "rank1" else "Rank-5")
    for i, value in enumerate(values):
        ax.text(i, value + errors[i] + 2, f"{value:.2f}", ha="center", fontsize=11)
axes[0].set_ylabel("Recognition rate (%)")
fig.suptitle("USF v1.7 A–C proxy-video validation · mean ± seed SD (n=3)", fontsize=15)
fig.tight_layout(pad=1.5)
fig.savefig(figures / "recognition.png", dpi=300)
fig.savefig(figures / "recognition.svg")
plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
for seed, color in zip(seeds, ["#0F4D92", "#42949E", "#B64342"]):
    history = json.loads((Path("artifacts") / f"seed-{seed}/ubm-history.json").read_text())
    axes[0].plot([d["iteration"] + 1 for d in history["em"]],
                 [d["mean_log_likelihood"] for d in history["em"]], color=color, label=f"seed {seed}", lw=2)
diagnostics = json.loads(Path("artifacts/seed-0/solver-default.json").read_text())
for split, color in zip(["A", "B", "C"], ["#0F4D92", "#42949E", "#B64342"]):
    row = next(d for d in diagnostics if d["split"] == split and d["method"] == "Gabor-PDF-LGSR")
    axes[1].plot([t["iteration"] for t in row["trace"]],
                 [t["objective"] / row["initial_objective"] for t in row["trace"]],
                 color=color, label=f"{split}, identity {row['subject']}", lw=2)
axes[0].set(xlabel="EM iteration", ylabel="Mean log likelihood", title="Gallery-only UBM")
axes[1].set(xlabel="Algorithm 1 sweep", ylabel="Objective / initial objective", title="LGSR traces (seed 0)")
for ax in axes:
    ax.legend(fontsize=11)
fig.tight_layout(pad=1.5)
fig.savefig(figures / "optimization.png", dpi=300)
fig.savefig(figures / "optimization.svg")
plt.close(fig)

paths = sorted([*out.rglob("*"), *figures.rglob("*")])
(out / "checksums.json").write_text(json.dumps({str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                                               for p in paths if p.is_file() and p.name != "checksums.json"}, indent=2))
print(json.dumps({method: summary[method]["overall"] for method in methods}, indent=2))
