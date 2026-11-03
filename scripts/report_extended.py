"""Figures from saved predictions, measured runtimes, and one fixed GEI example."""

import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import torch

from gait.features import gabor_descriptors
from gait.run import load_geis


OUT = Path("results/presentation")
FIG = Path("figures/extended")
SEEDS = [0, 1, 2]
BLUE, GREEN, RED = "#0F4D92", "#42949E", "#B64342"


def read_gzip(path):
    with gzip.open(path, "rt") as handle:
        return json.load(handle)


def save(fig, name):
    fig.savefig(FIG / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)


def write_json(name, payload):
    (OUT / name).write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")


def parameter_grid():
    source = Path("results/seed-0-results-grid.json.gz")
    data = read_gzip(source)
    lambdas, sigmas = [1 / 16, 1 / 8, 1 / 4], [1 / 40, 1 / 8, 1 / 4, 1 / 2]
    rows = []
    fig, axes = plt.subplots(2, 4, figsize=(15, 7), layout="constrained")
    for row, rank in enumerate(["rank1", "rank5"]):
        for col, split in enumerate(["A", "B", "C", "overall"]):
            matrix = np.empty((3, 4))
            for i, lam in enumerate(lambdas):
                for j, sigma in enumerate(sigmas):
                    method = f"Gabor-PDF-LGSR-maxWIRE/lambda={lam:g},sigma={sigma:g}"
                    value = data["summary"][method][split][rank + "_percent"]
                    matrix[i, j] = value
                    rows.append(dict(split=split, rank=rank, regularization=lam, sigma=sigma, percent=value))
            ax = axes[row, col]
            im = ax.imshow(matrix, cmap="Blues", vmin=50, vmax=100, aspect="auto")
            for (i, j), value in np.ndenumerate(matrix):
                ax.text(j, i, f"{value:.1f}", ha="center", va="center",
                        color="white" if value >= 79 else "#272727", fontsize=12)
            ax.set_xticks(range(4), ["1/40", "1/8", "1/4", "1/2"])
            ax.set_yticks(range(3), ["1/16", "1/8", "1/4"])
            ax.set(xlabel=r"$\sigma$", ylabel=r"$\lambda$",
                   title=f"{split.upper() if split == 'overall' else split} · Rank-{1 if row == 0 else 5}")
            # Mark the fixed configuration used in the three-seed comparison.
            ax.add_patch(plt.Rectangle((0.5, 0.5), 1, 1, fill=False, lw=2.5, edgecolor=RED))
    fig.colorbar(im, ax=axes, label="Recognition rate (%)", shrink=0.8, pad=0.02)
    fig.suptitle("LGSR parameter grid · seed 0 · A–C proxy-video validation\n"
                 "Red outline: fixed configuration (λ = σ = 1/8)", fontsize=16)
    save(fig, "parameter-grid")
    write_json("parameter-grid.json", {"source": str(source), "seed": 0,
                                       "criterion": "maxWIRE", "metrics": rows})


def runtimes():
    rows = []
    whole_runs = []
    for seed in SEEDS:
        source = f"results/seed-{seed}-solver-default.json.gz"
        rows.extend({"seed": seed, "source": source, **{key: row[key] for key in
                     ["method", "split", "subject", "seconds", "iterations"]}}
                    for row in read_gzip(source))
        run = read_gzip(f"results/seed-{seed}-results-default.json.gz")
        provenance = read_gzip(f"results/seed-{seed}-provenance.json.gz")
        whole_runs.append({"seed": seed, "feature_pipeline_seconds": provenance["seconds"],
                           "evaluation_seconds": run["seconds"], "gpu": provenance["gpu"]})
    groups = []
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5), layout="constrained")
    for offset, method, color in [(-0.18, "Gabor-PDF-GSR", GREEN), (0.18, "Gabor-PDF-LGSR", BLUE)]:
        values = []
        for split in ["A", "B", "C"]:
            seconds = np.array([r["seconds"] for r in rows if r["method"] == method and r["split"] == split])
            values.append(1000 * seconds)
            groups.append({"method": method, "split": split, "groups": len(seconds),
                           "mean_seconds": float(seconds.mean()), "median_seconds": float(np.median(seconds)),
                           "q25_seconds": float(np.quantile(seconds, 0.25)),
                           "q75_seconds": float(np.quantile(seconds, 0.75)), "total_seconds": float(seconds.sum())})
        box = axes[0].boxplot(values, positions=np.arange(3) + offset, widths=0.28,
                              patch_artist=True, showfliers=False,
                              medianprops={"color": "black", "linewidth": 1.5})
        for patch in box["boxes"]:
            patch.set_facecolor(color)
            patch.set_alpha(0.75)
        axes[0].plot([], [], color=color, linewidth=7, label=method.removeprefix("Gabor-PDF-"))
    axes[0].set(xticks=range(3), xticklabels=["A", "B", "C"], ylabel="Solve + score per proxy video (ms)",
                title="Algorithm 1 and identity scoring", yscale="log")
    axes[0].legend()
    x = np.arange(3)
    for offset, key, label, color in [(-0.18, "feature_pipeline_seconds", "Features + UBM + NN", GREEN),
                                    (0.18, "evaluation_seconds", "GSR + LGSR evaluation", BLUE)]:
        values = [run[key] for run in whole_runs]
        axes[1].bar(x + offset, values, width=0.34, color=color, edgecolor="black", linewidth=0.8, label=label)
        for position, value in zip(x + offset, values):
            axes[1].text(position, value + 2, f"{value:.1f}", ha="center", fontsize=10)
    axes[1].set(xticks=x, xticklabels=["Seed 0", "Seed 1", "Seed 2"], ylabel="Wall time (s)", title="Recorded complete stages")
    axes[1].set_ylim(0, max(run["feature_pipeline_seconds"] for run in whole_runs) * 1.24)
    axes[1].legend(fontsize=10)
    fig.suptitle("Measured runtime · debug / RTX 4090 · fixed parameters", fontsize=16)
    save(fig, "runtime")
    with (OUT / "runtime-groups.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json("runtime-summary.json", {
        "group_timing": "Algorithm 1, class scoring, and both ranking records; EMD, Gram matrices, and distances are prepared before this timer.",
        "feature_timing": "MAT loading, Gabor extraction, gallery UBM fitting, MAP encoding, and three NN baselines across gallery and A–C.",
        "evaluation_timing": "A–C cross-products, distances, EMD, GSR/LGSR optimization and scoring; gallery Gram preparation precedes this timer.",
        "boxplot": "All proxy groups from seeds 0,1,2; boxes: quartiles, line: median, whiskers: 1.5 IQR; fliers hidden.",
        "groups": groups, "runs": whole_runs})


def method_example():
    bundle_path = Path("artifacts/seed-0/features.pt")
    bundle = torch.load(bundle_path, weights_only=False, map_location="cpu")
    cfg = bundle["config"]
    images, labels = load_geis(Path(cfg["data_dir"]), "A", cfg["gei_size"])
    subject = int(labels.min())
    index = int(np.flatnonzero(labels == subject)[0])
    patches = gabor_descriptors(images[index:index + 1], cfg)[0].cpu().numpy()
    gei = images[index].cpu().numpy()
    pdf = bundle["features"]["A"]["pdf"][index].numpy().reshape(cfg["components"], -1)
    result_source = Path("results/seed-0-results-default.json.gz")
    result = read_gzip(result_source)
    record = next(row for row in result["records"] if row["split"] == "A" and row["subject"] == subject
                  and row["method"] == "Gabor-PDF-LGSR-maxWIRE")
    identities = np.unique(bundle["features"]["gallery"]["labels"])
    top = record["ranking"][:5]
    wire = [-record["scores"][int(np.flatnonzero(identities == identity)[0])] for identity in top]
    fig = plt.figure(figsize=(18, 5.6), layout="constrained")
    grid = fig.add_gridspec(1, 4, width_ratios=[1.1, 3.4, 1.9, 2.3], wspace=0.10)
    ax = fig.add_subplot(grid[0])
    ax.imshow(gei, cmap="gray", vmin=0, vmax=1)
    ax.set_title("1  Mean GEI\n128 × 88", fontsize=13)
    ax.set_axis_off()
    ax.text(0.5, -0.05, f"A · identity {subject}\ntensor index {index} (zero-based)",
            transform=ax.transAxes, ha="center", va="top", fontsize=10)
    bank_frame = grid[1].subgridspec(2, 1, height_ratios=[0.12, 1])
    heading = fig.add_subplot(bank_frame[0])
    heading.set_axis_off()
    heading.text(0.5, 0.5, "2  Gabor magnitudes\n5 scales × 8 directions", ha="center", va="center", fontsize=13)
    bank = bank_frame[1].subgridspec(5, 8, wspace=0.04, hspace=0.05)
    maximum = patches[:, :40].max()
    for scale in range(5):
        for orientation in range(8):
            ax = fig.add_subplot(bank[scale, orientation])
            ax.imshow(patches[:, scale * 8 + orientation].reshape(64, 44), cmap="gray", vmin=0, vmax=maximum)
            ax.set_xticks([])
            ax.set_yticks([])
            if scale == 0:
                ax.set_title(f"{orientation}π/8", fontsize=9)
            if orientation == 0:
                ax.set_ylabel(f"v={scale}", fontsize=10)
    ax = fig.add_subplot(grid[2])
    limit = np.abs(pdf).max()
    im = ax.imshow(pdf, cmap="RdBu_r", norm=TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit), aspect="auto")
    ax.set(title="3  PDF supervector\n300 × 42 = 12,600", xlabel="Gabor channels + XY", ylabel="Gaussian component")
    ax.set_xticks([0, 19, 40.5], ["1", "20", "XY"])
    fig.colorbar(im, ax=ax, orientation="horizontal", shrink=0.85, pad=0.08, label="Normalized coordinate")
    ax = fig.add_subplot(grid[3])
    ax.barh(range(len(top)), wire, color=[BLUE if identity == subject else "#CFCECE" for identity in top],
            edgecolor="black", linewidth=0.8)
    ax.set_yticks(range(len(top)), [f"#{rank}  ID {identity}" for rank, identity in enumerate(top, 1)])
    ax.invert_yaxis()
    ax.set(xlabel="WIRE score (higher ranks first)", title=f"4  LGSR identity ranking\nAll {int((labels == subject).sum())} GEIs of identity {subject}")
    ax.set_xlim(0, max(wire) * 1.16)
    for row, score in enumerate(wire):
        ax.text(score + max(wire) * 0.02, row, f"{score:.3f}", va="center", fontsize=10)
    fig.suptitle("Fixed example: smallest identity in A, first tensor · seed 0", fontsize=16)
    save(fig, "method-example")
    write_json("method-example.json", {
        "selection": "Smallest identity label in A, first tensor in original MAT order; selection precedes checking the prediction.",
        "split": "A", "subject": subject, "tensor_index_zero_based": index,
        "group_tensor_indices_zero_based": np.flatnonzero(labels == subject).tolist(),
        "gei": "Mean of the tensor's 10 frames, resized from 32×22 to 128×88 by bilinear interpolation.",
        "display": "GEI, Gabor bank and PDF show one tensor; ranking uses all tensors in its identity proxy group.",
        "gabor_display": "Shared grayscale limits from zero to the maximum magnitude across all 40 response maps.",
        "pdf_shape": list(pdf.shape), "prediction_source": str(result_source),
        "feature_source": str(bundle_path), "source_sha256": bundle["provenance"]["source_sha256"],
        "prediction_source_sha256": hashlib.sha256(result_source.read_bytes()).hexdigest(),
        "ranking": record, "gallery_identity_order": identities.tolist(),
        "top5_wire": [{"identity": identity, "wire": score} for identity, score in zip(top, wire)]})


def baseline_comparison():
    original = json.loads(Path("results/summary.json").read_text())["summary"]
    additional = json.loads(Path("results/baselines/summary.json").read_text())["summary"]
    summary = {**original, **additional}
    methods = ["Gray-NN", "Gabor-NN", "DCT-NN", "Gabor-PDF-NN", "DCT-PDF-NN",
               "Gabor-PDF-SR-minRE", "Gabor-PDF-LLC-minRE", "Gabor-PDF-GSR-maxWIRE", "Gabor-PDF-LGSR-maxWIRE"]
    labels = ["Gray\nNN", "Gabor\nNN", "DCT\nNN", "Gabor-PDF\nNN", "DCT-PDF\nNN",
              "PDF-SR\nminRE", "PDF-LLC\nminRE", "PDF-GSR\nmaxWIRE", "PDF-LGSR\nmaxWIRE"]
    colors = ["#CFCECE", "#E9A6A1", "#B64342", "#AADCA9", "#8BCF8B", "#767676", "#42949E", "#3775BA", BLUE]
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), layout="constrained", sharex=True)
    for ax, rank in zip(axes, ["rank1", "rank5"]):
        values = [summary[method]["overall"][rank]["mean"] for method in methods]
        errors = [summary[method]["overall"][rank]["sample_std"] for method in methods]
        ax.bar(range(len(methods)), values, yerr=errors, capsize=3, color=colors, edgecolor="black", linewidth=0.8)
        ax.set(ylim=(0, 108), ylabel=f"Rank-{1 if rank == 'rank1' else 5} (%)")
        for position, (value, error) in enumerate(zip(values, errors)):
            ax.text(position, value + error + 2, f"{value:.2f}", ha="center", fontsize=11)
    axes[1].set_xticks(range(len(methods)), labels)
    fig.suptitle("A–C baseline comparison · mean ± seed SD (n=3)\nSR / LLC: minRE · GSR / LGSR: maxWIRE", fontsize=16)
    save(fig, "baseline-comparison")
    write_json("baseline-comparison.json", {
        "sources": ["results/summary.json", "results/baselines/summary.json"],
        "seeds": SEEDS, "criteria": {"SR": "minRE", "LLC": "minRE", "GSR": "maxWIRE", "LGSR": "maxWIRE"},
        "summary": {method: summary[method] for method in methods}})


def ablation_comparison():
    data = json.loads(Path("results/ablations/summary.json").read_text())
    summary = {name: variant["summary"] for name, variant in data["variants"].items()}
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    banks = [("5x8", BLUE, "o"), ("5x4", GREEN, "s"), ("3x8", RED, "D"), ("3x4", "#767676", "^")]
    components = [100, 300, 500]
    for row, rank in enumerate(["rank1", "rank5"]):
        ax = axes[row, 0]
        for bank, color, marker in banks:
            stats = [summary[f"gabor-{bank}-xy-k{k}"]["overall"][rank] for k in components]
            ax.errorbar(components, [s["mean"] for s in stats], yerr=[s["sample_std"] for s in stats],
                        color=color, marker=marker, capsize=3, lw=2, label=f"{bank.replace('x', ' × ')} + XY")
        ax.set(xticks=components, xlabel="GMM components", ylabel=f"Rank-{1 if row == 0 else 5} (%)")
        ax = axes[row, 1]
        for tag, label, color, marker in [("xy", "With XY", BLUE, "o"), ("no-xy", "Without XY", RED, "s")]:
            stats = [summary[f"gabor-5x8-{tag}-k{k}"]["overall"][rank] for k in components]
            ax.errorbar(components, [s["mean"] for s in stats], yerr=[s["sample_std"] for s in stats],
                        color=color, marker=marker, capsize=3, lw=2, label=label)
        ax.set(xticks=components, xlabel="GMM components", ylabel=f"Rank-{1 if row == 0 else 5} (%)")
    axes[0, 0].set_title("Gabor scales × directions")
    axes[0, 1].set_title("Spatial coordinates · 5 × 8 Gabor bank")
    axes[0, 0].legend(fontsize=10, ncol=2)
    axes[0, 1].legend(fontsize=10)
    fig.suptitle("Gabor-PDF-NN feature ablations · A–C pooled\nMean ± seed SD (n=3)", fontsize=16)
    save(fig, "feature-ablations")
    write_json("feature-ablations.json", {"source": "results/ablations/summary.json", "seeds": SEEDS,
                                         "method": "Gabor-PDF-NN", "summary": summary})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--part", choices=["existing", "baselines", "ablations"], default="existing")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12, "axes.linewidth": 1.5,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "legend.frameon": False, "svg.fonttype": "none"})
    if args.part == "existing":
        parameter_grid()
        runtimes()
        method_example()
    elif args.part == "baselines":
        baseline_comparison()
    else:
        ablation_comparison()


if __name__ == "__main__":
    main()
