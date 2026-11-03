"""Run the approved A–C proxy-video verification protocol on a Slurm GPU node."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
from scipy.io import loadmat
import swanlab
import torch
import torch.nn.functional as F

from .features import fit_ubm, gabor_descriptors, map_encode
from .solver import algorithm1, class_scores, ranking_record, uniform_emd


def load_geis(root, split, size):
    data = loadmat(root / ("Gal.mat" if split == "gallery" else f"Prb{split}.mat"))
    key, label_key = ("fea3Dg", "gndg") if split == "gallery" else ("fea3D", "gnd")
    # Each provided tensor is treated as one 10-frame observation, not a
    # recovered original gait cycle. No inferred video/cycle boundaries.
    images = torch.from_numpy(data[key].mean(axis=2).transpose(2, 0, 1).copy()).float().cuda()
    images = F.interpolate(images[:, None], size=size, mode="bilinear", align_corners=False)[:, 0]
    return images, data[label_key].ravel().astype(np.int64)


def summarize(records):
    output = {}
    for method in sorted({r["method"] for r in records}):
        selected = [r for r in records if r["method"] == method]
        output[method] = {}
        for split in [*sorted({r["split"] for r in selected}), "overall"]:
            rows = selected if split == "overall" else [r for r in selected if r["split"] == split]
            output[method][split] = {"videos": len(rows),
                                    "rank1_correct": sum(r["rank1"] for r in rows),
                                    "rank5_correct": sum(r["rank5"] for r in rows),
                                    "rank1_percent": 100 * np.mean([r["rank1"] for r in rows]),
                                    "rank5_percent": 100 * np.mean([r["rank5"] for r in rows])}
    return output


def nn_records(distance, gallery_labels, probe_labels, split, method):
    identities = np.unique(gallery_labels)
    records = []
    for target in np.unique(probe_labels):
        # numpy.median averages the middle two values, matching MATLAB.
        scores = np.array([np.median(distance[gallery_labels == identity][:, probe_labels == target].min(axis=0))
                           for identity in identities])
        record = ranking_record(scores, identities, target)
        record.update(split=split, method=method, scores=scores.tolist())
        records.append(record)
    return records


def extract(cfg, run, directory):
    start = time.perf_counter()
    root = Path(cfg["data_dir"])
    gallery_images, gallery_labels = load_geis(root, "gallery", cfg["gei_size"])
    gallery = gabor_descriptors(gallery_images, cfg)
    model, history = fit_ubm(gallery, cfg, run.log)
    gallery_pdf = map_encode(gallery, model, cfg["map_relevance"], cfg["image_batch_size"])
    gabor_dimension = len(cfg["gabor_scales"]) * len(cfg["gabor_orientations"])
    gallery_gabor = gallery[..., :gabor_dimension].flatten(1).contiguous()
    features = {"gallery": {"pdf": gallery_pdf.cpu(), "labels": gallery_labels.tolist()}}
    records = []
    for split in cfg["probe_splits"]:
        images, labels = load_geis(root, split, cfg["gei_size"])
        patches = gabor_descriptors(images, cfg)
        pdf = map_encode(patches, model, cfg["map_relevance"], cfg["image_batch_size"])
        features[split] = {"pdf": pdf.cpu(), "labels": labels.tolist()}
        for method, x, y in [("Gray-NN", gallery_images.flatten(1), images.flatten(1)),
                             ("Gabor-NN", gallery_gabor, patches[..., :gabor_dimension].flatten(1).contiguous()),
                             ("Gabor-PDF-NN", gallery_pdf, pdf)]:
            distances = torch.cdist(x, y).cpu().numpy()
            records += nn_records(distances, gallery_labels, labels, split, method)
        print("Extracted", split, len(labels), "GEIs", flush=True)
        del images, patches, pdf
    files = ["Gal.mat"] + [f"Prb{s}.mat" for s in cfg["probe_splits"]]
    provenance = {"source_sha256": {f: hashlib.sha256((root / f).read_bytes()).hexdigest() for f in files},
                  "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                  "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
                  "slurm_job_id": os.environ["SLURM_JOB_ID"], "partition": os.environ["SLURM_JOB_PARTITION"],
                  "seconds": time.perf_counter() - start}
    torch.save({"config": cfg, "features": features,
                "ubm": {k: v.cpu() for k, v in model.items()}, "provenance": provenance}, directory / "features.pt")
    (directory / "baselines.json").write_text(json.dumps({"records": records, "summary": summarize(records)}, indent=2))
    (directory / "ubm-history.json").write_text(json.dumps(history, indent=2))
    (directory / "provenance.json").write_text(json.dumps(provenance, indent=2))
    run.log({"timing/features_seconds": provenance["seconds"],
             "features/dimension": gallery_pdf.shape[1],
             "features/gallery_geis": len(gallery_labels)})


def evaluate(directory, run, grid):
    bundle = torch.load(directory / "features.pt", weights_only=False, map_location="cpu")
    cfg = bundle["config"]
    features = bundle["features"]
    gallery = features["gallery"]["pdf"].double().cuda()
    gallery_labels = np.asarray(features["gallery"]["labels"])
    identities = np.unique(gallery_labels)
    groups = [np.flatnonzero(gallery_labels == c) for c in identities]
    gram = (gallery @ gallery.T).cpu().numpy()
    baseline = json.loads((directory / "baselines.json").read_text())
    records = baseline["records"].copy()
    diagnostics = []
    lambdas = [1 / 16, 1 / 8, 1 / 4] if grid else [cfg["lambda"]]
    sigmas = [1 / 40, 1 / 8, 1 / 4, 1 / 2] if grid else [cfg["sigma"]]
    start = time.perf_counter()
    for split in cfg["probe_splits"]:
        probe = features[split]["pdf"].double().cuda()
        labels = np.asarray(features[split]["labels"])
        cross_all = (gallery @ probe.T).cpu().numpy()
        norms = probe.square().sum(1).cpu().numpy()
        euclidean = torch.cdist(gallery, probe).cpu().numpy()
        for index, target in enumerate(np.unique(labels)):
            columns = np.flatnonzero(labels == target)
            cross = cross_all[:, columns]
            norm_sq = norms[columns].sum()
            costs = [euclidean[np.ix_(g, columns)] for g in groups]
            emd = np.array([uniform_emd(cost) for cost in costs])
            configurations = [("GSR", lam, None) for lam in lambdas]
            configurations += [("LGSR", lam, sigma) for lam in lambdas for sigma in sigmas]
            for method, lam, sigma in configurations:
                weights = [np.ones_like(c) for c in costs] if method == "GSR" else [
                    cost * np.exp((distance - emd.min()) / sigma) for cost, distance in zip(costs, emd)]
                tick = time.perf_counter()
                coefficients, diagnostic = algorithm1(gram, cross, norm_sq, groups, weights, lam, cfg)
                residual, wire, active = class_scores(gram, cross, norm_sq, groups, coefficients)
                name = f"Gabor-PDF-{method}"
                tag = f"lambda={lam:g}" + (f",sigma={sigma:g}" if sigma is not None else "")
                for criterion, scores in [("minRE", residual), ("maxWIRE", -wire)]:
                    record = ranking_record(scores, identities, target, active)
                    record.update(split=split, method=f"{name}-{criterion}" + (f"/{tag}" if grid else ""),
                                  scores=scores.tolist(), regularization=lam, sigma=sigma)
                    records.append(record)
                diagnostic.update(subject=int(target), split=split, method=name,
                                  regularization=lam, sigma=sigma, seconds=time.perf_counter() - tick,
                                  active_groups=int(active.sum()), emd=emd.tolist())
                diagnostics.append(diagnostic)
            if index % 10 == 0:
                print(split, index + 1, "/", len(np.unique(labels)), "proxy videos", flush=True)
        current = summarize([r for r in records if r["split"] == split])
        run.log({f"{split}/{name}/rank1": data["overall"]["rank1_percent"] for name, data in current.items()})
    summary = summarize(records)
    result = {"scope": "USF-v1.7 MPCA A-C identity-grouped proxy-video validation; not the full paper benchmark",
              "config": cfg, "grid_search": grid, "summary": summary,
              "seconds": time.perf_counter() - start, "records": records,
              "feature_provenance": bundle["provenance"]}
    suffix = "grid" if grid else "default"
    (directory / f"results-{suffix}.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    (directory / f"solver-{suffix}.json").write_text(json.dumps(diagnostics, indent=2, allow_nan=False))
    run.log({f"overall/{name}/{rank}": data["overall"][rank + "_percent"]
             for name, data in summary.items() for rank in ["rank1", "rank5"]})
    run.log({"timing/evaluation_seconds": result["seconds"],
             "solver/max_kkt_residual": max(d["max_kkt_residual"] for d in diagnostics),
             "solver/iteration_limit_count": sum(d["stop_reason"] == "iteration_limit" for d in diagnostics)})
    print(json.dumps({k: v["overall"] for k, v in summary.items()}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--stage", choices=["features", "evaluate", "all"], default="all")
    parser.add_argument("--grid", action="store_true")
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    cfg["seed"] = args.seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    directory = Path("artifacts") / f"seed-{args.seed}"
    directory.mkdir(parents=True, exist_ok=True)
    run = swanlab.init(project=cfg["swanlab_project"],
                       experiment_name=f"mpca-ac-{args.stage}-seed{args.seed}" + ("-grid" if args.grid else ""),
                       description="A–C proxy-video validation; 71 gallery identities; not the 122-person paper benchmark.",
                       config={**cfg, "stage": args.stage, "grid_search": args.grid,
                               "slurm_job_id": os.environ["SLURM_JOB_ID"]})
    if args.stage in ("features", "all"):
        extract(cfg, run, directory)
    if args.stage in ("evaluate", "all"):
        evaluate(directory, run, args.grid)
    run.finish()


if __name__ == "__main__":
    main()
