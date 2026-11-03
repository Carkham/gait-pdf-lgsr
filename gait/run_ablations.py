"""Tables III and V: Gabor bank, mixture count, and XY ablations with NN."""

import argparse
import csv
import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
import swanlab
import torch

from .features import fit_ubm, gabor_descriptors, map_encode
from .run import load_geis, nn_records, summarize


VARIANTS = [
    {"name": f"gabor-{scales}x{orientations}-xy-k{k}",
     "gabor_scales": list(range(5)) if scales == 5 else [0, 2, 4],
     "gabor_orientations": list(range(8)) if orientations == 8 else [0, 2, 4, 6],
     "use_xy": True, "components": k}
    for scales, orientations in [(5, 8), (5, 4), (3, 8), (3, 4)]
    for k in [100, 300, 500]
] + [
    {"name": f"gabor-5x8-no-xy-k{k}", "gabor_scales": list(range(5)),
     "gabor_orientations": list(range(8)), "use_xy": False, "components": k}
    for k in [100, 300, 500]
]


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def timed(function):
    torch.cuda.synchronize()
    start = time.perf_counter()
    value = function()
    torch.cuda.synchronize()
    return value, time.perf_counter() - start


def save_result(result, records, history):
    directory = Path("results/ablations") / result["variant"] / f"seed-{result['seed']}"
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory / "result.json", result)
    write_json(directory / "ubm-history.json", history)
    with gzip.open(directory / "records.json.gz", "wt") as stream:
        json.dump(records, stream, allow_nan=False)


def reuse_default():
    variant = VARIANTS[1]["name"]
    for seed in range(3):
        source = Path("artifacts") / f"seed-{seed}"
        original = json.loads((source / "results-default.json").read_text())
        records = [r for r in original["records"] if r["method"] == "Gabor-PDF-NN"]
        history = json.loads((source / "ubm-history.json").read_text())
        result = {"variant": variant, "seed": seed, "config": original["config"],
                  "summary": summarize(records), "reused": True,
                  "timings": {"original_all_baseline_feature_seconds": original["feature_provenance"]["seconds"]},
                  "provenance": original["feature_provenance"],
                  "source_result_sha256": hashlib.sha256((source / "results-default.json").read_bytes()).hexdigest(),
                  "em_iterations": len(history["em"]),
                  "em_reached_tolerance": history["em_reached_tolerance"]}
        save_result(result, records, history)


def experiment(index):
    base = json.loads(Path("config.json").read_text())
    variant = VARIANTS[index]
    cfg = {**base, **{key: value for key, value in variant.items() if key != "name"}}
    root = Path(cfg["data_dir"])
    run = swanlab.init(project=cfg["swanlab_project"], experiment_name=f"ablation-{variant['name']}",
                       description="Gabor-PDF NN; A–C identity-grouped proxy videos; seeds 0, 1, 2.",
                       config={**cfg, "variant": variant["name"], "seeds": [0, 1, 2],
                               "slurm_job_id": os.environ["SLURM_JOB_ID"]})
    provenance = {"git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                  "source_sha256": {filename: hashlib.sha256((root / filename).read_bytes()).hexdigest()
                                    for filename in ["Gal.mat", "PrbA.mat", "PrbB.mat", "PrbC.mat"]},
                  "code_sha256": {filename: hashlib.sha256(Path(filename).read_bytes()).hexdigest()
                                  for filename in ["gait/run_ablations.py", "gait/features.py", "gait/run.py", "config.json", "uv.lock"]},
                  "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
                  "slurm_job_id": os.environ["SLURM_JOB_ID"], "partition": os.environ["SLURM_JOB_PARTITION"]}
    patches, labels, preparation = {}, {}, {}
    for split in ["gallery", *cfg["probe_splits"]]:
        (images, labels[split]), load_seconds = timed(lambda: load_geis(root, split, cfg["gei_size"]))
        patches[split], gabor_seconds = timed(lambda: gabor_descriptors(images, cfg))
        preparation[split] = {"load_gei_seconds": load_seconds, "gabor_seconds": gabor_seconds}
        del images
    for seed in range(3):
        cfg["seed"] = seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        start = time.perf_counter()
        log = lambda values: run.log({f"seed{seed}/{key}": value for key, value in values.items()})
        (model, history), fit_seconds = timed(lambda: fit_ubm(patches["gallery"], cfg, log))
        encoded, map_seconds = {}, {}
        for split, descriptors in patches.items():
            encoded[split], map_seconds[split] = timed(
                lambda: map_encode(descriptors, model, cfg["map_relevance"], cfg["image_batch_size"]))
        records, nn_seconds = [], {}
        for split in cfg["probe_splits"]:
            def classify():
                distances = torch.cdist(encoded["gallery"], encoded[split]).cpu().numpy()
                return nn_records(distances, labels["gallery"], labels[split], split, "Gabor-PDF-NN")
            rows, nn_seconds[split] = timed(classify)
            records.extend(rows)
        timings = {"shared_preparation_by_split": preparation, "ubm_fit_seconds": fit_seconds,
                   "map_by_split_seconds": map_seconds, "nn_by_split_seconds": nn_seconds,
                   "per_seed_seconds": time.perf_counter() - start,
                   "map_seconds_per_gei": sum(map_seconds.values()) / sum(len(x) for x in labels.values())}
        result = {"variant": variant["name"], "seed": seed, "config": dict(cfg),
                  "summary": summarize(records), "timings": timings, "provenance": provenance,
                  "reused": False, "em_iterations": len(history["em"]),
                  "em_reached_tolerance": history["em_reached_tolerance"]}
        save_result(result, records, history)
        directory = Path("artifacts/ablations") / variant["name"] / f"seed-{seed}"
        directory.mkdir(parents=True, exist_ok=True)
        torch.save({"config": dict(cfg), "ubm": {key: value.cpu() for key, value in model.items()},
                    "provenance": provenance}, directory / "ubm.pt")
        log({f"{split}/{rank}": values[f"{rank}_percent"]
             for split, values in result["summary"]["Gabor-PDF-NN"].items() for rank in ["rank1", "rank5"]})
        log({"timing/ubm_seconds": fit_seconds, "timing/map_seconds": sum(map_seconds.values()),
             "timing/nn_seconds": sum(nn_seconds.values()), "ubm/em_reached_tolerance": history["em_reached_tolerance"]})
        print(json.dumps({"variant": variant["name"], "seed": seed,
                          "overall": result["summary"]["Gabor-PDF-NN"]["overall"],
                          "timings": timings, "em_reached_tolerance": history["em_reached_tolerance"]}), flush=True)
        del encoded, model
    run.finish()


def report():
    reuse_default()
    rows, aggregate = [], {}
    for variant in VARIANTS:
        name = variant["name"]
        results = [json.loads((Path("results/ablations") / name / f"seed-{seed}" / "result.json").read_text())
                   for seed in range(3)]
        aggregate[name] = {"configuration": variant, "summary": {}}
        for split in ["A", "B", "C", "overall"]:
            aggregate[name]["summary"][split] = {}
            for rank in ["rank1", "rank5"]:
                values = [result["summary"]["Gabor-PDF-NN"][split][f"{rank}_percent"] for result in results]
                aggregate[name]["summary"][split][rank] = {
                    "mean": float(np.mean(values)), "sample_std": float(np.std(values, ddof=1)), "seeds": values}
        for result in results:
            for split, metrics in result["summary"]["Gabor-PDF-NN"].items():
                rows.append({"variant": name, "seed": result["seed"], "split": split,
                             "components": variant["components"], "scales": len(variant["gabor_scales"]),
                             "orientations": len(variant["gabor_orientations"]), "use_xy": variant["use_xy"],
                             "reused": result["reused"], "em_iterations": result["em_iterations"],
                             "em_reached_tolerance": result["em_reached_tolerance"], **metrics})
    write_json(Path("results/ablations/summary.json"),
               {"protocol": "USF-v1.7 MPCA A–C identity-grouped proxy videos", "method": "Gabor-PDF-NN",
                "seeds": [0, 1, 2], "variants": aggregate})
    with Path("results/ablations/metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant-index", type=int)
    parser.add_argument("--report", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if args.report:
        report()
    else:
        experiment(args.variant_index)


if __name__ == "__main__":
    main()
