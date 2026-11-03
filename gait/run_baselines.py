"""Run DCT-NN, DCT-PDF-NN, SR and exact LLC on the A–C protocol."""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
import swanlab
import torch

from .baselines import dct_descriptors, exact_llc, sparse_representation
from .features import fit_ubm, map_encode
from .run import load_geis, nn_records, summarize
from .solver import class_scores, ranking_record


def elapsed(start):
    torch.cuda.synchronize()
    return time.perf_counter() - start


def representation_records(gram, cross, norms, labels, gallery_labels, coefficients, split, name):
    identities = np.unique(gallery_labels)
    groups = [np.flatnonzero(gallery_labels == identity) for identity in identities]
    records = []
    for target in np.unique(labels):
        columns = np.flatnonzero(labels == target)
        residual, wire, active = class_scores(gram, cross[:, columns], norms[columns].sum(),
                                             groups, coefficients[:, columns])
        for criterion, scores in [("minRE", residual), ("maxWIRE", -wire)]:
            record = ranking_record(scores, identities, target, active)
            record.update(split=split, method=f"Gabor-PDF-{name}-{criterion}", scores=scores.tolist())
            records.append(record)
    return records


def run(seed):
    cfg = json.loads(Path("config.json").read_text())
    cfg.update(seed=seed, dct_patch_size=8, dct_frequencies=5, dct_boundary="reflect",
               dct_coefficient_normalization="orthonormal_dct_ii", dct_include_dc=True,
               dct_patch_offsets=[-3, 4], dct_response_grid=[64, 44],
               sr_solver="fista", sr_iterations=10000, sr_kkt_tolerance=1e-6,
               sr_check_every=10, llc_batch_size=16)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    artifact = Path("artifacts/baselines") / f"seed-{seed}"
    artifact.mkdir(parents=True, exist_ok=True)
    output = Path("results/baselines")
    output.mkdir(parents=True, exist_ok=True)
    experiment = swanlab.init(project=cfg["swanlab_project"], experiment_name=f"baselines-seed{seed}",
                              description="A–C identity-grouped validation: DCT, DCT-PDF, SR, exact LLC.",
                              config={**cfg, "slurm_job_id": os.environ["SLURM_JOB_ID"]})
    feature_file = Path("artifacts") / f"seed-{seed}/features.pt"
    bundle = torch.load(feature_file, map_location="cpu", weights_only=False)
    features = bundle["features"]
    gallery = features["gallery"]["pdf"].double().cuda()
    gallery_labels = np.asarray(features["gallery"]["labels"])
    gram = gallery @ gallery.T
    gram_cpu = gram.cpu().numpy()
    records, diagnostics, timing = [], {}, {}
    for split in cfg["probe_splits"]:
        probe = features[split]["pdf"].double().cuda()
        labels = np.asarray(features[split]["labels"])
        cross = gallery @ probe.T
        norms = probe.square().sum(1)
        torch.cuda.synchronize()
        for name in ["SR", "LLC"]:
            tick = time.perf_counter()
            if name == "SR":
                coefficients, diagnostic = sparse_representation(gram, cross, norms, cfg["lambda"], cfg)
            else:
                coefficients, diagnostic = exact_llc(gram, cross, norms, cfg["lambda"],
                                                     cfg["sigma"], cfg["llc_batch_size"])
            timing[f"{split}/{name}_coefficient_seconds"] = elapsed(tick)
            diagnostics[f"{split}/{name}"] = diagnostic
            records.extend(representation_records(gram_cpu, cross.cpu().numpy(), norms.cpu().numpy(), labels,
                                                   gallery_labels, coefficients.cpu().numpy(), split, name))
            print(split, name, timing[f"{split}/{name}_coefficient_seconds"], "seconds", flush=True)
    del gallery, gram, cross, probe, coefficients

    root = Path(cfg["data_dir"])
    tick = time.perf_counter()
    images, gallery_labels = load_geis(root, "gallery", cfg["gei_size"])
    patches = dct_descriptors(images, cfg)
    timing["DCT/gallery_descriptor_seconds"] = elapsed(tick)
    tick = time.perf_counter()
    model, history = fit_ubm(patches, cfg, experiment.log)
    timing["DCT/ubm_seconds"] = elapsed(tick)
    tick = time.perf_counter()
    gallery_pdf = map_encode(patches, model, cfg["map_relevance"], cfg["image_batch_size"])
    timing["DCT/gallery_map_seconds"] = elapsed(tick)
    gallery_dct = patches[..., :25].flatten(1).contiguous()
    dct_features = {"gallery": {"pdf": gallery_pdf.cpu(), "labels": gallery_labels.tolist()}}
    del patches, images
    for split in cfg["probe_splits"]:
        tick = time.perf_counter()
        images, labels = load_geis(root, split, cfg["gei_size"])
        patches = dct_descriptors(images, cfg)
        timing[f"DCT/{split}_descriptor_seconds"] = elapsed(tick)
        tick = time.perf_counter()
        pdf = map_encode(patches, model, cfg["map_relevance"], cfg["image_batch_size"])
        timing[f"DCT/{split}_map_seconds"] = elapsed(tick)
        dct_features[split] = {"pdf": pdf.cpu(), "labels": labels.tolist()}
        for name, x, y in [("DCT-NN", gallery_dct, patches[..., :25].flatten(1).contiguous()),
                            ("DCT-PDF-NN", gallery_pdf, pdf)]:
            tick = time.perf_counter()
            distance = torch.cdist(x, y).cpu().numpy()
            records.extend(nn_records(distance, gallery_labels, labels, split, name))
            timing[f"{split}/{name}_classification_seconds"] = elapsed(tick)
        print("DCT extracted", split, flush=True)
        del patches, images, pdf
    summary = summarize(records)
    provenance = {"git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                  "slurm_job_id": os.environ["SLURM_JOB_ID"], "partition": os.environ["SLURM_JOB_PARTITION"],
                  "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
                  "gabor_feature_sha256": hashlib.sha256(feature_file.read_bytes()).hexdigest(),
                  "gabor_feature_provenance": bundle["provenance"],
                  "source_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                                    for path in [Path("gait/baselines.py"), Path("gait/run_baselines.py")]}}
    result = {"scope": "USF v1.7 A–C identity-grouped proxy videos", "config": cfg,
              "summary": summary, "records": records, "diagnostics": diagnostics,
              "timing": timing, "dct_ubm_history": history, "provenance": provenance}
    (output / f"seed-{seed}.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    torch.save({"config": cfg, "features": dct_features,
                "ubm": {k: v.cpu() for k, v in model.items()}}, artifact / "dct-features.pt")
    experiment.log({f"overall/{method}/{rank}": values["overall"][f"{rank}_percent"]
                    for method, values in summary.items() for rank in ["rank1", "rank5"]})
    experiment.log({f"timing/{key}": value for key, value in timing.items()})
    print(json.dumps({method: values["overall"] for method, values in summary.items()}, indent=2), flush=True)
    experiment.finish()


def report():
    output = Path("results/baselines")
    runs = [json.loads((output / f"seed-{seed}.json").read_text()) for seed in range(3)]
    summary = {}
    for method in runs[0]["summary"]:
        summary[method] = {}
        for split in ["A", "B", "C", "overall"]:
            summary[method][split] = {}
            for rank in ["rank1", "rank5"]:
                values = [run["summary"][method][split][f"{rank}_percent"] for run in runs]
                summary[method][split][rank] = {"mean": float(np.mean(values)),
                                               "sample_std": float(np.std(values, ddof=1)), "seeds": values}
    (output / "summary.json").write_text(json.dumps({"seeds": [0, 1, 2], "summary": summary}, indent=2))
    with (output / "metrics.csv").open("w") as handle:
        writer = csv.writer(handle)
        writer.writerow(["method", "split", "rank", "mean_percent", "sample_std_percent", "seed0", "seed1", "seed2"])
        for method, splits in summary.items():
            for split, ranks in splits.items():
                for rank, values in ranks.items():
                    writer.writerow([method, split, rank, values["mean"], values["sample_std"], *values["seeds"]])
    print(json.dumps({method: splits["overall"] for method, splits in summary.items()}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--report", action="store_true")
    args = parser.parse_args()
    if args.report:
        report()
    else:
        run(args.seed)
