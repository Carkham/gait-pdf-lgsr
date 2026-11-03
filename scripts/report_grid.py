"""Summarize the paper's parameter grid, explicitly labelled post-hoc oracle."""

import csv
import gzip
import hashlib
import json
from pathlib import Path


root = Path("artifacts/seed-0")
out = Path("results")
out.mkdir(exist_ok=True)
data = json.loads((root / "results-grid.json").read_text())
with (out / "grid-metrics.csv").open("w") as handle:
    writer = csv.writer(handle)
    writer.writerow(["method", "split", "videos", "rank1_correct", "rank5_correct", "rank1_percent", "rank5_percent"])
    for method, splits in data["summary"].items():
        for split, stats in splits.items():
            writer.writerow([method, split, *[stats[k] for k in ["videos", "rank1_correct", "rank5_correct", "rank1_percent", "rank5_percent"]]])
oracle = {}
for family in ["Gabor-PDF-GSR-maxWIRE", "Gabor-PDF-LGSR-maxWIRE"]:
    options = {k: v for k, v in data["summary"].items() if k.startswith(family + "/")}
    oracle[family] = {}
    for rank in ["rank1", "rank5"]:
        best = {}
        for split in ["A", "B", "C"]:
            chosen = max(options, key=lambda m: options[m][split][rank + "_correct"])
            best[split] = {"method": chosen, **options[chosen][split]}
        correct = sum(x[rank + "_correct"] for x in best.values())
        count = sum(x["videos"] for x in best.values())
        oracle[family][rank] = {"correct": correct, "videos": count,
                               "percent": 100 * correct / count, "selected_by_split": best}
(out / "grid-oracle.json").write_text(json.dumps({
    "warning": "Post-hoc selection using probe labels, separately per probe set and metric; not held-out generalization performance.",
    "seed": 0, "oracle": oracle}, indent=2))
for name in ["results-grid.json", "solver-grid.json"]:
    payload = json.loads((root / name).read_text())
    with gzip.GzipFile(filename=str(out / ("seed-0-" + name + ".gz")), mode="wb", mtime=0) as handle:
        handle.write(json.dumps(payload, separators=(",", ":"), allow_nan=False).encode())
paths = sorted([*out.rglob("*"), *Path("figures").rglob("*")])
(out / "checksums.json").write_text(json.dumps({str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                                               for p in paths if p.is_file() and p.name != "checksums.json"}, indent=2))
print(json.dumps(oracle, indent=2))
