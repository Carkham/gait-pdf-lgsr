"""Download the small USF v1.7 MPCA tensors for pipeline validation."""

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.request import urlopen

import numpy as np
from scipy.io import loadmat, whosmat


COMMIT = "9757930e0bbdd0b6b2f499410d9c1cbee2f3348e"
REPOSITORY = "zeke75/Multilinear-PCA"
DESTINATION = Path("data/usf-v1.7-mpca")
FILES = ["ReadMe.txt", "ReadMe1.2.txt", "Gal.mat"] + [
    f"Prb{split}.mat" for split in "ABCDEFG"
]


def download(name):
    url = f"https://raw.githubusercontent.com/{REPOSITORY}/{COMMIT}/{name}"
    with urlopen(url, timeout=60) as response:
        content = response.read()
    path = DESTINATION / name
    path.write_bytes(content)
    result = {
        "file": name,
        "url": url,
        "bytes": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
    }
    if name.endswith(".mat"):
        result["variables"] = whosmat(path)
        data = loadmat(path)
        feature_key, label_key = (
            ("fea3Dg", "gndg") if name == "Gal.mat" else ("fea3D", "gnd")
        )
        features = data[feature_key]
        labels = data[label_key].ravel()
        subjects, counts = np.unique(labels, return_counts=True)
        result["audit"] = {
            "feature_key": feature_key,
            "label_key": label_key,
            "sample_count": features.shape[-1],
            "label_count": labels.size,
            "subject_count": subjects.size,
            "subject_ids": subjects.tolist(),
            "samples_per_subject": counts.tolist(),
            "all_finite": bool(np.isfinite(features).all()),
            "value_range": [float(features.min()), float(features.max())],
        }
    return result


DESTINATION.mkdir(parents=True, exist_ok=True)
with ThreadPoolExecutor(max_workers=2) as pool:
    files = list(pool.map(download, FILES))
gallery_ids = set(files[2]["audit"]["subject_ids"])
for entry in files[3:]:
    entry["audit"]["subjects_missing_from_gallery"] = sorted(
        set(entry["audit"]["subject_ids"]) - gallery_ids
    )
manifest = {
    "dataset": "USF/NIST v1.7, MPCA 32x22x10 tensors",
    "purpose": "Small pipeline validation dataset, not a held-out tuning split or full paper reproduction",
    "source_repository": f"https://github.com/{REPOSITORY}",
    "source_commit": COMMIT,
    "files": files,
}
(DESTINATION / "manifest.json").write_text(
    json.dumps(manifest, indent=2) + "\n"
)
for entry in files[2:]:
    audit = entry["audit"]
    print(entry["file"], audit["sample_count"], "samples,",
          audit["subject_count"], "subjects, range", audit["value_range"])
print("Saved", DESTINATION / "manifest.json")
