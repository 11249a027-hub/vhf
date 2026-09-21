import os
import random
import json
import torch
import numpy as np
from torch.utils.data import Dataset
from preprocess import preprocess_signature
import re

# Fixed random seed for reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

class SignaturePairDataset(Dataset):
    """Dataset that yields (img1, img2, label) pairs.

    Writer‑aware implementation with deterministic train/val/test splits.
    Generates three pair types:
        * genuine (same writer, two originals) – label 0
        * forgery (original vs forged, same writer) – label 1
        * cross‑writer negative (original vs original, different writers) – label 1
    """

    def __init__(self, full_org_path, full_forg_path, split="train", split_mapping_path=None):
        """Create the dataset.

        Args:
            full_org_path: directory with original signatures (CEDAR naming).
            full_forg_path: directory with forged signatures (CEDAR naming).
            split: "train" | "val" | "test".
            split_mapping_path: optional JSON file mapping writers to splits.
        """
        self.full_org_path = full_org_path
        self.full_forg_path = full_forg_path
        self.split = split
        self.split_mapping_path = split_mapping_path

        # Load file lists, ignore non‑image files (e.g., Thumbs.db)
        self.org_files = [f for f in os.listdir(full_org_path) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
        self.forg_files = [f for f in os.listdir(full_forg_path) if f.lower().endswith((".png", ".jpg", ".jpeg"))]

        # Parse writer IDs from filenames (CEDAR convention)
        org_pat = re.compile(r"original_(\d+)_(\d+)\.(png|jpg|jpeg)", re.IGNORECASE)
        forg_pat = re.compile(r"forgeries_(\d+)_(\d+)\.(png|jpg|jpeg)", re.IGNORECASE)
        self.org_by_writer = {}
        for f in self.org_files:
            m = org_pat.match(f)
            if not m:
                continue
            writer_id = int(m.group(1))
            self.org_by_writer.setdefault(writer_id, []).append(f)
        self.forg_by_writer = {}
        for f in self.forg_files:
            m = forg_pat.match(f)
            if not m:
                continue
            writer_id = int(m.group(1))
            self.forg_by_writer.setdefault(writer_id, []).append(f)

        # Load or create deterministic writer split mapping
        if split_mapping_path and os.path.exists(split_mapping_path):
            with open(split_mapping_path, "r") as jf:
                split_map = json.load(jf)
        else:
            all_writers = sorted(set(self.org_by_writer.keys()))
            random.shuffle(all_writers)
            n = len(all_writers)
            train_end = int(0.70 * n)
            val_end = train_end + int(0.15 * n)
            split_map = {
                "train": all_writers[:train_end],
                "val": all_writers[train_end:val_end],
                "test": all_writers[val_end:]
            }
            if split_mapping_path:
                os.makedirs(os.path.dirname(split_mapping_path), exist_ok=True)
                with open(split_mapping_path, "w") as jf:
                    json.dump(split_map, jf, indent=2)

        # Keep only writers belonging to the requested split
        allowed = set(split_map.get(split, []))
        self.org_by_writer = {w: v for w, v in self.org_by_writer.items() if w in allowed}
        self.forg_by_writer = {w: v for w, v in self.forg_by_writer.items() if w in allowed}

        # Build pair list
        self.pairs = []
        self._create_pairs()

    def _create_pairs(self):
        # Genuine pairs (same writer, two different originals) -> label 0
        for writer, files in self.org_by_writer.items():
            for i in range(len(files)):
                for j in range(i + 1, len(files)):
                    self.pairs.append((os.path.join(self.full_org_path, files[i]),
                                       os.path.join(self.full_org_path, files[j]),
                                       0.0))
        # Forgery pairs (original vs forged, same writer) -> label 1
        for writer, org_files in self.org_by_writer.items():
            forg_files = self.forg_by_writer.get(writer, [])
            for org in org_files:
                for forg in forg_files:
                    self.pairs.append((os.path.join(self.full_org_path, org),
                                       os.path.join(self.full_forg_path, forg),
                                       1.0))
        # Cross‑writer negative pairs (original vs original, different writers) -> label 1
        writers = list(self.org_by_writer.keys())
        for i in range(len(writers)):
            for j in range(i + 1, len(writers)):
                org_i = self.org_by_writer[writers[i]]
                org_j = self.org_by_writer[writers[j]]
                if org_i and org_j:
                    self.pairs.append((os.path.join(self.full_org_path, org_i[0]),
                                       os.path.join(self.full_org_path, org_j[0]),
                                       1.0))
        random.shuffle(self.pairs)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img1_path, img2_path, label = self.pairs[idx]
        img1 = preprocess_signature(img1_path)
        img2 = preprocess_signature(img2_path)
        img1 = torch.tensor(img1, dtype=torch.float32)
        img2 = torch.tensor(img2, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.float32)
        return img1, img2, label