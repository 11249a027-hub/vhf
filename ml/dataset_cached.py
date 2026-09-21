import os
import json
import random
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
import re

RANDOM_SEED = 42

class CachedPairDataset(Dataset):
    """
    High-performance in-memory dataset for pairwise Siamese training & evaluation.
    Zero disk I/O during training loop.
    """
    def __init__(self, cache: dict, full_org_path: str, full_forg_path: str,
                 split: str = "train", split_mapping_path: str = "ml/split_mapping.json",
                 augment: bool = False):
        self.cache = cache
        self.split = split
        self.augment = augment

        # Load writer split mapping
        with open(split_mapping_path, "r") as f:
            split_map = json.load(f)
        allowed_writers = set(split_map[split])

        # Parse files for allowed writers
        org_pat = re.compile(r"original_(\d+)_(\d+)\.(png|jpg|jpeg)", re.IGNORECASE)
        forg_pat = re.compile(r"forgeries_(\d+)_(\d+)\.(png|jpg|jpeg)", re.IGNORECASE)

        self.org_by_writer = {}
        for f in os.listdir(full_org_path):
            m = org_pat.match(f)
            if m and int(m.group(1)) in allowed_writers:
                w = int(m.group(1))
                norm_p = os.path.normpath(os.path.join(full_org_path, f)).replace("\\", "/")
                self.org_by_writer.setdefault(w, []).append(norm_p)

        self.forg_by_writer = {}
        for f in os.listdir(full_forg_path):
            m = forg_pat.match(f)
            if m and int(m.group(1)) in allowed_writers:
                w = int(m.group(1))
                norm_p = os.path.normpath(os.path.join(full_forg_path, f)).replace("\\", "/")
                self.forg_by_writer.setdefault(w, []).append(norm_p)

        self.pairs = []
        self._build_pairs()

    def _build_pairs(self):
        rng = random.Random(RANDOM_SEED + (0 if self.split == 'train' else 1))
        writers = sorted(list(self.org_by_writer.keys()))

        # 1. Genuine pairs: same writer, two authentic signatures -> label 0.0
        genuine_pairs = []
        for w in writers:
            orgs = sorted(self.org_by_writer[w])
            for i in range(len(orgs)):
                for j in range(i + 1, len(orgs)):
                    genuine_pairs.append((orgs[i], orgs[j], 0.0))

        # 2. Skilled Forgery pairs: same writer, authentic vs forged -> label 1.0
        forg_pairs = []
        for w in writers:
            orgs = sorted(self.org_by_writer[w])
            forgs = sorted(self.forg_by_writer.get(w, []))
            for o in orgs:
                for f in forgs:
                    forg_pairs.append((o, f, 1.0))

        # 3. Cross-writer negative pairs: different writers, authentic signatures -> label 1.0
        cross_pairs = []
        if self.split == 'train':
            # For training, balance negative types
            target_cross = len(genuine_pairs) // 2
            for _ in range(target_cross):
                w1, w2 = rng.sample(writers, 2)
                o1 = rng.choice(self.org_by_writer[w1])
                o2 = rng.choice(self.org_by_writer[w2])
                cross_pairs.append((o1, o2, 1.0))
            
            # Combine balanced training pairs
            self.pairs = genuine_pairs + forg_pairs + cross_pairs
        else:
            # For validation and test: maintain standard CEDAR evaluation sets
            for i in range(len(writers)):
                for j in range(i + 1, len(writers)):
                    o_i = self.org_by_writer[writers[i]][0]
                    o_j = self.org_by_writer[writers[j]][0]
                    cross_pairs.append((o_i, o_j, 1.0))
            self.pairs = genuine_pairs + forg_pairs + cross_pairs

        rng.shuffle(self.pairs)

    def __len__(self):
        return len(self.pairs)

    def _apply_augmentation(self, img_np: np.ndarray) -> np.ndarray:
        # Subtle rotation and translation for training robustness
        if random.random() < 0.5:
            angle = random.uniform(-3.0, 3.0)
            dx = random.uniform(-3.0, 3.0)
            dy = random.uniform(-2.0, 2.0)
            c, h, w = img_np.shape
            M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
            M[0, 2] += dx
            M[1, 2] += dy
            warped = cv2.warpAffine(img_np[0], M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            return np.expand_dims(warped, axis=0)
        return img_np

    def __getitem__(self, idx):
        p1, p2, label = self.pairs[idx]
        img1 = self.cache[p1]
        img2 = self.cache[p2]

        if self.augment:
            img1 = self._apply_augmentation(img1)
            img2 = self._apply_augmentation(img2)

        return (
            torch.from_numpy(img1),
            torch.from_numpy(img2),
            torch.tensor(label, dtype=torch.float32)
        )


class CachedTripletDataset(Dataset):
    """
    In-memory Triplet Dataset: (Anchor, Positive, Negative).
    Anchor & Positive: Genuine signatures from same writer.
    Negative: 50% Skilled Forgery (Hard Negative), 50% Cross-Writer signature.
    """
    def __init__(self, cache: dict, full_org_path: str, full_forg_path: str,
                 split_mapping_path: str = "ml/split_mapping.json", num_triplets: int = 30000):
        self.cache = cache
        self.num_triplets = num_triplets

        with open(split_mapping_path, "r") as f:
            split_map = json.load(f)
        allowed_writers = set(split_map["train"])

        org_pat = re.compile(r"original_(\d+)_(\d+)\.(png|jpg|jpeg)", re.IGNORECASE)
        forg_pat = re.compile(r"forgeries_(\d+)_(\d+)\.(png|jpg|jpeg)", re.IGNORECASE)

        self.org_by_writer = {}
        for f in os.listdir(full_org_path):
            m = org_pat.match(f)
            if m and int(m.group(1)) in allowed_writers:
                w = int(m.group(1))
                p = os.path.normpath(os.path.join(full_org_path, f)).replace("\\", "/")
                self.org_by_writer.setdefault(w, []).append(p)

        self.forg_by_writer = {}
        for f in os.listdir(full_forg_path):
            m = forg_pat.match(f)
            if m and int(m.group(1)) in allowed_writers:
                w = int(m.group(1))
                p = os.path.normpath(os.path.join(full_forg_path, f)).replace("\\", "/")
                self.forg_by_writer.setdefault(w, []).append(p)

        self.writers = sorted(list(self.org_by_writer.keys()))
        self.rng = random.Random(RANDOM_SEED)

    def __len__(self):
        return self.num_triplets

    def __getitem__(self, idx):
        # Pick writer
        w = self.rng.choice(self.writers)
        orgs = self.org_by_writer[w]
        # Anchor and Positive
        anc_path, pos_path = self.rng.sample(orgs, 2)

        # 50% chance skilled forgery, 50% chance cross-writer
        if self.rng.random() < 0.5 and w in self.forg_by_writer:
            neg_path = self.rng.choice(self.forg_by_writer[w])
        else:
            w_other = self.rng.choice([ow for ow in self.writers if ow != w])
            neg_path = self.rng.choice(self.org_by_writer[w_other])

        return (
            torch.from_numpy(self.cache[anc_path]),
            torch.from_numpy(self.cache[pos_path]),
            torch.from_numpy(self.cache[neg_path])
        )
