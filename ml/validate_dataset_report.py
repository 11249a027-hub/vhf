import sys, os
# Ensure project root is in PYTHONPATH
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from ml.dataset_pairs import SignaturePairDataset

FULL_ORG = "signatures/full_org"
FULL_FORG = "signatures/full_forg"
SPLIT_MAP = "ml/writer_split.json"

splits = ["train", "val", "test"]

for split in splits:
    ds = SignaturePairDataset(FULL_ORG, FULL_FORG, split=split, split_mapping_path=SPLIT_MAP)
    total_pairs = len(ds)
    # Count pair types
    genuine = 0
    forgery = 0
    cross = 0
    examples = {"genuine": [], "forgery": [], "cross": []}
    for idx in range(min(total_pairs, 2000)):
        img1_path, img2_path, label = ds.pairs[idx]
        if label == 0.0:
            genuine += 1
            if len(examples["genuine"]) < 5:
                examples["genuine"].append((img1_path, img2_path, label))
        else:
            # distinguish forgery vs cross using directory of second image
            if "full_forg" in img2_path:
                forgery += 1
                if len(examples["forgery"]) < 5:
                    examples["forgery"].append((img1_path, img2_path, label))
            else:
                cross += 1
                if len(examples["cross"]) < 5:
                    examples["cross"].append((img1_path, img2_path, label))
    # Writers count
    writers = len(ds.org_by_writer)
    # Print report
    print(f"=== Split: {split.upper()} ===")
    print(f"Writers in split: {writers}")
    print(f"Total pairs: {total_pairs}")
    print(f"Genuine pairs (label 0): {genuine}")
    print(f"Forgery pairs (original vs forged, label 1): {forgery}")
    print(f"Cross-writer negative pairs (label 1): {cross}\n")
    for cat, ex in examples.items():
        print(f"--- {cat.title()} examples (up to 5) ---")
        for a, b, l in ex:
            print(f"{a}  <->  {b}   label={l}")
        print()
