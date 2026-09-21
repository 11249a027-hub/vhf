import os, json, random

# Project root is two levels up from this script (ml folder)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
full_org_path = os.path.join(project_root, 'signatures', 'full_org')

# Gather writer IDs from original signatures (CEDAR naming)
org_files = [f for f in os.listdir(full_org_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
writer_ids = set()
for f in org_files:
    # Expect pattern original_<writer>_<sample>.png
    parts = f.split('_')
    if len(parts) >= 3:
        try:
            writer_ids.add(int(parts[1]))
        except ValueError:
            pass
writer_ids = sorted(writer_ids)
random.seed(42)
random.shuffle(writer_ids)
n = len(writer_ids)
train_end = int(0.70 * n)
val_end = train_end + int(0.15 * n)
split_map = {
    "train": writer_ids[:train_end],
    "val": writer_ids[train_end:val_end],
    "test": writer_ids[val_end:]
}

# Write to ml/writer_split.json
output_path = os.path.join(project_root, 'ml', 'writer_split.json')
with open(output_path, 'w') as f:
    json.dump(split_map, f, indent=2)
print('Writer split mapping written to', output_path)
print('Counts -> train:', len(split_map['train']), 'val:', len(split_map['val']), 'test:', len(split_map['test']))
