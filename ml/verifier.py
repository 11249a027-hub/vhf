import os
import json
import torch
import torch.nn.functional as F
import numpy as np

from ml.preprocess_v3 import preprocess_signature_v3
from ml.models_v3 import SiameseDeepV3

# --------------------------------------------------
# Configuration & V3 Model Loading
# --------------------------------------------------

MODEL_PATH = "ml_models/siamese_signature_v4.pth"
CONFIG_PATH = "ml_models/model_v4_config.json"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load configuration and calibrated threshold
if os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH, "r") as f:
        _cfg = json.load(f)
    DISTANCE_THRESHOLD = float(_cfg["threshold"]["locked_validation_distance_threshold"])
    SIMILARITY_THRESHOLD = float(_cfg["threshold"]["similarity_percentage_threshold"])
    EMBEDDING_DIM = int(_cfg["architecture"]["embedding_dimension"])
else:
    DISTANCE_THRESHOLD = 0.4230
    SIMILARITY_THRESHOLD = 95.53
    EMBEDDING_DIM = 256

# Initialize SiameseDeepV3 architecture
model = SiameseDeepV3(emb_dim=EMBEDDING_DIM).to(DEVICE)

if os.path.exists(MODEL_PATH):
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
else:
    # Fallback if v4 not found
    fallback = "ml_models/siamese_signature_v3.pth"
    if os.path.exists(fallback):
        model.load_state_dict(torch.load(fallback, map_location=DEVICE))

model.eval()


# --------------------------------------------------
# Convert signature image to 256-D L2-normalized embedding
# --------------------------------------------------

def get_embedding(image_path: str) -> torch.Tensor:
    if not image_path:
        raise ValueError("Signature path is empty")

    if not os.path.exists(image_path):
        raise ValueError(f"Signature file not found: {image_path}")

    # Use V3 preprocessing (Otsu crop + aspect-ratio letterbox resize + inverted grayscale)
    image_np = preprocess_signature_v3(image_path)
    if image_np is None:
        raise ValueError(f"Could not preprocess signature: {image_path}")

    image_tensor = torch.from_numpy(image_np).to(DEVICE)

    # Ensure shape (1, 1, H, W)
    if image_tensor.dim() == 3:
        image_tensor = image_tensor.unsqueeze(0)
    elif image_tensor.dim() == 2:
        image_tensor = image_tensor.unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        embedding = model.forward_once(image_tensor)

    return embedding


# --------------------------------------------------
# Compare two signatures (distance & similarity)
# --------------------------------------------------

def compare_signatures(reference_path: str, test_path: str):
    """
    Returns (similarity_score_0_to_100, euclidean_distance).
    Direction:
      distance <= DISTANCE_THRESHOLD -> GENUINE
      distance >  DISTANCE_THRESHOLD -> FORGED
    """
    ref_emb = get_embedding(reference_path)
    test_emb = get_embedding(test_path)

    # Euclidean distance between L2-normalized vectors
    dist = torch.norm(ref_emb - test_emb, p=2).item()

    # Exact cosine similarity mapping for L2 unit vectors: cos = 1 - (d^2)/2
    cos_sim = 1.0 - (dist ** 2) / 2.0
    # Map cosine [-1, 1] to [0, 100]%
    score_pct = ((cos_sim + 1.0) / 2.0) * 100.0
    score_pct = max(0.0, min(100.0, score_pct))

    return round(score_pct, 2), round(dist, 4)


# --------------------------------------------------
# Compare against all enrolled reference signatures
# --------------------------------------------------

def verify_against_employee_signatures(test_path: str, signature_paths: list[str]):
    """
    Compares query signature against enrolled references.
    Direction:
      min_distance <= DISTANCE_THRESHOLD -> GENUINE
      min_distance >  DISTANCE_THRESHOLD -> FORGED
    """
    if not signature_paths:
        raise ValueError("No stored signatures found for this borrower/employee")

    valid_paths = [p for p in signature_paths if p and os.path.exists(p)]
    if not valid_paths:
        raise ValueError("No valid stored signatures found on disk")

    best_score = -1.0
    min_dist = float("inf")
    best_match = None

    scores = []

    for ref_path in valid_paths:
        score, dist = compare_signatures(ref_path, test_path)
        scores.append({
            "signature_path": ref_path,
            "score": score,
            "distance": dist
        })

        if dist < min_dist:
            min_dist = dist
            best_score = score
            best_match = ref_path

    # Sort: most similar first (highest score / lowest distance)
    scores.sort(key=lambda x: x["distance"])

    # Take top 3
    top_scores = scores[:3]
    top_average = round(sum(item["score"] for item in top_scores) / len(top_scores), 2)
    top_avg_dist = round(sum(item["distance"] for item in top_scores) / len(top_scores), 4)

    is_genuine = (min_dist <= DISTANCE_THRESHOLD)
    verdict = "GENUINE" if is_genuine else "FORGED"

    return {
        "best_score": round(best_score, 2),
        "best_distance": round(min_dist, 4),
        "top_average_score": top_average,
        "top_average_distance": top_avg_dist,
        "matched_signature": best_match,
        "number_of_signatures_checked": len(valid_paths),
        "distance_threshold": DISTANCE_THRESHOLD,
        "similarity_threshold": SIMILARITY_THRESHOLD,
        "is_genuine": is_genuine,
        "verdict": verdict,
        "scores": scores
    }

# Alias for loan applicant terminology
verify_against_applicant_signatures = verify_against_employee_signatures