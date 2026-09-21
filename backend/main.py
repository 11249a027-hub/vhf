from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os

from backend.applicant_routes import router as applicant_router
from backend.employee_routes import router as employee_router
from backend.signature_routes import router as signature_router
from backend.verification_routes import router as verification_router


app = FastAPI(
    title="VHF Signature Verification API"
)

# Enable CORS for enterprise frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount signatures directory for visual inspection of reference and submitted signatures
signatures_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "signatures")
if os.path.exists(signatures_dir):
    app.mount("/signatures", StaticFiles(directory=signatures_dir), name="signatures")


@app.get("/")
def root():
    return {
        "status": "VHF Signature Verification API running",
        "product": "VHF Signature Fraud Detection",
        "organization": "Vastu Housing Finance"
    }


app.include_router(applicant_router)
app.include_router(employee_router)
app.include_router(signature_router)
app.include_router(verification_router)