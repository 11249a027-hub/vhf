from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import Optional
import os
import shutil

from backend.database import SessionLocal
from backend.models import LoanApplicant, ApplicantSignature


router = APIRouter(tags=["Signatures"])


UPLOAD_DIR = "signatures/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@router.post("/upload-signature")
def upload_signature(
    applicant_id: Optional[str] = Form(None),
    employee_id: Optional[str] = Form(None),
    signature_no: int = Form(...),
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    target_id = applicant_id or employee_id
    if not target_id:
        raise HTTPException(
            status_code=400,
            detail="applicant_id is required"
        )

    # --------------------------------------------------
    # 1. Validate signature number
    # --------------------------------------------------
    if signature_no < 1 or signature_no > 10:
        raise HTTPException(
            status_code=400,
            detail="signature_no must be between 1 and 10"
        )

    # --------------------------------------------------
    # 2. Find applicant
    # --------------------------------------------------
    applicant = (
        db.query(LoanApplicant)
        .filter(LoanApplicant.applicant_id == target_id)
        .first()
    )

    if not applicant:
        raise HTTPException(
            status_code=404,
            detail="Applicant not found"
        )

    # --------------------------------------------------
    # 3. Validate file
    # --------------------------------------------------
    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="No signature file provided"
        )

    # --------------------------------------------------
    # 4. Create safe filename
    # --------------------------------------------------
    extension = os.path.splitext(file.filename)[1].lower()
    if extension not in [".png", ".jpg", ".jpeg"]:
        raise HTTPException(
            status_code=400,
            detail="Only PNG, JPG and JPEG files are allowed"
        )

    filename = f"{target_id}_sign{signature_no}{extension}"
    save_path = os.path.join(UPLOAD_DIR, filename)

    # --------------------------------------------------
    # 5. Save file
    # --------------------------------------------------
    with open(save_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # --------------------------------------------------
    # 6. Check whether this sample already exists
    # --------------------------------------------------
    existing_signature = (
        db.query(ApplicantSignature)
        .filter(
            ApplicantSignature.applicant_id == target_id,
            ApplicantSignature.sample_no == signature_no
        )
        .first()
    )

    # --------------------------------------------------
    # 7. Update existing sample
    # --------------------------------------------------
    if existing_signature:
        existing_signature.signature_path = save_path
        db.commit()
        db.refresh(existing_signature)
        return {
            "status": "updated",
            "applicant_id": target_id,
            "employee_id": target_id,
            "signature_no": signature_no,
            "saved_path": save_path
        }

    # --------------------------------------------------
    # 8. Create new sample
    # --------------------------------------------------
    new_signature = ApplicantSignature(
        applicant_id=target_id,
        sample_no=signature_no,
        signature_path=save_path
    )
    db.add(new_signature)
    db.commit()
    db.refresh(new_signature)

    # --------------------------------------------------
    # 9. Count uploaded signatures
    # --------------------------------------------------
    total_signatures = (
        db.query(ApplicantSignature)
        .filter(
            ApplicantSignature.applicant_id == target_id
        )
        .count()
    )

    return {
        "status": "uploaded",
        "applicant_id": target_id,
        "employee_id": target_id,
        "signature_no": signature_no,
        "saved_path": save_path,
        "total_signatures": total_signatures
    }


@router.get("/applicants/{applicant_id}/signatures")
def get_applicant_signatures(
    applicant_id: str,
    db: Session = Depends(get_db)
):
    signatures = (
        db.query(ApplicantSignature)
        .filter(ApplicantSignature.applicant_id == applicant_id)
        .order_by(ApplicantSignature.sample_no.asc())
        .all()
    )

    return [
        {
            "id": s.id,
            "applicant_id": s.applicant_id,
            "sample_no": s.sample_no,
            "signature_path": s.signature_path,
            "url": f"/{s.signature_path.replace(chr(92), '/')}" if not s.signature_path.startswith("/") else s.signature_path,
            "created_at": s.created_at
        }
        for s in signatures
    ]