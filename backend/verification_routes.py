from fastapi import (
    APIRouter,
    UploadFile,
    File,
    Form,
    Depends,
    HTTPException
)
from sqlalchemy.orm import Session
from typing import Optional
import os
import shutil
import uuid

from backend.database import SessionLocal
from backend.models import (
    LoanApplicant,
    ApplicantSignature,
    VerificationLog
)

from ml.verifier import (
    verify_against_employee_signatures
)


router = APIRouter(tags=["Verification"])


VERIFY_UPLOAD_DIR = "signatures/verify_uploads"
os.makedirs(VERIFY_UPLOAD_DIR, exist_ok=True)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@router.post("/verify-signature")
def verify_signature(
    applicant_id: Optional[str] = Form(None),
    employee_id: Optional[str] = Form(None),
    application_id: Optional[str] = Form(None),
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
    # 1. Find applicant
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
    # 2. Get ALL stored reference signatures
    # --------------------------------------------------
    applicant_signatures = (
        db.query(ApplicantSignature)
        .filter(ApplicantSignature.applicant_id == target_id)
        .order_by(ApplicantSignature.sample_no.asc())
        .all()
    )

    if not applicant_signatures:
        raise HTTPException(
            status_code=400,
            detail="No reference signatures found for this applicant"
        )

    # --------------------------------------------------
    # 3. Extract paths
    # --------------------------------------------------
    signature_paths = [
        sig.signature_path
        for sig in applicant_signatures
    ]

    # --------------------------------------------------
    # 4. Save uploaded test signature
    # --------------------------------------------------
    extension = os.path.splitext(file.filename or "")[1].lower()
    if extension not in [".png", ".jpg", ".jpeg"]:
        raise HTTPException(
            status_code=400,
            detail="Only PNG, JPG and JPEG files are allowed"
        )

    filename = f"{target_id}_{uuid.uuid4().hex}{extension}"
    temp_path = os.path.join(VERIFY_UPLOAD_DIR, filename)

    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        # --------------------------------------------------
        # 5. Run Siamese verification
        # --------------------------------------------------
        result = verify_against_employee_signatures(
            temp_path,
            signature_paths
        )

        best_score = result["best_score"]
        best_dist = result.get("best_distance", 0.0)
        dist_threshold = result.get("distance_threshold", 0.4230)
        top_average_score = result["top_average_score"]

        # --------------------------------------------------
        # 6. Calibrated Decision Threshold & Semantic Risk Mapping (V4)
        # Direction: distance <= threshold -> GENUINE, distance > threshold -> FORGED
        # --------------------------------------------------
        if best_dist <= dist_threshold:
            verdict = "GENUINE"
            risk_level = "LOW"
            recommendation = "VERIFIED"
            review_status = "CLEARED"
        elif best_dist <= dist_threshold * 1.35:
            verdict = "FORGED"
            risk_level = "MEDIUM"
            recommendation = "REVIEW_REQUIRED"
            review_status = "PENDING_REVIEW"
        elif best_dist <= dist_threshold * 2.0:
            verdict = "FORGED"
            risk_level = "HIGH"
            recommendation = "SUSPICIOUS_SIGNATURE"
            review_status = "PENDING_REVIEW"
        else:
            verdict = "FORGED"
            risk_level = "CRITICAL"
            recommendation = "FRAUD_SUSPECTED"
            review_status = "PENDING_REVIEW"

        # --------------------------------------------------
        # 7. Save verification log
        # --------------------------------------------------
        log = VerificationLog(
            applicant_id=applicant.applicant_id,
            application_id=application_id,
            similarity_score=best_score,
            result=verdict,
            risk_level=risk_level,
            review_status=review_status
        )
        db.add(log)
        db.commit()
        db.refresh(log)

        # --------------------------------------------------
        # 8. Return complete result
        # --------------------------------------------------
        return {
            "status": "success",
            "log_id": log.id,
            "applicant_id": applicant.applicant_id,
            "applicant_name": applicant.full_name,
            "application_id": application_id,
            "similarity_score": best_score,
            "accuracy_score": best_score,
            "distance": best_dist,
            "distance_threshold": dist_threshold,
            "risk_level": risk_level,
            "recommendation": recommendation,
            "review_status": review_status,
            "top_3_average_score": top_average_score,
            "result": verdict,
            "matched_signature": result["matched_signature"],
            "signatures_checked": result["number_of_signatures_checked"],
            "signature_scores": result["scores"],
            "model_version": "Siamese-V4-WriterIdentity",
            "created_at": log.created_at
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Verification failed: {str(e)}")
    finally:
        # --------------------------------------------------
        # 9. Delete temporary test image
        # --------------------------------------------------
        if os.path.exists(temp_path):
            os.remove(temp_path)


@router.get("/dashboard/kpis")
def get_dashboard_kpis(db: Session = Depends(get_db)):
    from backend.models import LoanApplication
    total_apps = db.query(LoanApplication).count()
    total_applicants = db.query(LoanApplicant).count()
    signatures_verified = db.query(VerificationLog).count()
    genuine_count = db.query(VerificationLog).filter(VerificationLog.result == "GENUINE").count()
    suspicious_forged_count = db.query(VerificationLog).filter(VerificationLog.result == "FORGED").count()
    pending_reviews_count = db.query(VerificationLog).filter(
        VerificationLog.review_status.in_(["PENDING_REVIEW", "UNDER_REVIEW"])
    ).count()

    recent_logs = (
        db.query(VerificationLog)
        .order_by(VerificationLog.created_at.desc())
        .limit(6)
        .all()
    )

    recent_activity = []
    for r in recent_logs:
        app = db.query(LoanApplicant).filter(LoanApplicant.applicant_id == r.applicant_id).first()
        recent_activity.append({
            "id": r.id,
            "applicant_id": r.applicant_id,
            "applicant_name": app.full_name if app else "Unknown",
            "application_id": r.application_id or "N/A",
            "similarity_score": r.similarity_score,
            "result": r.result,
            "risk_level": r.risk_level or ("LOW" if r.result == "GENUINE" else "HIGH"),
            "review_status": r.review_status or "PENDING_REVIEW",
            "created_at": r.created_at
        })

    total_signatures = db.query(ApplicantSignature).count()

    return {
        "total_applications": total_apps,
        "total_applicants": total_applicants,
        "total_employees": total_applicants,
        "total_signatures": total_signatures,
        "signatures_verified": signatures_verified,
        "total_verifications": signatures_verified,
        "genuine_count": genuine_count,
        "suspicious_forged_count": suspicious_forged_count,
        "pending_reviews_count": pending_reviews_count,
        "model_version": "Siamese-V3-VHF",
        "calibrated_threshold": 0.3817,
        "recent_activity": recent_activity
    }


@router.get("/verification-logs")
def list_verification_logs(
    applicant_id: Optional[str] = None,
    result: Optional[str] = None,
    risk_level: Optional[str] = None,
    review_status: Optional[str] = None,
    limit: int = 50,
    db: Session = Depends(get_db)
):
    query = db.query(VerificationLog)
    if applicant_id:
        query = query.filter(VerificationLog.applicant_id.ilike(f"%{applicant_id}%"))
    if result:
        query = query.filter(VerificationLog.result == result)
    if risk_level:
        query = query.filter(VerificationLog.risk_level == risk_level)
    if review_status:
        query = query.filter(VerificationLog.review_status == review_status)

    logs = query.order_by(VerificationLog.created_at.desc()).limit(limit).all()

    items = []
    for log in logs:
        applicant = db.query(LoanApplicant).filter(LoanApplicant.applicant_id == log.applicant_id).first()
        items.append({
            "id": log.id,
            "applicant_id": log.applicant_id,
            "applicant_name": applicant.full_name if applicant else "Unknown",
            "application_id": log.application_id or "N/A",
            "similarity_score": log.similarity_score,
            "result": log.result,
            "risk_level": log.risk_level or ("LOW" if log.result == "GENUINE" else "HIGH"),
            "review_status": log.review_status or "PENDING_REVIEW",
            "reviewer_remarks": log.reviewer_remarks,
            "created_at": log.created_at
        })
    return items


@router.get("/fraud-review/cases")
def list_fraud_review_cases(
    status: Optional[str] = None,
    db: Session = Depends(get_db)
):
    query = db.query(VerificationLog)
    if status:
        query = query.filter(VerificationLog.review_status == status)
    else:
        query = query.filter(
            (VerificationLog.review_status.in_(["PENDING_REVIEW", "UNDER_REVIEW", "CONFIRMED_SUSPICIOUS", "CONFIRMED_FRAUD"])) |
            (VerificationLog.result == "FORGED")
        )

    cases = query.order_by(VerificationLog.created_at.desc()).limit(50).all()

    items = []
    for c in cases:
        applicant = db.query(LoanApplicant).filter(LoanApplicant.applicant_id == c.applicant_id).first()
        from backend.models import LoanApplication, ApplicantSignature
        loan_app = None
        if c.application_id:
            loan_app = db.query(LoanApplication).filter(LoanApplication.application_id == c.application_id).first()

        signatures = db.query(ApplicantSignature).filter(
            ApplicantSignature.applicant_id == c.applicant_id
        ).order_by(ApplicantSignature.sample_no.asc()).all()

        items.append({
            "id": c.id,
            "applicant_id": c.applicant_id,
            "applicant_name": applicant.full_name if applicant else "Unknown",
            "application_id": c.application_id or "N/A",
            "loan_type": loan_app.loan_type if loan_app else "Home Loan",
            "loan_amount": loan_app.loan_amount if loan_app else 2500000.0,
            "branch": loan_app.branch if loan_app else "Mumbai Main",
            "similarity_score": c.similarity_score,
            "result": c.result,
            "risk_level": c.risk_level or ("HIGH" if c.result == "FORGED" else "LOW"),
            "review_status": c.review_status or "PENDING_REVIEW",
            "reviewer_remarks": c.reviewer_remarks or "",
            "created_at": c.created_at,
            "reference_signatures": [
                {
                    "sample_no": s.sample_no,
                    "url": f"/{s.signature_path.replace(chr(92), '/')}" if not s.signature_path.startswith("/") else s.signature_path
                }
                for s in signatures
            ]
        })
    return items


@router.post("/fraud-review/{log_id}")
def update_fraud_review_case(
    log_id: int,
    action: str = Form(...),
    remarks: Optional[str] = Form(""),
    db: Session = Depends(get_db)
):
    valid_actions = ["CLEARED", "CONFIRMED_SUSPICIOUS", "CONFIRMED_FRAUD", "UNDER_REVIEW"]
    if action not in valid_actions:
        raise HTTPException(status_code=400, detail=f"Invalid review action. Allowed: {valid_actions}")

    log = db.query(VerificationLog).filter(VerificationLog.id == log_id).first()
    if not log:
        raise HTTPException(status_code=404, detail="Verification case not found")

    log.review_status = action
    log.reviewer_remarks = remarks
    db.commit()
    db.refresh(log)

    return {
        "status": "success",
        "log_id": log.id,
        "applicant_id": log.applicant_id,
        "review_status": log.review_status,
        "reviewer_remarks": log.reviewer_remarks,
        "updated_at": log.updated_at
    }


@router.get("/verification-history/{applicant_id}")
def get_verification_history(
    applicant_id: str,
    limit: int = 50,
    db: Session = Depends(get_db)
):
    applicant = (
        db.query(LoanApplicant)
        .filter(LoanApplicant.applicant_id == applicant_id)
        .first()
    )
    if not applicant:
        raise HTTPException(status_code=404, detail="Applicant not found")

    logs = (
        db.query(VerificationLog)
        .filter(VerificationLog.applicant_id == applicant_id)
        .order_by(VerificationLog.created_at.desc())
        .limit(limit)
        .all()
    )

    return {
        "applicant_id": applicant_id,
        "applicant_name": applicant.full_name,
        "total_verifications": len(logs),
        "history": [
            {
                "id": log.id,
                "application_id": log.application_id,
                "similarity_score": log.similarity_score,
                "result": log.result,
                "risk_level": log.risk_level or ("LOW" if log.result == "GENUINE" else "HIGH"),
                "review_status": log.review_status or "PENDING_REVIEW",
                "reviewer_remarks": log.reviewer_remarks,
                "created_at": log.created_at
            }
            for log in logs
        ]
    }