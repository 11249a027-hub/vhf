from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import Optional, List

from backend.database import SessionLocal
from backend.models import LoanApplicant, LoanApplication, ApplicantSignature, VerificationLog

router = APIRouter(prefix="/applicants", tags=["Loan Applicants"])


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@router.get("")
@router.get("/")
def list_applicants(
    search: Optional[str] = Query(None),
    db: Session = Depends(get_db)
):
    query = db.query(LoanApplicant)
    if search:
        s = f"%{search}%"
        query = query.filter(
            (LoanApplicant.applicant_id.ilike(s)) |
            (LoanApplicant.full_name.ilike(s))
        )
    applicants = query.order_by(LoanApplicant.created_at.desc()).all()

    result = []
    for app in applicants:
        sig_count = db.query(ApplicantSignature).filter(
            ApplicantSignature.applicant_id == app.applicant_id
        ).count()
        app_count = db.query(LoanApplication).filter(
            LoanApplication.applicant_id == app.applicant_id
        ).count()
        latest_verif = db.query(VerificationLog).filter(
            VerificationLog.applicant_id == app.applicant_id
        ).order_by(VerificationLog.created_at.desc()).first()

        result.append({
            "applicant_id": app.applicant_id,
            "full_name": app.full_name,
            "created_at": app.created_at,
            "total_signatures": sig_count,
            "total_applications": app_count,
            "latest_verification": {
                "result": latest_verif.result if latest_verif else None,
                "similarity_score": latest_verif.similarity_score if latest_verif else None,
                "risk_level": latest_verif.risk_level if latest_verif else None,
                "created_at": latest_verif.created_at if latest_verif else None,
            } if latest_verif else None
        })

    return result


@router.get("/applications/list")
def list_all_applications(
    status: Optional[str] = Query(None),
    db: Session = Depends(get_db)
):
    query = db.query(LoanApplication)
    if status:
        query = query.filter(LoanApplication.status == status)
    applications = query.order_by(LoanApplication.created_at.desc()).all()

    results = []
    for a in applications:
        applicant = db.query(LoanApplicant).filter(
            LoanApplicant.applicant_id == a.applicant_id
        ).first()
        latest_verif = db.query(VerificationLog).filter(
            (VerificationLog.application_id == a.application_id) |
            (VerificationLog.applicant_id == a.applicant_id)
        ).order_by(VerificationLog.created_at.desc()).first()

        results.append({
            "application_id": a.application_id,
            "applicant_id": a.applicant_id,
            "applicant_name": applicant.full_name if applicant else "Unknown",
            "loan_type": a.loan_type or "Home Loan",
            "loan_amount": a.loan_amount or 2500000.0,
            "branch": a.branch or "Mumbai Main",
            "status": a.status,
            "created_at": a.created_at,
            "verification_status": latest_verif.result if latest_verif else "UNVERIFIED",
            "similarity_score": latest_verif.similarity_score if latest_verif else None,
            "risk_level": latest_verif.risk_level if latest_verif else None,
        })
    return results


@router.post("/register")
def register_applicant(
    applicant_id: str,
    full_name: str,
    application_id: Optional[str] = None,
    loan_type: Optional[str] = "Home Loan",
    loan_amount: Optional[float] = 2500000.0,
    branch: Optional[str] = "Mumbai Main",
    db: Session = Depends(get_db)
):
    existing = db.query(LoanApplicant).filter(
        LoanApplicant.applicant_id == applicant_id
    ).first()
    if existing:
        raise HTTPException(status_code=400, detail="Applicant already exists")

    applicant = LoanApplicant(
        applicant_id=applicant_id,
        full_name=full_name
    )
    db.add(applicant)
    db.commit()
    db.refresh(applicant)

    created_app_id = None
    if application_id:
        loan_app = LoanApplication(
            application_id=application_id,
            applicant_id=applicant.applicant_id,
            loan_type=loan_type or "Home Loan",
            loan_amount=loan_amount if loan_amount is not None else 2500000.0,
            branch=branch or "Mumbai Main",
            status="PENDING"
        )
        db.add(loan_app)
        db.commit()
        created_app_id = loan_app.application_id

    return {
        "status": "Applicant Registered Successfully",
        "applicant_id": applicant.applicant_id,
        "full_name": applicant.full_name,
        "application_id": created_app_id
    }


@router.post("/applications/create")
def create_loan_application(
    applicant_id: str,
    application_id: str,
    loan_type: Optional[str] = "Home Loan",
    loan_amount: Optional[float] = 2500000.0,
    branch: Optional[str] = "Mumbai Main",
    status: str = "PENDING",
    db: Session = Depends(get_db)
):
    applicant = db.query(LoanApplicant).filter(
        LoanApplicant.applicant_id == applicant_id
    ).first()
    if not applicant:
        raise HTTPException(status_code=404, detail="Applicant not found")

    existing_app = db.query(LoanApplication).filter(
        LoanApplication.application_id == application_id
    ).first()
    if existing_app:
        raise HTTPException(status_code=400, detail="Application ID already exists")

    loan_app = LoanApplication(
        application_id=application_id,
        applicant_id=applicant_id,
        loan_type=loan_type or "Home Loan",
        loan_amount=loan_amount if loan_amount is not None else 2500000.0,
        branch=branch or "Mumbai Main",
        status=status
    )
    db.add(loan_app)
    db.commit()
    db.refresh(loan_app)

    return {
        "status": "Application Created Successfully",
        "application_id": loan_app.application_id,
        "applicant_id": loan_app.applicant_id,
        "loan_type": loan_app.loan_type,
        "loan_amount": loan_app.loan_amount,
        "branch": loan_app.branch,
        "application_status": loan_app.status
    }


@router.get("/{applicant_id}")
def get_applicant_profile(
    applicant_id: str,
    db: Session = Depends(get_db)
):
    applicant = db.query(LoanApplicant).filter(
        LoanApplicant.applicant_id == applicant_id
    ).first()
    if not applicant:
        raise HTTPException(status_code=404, detail="Applicant not found")

    sig_count = db.query(ApplicantSignature).filter(
        ApplicantSignature.applicant_id == applicant_id
    ).count()

    apps = db.query(LoanApplication).filter(
        LoanApplication.applicant_id == applicant_id
    ).all()

    signatures = db.query(ApplicantSignature).filter(
        ApplicantSignature.applicant_id == applicant_id
    ).order_by(ApplicantSignature.sample_no.asc()).all()

    recent_verifications = db.query(VerificationLog).filter(
        VerificationLog.applicant_id == applicant_id
    ).order_by(VerificationLog.created_at.desc()).limit(10).all()

    return {
        "applicant_id": applicant.applicant_id,
        "full_name": applicant.full_name,
        "total_signatures": sig_count,
        "created_at": applicant.created_at,
        "signatures": [
            {
                "sample_no": s.sample_no,
                "signature_path": s.signature_path,
                "url": f"/{s.signature_path.replace(chr(92), '/')}" if not s.signature_path.startswith("/") else s.signature_path,
                "created_at": s.created_at
            }
            for s in signatures
        ],
        "applications": [
            {
                "application_id": a.application_id,
                "loan_type": a.loan_type or "Home Loan",
                "loan_amount": a.loan_amount or 2500000.0,
                "branch": a.branch or "Mumbai Main",
                "status": a.status,
                "created_at": a.created_at
            }
            for a in apps
        ],
        "verification_history": [
            {
                "id": v.id,
                "application_id": v.application_id,
                "similarity_score": v.similarity_score,
                "result": v.result,
                "risk_level": v.risk_level,
                "review_status": v.review_status,
                "created_at": v.created_at
            }
            for v in recent_verifications
        ]
    }
