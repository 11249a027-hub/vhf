from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import Optional, List
import os

from backend.database import SessionLocal
from backend.models import LoanApplicant, ApplicantSignature, VerificationLog, LoanApplication

router = APIRouter(prefix="/employees", tags=["Employees"])


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@router.get("")
@router.get("/")
def list_employees(
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
    employees = query.order_by(LoanApplicant.created_at.desc()).all()

    result = []
    for emp in employees:
        sig_count = db.query(ApplicantSignature).filter(
            ApplicantSignature.applicant_id == emp.applicant_id
        ).count()
        latest_verif = db.query(VerificationLog).filter(
            VerificationLog.applicant_id == emp.applicant_id
        ).order_by(VerificationLog.created_at.desc()).first()

        # Derive clean designation and department for display
        designation = "Senior Underwriter" if "EMP" in emp.applicant_id else "Loan Officer"
        department = "Credit & Risk Operations"

        result.append({
            "employee_id": emp.applicant_id,
            "applicant_id": emp.applicant_id,
            "name": emp.full_name,
            "full_name": emp.full_name,
            "designation": designation,
            "department": department,
            "total_signatures": sig_count,
            "created_at": emp.created_at,
            "latest_verification": {
                "result": latest_verif.result if latest_verif else None,
                "similarity_score": latest_verif.similarity_score if latest_verif else None,
                "risk_level": latest_verif.risk_level if latest_verif else None,
                "created_at": latest_verif.created_at if latest_verif else None,
            } if latest_verif else None
        })

    return result


@router.post("/register")
@router.post("")
def register_employee(
    employee_id: str,
    name: str,
    designation: Optional[str] = "Loan Officer",
    department: Optional[str] = "Credit & Risk Operations",
    db: Session = Depends(get_db)
):
    clean_id = employee_id.strip()
    clean_name = name.strip()

    if not clean_id or not clean_name:
        raise HTTPException(status_code=400, detail="employee_id and name are required")

    existing = db.query(LoanApplicant).filter(
        LoanApplicant.applicant_id == clean_id
    ).first()
    if existing:
        raise HTTPException(status_code=400, detail="Employee ID already exists")

    new_emp = LoanApplicant(
        applicant_id=clean_id,
        full_name=clean_name
    )
    db.add(new_emp)
    db.commit()
    db.refresh(new_emp)

    return {
        "status": "success",
        "employee_id": new_emp.applicant_id,
        "name": new_emp.full_name,
        "designation": designation,
        "department": department,
        "created_at": new_emp.created_at
    }


@router.get("/{employee_id}")
def get_employee_details(
    employee_id: str,
    db: Session = Depends(get_db)
):
    emp = db.query(LoanApplicant).filter(
        LoanApplicant.applicant_id == employee_id
    ).first()

    if not emp:
        raise HTTPException(status_code=404, detail="Employee not found")

    sig_count = db.query(ApplicantSignature).filter(
        ApplicantSignature.applicant_id == employee_id
    ).count()

    signatures = db.query(ApplicantSignature).filter(
        ApplicantSignature.applicant_id == employee_id
    ).order_by(ApplicantSignature.sample_no.asc()).all()

    recent_verifications = db.query(VerificationLog).filter(
        VerificationLog.applicant_id == employee_id
    ).order_by(VerificationLog.created_at.desc()).limit(10).all()

    return {
        "employee_id": emp.applicant_id,
        "applicant_id": emp.applicant_id,
        "name": emp.full_name,
        "full_name": emp.full_name,
        "designation": "Senior Underwriter" if "EMP" in emp.applicant_id else "Loan Officer",
        "department": "Credit & Risk Operations",
        "total_signatures": sig_count,
        "created_at": emp.created_at,
        "signatures": [
            {
                "sample_no": s.sample_no,
                "signature_path": s.signature_path,
                "url": f"/{s.signature_path.replace(chr(92), '/')}" if not s.signature_path.startswith("/") else s.signature_path,
                "created_at": s.created_at
            }
            for s in signatures
        ],
        "verification_history": [
            {
                "id": v.id,
                "similarity_score": v.similarity_score,
                "result": v.result,
                "risk_level": v.risk_level,
                "review_status": v.review_status,
                "created_at": v.created_at
            }
            for v in recent_verifications
        ]
    }


@router.get("/{employee_id}/signatures")
def get_employee_signatures(
    employee_id: str,
    db: Session = Depends(get_db)
):
    emp = db.query(LoanApplicant).filter(
        LoanApplicant.applicant_id == employee_id
    ).first()

    if not emp:
        raise HTTPException(status_code=404, detail="Employee not found")

    signatures = db.query(ApplicantSignature).filter(
        ApplicantSignature.applicant_id == employee_id
    ).order_by(ApplicantSignature.sample_no.asc()).all()

    return [
        {
            "id": s.id,
            "sample_no": s.sample_no,
            "signature_path": s.signature_path,
            "url": f"/{s.signature_path.replace(chr(92), '/')}" if not s.signature_path.startswith("/") else s.signature_path,
            "created_at": s.created_at
        }
        for s in signatures
    ]
