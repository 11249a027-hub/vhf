from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey
from sqlalchemy.orm import declarative_base, relationship
from datetime import datetime


Base = declarative_base()


class LoanApplicant(Base):
    __tablename__ = "loan_applicants"

    id = Column(Integer, primary_key=True, index=True)
    applicant_id = Column(String, unique=True, nullable=False, index=True)
    full_name = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    signatures = relationship(
        "ApplicantSignature",
        back_populates="applicant",
        cascade="all, delete-orphan"
    )
    applications = relationship(
        "LoanApplication",
        back_populates="applicant",
        cascade="all, delete-orphan"
    )


class LoanApplication(Base):
    __tablename__ = "loan_applications"

    id = Column(Integer, primary_key=True, index=True)
    application_id = Column(String, unique=True, nullable=False, index=True)
    applicant_id = Column(
        String,
        ForeignKey("loan_applicants.applicant_id"),
        nullable=False,
        index=True
    )
    loan_type = Column(String, default="Home Loan")
    loan_amount = Column(Float, default=2500000.0)
    branch = Column(String, default="Mumbai Main")
    status = Column(String, default="PENDING")
    created_at = Column(DateTime, default=datetime.utcnow)

    applicant = relationship(
        "LoanApplicant",
        back_populates="applications"
    )


class ApplicantSignature(Base):
    __tablename__ = "applicant_signatures"

    id = Column(Integer, primary_key=True, index=True)
    applicant_id = Column(
        String,
        ForeignKey("loan_applicants.applicant_id"),
        nullable=False,
        index=True
    )
    sample_no = Column(Integer, nullable=False)
    signature_path = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    applicant = relationship(
        "LoanApplicant",
        back_populates="signatures"
    )


class VerificationLog(Base):
    __tablename__ = "verification_logs"

    id = Column(Integer, primary_key=True, index=True)
    applicant_id = Column(String, nullable=False, index=True)
    application_id = Column(String, nullable=True, index=True)
    similarity_score = Column(Float, nullable=False)
    result = Column(String, nullable=False)
    risk_level = Column(String, default="LOW")
    review_status = Column(String, default="PENDING_REVIEW")
    reviewer_remarks = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)