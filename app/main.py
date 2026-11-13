"""
FastAPI application entrypoint for ViralBrandEngine.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.api.v1.endpoints import generate, status, analytics
from app.db import models
from app.auth import require_admin_access, require_api_access


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown lifecycle hook."""
    models.Base.metadata.create_all(bind=__import__("app.db.session", fromlist=["engine"]).engine)
    yield


settings = get_settings()
cors_origins = [origin.strip() for origin in settings.CORS_ORIGINS.split(",") if origin.strip()]

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Autonomous Viral Brand B-Roll & Social Media Engine",
    lifespan=lifespan,
)

# ── CORS ─────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "Content-Type", "X-API-Key"],
)


# ── Security Headers ─────────────────────────────────────────────────────
@app.middleware("http")
async def add_security_headers(request, call_next):
    response = await call_next(request)
    response.headers["X-Frame-Options"] = "SAMEORIGIN"
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Content-Security-Policy"] = (
        "default-src 'self' 'unsafe-inline' 'unsafe-eval' https: data: blob:; "
        "frame-ancestors 'self';"
    )
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    return response

import os
from pathlib import Path
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from sqlalchemy.orm import Session
from app.db.session import get_db
from io import BytesIO
import hashlib
import hmac
import secrets
import smtplib
from email.message import EmailMessage
from datetime import datetime, timedelta, timezone

STATIC_DIR = Path(__file__).parent / "static"
MEDIA_DIR = Path(__file__).parent.parent / "media"
MEDIA_DIR.mkdir(parents=True, exist_ok=True)
(MEDIA_DIR / "renders").mkdir(parents=True, exist_ok=True)

# ── Routers ──────────────────────────────────────────────────────────────
app.include_router(generate.router, prefix="/api/v1", tags=["generate"], dependencies=[Depends(require_api_access)])
app.include_router(status.router, prefix="/api/v1", tags=["status"], dependencies=[Depends(require_api_access)])
app.include_router(analytics.router, prefix="/api/v1", tags=["analytics"], dependencies=[Depends(require_api_access)])

# ── Static & Media Serving ───────────────────────────────────────────────
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
app.mount("/media", StaticFiles(directory=str(MEDIA_DIR)), name="media")


@app.get("/", tags=["ui"])
async def serve_index():
    """Serve the Web Studio Dashboard."""
    index_file = STATIC_DIR / "index.html"
    return FileResponse(index_file)


@app.get("/admin", tags=["ui"])
async def serve_admin_login():
    return FileResponse(STATIC_DIR / "admin.html")


@app.get("/api/v1/default-campaign", tags=["ui"], dependencies=[Depends(require_api_access)])
async def get_default_campaign():
    """Helper for UI to get or create an active campaign ID."""
    from app.db.session import SessionLocal
    from app.db.models import User, Campaign
    import uuid

    db = SessionLocal()
    try:
        camp = db.query(Campaign).first()
        if not camp:
            user = db.query(User).first()
            if not user:
                user = User(id=uuid.uuid4(), email="studio@viralbrand.ai", display_name="Studio User")
                db.add(user)
                db.commit()
                db.refresh(user)
            camp = Campaign(id=uuid.uuid4(), user_id=user.id, name="Default Studio Campaign")
            db.add(camp)
            db.commit()
            db.refresh(camp)
        return {"campaign_id": str(camp.id)}
    finally:
        db.close()


from pydantic import BaseModel
from typing import Optional


class ConfigKeysUpdate(BaseModel):
    openai_api_key: Optional[str] = None
    gemini_api_key: Optional[str] = None
    google_flow_api_key: Optional[str] = None
    luma_api_key: Optional[str] = None
    kling_api_key: Optional[str] = None
    elevenlabs_api_key: Optional[str] = None
    google_tts_api_key: Optional[str] = None
    pexels_api_key: Optional[str] = None
    pixabay_api_key: Optional[str] = None
    meta_access_token: Optional[str] = None
    meta_ig_user_id: Optional[str] = None
    tiktok_access_token: Optional[str] = None


class LoginRequest(BaseModel):
    username: str
    password: str


class RegisterRequest(BaseModel):
    display_name: str
    email: str
    password: str


class VerifyEmailRequest(BaseModel):
    email: str
    code: str


def _hash_password(password: str) -> str:
    salt = secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 120_000).hex()
    return f"pbkdf2_sha256${salt}${digest}"


def _verify_password(password: str, encoded: str) -> bool:
    try:
        _, salt, digest = encoded.split("$", 2)
        candidate = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 120_000).hex()
        return hmac.compare_digest(candidate, digest)
    except (ValueError, TypeError):
        return False


def _send_verification_email(email: str, code: str) -> bool:
    settings = get_settings()
    if not settings.SMTP_HOST or not settings.SMTP_USERNAME or not settings.SMTP_FROM_EMAIL:
        return False
    message = EmailMessage()
    message["Subject"] = "ViralBrandEngine e-posta dogrulama kodu"
    message["From"] = settings.SMTP_FROM_EMAIL
    message["To"] = email
    message.set_content(f"ViralBrandEngine kaydinizi tamamlamak icin dogrulama kodunuz: {code}\nKod 15 dakika gecerlidir.")
    with smtplib.SMTP(settings.SMTP_HOST, settings.SMTP_PORT, timeout=20) as smtp:
        if settings.SMTP_USE_TLS:
            smtp.starttls()
        smtp.login(settings.SMTP_USERNAME, "".join(settings.SMTP_PASSWORD.split()))
        smtp.send_message(message)
    return True


@app.post("/api/v1/auth/login", tags=["auth"])
async def login(payload: LoginRequest):
    """Authenticate the demo administrator or standard user."""
    settings = get_settings()
    if payload.username == settings.ADMIN_USERNAME and payload.password == settings.ADMIN_PASSWORD:
        if not settings.ADMIN_API_KEY:
            raise HTTPException(status_code=503, detail="Administrator API key is not configured.")
        return {
            "authenticated": True,
            "username": payload.username,
            "role": "admin",
            "display_name": "Yönetici",
            "api_key": settings.ADMIN_API_KEY,
        }
    if payload.username == settings.USER_USERNAME and payload.password == settings.USER_PASSWORD:
        if not settings.API_KEY:
            raise HTTPException(status_code=503, detail="User API key is not configured.")
        return {
            "authenticated": True,
            "username": payload.username,
            "role": "user",
            "display_name": "Kullanıcı",
            "api_key": settings.API_KEY,
        }
    raise HTTPException(status_code=401, detail="Kullanıcı adı veya şifre hatalı.")


@app.post("/api/v1/auth/register", tags=["auth"])
async def register(payload: RegisterRequest, db: Session = Depends(get_db)):
    from app.db.models import Account

    email = payload.email.strip().lower()
    if len(payload.password) < 8:
        raise HTTPException(status_code=422, detail="Şifre en az 8 karakter olmalıdır.")
    if db.query(Account).filter(Account.email == email).first():
        raise HTTPException(status_code=409, detail="Bu e-posta adresi zaten kayıtlı.")

    code = f"{secrets.randbelow(900000) + 100000}"
    account = Account(
        email=email,
        display_name=payload.display_name.strip(),
        password_hash=_hash_password(payload.password),
        email_verified=False,
        verification_code=code,
        verification_expires_at=datetime.now(timezone.utc) + timedelta(minutes=15),
    )
    db.add(account)
    db.commit()

    try:
        sent = _send_verification_email(email, code)
    except Exception as exc:
        message = str(exc)
        if "Connection unexpectedly closed" in message or isinstance(exc, smtplib.SMTPServerDisconnected):
            message = (
                "SMTP oturumu kimlik doğrulama sırasında kapandı. "
                "Gmail hesapta uygulama şifresi kullanın ve SMTP_PASSWORD alanının 16 karakter olduğunu kontrol edin."
            )
        elif isinstance(exc, smtplib.SMTPAuthenticationError):
            message = (
                "SMTP kimlik doğrulaması reddedildi. SMTP_USERNAME ve Gmail uygulama şifresini kontrol edin."
            )
        raise HTTPException(status_code=502, detail=f"Dogrulama e-postasi gonderilemedi: {message}")
    if not sent:
        raise HTTPException(
            status_code=503,
            detail="SMTP ayarlari bulunamadi. Administratore SMTP_HOST, SMTP_USERNAME ve SMTP_FROM_EMAIL bilgilerini tanimlatin.",
        )
    return {"registered": True, "email": email, "message": "Dogrulama kodu e-posta adresinize gonderildi."}


@app.post("/api/v1/auth/verify-email", tags=["auth"])
async def verify_email(payload: VerifyEmailRequest, db: Session = Depends(get_db)):
    from app.db.models import Account

    email = payload.email.strip().lower()
    account = db.query(Account).filter(Account.email == email).first()
    if not account:
        raise HTTPException(status_code=404, detail="Hesap bulunamadı.")
    if account.email_verified:
        return {"verified": True, "email": email}
    expires_at = account.verification_expires_at
    if expires_at and expires_at.tzinfo is None:
        expires_at = expires_at.replace(tzinfo=timezone.utc)
    if not expires_at or expires_at < datetime.now(timezone.utc):
        raise HTTPException(status_code=410, detail="Dogrulama kodunun süresi dolmuş.")
    if not hmac.compare_digest(payload.code.strip(), account.verification_code or ""):
        raise HTTPException(status_code=400, detail="Dogrulama kodu hatalı.")
    account.email_verified = True
    account.verification_code = None
    db.commit()
    return {"verified": True, "email": email}


@app.post("/api/v1/auth/user-login", tags=["auth"])
async def user_login(payload: LoginRequest, db: Session = Depends(get_db)):
    from app.db.models import Account

    account = db.query(Account).filter(Account.email == payload.username.strip().lower()).first()
    if not account or not _verify_password(payload.password, account.password_hash):
        raise HTTPException(status_code=401, detail="E-posta veya şifre hatalı.")
    if not account.email_verified:
        raise HTTPException(status_code=403, detail="E-posta adresi henüz doğrulanmadı.")
    return {
        "authenticated": True,
        "username": account.email,
        "role": "user",
        "display_name": account.display_name,
        "api_key": settings.API_KEY,
    }


def _dashboard_report_rows(db):
    from app.db.models import VideoJob
    jobs = db.query(VideoJob).all()
    return [
        {
            "Konu": job.input_topic or "",
            "Durum": job.status.value if job.status else "",
            "Dil": job.target_language or "",
            "Ton": job.brand_tone or "",
            "Sure (sn)": job.target_duration_secs or 0,
            "Olusturma Tarihi": job.created_at.isoformat() if job.created_at else "",
        }
        for job in jobs
    ]


@app.get("/api/v1/reports/pdf", tags=["reports"], dependencies=[Depends(require_admin_access)])
async def download_pdf_report(db: Session = Depends(get_db)):
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
    from reportlab.lib.styles import getSampleStyleSheet

    rows = _dashboard_report_rows(db)
    buffer = BytesIO()
    document = SimpleDocTemplate(buffer, pagesize=A4, title="ViralBrandEngine Raporu")
    styles = getSampleStyleSheet()
    story = [
        Paragraph("ViralBrandEngine Yönetici Raporu", styles["Title"]),
        Paragraph(f"Toplam video kaydı: {len(rows)}", styles["Normal"]),
        Spacer(1, 12),
    ]
    headers = ["Konu", "Durum", "Dil", "Ton", "Sure (sn)", "Olusturma Tarihi"]
    data = [headers] + [[row[header] for header in headers] for row in rows[:100]]
    table = Table(data, repeatRows=1, colWidths=[150, 55, 35, 50, 45, 100])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#6366f1")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
        ("FONTSIZE", (0, 0), (-1, -1), 7),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    story.append(table)
    document.build(story)
    buffer.seek(0)
    return StreamingResponse(buffer, media_type="application/pdf", headers={
        "Content-Disposition": "attachment; filename=viral_brand_engine_raporu.pdf"
    })


@app.get("/api/v1/reports/excel", tags=["reports"], dependencies=[Depends(require_admin_access)])
async def download_excel_report(db: Session = Depends(get_db)):
    from openpyxl import Workbook

    rows = _dashboard_report_rows(db)
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Video Raporu"
    headers = ["Konu", "Durum", "Dil", "Ton", "Sure (sn)", "Olusturma Tarihi"]
    sheet.append(headers)
    for row in rows:
        sheet.append([row["Konu"], row["Durum"], row["Dil"], row["Ton"], row["Sure (sn)"], row["Olusturma Tarihi"]])
    sheet.freeze_panes = "A2"
    sheet.auto_filter.ref = sheet.dimensions
    for column in sheet.columns:
        max_width = min(max(len(str(cell.value or "")) for cell in column) + 2, 50)
        sheet.column_dimensions[column[0].column_letter].width = max_width

    buffer = BytesIO()
    workbook.save(buffer)
    buffer.seek(0)
    return StreamingResponse(
        buffer,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": "attachment; filename=viral_brand_engine_raporu.xlsx"},
    )


@app.get("/api/v1/config/status", tags=["config"], dependencies=[Depends(require_admin_access)])
async def get_config_status():
    """Return configured status for each AI and media provider."""
    from app.config import get_settings
    s = get_settings()
    return {
        "openai_configured": bool(s.OPENAI_API_KEY.strip() if s.OPENAI_API_KEY else False),
        "gemini_configured": bool(s.GEMINI_API_KEY.strip() if s.GEMINI_API_KEY else False),
        "google_flow_configured": bool(s.GOOGLE_FLOW_API_KEY.strip() if s.GOOGLE_FLOW_API_KEY else False),
        "luma_configured": bool(s.LUMA_API_KEY.strip() if s.LUMA_API_KEY else False),
        "kling_configured": bool(s.KLING_API_KEY.strip() if s.KLING_API_KEY else False),
        "elevenlabs_configured": bool(s.ELEVENLABS_API_KEY.strip() if s.ELEVENLABS_API_KEY else False),
        "google_tts_configured": bool(s.GOOGLE_TTS_API_KEY.strip() if s.GOOGLE_TTS_API_KEY else False),
        "pexels_configured": bool(s.PEXELS_API_KEY.strip() if s.PEXELS_API_KEY else False),
        "pixabay_configured": bool(s.PIXABAY_API_KEY.strip() if s.PIXABAY_API_KEY else False),
        "meta_configured": bool(s.META_ACCESS_TOKEN.strip() if s.META_ACCESS_TOKEN else False),
        "tiktok_configured": bool(s.TIKTOK_ACCESS_TOKEN.strip() if s.TIKTOK_ACCESS_TOKEN else False),
    }


@app.post("/api/v1/config/keys", tags=["config"], dependencies=[Depends(require_admin_access)])
async def update_config_keys(payload: ConfigKeysUpdate):
    """Reject runtime secret writes; secrets must be injected by deployment."""
    raise HTTPException(
        status_code=410,
        detail="Runtime secret updates are disabled. Use the deployment secret store.",
    )


@app.get("/health", tags=["infra"])
async def health_check():
    """Basic liveness probe."""
    return {"status": "ok", "service": settings.APP_NAME}
