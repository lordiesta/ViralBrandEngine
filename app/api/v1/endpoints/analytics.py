"""
Analytics & Data Collection API endpoints for ViralBrandEngine.

These endpoints are designed for:
  - Teacher/instructor live demonstrations and data collection
  - REST API concept showcasing (full CRUD, filtering, aggregation)
  - JSON dataset export for data analysis
  - Real-time system monitoring

Endpoints:
  GET  /api/v1/analytics/summary          → Aggregated platform-wide statistics
  GET  /api/v1/analytics/jobs             → Full job dataset (paginated, filterable)
  GET  /api/v1/analytics/jobs/{job_id}    → Single job detail with full metadata
  POST /api/v1/analytics/query            → Filtered/sorted job query
  GET  /api/v1/analytics/providers        → AI provider usage breakdown
  GET  /api/v1/analytics/export           → Full JSON export (for data analysis)
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field
from sqlalchemy import func, case
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.db.models import (
    VideoJob,
    JobStatus,
    Campaign,
    User,
    ProductEvent,
    CustomerFeedback,
    ContentCost,
    SocialPerformance,
    SocialPost,
)
from app.config import get_settings

settings = get_settings()
router = APIRouter()


# ─────────────────────────────────────────────────────────────────────────────
# Response Models
# ─────────────────────────────────────────────────────────────────────────────

class JobSummaryItem(BaseModel):
    """Compact job record for dataset/list responses."""
    job_id: str
    topic: str
    language: str
    tone: str
    duration_secs: int
    status: str
    created_at: Optional[str]
    completed_at: Optional[str]
    processing_time_secs: Optional[float]
    has_audio: bool
    has_video: bool
    video_url: Optional[str] = None
    audio_url: Optional[str] = None
    broll_count: int
    script_provider: Optional[str]
    tts_provider: Optional[str]
    broll_provider: Optional[str]

    class Config:
        from_attributes = True


class VideoListItem(BaseModel):
    """Lightweight item containing just video info and direct watch links."""
    job_id: str
    topic: str
    duration_secs: int
    language: str
    tone: str
    status: str
    watch_url: str
    audio_url: Optional[str] = None
    created_at: Optional[str] = None


class VideoListResponse(BaseModel):
    total_videos: int
    description: str
    videos: List[VideoListItem]


class JobDetailItem(JobSummaryItem):
    """Full job record including script data and artifact paths."""
    viral_hook: Optional[str]
    voiceover_script: Optional[str]
    broll_prompts: Optional[List[str]]
    audio_path: Optional[str]
    render_path: Optional[str]
    error_message: Optional[str]
    subtitles_srt: Optional[str]
    caption: Optional[str]
    hashtags: Optional[List[str]]


class ProviderStats(BaseModel):
    """Usage statistics for a single AI/service provider."""
    provider_name: str
    jobs_handled: int
    success_rate_pct: float
    total_seconds_generated: int


class AnalyticsSummary(BaseModel):
    """Platform-wide aggregated statistics."""
    generated_at: str
    platform: str
    version: str
    total_jobs: int
    completed_jobs: int
    failed_jobs: int
    pending_jobs: int
    in_progress_jobs: int
    success_rate_pct: float
    total_video_seconds_generated: int
    languages_used: List[str]
    tones_used: List[str]
    avg_duration_secs: float
    top_topics: List[str]
    provider_breakdown: Dict[str, Any]
    configured_services: Dict[str, bool]


class QueryRequest(BaseModel):
    """Flexible filter/sort request for the analytics query endpoint."""
    status: Optional[str] = Field(None, description="Filter by job status (PENDING, COMPLETED, FAILED, ...)")
    language: Optional[str] = Field(None, description="Filter by target language (tr, en, ...)")
    topic_contains: Optional[str] = Field(None, description="Filter topics containing this string")
    min_duration: Optional[int] = Field(None, description="Minimum video duration in seconds")
    max_duration: Optional[int] = Field(None, description="Maximum video duration in seconds")
    from_date: Optional[str] = Field(None, description="ISO 8601 date string (inclusive)")
    to_date: Optional[str] = Field(None, description="ISO 8601 date string (inclusive)")
    sort_by: str = Field(default="created_at", description="Sort field: created_at | duration | status")
    sort_order: str = Field(default="desc", description="asc or desc")
    limit: int = Field(default=50, ge=1, le=500)
    offset: int = Field(default=0, ge=0)


class QueryResponse(BaseModel):
    total_count: int
    returned_count: int
    filters_applied: Dict[str, Any]
    jobs: List[JobSummaryItem]


class ProductEventRequest(BaseModel):
    event_name: str = Field(..., min_length=1, max_length=128)
    user_id: Optional[uuid.UUID] = None
    campaign_id: Optional[uuid.UUID] = None
    job_id: Optional[uuid.UUID] = None
    properties: Optional[Dict[str, Any]] = None


class FeedbackRequest(BaseModel):
    feedback_type: str = Field(..., pattern="^(NPS|CSAT)$")
    score: int
    comment: Optional[str] = Field(None, max_length=2000)
    user_id: Optional[uuid.UUID] = None
    job_id: Optional[uuid.UUID] = None


class ContentCostRequest(BaseModel):
    job_id: uuid.UUID
    provider: str = Field(..., min_length=1, max_length=128)
    amount: float = Field(..., ge=0)
    currency: str = Field(default="USD", min_length=3, max_length=8)


class SocialPerformanceRequest(BaseModel):
    social_post_id: uuid.UUID
    views: int = Field(default=0, ge=0)
    likes: int = Field(default=0, ge=0)
    comments: int = Field(default=0, ge=0)
    shares: int = Field(default=0, ge=0)
    saves: int = Field(default=0, ge=0)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _job_to_summary(job: VideoJob) -> JobSummaryItem:
    """Convert a VideoJob ORM row to a JobSummaryItem."""
    sd = job.script_data or {}
    processing_time = None
    if job.completed_at and job.created_at:
        processing_time = round(
            (job.completed_at - job.created_at).total_seconds(), 1
        )

    broll_paths = job.broll_paths or []
    broll_count = len(broll_paths) if isinstance(broll_paths, list) else 0

    video_url = f"http://127.0.0.1:8000/media/renders/{job.id}.mp4" if job.render_path else None
    audio_url = f"http://127.0.0.1:8000/media/audio/{job.id}/voiceover.mp3" if job.audio_path else None

    return JobSummaryItem(
        job_id=str(job.id),
        topic=job.input_topic,
        language=job.target_language,
        tone=job.brand_tone,
        duration_secs=job.target_duration_secs,
        status=job.status.value if job.status else "UNKNOWN",
        created_at=job.created_at.isoformat() if job.created_at else None,
        completed_at=job.completed_at.isoformat() if job.completed_at else None,
        processing_time_secs=processing_time,
        has_audio=bool(job.audio_path),
        has_video=bool(job.render_path),
        video_url=video_url,
        audio_url=audio_url,
        broll_count=broll_count,
        script_provider=sd.get("provider"),
        tts_provider=sd.get("tts_provider"),
        broll_provider=sd.get("broll_provider"),
    )


def _job_to_detail(job: VideoJob) -> JobDetailItem:
    """Convert a VideoJob ORM row to a full JobDetailItem."""
    summary = _job_to_summary(job)
    sd = job.script_data or {}

    broll_prompts = sd.get("broll_prompts") or []
    if not isinstance(broll_prompts, list):
        broll_prompts = []

    return JobDetailItem(
        **summary.model_dump(),
        viral_hook=sd.get("viral_hook"),
        voiceover_script=sd.get("voiceover_script"),
        broll_prompts=broll_prompts,
        audio_path=job.audio_path,
        render_path=job.render_path,
        error_message=job.error_message,
        subtitles_srt=sd.get("subtitles_srt"),
        caption=sd.get("caption"),
        hashtags=sd.get("hashtags", []),
    )


def _get_provider_stats(jobs: List[VideoJob]) -> Dict[str, Any]:
    """Build provider usage counts from job records."""
    script_providers: Dict[str, int] = {}
    tts_providers: Dict[str, int] = {}
    broll_providers: Dict[str, int] = {}

    for job in jobs:
        sd = job.script_data or {}
        sp = sd.get("provider", "unknown")
        tp = sd.get("tts_provider", "unknown")
        bp = sd.get("broll_provider", "unknown")

        script_providers[sp] = script_providers.get(sp, 0) + 1
        tts_providers[tp] = tts_providers.get(tp, 0) + 1
        broll_providers[bp] = broll_providers.get(bp, 0) + 1

    return {
        "script_generation": script_providers,
        "text_to_speech": tts_providers,
        "broll_video": broll_providers,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────────────────

@router.get(
    "/analytics/summary",
    summary="Platform Analytics Summary",
    description="""
Returns a **complete aggregated statistics object** for the ViralBrandEngine platform.

This endpoint is designed for:
- **Data analysis** — pass to teacher/external systems for structured analysis
- **Dashboard** — overview of all video generation activity
- **API demonstration** — showcases JSON response format and REST conventions

Returns counts, rates, provider breakdowns, and configured services.
""",
    tags=["analytics"],
)
async def get_analytics_summary(db: Session = Depends(get_db)) -> Dict[str, Any]:
    """
    Aggregated platform statistics.

    **Use case:** Teacher calls this endpoint to receive a JSON dataset
    representing all video generation activity for analysis.
    """
    all_jobs = db.query(VideoJob).all()

    total = len(all_jobs)
    completed = sum(1 for j in all_jobs if j.status == JobStatus.COMPLETED)
    failed = sum(1 for j in all_jobs if j.status == JobStatus.FAILED)
    pending = sum(1 for j in all_jobs if j.status == JobStatus.PENDING)
    in_progress = total - completed - failed - pending

    success_rate = round((completed / total * 100) if total > 0 else 0.0, 1)

    total_video_secs = sum(
        j.target_duration_secs for j in all_jobs if j.status == JobStatus.COMPLETED
    )

    languages = list(set(j.target_language for j in all_jobs if j.target_language))
    tones = list(set(j.brand_tone for j in all_jobs if j.brand_tone))
    topics = [j.input_topic for j in all_jobs if j.input_topic][:5]  # top 5 recent

    avg_dur = 0.0
    if all_jobs:
        avg_dur = round(sum(j.target_duration_secs for j in all_jobs) / len(all_jobs), 1)

    s = get_settings()
    configured_services = {
        "openai_gpt4": bool(s.OPENAI_API_KEY.strip()),
        "google_gemini": bool(s.GEMINI_API_KEY.strip()),
        "elevenlabs_tts": bool(s.ELEVENLABS_API_KEY.strip()),
        "google_cloud_tts": bool(s.GOOGLE_TTS_API_KEY.strip()),
        "gtts_free": True,  # always available
        "pexels_video": bool(s.PEXELS_API_KEY.strip()),
        "pixabay_video": bool(s.PIXABAY_API_KEY.strip()),
        "meta_instagram": bool(s.META_ACCESS_TOKEN.strip()),
        "tiktok": bool(s.TIKTOK_ACCESS_TOKEN.strip()),
    }

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "platform": "ViralBrandEngine",
        "version": settings.APP_VERSION,
        "total_jobs": total,
        "completed_jobs": completed,
        "failed_jobs": failed,
        "pending_jobs": pending,
        "in_progress_jobs": in_progress,
        "success_rate_pct": success_rate,
        "total_video_seconds_generated": total_video_secs,
        "languages_used": languages,
        "tones_used": tones,
        "avg_duration_secs": avg_dur,
        "top_topics": topics,
        "provider_breakdown": _get_provider_stats(all_jobs),
        "configured_services": configured_services,
    }


@router.get(
    "/analytics/jobs",
    summary="List All Jobs (Dataset)",
    description="""
Returns a **paginated JSON dataset** of all video generation jobs.

Each record includes:
- Job ID, topic, language, tone, duration
- Status, timestamps, processing time
- Provider used for script/TTS/B-roll
- Artifact availability (audio, video)

**Designed for teacher data collection:** The teacher's system can call this
endpoint and receive structured JSON records for analysis.
""",
    tags=["analytics"],
)
async def list_all_jobs(
    page: int = Query(default=1, ge=1, description="Page number (1-indexed)"),
    per_page: int = Query(default=20, ge=1, le=100, description="Jobs per page"),
    status_filter: Optional[str] = Query(default=None, alias="status", description="Filter by status"),
    language: Optional[str] = Query(default=None, description="Filter by language code (tr, en)"),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """
    Paginated job list — full JSON dataset suitable for external analysis.
    """
    query = db.query(VideoJob)

    if status_filter:
        try:
            status_enum = JobStatus(status_filter.upper())
            query = query.filter(VideoJob.status == status_enum)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid status value '{status_filter}'. Valid: {[s.value for s in JobStatus]}",
            )

    if language:
        query = query.filter(VideoJob.target_language == language.lower())

    total_count = query.count()
    jobs = (
        query.order_by(VideoJob.created_at.desc())
        .offset((page - 1) * per_page)
        .limit(per_page)
        .all()
    )

    return {
        "dataset": "ViralBrandEngine Video Jobs",
        "total_count": total_count,
        "page": page,
        "per_page": per_page,
        "total_pages": (total_count + per_page - 1) // per_page if total_count > 0 else 0,
        "returned_count": len(jobs),
        "filters": {"status": status_filter, "language": language},
        "jobs": [_job_to_summary(j).model_dump() for j in jobs],
    }


@router.get(
    "/analytics/jobs/{job_id}",
    summary="Get Job Detail",
    description="Returns complete metadata for a single video generation job, including script content, B-roll prompts, and artifact paths.",
    tags=["analytics"],
)
async def get_job_detail(
    job_id: str,
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """Full detail for one job — script, audio path, render path, all metadata."""
    try:
        job_uuid = uuid.UUID(job_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid job ID format.")

    job = db.query(VideoJob).filter(VideoJob.id == job_uuid).first()
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")

    return {
        "job": _job_to_detail(job).model_dump(),
        "metadata": {
            "api_version": "v1",
            "fetched_at": datetime.now(timezone.utc).isoformat(),
        },
    }


@router.post(
    "/analytics/query",
    summary="Query Jobs with Filters",
    description="""
**Advanced filtered query** for job records. Supports:
- Filter by status, language, topic keyword, date range, duration range
- Sorting by any field (asc/desc)
- Pagination

This endpoint demonstrates **POST-based querying** (a common REST API pattern
for complex filter logic that would overflow GET query parameters).
""",
    tags=["analytics"],
)
async def query_jobs(
    request: QueryRequest,
    db: Session = Depends(get_db),
) -> QueryResponse:
    """Flexible filtered job query for data analysis."""
    query = db.query(VideoJob)

    if request.status:
        try:
            status_enum = JobStatus(request.status.upper())
            query = query.filter(VideoJob.status == status_enum)
        except ValueError:
            raise HTTPException(400, f"Invalid status: {request.status}")

    if request.language:
        query = query.filter(VideoJob.target_language == request.language.lower())

    if request.topic_contains:
        query = query.filter(
            VideoJob.input_topic.ilike(f"%{request.topic_contains}%")
        )

    if request.min_duration is not None:
        query = query.filter(VideoJob.target_duration_secs >= request.min_duration)

    if request.max_duration is not None:
        query = query.filter(VideoJob.target_duration_secs <= request.max_duration)

    if request.from_date:
        try:
            from_dt = datetime.fromisoformat(request.from_date)
            query = query.filter(VideoJob.created_at >= from_dt)
        except ValueError:
            raise HTTPException(400, "Invalid from_date format. Use ISO 8601.")

    if request.to_date:
        try:
            to_dt = datetime.fromisoformat(request.to_date)
            query = query.filter(VideoJob.created_at <= to_dt)
        except ValueError:
            raise HTTPException(400, "Invalid to_date format. Use ISO 8601.")

    # Sorting
    sort_col_map = {
        "created_at": VideoJob.created_at,
        "duration": VideoJob.target_duration_secs,
        "status": VideoJob.status,
        "topic": VideoJob.input_topic,
    }
    sort_col = sort_col_map.get(request.sort_by, VideoJob.created_at)
    if request.sort_order.lower() == "asc":
        query = query.order_by(sort_col.asc())
    else:
        query = query.order_by(sort_col.desc())

    total_count = query.count()
    jobs = query.offset(request.offset).limit(request.limit).all()

    filters_applied = {
        k: v for k, v in {
            "status": request.status,
            "language": request.language,
            "topic_contains": request.topic_contains,
            "min_duration": request.min_duration,
            "max_duration": request.max_duration,
            "from_date": request.from_date,
            "to_date": request.to_date,
        }.items() if v is not None
    }

    return QueryResponse(
        total_count=total_count,
        returned_count=len(jobs),
        filters_applied=filters_applied,
        jobs=[_job_to_summary(j) for j in jobs],
    )


@router.get(
    "/analytics/providers",
    summary="AI Provider Usage Breakdown",
    description="Returns a breakdown of which AI and service providers were used across all jobs: script generation (OpenAI / Gemini / Local), TTS (ElevenLabs / Google Cloud / gTTS), and B-roll (Pexels / Pixabay / Procedural).",
    tags=["analytics"],
)
async def get_provider_stats(db: Session = Depends(get_db)) -> Dict[str, Any]:
    """Provider usage statistics for demonstrating multi-provider architecture."""
    all_jobs = db.query(VideoJob).all()

    provider_data = _get_provider_stats(all_jobs)

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_jobs_analyzed": len(all_jobs),
        "provider_usage": provider_data,
        "available_providers": {
            "script_generation": [
                {"name": "OpenAI GPT-4o", "status": "configured" if settings.OPENAI_API_KEY.strip() else "not configured", "tier": "paid"},
                {"name": "Google Gemini", "status": "configured" if settings.GEMINI_API_KEY.strip() else "not configured", "tier": "free/paid"},
                {"name": "Smart Local Fallback", "status": "always available", "tier": "free"},
            ],
            "text_to_speech": [
                {"name": "ElevenLabs", "status": "configured" if settings.ELEVENLABS_API_KEY.strip() else "not configured", "tier": "paid"},
                {"name": "Google Cloud TTS", "status": "configured" if settings.GOOGLE_TTS_API_KEY.strip() else "not configured", "tier": "paid"},
                {"name": "gTTS (Google Translate TTS)", "status": "always available", "tier": "free", "supports_turkish": True},
                {"name": "SAPI5 (Windows)", "status": "always available", "tier": "free"},
            ],
            "broll_video": [
                {"name": "Kling AI Video", "status": "configured" if settings.KLING_API_KEY.strip() else "not configured", "tier": "paid/quota"},
                {"name": "Luma Ray 3.2 (AI Video)", "status": "configured" if settings.LUMA_API_KEY.strip() else "not configured", "tier": "paid/quota"},
                {"name": "Google Gemini Omni Flash (AI Video)", "status": "configured" if settings.GEMINI_API_KEY.strip() else "not configured", "tier": "paid/quota"},
                {"name": "Pexels Stock Video", "status": "configured" if settings.PEXELS_API_KEY.strip() else "not configured", "tier": "free"},
                {"name": "Pixabay Stock Video", "status": "configured" if settings.PIXABAY_API_KEY.strip() else "not configured", "tier": "free"},
                {"name": "Procedural Color Clip", "status": "always available", "tier": "free"},
            ],
        },
    }


@router.post("/analytics/events", summary="Record a product funnel event")
async def record_product_event(payload: ProductEventRequest, db: Session = Depends(get_db)) -> Dict[str, Any]:
    event = ProductEvent(
        event_name=payload.event_name.strip(),
        user_id=payload.user_id,
        campaign_id=payload.campaign_id,
        job_id=payload.job_id,
        properties=payload.properties or {},
    )
    db.add(event)
    db.commit()
    db.refresh(event)
    return {"event_id": str(event.id), "event_name": event.event_name}


@router.post("/analytics/feedback", summary="Record NPS or CSAT feedback")
async def record_feedback(payload: FeedbackRequest, db: Session = Depends(get_db)) -> Dict[str, Any]:
    feedback_type = payload.feedback_type.upper()
    score_range = (0, 10) if feedback_type == "NPS" else (1, 5)
    if not score_range[0] <= payload.score <= score_range[1]:
        raise HTTPException(400, f"{feedback_type} score must be between {score_range[0]} and {score_range[1]}.")

    feedback = CustomerFeedback(
        feedback_type=feedback_type,
        score=payload.score,
        comment=payload.comment,
        user_id=payload.user_id,
        job_id=payload.job_id,
    )
    db.add(feedback)
    db.commit()
    db.refresh(feedback)
    return {"feedback_id": str(feedback.id), "feedback_type": feedback_type, "score": payload.score}


@router.post("/analytics/costs", summary="Record video generation cost")
async def record_content_cost(payload: ContentCostRequest, db: Session = Depends(get_db)) -> Dict[str, Any]:
    if not db.query(VideoJob).filter(VideoJob.id == payload.job_id).first():
        raise HTTPException(404, "Video job not found.")
    cost = ContentCost(
        job_id=payload.job_id,
        provider=payload.provider.strip(),
        amount=payload.amount,
        currency=payload.currency.upper(),
    )
    db.add(cost)
    db.commit()
    db.refresh(cost)
    return {"cost_id": str(cost.id)}


@router.post("/analytics/social-performance", summary="Record social channel metrics")
async def record_social_performance(payload: SocialPerformanceRequest, db: Session = Depends(get_db)) -> Dict[str, Any]:
    if not db.query(SocialPost).filter(SocialPost.id == payload.social_post_id).first():
        raise HTTPException(404, "Social post not found.")
    performance = SocialPerformance(**payload.model_dump())
    db.add(performance)
    db.commit()
    db.refresh(performance)
    return {"measurement_id": str(performance.id)}


@router.get("/analytics/kpis", summary="Customer, funnel, cost and social KPI summary")
async def get_product_kpis(db: Session = Depends(get_db)) -> Dict[str, Any]:
    feedback = db.query(CustomerFeedback).all()
    nps_scores = [row.score for row in feedback if row.feedback_type == "NPS"]
    csat_scores = [row.score for row in feedback if row.feedback_type == "CSAT"]
    nps = None
    if nps_scores:
        promoters = sum(score >= 9 for score in nps_scores)
        detractors = sum(score <= 6 for score in nps_scores)
        nps = round((promoters - detractors) / len(nps_scores) * 100, 1)

    csat = round(sum(csat_scores) / len(csat_scores), 2) if csat_scores else None
    events = db.query(ProductEvent).all()
    funnel_names = ("studio_opened", "generation_started", "job_completed", "video_downloaded")
    funnel = {name: sum(event.event_name == name for event in events) for name in funnel_names}
    costs = db.query(ContentCost).all()
    jobs = db.query(VideoJob).all()
    completed_jobs = sum(job.status == JobStatus.COMPLETED for job in jobs)
    total_cost = round(sum(cost.amount for cost in costs), 2)
    cost_per_completed_video = (
        round(total_cost / completed_jobs, 2)
        if completed_jobs and costs else None
    )

    social_rows = db.query(SocialPerformance).all()
    social_totals = {
        "views": sum(row.views for row in social_rows),
        "likes": sum(row.likes for row in social_rows),
        "comments": sum(row.comments for row in social_rows),
        "shares": sum(row.shares for row in social_rows),
        "saves": sum(row.saves for row in social_rows),
    }
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "customer": {
            "nps": nps,
            "nps_responses": len(nps_scores),
            "csat": csat,
            "csat_responses": len(csat_scores),
        },
        "funnel": funnel,
        "funnel_completion_rate_pct": round(
            (funnel["job_completed"] / funnel["generation_started"] * 100)
            if funnel["generation_started"] else 0, 1
        ),
        "cost": {
            "total": total_cost,
            "currency": "mixed",
            "cost_per_completed_video": cost_per_completed_video,
        },
        "social_performance": social_totals,
    }


@router.get(
    "/analytics/export",
    summary="Full JSON Export (Data Analysis)",
    description="""
**Complete data export** endpoint — returns the entire job dataset plus summary statistics
in a single JSON response, designed for import into data analysis tools (Excel, Python pandas, etc.).

**Teacher use case:** Call this endpoint to receive a complete snapshot of all platform
data for analysis, grading, or demonstration purposes.
""",
    tags=["analytics"],
)
async def export_full_dataset(db: Session = Depends(get_db)) -> Dict[str, Any]:
    """Full platform data export — summary + all jobs in one response."""
    all_jobs = db.query(VideoJob).all()
    users = db.query(User).all()
    campaigns = db.query(Campaign).all()

    total = len(all_jobs)
    completed = sum(1 for j in all_jobs if j.status == JobStatus.COMPLETED)
    failed = sum(1 for j in all_jobs if j.status == JobStatus.FAILED)

    return {
        "export_metadata": {
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "platform": "ViralBrandEngine",
            "version": settings.APP_VERSION,
            "description": "Complete dataset export for analysis",
            "total_records": total,
        },
        "summary": {
            "total_users": len(users),
            "total_campaigns": len(campaigns),
            "total_jobs": total,
            "completed_jobs": completed,
            "failed_jobs": failed,
            "pending_jobs": total - completed - failed,
            "success_rate_pct": round((completed / total * 100) if total > 0 else 0, 1),
            "total_video_minutes_generated": round(
                sum(j.target_duration_secs for j in all_jobs if j.status == JobStatus.COMPLETED) / 60, 2
            ),
            "languages": list(set(j.target_language for j in all_jobs)),
            "tones": list(set(j.brand_tone for j in all_jobs)),
        },
        "users": [
            {
                "id": str(u.id),
                "email": u.email,
                "display_name": u.display_name,
                "created_at": u.created_at.isoformat() if u.created_at else None,
            }
            for u in users
        ],
        "campaigns": [
            {
                "id": str(c.id),
                "name": c.name,
                "brand_tone": c.brand_tone,
                "target_language": c.target_language,
                "created_at": c.created_at.isoformat() if c.created_at else None,
                "job_count": len(c.video_jobs),
            }
            for c in campaigns
        ],
        "video_jobs": [_job_to_detail(j).model_dump() for j in all_jobs],
    }


@router.get(
    "/analytics/health",
    summary="API Health & System Status",
    description="Returns current system health, database connectivity, and configured service status.",
    tags=["analytics"],
)
async def analytics_health(db: Session = Depends(get_db)) -> Dict[str, Any]:
    """Liveness + readiness probe with service configuration status."""
    # DB check
    try:
        db.execute(__import__("sqlalchemy").text("SELECT 1"))
        db_status = "connected"
    except Exception as e:
        db_status = f"error: {str(e)}"

    s = get_settings()

    return {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "database": db_status,
        "api_version": "v1",
        "platform": "ViralBrandEngine",
        "version": s.APP_VERSION,
        "services": {
            "openai": "configured" if s.OPENAI_API_KEY.strip() else "not configured",
            "gemini": "configured" if s.GEMINI_API_KEY.strip() else "not configured",
            "elevenlabs": "configured" if s.ELEVENLABS_API_KEY.strip() else "not configured",
            "google_tts": "configured" if s.GOOGLE_TTS_API_KEY.strip() else "not configured",
            "gtts_free": "always available",
            "pexels": "configured" if s.PEXELS_API_KEY.strip() else "not configured",
            "pixabay": "configured" if s.PIXABAY_API_KEY.strip() else "not configured",
        },
        "documentation": {
            "swagger_ui": "http://127.0.0.1:8000/docs",
            "redoc": "http://127.0.0.1:8000/redoc",
            "openapi_json": "http://127.0.0.1:8000/openapi.json",
        },
    }


@router.get(
    "/analytics/videos",
    response_model=VideoListResponse,
    summary="Sadece Üretilen Videolar ve İzleme Linkleri (Basit Liste)",
    description="""
Sadece başarıyla üretilmiş (hazır) videoları, başlıklarını ve **doğrudan tarayıcıda tıklandığında açılan izleme linklerini (MP4)** listeleyen sade endpoint.

**Örnek kullanım:** Kullanıcı veya öğretmen sadece üretilen videoların listesini ve izleme linklerini almak istediğinde bu endpoint'i çağırır.
""",
    tags=["analytics"],
)
@router.get(
    "/videos",
    response_model=VideoListResponse,
    include_in_schema=False,
)
async def list_videos_simple(
    limit: int = Query(default=50, ge=1, le=200, description="Kaç video getirilsin?"),
    db: Session = Depends(get_db),
) -> VideoListResponse:
    """Tamamlanmış videoların doğrudan izleme linkleriyle birlikte sade listesi."""
    completed_jobs = (
        db.query(VideoJob)
        .filter(VideoJob.status == JobStatus.COMPLETED)
        .order_by(VideoJob.completed_at.desc())
        .limit(limit)
        .all()
    )

    items = []
    for j in completed_jobs:
        items.append(
            VideoListItem(
                job_id=str(j.id),
                topic=j.input_topic,
                duration_secs=j.target_duration_secs,
                language=j.target_language,
                tone=j.brand_tone,
                status="HAZIR (COMPLETED)",
                watch_url=f"http://127.0.0.1:8000/media/renders/{j.id}.mp4",
                audio_url=f"http://127.0.0.1:8000/media/audio/{j.id}/voiceover.mp3" if j.audio_path else None,
                created_at=j.created_at.isoformat() if j.created_at else None,
            )
        )

    return VideoListResponse(
        total_videos=len(items),
        description="Üretilen tüm videoların doğrudan izleme ve indirme linkleri listesi",
        videos=items,
    )
