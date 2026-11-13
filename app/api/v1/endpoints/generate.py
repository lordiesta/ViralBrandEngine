"""
POST /api/v1/generate — Trigger a new video generation pipeline.
"""

from __future__ import annotations

import uuid
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.db.models import VideoJob, JobStatus, Campaign
from app.workers.tasks import process_video_pipeline
from app.config import get_settings

settings = get_settings()

router = APIRouter()


# ── Request / Response schemas ───────────────────────────────────────────

class GenerateRequest(BaseModel):
    """Payload to kick off a new video job."""

    campaign_id: uuid.UUID = Field(..., description="Parent campaign UUID.")
    input_topic: str = Field(
        ...,
        min_length=3,
        max_length=1000,
        description="Topic / prompt for the video (e.g. '5 benefits of cold showers').",
    )
    brand_tone: str = Field(default="professional", max_length=128)
    target_language: str = Field(default="en", max_length=16)
    target_duration_secs: int = Field(default=20, ge=10, le=60)
    auto_publish: bool = Field(default=False)
    subtitles_enabled: bool = Field(default=False)


class GenerateResponse(BaseModel):
    job_id: uuid.UUID
    status: str
    message: str


# ── Endpoint ─────────────────────────────────────────────────────────────

@router.post(
    "/generate",
    response_model=GenerateResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Submit a video generation job",
)
def create_video_job(
    payload: GenerateRequest,
    db: Session = Depends(get_db),
):
    """
    Creates a new ``VideoJob`` row and dispatches the Celery pipeline task.
    Returns immediately with a ``202 Accepted`` and the job ID for polling.
    """

    # Validate that the campaign exists
    campaign = db.query(Campaign).filter(Campaign.id == payload.campaign_id).first()
    if not campaign:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Campaign {payload.campaign_id} not found.",
        )

    # Persist the job
    job = VideoJob(
        campaign_id=payload.campaign_id,
        input_topic=payload.input_topic,
        brand_tone=payload.brand_tone,
        target_language=payload.target_language,
        target_duration_secs=payload.target_duration_secs,
        auto_publish=payload.auto_publish,
        subtitles_enabled=payload.subtitles_enabled,
        status=JobStatus.PENDING,
    )
    db.add(job)
    db.commit()
    db.refresh(job)

    # Dispatch async pipeline
    # Dispatch only through the durable Celery broker. In-process threads do
    # not survive API restarts and cannot provide queue guarantees.
    try:
        process_video_pipeline.apply_async(
            args=(str(job.id),),
            task_id=f"video-{job.id}",
        )
    except Exception:
        db.delete(job)
        db.commit()
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The job queue is unavailable. No job was dispatched.",
        )

    return GenerateResponse(
        job_id=job.id,
        status=job.status.value,
        message="Video generation pipeline queued successfully.",
    )
