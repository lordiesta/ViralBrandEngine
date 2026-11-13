"""
GET /api/v1/status/{job_id} — Poll the progress of a video generation job.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from typing import Optional, List
from pathlib import Path
import shutil

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.db.models import JobStatus, VideoJob
from app.config import get_settings
from app.workers.tasks import process_video_pipeline

router = APIRouter()


# ── Response schema ──────────────────────────────────────────────────────

class JobStatusResponse(BaseModel):
    job_id: uuid.UUID
    status: str
    input_topic: Optional[str] = None
    brand_tone: Optional[str] = None
    error_message: Optional[str] = None
    render_path: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    completed_at: Optional[str] = None


class JobListResponse(BaseModel):
    jobs: List[JobStatusResponse]
    total: int


class DeleteJobsRequest(BaseModel):
    job_ids: List[uuid.UUID]


def _delete_job_artifacts(job: VideoJob) -> None:
    settings = get_settings()
    job_id = str(job.id)
    paths = []

    if job.render_path:
        paths.append(Path(job.render_path))
    if job.audio_path:
        paths.append(Path(job.audio_path))
    for artifact in job.broll_paths or []:
        paths.append(Path(artifact))

    for artifact in paths:
        if artifact.name:
            artifact.unlink(missing_ok=True)

    for artifact_dir in (
        settings.audio_dir / job_id,
        settings.broll_dir / job_id,
    ):
        if artifact_dir.exists():
            shutil.rmtree(artifact_dir)

    render_path = settings.renders_dir / f"{job_id}.mp4"
    render_path.unlink(missing_ok=True)


# ── Endpoints ────────────────────────────────────────────────────────────

@router.get(
    "/jobs",
    response_model=JobListResponse,
    summary="List all video jobs (paginated)",
)
def list_jobs(
    campaign_id: Optional[uuid.UUID] = None,
    limit: int = 20,
    offset: int = 0,
    db: Session = Depends(get_db),
):
    """Return a paginated list of video jobs, optionally filtered by campaign."""
    query = db.query(VideoJob)
    if campaign_id:
        query = query.filter(VideoJob.campaign_id == campaign_id)

    total = query.count()
    jobs = query.order_by(VideoJob.created_at.desc()).offset(offset).limit(limit).all()

    return JobListResponse(
        jobs=[
            JobStatusResponse(
                job_id=j.id,
                status=j.status.value,
                input_topic=j.input_topic,
                brand_tone=j.brand_tone,
                error_message=j.error_message,
                render_path=j.render_path,
                created_at=j.created_at.isoformat() if j.created_at else None,
                updated_at=j.updated_at.isoformat() if j.updated_at else None,
                completed_at=j.completed_at.isoformat() if j.completed_at else None,
            )
            for j in jobs
        ],
        total=total,
    )


@router.get(
    "/status/{job_id}",
    response_model=JobStatusResponse,
    summary="Get status of a single video job",
)
def get_job_status(
    job_id: uuid.UUID,
    db: Session = Depends(get_db),
):
    """Return current pipeline status and artifact path (if completed)."""
    job = db.query(VideoJob).filter(VideoJob.id == job_id).first()
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"VideoJob {job_id} not found.",
        )

    return JobStatusResponse(
        job_id=job.id,
        status=job.status.value,
        input_topic=job.input_topic,
        brand_tone=job.brand_tone,
        error_message=job.error_message,
        render_path=job.render_path,
        created_at=job.created_at.isoformat() if job.created_at else None,
        updated_at=job.updated_at.isoformat() if job.updated_at else None,
        completed_at=job.completed_at.isoformat() if job.completed_at else None,
    )


@router.post(
    "/jobs/delete",
    summary="Delete selected video jobs and their generated files",
)
def delete_jobs(payload: DeleteJobsRequest, db: Session = Depends(get_db)):
    if not payload.job_ids:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No video jobs selected.")

    jobs = db.query(VideoJob).filter(VideoJob.id.in_(payload.job_ids)).all()
    for job in jobs:
        _delete_job_artifacts(job)
        db.delete(job)
    db.commit()

    return {"deleted": len(jobs), "requested": len(payload.job_ids)}


@router.post("/jobs/requeue-stale", summary="Retry jobs stuck outside terminal states")
def requeue_stale_jobs(
    stale_after_minutes: int = 30,
    db: Session = Depends(get_db),
):
    cutoff = datetime.now(timezone.utc) - timedelta(minutes=max(stale_after_minutes, 1))
    terminal = {JobStatus.COMPLETED, JobStatus.FAILED}
    jobs = db.query(VideoJob).filter(
        VideoJob.status.notin_(terminal),
        VideoJob.updated_at < cutoff,
    ).all()
    queued = 0
    for job in jobs:
        try:
            process_video_pipeline.apply_async(
                args=(str(job.id),),
                task_id=f"video-{job.id}-retry-{job.retry_count + 1}",
            )
            job.retry_count = (job.retry_count or 0) + 1
            queued += 1
        except Exception:
            continue
    db.commit()
    return {"requeued": queued}
