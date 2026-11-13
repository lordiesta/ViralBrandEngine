"""
Celery tasks — Asynchronous video generation pipeline orchestrator.

The master task ``process_video_pipeline`` executes the full pipeline
sequentially within a single Celery worker:

  1. Fetch VideoJob from DB
  2. Generate script via OpenAI
  3. Synthesise voiceover via ElevenLabs
  4. Fetch / generate B-roll clips
  5. Edit & render final video (FFmpeg / MoviePy)
  6. (Optional) Publish to social platforms
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path

from app.workers.celery_app import celery_app
from app.config import get_settings
from app.db.session import SessionLocal
from app.db.models import VideoJob, JobStatus, SocialPost, SocialPlatform

from app.services.script_generator import ScriptGeneratorService
from app.services.voice_synthesizer import VoiceSynthesizerService
from app.services.broll_fetcher import BRollFetcherService, GeminiVideoQuotaError
from app.services.video_editor import VideoEditorService
from app.services.social_publisher import SocialPublisherService

logger = logging.getLogger(__name__)
settings = get_settings()


def _update_job_status(
    job_id: str | uuid.UUID, status: JobStatus, error_message: str | None = None, **kwargs
) -> None:
    """Helper to update a VideoJob's status in a fresh DB session."""
    uid = uuid.UUID(str(job_id)) if not isinstance(job_id, uuid.UUID) else job_id
    db = SessionLocal()
    try:
        job = db.query(VideoJob).filter(VideoJob.id == uid).first()
        if job:
            job.status = status
            if error_message:
                job.error_message = error_message
            for key, value in kwargs.items():
                setattr(job, key, value)
            db.commit()
    finally:
        db.close()


@celery_app.task(
    bind=True,
    name="process_video_pipeline",
    max_retries=2,
    default_retry_delay=30,
    acks_late=True,
)
def process_video_pipeline(self=None, video_job_id: str | None = None) -> dict:
    """
    Master pipeline task.  Runs each stage sequentially and updates the
    job status in the database at every transition.
    """
    if video_job_id is None:
        video_job_id = self
        self = None
    """
    Master pipeline task.  Runs each stage sequentially and updates the
    job status in the database at every transition.
    """
    logger.info("▶ Starting pipeline for VideoJob %s", video_job_id)

    # ── 0. Fetch job details ─────────────────────────────────────────────
    import uuid as _uuid
    job_uuid = _uuid.UUID(str(video_job_id)) if not isinstance(video_job_id, _uuid.UUID) else video_job_id
    db = SessionLocal()
    try:
        job = db.query(VideoJob).filter(VideoJob.id == job_uuid).first()
        if not job:
            raise ValueError(f"VideoJob {video_job_id} not found in database.")
        if job.status == JobStatus.COMPLETED:
            return {"job_id": video_job_id, "status": "COMPLETED", "render_path": job.render_path}

        input_topic = job.input_topic
        brand_tone = job.brand_tone
        target_language = job.target_language
        target_duration = job.target_duration_secs
        auto_publish = job.auto_publish
        subtitles_enabled = job.subtitles_enabled
    finally:
        db.close()

    try:
        # ── 1. Script Generation ─────────────────────────────────────────
        _update_job_status(video_job_id, JobStatus.GENERATING_SCRIPT)
        logger.info("  [1/5] Generating script for topic: %s", input_topic)

        script_service = ScriptGeneratorService()
        script = script_service.generate(
            topic=input_topic,
            brand_tone=brand_tone,
            target_language=target_language,
            target_duration_secs=target_duration,
        )

        # Persist script data
        script_data = {
            "viral_hook": script.viral_hook,
            "voiceover_script": script.voiceover_script,
            "broll_prompts": script.broll_prompts,
            "subtitles_srt": script.subtitles_srt,
            "caption": script.caption,
            "hashtags": script.hashtags,
            "scenes": getattr(script, "scenes", []),
            "provider": getattr(script, "provider", "smart_local"),
            "tts_provider": "gtts" if not settings.ELEVENLABS_API_KEY else "elevenlabs",
            "broll_provider": (
                "kling_video"
                if get_settings().KLING_API_KEY
                else "luma_video"
                if get_settings().LUMA_API_KEY
                else "gemini_video"
                if settings.GOOGLE_FLOW_API_KEY or settings.GEMINI_API_KEY
                else "pexels"
                if settings.PEXELS_API_KEY
                else "pixabay"
                if settings.PIXABAY_API_KEY
                else "procedural"
            ),
        }
        _update_job_status(video_job_id, JobStatus.GENERATING_SCRIPT, script_data=script_data)

        # ── 2. Audio Synthesis ───────────────────────────────────────────
        _update_job_status(video_job_id, JobStatus.SYNTHESIZING_AUDIO)
        logger.info("  [2/5] Synthesising voiceover audio…")

        voice_service = VoiceSynthesizerService()
        audio_dir = settings.audio_dir / video_job_id
        audio_path = voice_service.synthesize(
            text=script.voiceover_script,
            output_path=audio_dir / "voiceover.mp3",
            language=target_language,
        )
        _update_job_status(video_job_id, JobStatus.SYNTHESIZING_AUDIO, audio_path=str(audio_path))

        # ── 3. B-Roll Fetching ───────────────────────────────────────────
        _update_job_status(video_job_id, JobStatus.FETCHING_BROLL)
        logger.info("  [3/5] Fetching B-roll clips…")

        broll_service = BRollFetcherService()
        broll_dir = settings.broll_dir / video_job_id
        # Build a list of visual search queries from the scene storyboard for tighter visual‑narrative alignment
        scene_prompts = [scene.get("visual_search_query", "") for scene in getattr(script, "scenes", [])]
        # Fall back to the original broll_prompts if scenes are missing (unlikely)
        prompts_to_use = scene_prompts if any(scene_prompts) else script.broll_prompts

        broll_paths = broll_service.fetch_clips(
            prompts=prompts_to_use,
            output_dir=broll_dir,
            topic=input_topic,
            scenes=getattr(script, "scenes", None),
        )
        _update_job_status(
            video_job_id,
            JobStatus.FETCHING_BROLL,
            broll_paths=[str(p) for p in broll_paths],
        )

        # ── 4. Video Editing & Render ────────────────────────────────────
        _update_job_status(video_job_id, JobStatus.EDITING_VIDEO)
        logger.info("  [4/5] Rendering final video…")

        editor_service = VideoEditorService()
        render_path = settings.renders_dir / f"{video_job_id}.mp4"
        editor_service.render(
            broll_paths=[Path(p) for p in broll_paths],
            audio_path=Path(audio_path),
            srt_text=script.subtitles_srt,
            output_path=render_path,
            target_duration_secs=target_duration,
            scenes=getattr(script, "scenes", None),
            subtitles_enabled=subtitles_enabled,
        )
        _update_job_status(video_job_id, JobStatus.EDITING_VIDEO, render_path=str(render_path))

        # ── 5. Social Publishing (optional) ──────────────────────────────
        if auto_publish:
            _update_job_status(video_job_id, JobStatus.PUBLISHING)
            logger.info("  [5/5] Publishing to social platforms…")
            _publish_to_socials(video_job_id, render_path, script.caption)
        else:
            logger.info("  [5/5] Auto-publish disabled — skipping.")

        # ── Done ─────────────────────────────────────────────────────────
        _update_job_status(
            video_job_id,
            JobStatus.COMPLETED,
            render_path=str(render_path),
            completed_at=datetime.now(timezone.utc),
        )
        logger.info("✅ Pipeline complete for VideoJob %s", video_job_id)
        return {"job_id": video_job_id, "status": "COMPLETED", "render_path": str(render_path)}

    except GeminiVideoQuotaError as exc:
        logger.error("❌ Pipeline stopped because Gemini video quota is unavailable: %s", exc)
        _update_job_status(video_job_id, JobStatus.FAILED, error_message=str(exc))
        raise
    except Exception as exc:
        logger.exception("❌ Pipeline failed for VideoJob %s: %s", video_job_id, exc)
        _update_job_status(video_job_id, JobStatus.FAILED, error_message=str(exc))
        if self and hasattr(self, "retry"):
            raise self.retry(exc=exc)
        raise exc


# ── Social publishing helper ─────────────────────────────────────────────

def _publish_to_socials(job_id: str, render_path: Path, caption: str) -> None:
    """Attempt to publish to all configured platforms and record results."""
    publisher = SocialPublisherService()
    db = SessionLocal()

    try:
        # Instagram Reels
        public_base_url = settings.PUBLIC_MEDIA_BASE_URL.rstrip("/")
        public_video_url = f"{public_base_url}/{render_path.name}" if public_base_url else ""

        if settings.META_ACCESS_TOKEN:
            try:
                if not public_video_url.startswith("https://"):
                    raise RuntimeError("Instagram publishing requires PUBLIC_MEDIA_BASE_URL with an HTTPS URL.")
                ig_id = publisher.publish_instagram_reel(
                    video_url=public_video_url,
                    caption=caption,
                )
                post = SocialPost(
                    video_job_id=job_id,
                    platform=SocialPlatform.INSTAGRAM_REELS,
                    platform_post_id=ig_id,
                    caption=caption,
                    published=bool(ig_id),
                    published_at=datetime.now(timezone.utc) if ig_id else None,
                    provider_status="PUBLISHED" if ig_id else "SKIPPED",
                )
                db.add(post)
            except Exception as exc:
                logger.error("IG publish failed: %s", exc)
                post = SocialPost(
                    video_job_id=job_id,
                    platform=SocialPlatform.INSTAGRAM_REELS,
                    published=False,
                    error_message=str(exc),
                )
                db.add(post)

        # TikTok
        if settings.TIKTOK_ACCESS_TOKEN:
            try:
                tt_id = publisher.publish_tiktok(
                    video_path=render_path,
                    caption=caption,
                )
                post = SocialPost(
                    video_job_id=job_id,
                    platform=SocialPlatform.TIKTOK,
                    platform_post_id=tt_id,
                    caption=caption,
                    published=False,
                    provider_status="UPLOAD_COMPLETE",
                )
                db.add(post)
            except Exception as exc:
                logger.error("TikTok publish failed: %s", exc)
                post = SocialPost(
                    video_job_id=job_id,
                    platform=SocialPlatform.TIKTOK,
                    published=False,
                    error_message=str(exc),
                )
                db.add(post)

        db.commit()
    finally:
        db.close()
