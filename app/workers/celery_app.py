"""
Celery application configuration.

Usage:
  celery -A app.workers.celery_app worker --loglevel=info
"""

from __future__ import annotations

from celery import Celery

from app.config import get_settings

settings = get_settings()

celery_app = Celery(
    "viral_brand_engine",
    broker=settings.REDIS_URL,
    backend=(
        "db+sqlite:///celery_results.sqlite"
        if settings.REDIS_URL.startswith("sqla+sqlite")
        else settings.REDIS_URL
    ),
    include=["app.workers.tasks"],
)

# ── Celery configuration ────────────────────────────────────────────────
celery_app.conf.update(
    # Serialisation
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],

    # Timezone
    timezone="UTC",
    enable_utc=True,

    # Task behaviour
    task_acks_late=True,                   # ack after execution (safer)
    worker_prefetch_multiplier=1,          # one task at a time per worker
    task_track_started=True,               # report STARTED state

    # Result expiration
    result_expires=86400,                  # 24 hours
)
