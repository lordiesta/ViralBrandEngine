"""
SQLAlchemy ORM models for ViralBrandEngine.

Entities:
  - User          – API consumer / brand account.
  - Campaign      – A themed collection of video jobs.
  - VideoJob      – Single video generation pipeline run.
  - SocialPost    – Record of a published social media post.
"""

from __future__ import annotations

import enum
import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    Column,
    DateTime,
    Enum,
    ForeignKey,
    Integer,
    String,
    Text,
    Boolean,
    JSON,
    Uuid,
    false,
    Float,
)
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    """Shared base class for all ORM models."""
    pass


# ── Enums ────────────────────────────────────────────────────────────────

class JobStatus(str, enum.Enum):
    PENDING = "PENDING"
    GENERATING_SCRIPT = "GENERATING_SCRIPT"
    SYNTHESIZING_AUDIO = "SYNTHESIZING_AUDIO"
    FETCHING_BROLL = "FETCHING_BROLL"
    EDITING_VIDEO = "EDITING_VIDEO"
    PUBLISHING = "PUBLISHING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class SocialPlatform(str, enum.Enum):
    INSTAGRAM_REELS = "INSTAGRAM_REELS"
    TIKTOK = "TIKTOK"


# ── Models ───────────────────────────────────────────────────────────────

class User(Base):
    __tablename__ = "users"

    id = Column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email = Column(String(255), unique=True, nullable=False, index=True)
    display_name = Column(String(128), nullable=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    campaigns = relationship("Campaign", back_populates="user", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<User {self.email}>"


class Account(Base):
    __tablename__ = "accounts"

    id = Column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email = Column(String(255), unique=True, nullable=False, index=True)
    display_name = Column(String(128), nullable=False)
    password_hash = Column(String(512), nullable=False)
    email_verified = Column(Boolean, default=False, nullable=False)
    verification_code = Column(String(6), nullable=True)
    verification_expires_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))


class Campaign(Base):
    __tablename__ = "campaigns"

    id = Column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(Uuid(as_uuid=True), ForeignKey("users.id"), nullable=False)
    name = Column(String(256), nullable=False)
    brand_tone = Column(String(128), default="professional")
    target_language = Column(String(16), default="en")
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    user = relationship("User", back_populates="campaigns")
    video_jobs = relationship("VideoJob", back_populates="campaign", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<Campaign {self.name}>"


class VideoJob(Base):
    __tablename__ = "video_jobs"

    id = Column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    campaign_id = Column(Uuid(as_uuid=True), ForeignKey("campaigns.id"), nullable=False)

    # Input parameters
    input_topic = Column(Text, nullable=False)
    brand_tone = Column(String(128), default="professional")
    target_language = Column(String(16), default="en")
    target_duration_secs = Column(Integer, default=20)

    # Pipeline state
    status = Column(Enum(JobStatus), default=JobStatus.PENDING, nullable=False, index=True)
    error_message = Column(Text, nullable=True)
    retry_count = Column(Integer, default=0, nullable=False)
    started_at = Column(DateTime(timezone=True), nullable=True)

    # Generated artifacts (JSON blobs / file paths)
    script_data = Column(JSON, nullable=True)        # structured script output from LLM
    audio_path = Column(String(512), nullable=True)   # path to .mp3
    broll_paths = Column(JSON, nullable=True)          # list of .mp4 clip paths
    render_path = Column(String(512), nullable=True)   # final rendered video path

    # Scheduling
    auto_publish = Column(Boolean, default=False)
    subtitles_enabled = Column(Boolean, default=False, nullable=False, server_default=false())

    # Timestamps
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )
    completed_at = Column(DateTime(timezone=True), nullable=True)

    campaign = relationship("Campaign", back_populates="video_jobs")
    social_posts = relationship("SocialPost", back_populates="video_job", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<VideoJob {self.id} [{self.status}]>"


class SocialPost(Base):
    __tablename__ = "social_posts"

    id = Column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    video_job_id = Column(Uuid(as_uuid=True), ForeignKey("video_jobs.id"), nullable=False)

    platform = Column(Enum(SocialPlatform), nullable=False)
    platform_post_id = Column(String(256), nullable=True)  # ID returned by platform
    caption = Column(Text, nullable=True)
    published = Column(Boolean, default=False)
    published_at = Column(DateTime(timezone=True), nullable=True)
    error_message = Column(Text, nullable=True)
    provider_status = Column(String(64), nullable=True)

    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    video_job = relationship("VideoJob", back_populates="social_posts")

    def __repr__(self) -> str:
        return f"<SocialPost {self.platform} published={self.published}>"


class ProductEvent(Base):
    __tablename__ = "product_events"

    id = Column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    event_name = Column(String(128), nullable=False, index=True)
    user_id = Column(Uuid(as_uuid=True), ForeignKey("users.id"), nullable=True, index=True)
    campaign_id = Column(Uuid(as_uuid=True), ForeignKey("campaigns.id"), nullable=True, index=True)
    job_id = Column(Uuid(as_uuid=True), ForeignKey("video_jobs.id"), nullable=True, index=True)
    properties = Column(JSON, nullable=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), index=True)


class CustomerFeedback(Base):
    __tablename__ = "customer_feedback"

    id = Column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    feedback_type = Column(String(16), nullable=False, index=True)
    score = Column(Integer, nullable=False)
    comment = Column(Text, nullable=True)
    user_id = Column(Uuid(as_uuid=True), ForeignKey("users.id"), nullable=True, index=True)
    job_id = Column(Uuid(as_uuid=True), ForeignKey("video_jobs.id"), nullable=True, index=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), index=True)


class ContentCost(Base):
    __tablename__ = "content_costs"

    id = Column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    job_id = Column(Uuid(as_uuid=True), ForeignKey("video_jobs.id"), nullable=False, index=True)
    provider = Column(String(128), nullable=False, index=True)
    amount = Column(Float, nullable=False)
    currency = Column(String(8), nullable=False, default="USD")
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), index=True)


class SocialPerformance(Base):
    __tablename__ = "social_performance"

    id = Column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    social_post_id = Column(Uuid(as_uuid=True), ForeignKey("social_posts.id"), nullable=False, index=True)
    views = Column(Integer, nullable=False, default=0)
    likes = Column(Integer, nullable=False, default=0)
    comments = Column(Integer, nullable=False, default=0)
    shares = Column(Integer, nullable=False, default=0)
    saves = Column(Integer, nullable=False, default=0)
    measured_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), index=True)
