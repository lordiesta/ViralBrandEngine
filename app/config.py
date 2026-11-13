"""
Application configuration via Pydantic Settings.

All secrets and environment-specific values are loaded from environment
variables (or a `.env` file at the project root).  Pydantic validates types
and raises clear errors on missing required keys at startup.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central configuration object – one source of truth for all env vars."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Application ──────────────────────────────────────────────────────
    APP_NAME: str = "ViralBrandEngine"
    APP_VERSION: str = "0.1.0"
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"
    API_KEY: str = Field(default="", description="API key required by application endpoints.")
    ADMIN_API_KEY: str = Field(default="", description="Separate key required by administrative endpoints.")
    ADMIN_USERNAME: str = Field(default="admin", description="Administrator login username.")
    ADMIN_PASSWORD: str = Field(default="admin123", description="Administrator login password.")
    USER_USERNAME: str = Field(default="kullanici", description="Standard user login username.")
    USER_PASSWORD: str = Field(default="kullanici123", description="Standard user login password.")
    SMTP_HOST: str = Field(default="", description="SMTP server host.")
    SMTP_PORT: int = Field(default=587, description="SMTP server port.")
    SMTP_USERNAME: str = Field(default="", description="SMTP account username.")
    SMTP_PASSWORD: str = Field(default="", description="SMTP account password.")
    SMTP_FROM_EMAIL: str = Field(default="", description="Sender email address.")
    SMTP_USE_TLS: bool = Field(default=True, description="Use STARTTLS for SMTP.")
    CORS_ORIGINS: str = Field(default="", description="Comma-separated browser origins allowed to call the API.")
    PUBLIC_MEDIA_BASE_URL: str = Field(default="", description="Public HTTPS base URL for rendered videos.")

    # ── Database (PostgreSQL) ────────────────────────────────────────────
    DATABASE_URL: str = Field(
        default="postgresql+psycopg2://postgres:postgres@localhost:5432/viral_brand_engine",
        description="SQLAlchemy-style connection string for PostgreSQL.",
    )

    # ── Redis / Celery ───────────────────────────────────────────────────
    REDIS_URL: str = Field(
        default="redis://localhost:6379/0",
        description="Redis connection URL used as Celery broker & result backend.",
    )

    # ── OpenAI (GPT-4o) ─────────────────────────────────────────────────
    OPENAI_API_KEY: str = Field(
        default="",
        description="API key for OpenAI GPT-4o script generation.",
    )
    OPENAI_MODEL: str = "gpt-4o"

    # ── Google Gemini (Free Script Generator Alternative) ─────────────────
    GEMINI_API_KEY: str = Field(
        default="",
        description="Google Gemini API key (free tier available via Google AI Studio).",
    )
    GEMINI_MODEL: str = "gemini-3.8-flash"

    # ── ElevenLabs ───────────────────────────────────────────────────────
    ELEVENLABS_API_KEY: str = Field(
        default="",
        description="API key for ElevenLabs voice synthesis.",
    )
    ELEVENLABS_VOICE_ID: str = Field(
        default="21m00Tcm4TlvDq8ikWAM",  # "Rachel" – default ElevenLabs voice
        description="Default ElevenLabs voice ID.",
    )

    # ── Google Cloud TTS (Turkish / multilingual) ─────────────────────────
    GOOGLE_TTS_API_KEY: str = Field(
        default="",
        description="Google Cloud TTS API key for Turkish (tr-TR) voiceover synthesis.",
    )

    # ── B-Roll Video Sources ─────────────────────────────────────────────
    KLING_API_KEY: str = Field(
        default="",
        description="Kling AI Open Platform API key for video generation.",
    )
    GOOGLE_FLOW_API_KEY: str = Field(
        default="",
        description="Google AI Studio Gemini API key used for continuous video generation.",
    )
    PEXELS_API_KEY: str = Field(default="", description="Pexels stock video API key.")
    PIXABAY_API_KEY: str = Field(default="", description="Pixabay stock video API key (instant free alternative to Pexels).")
    RUNWAY_API_KEY: str = Field(default="", description="Runway ML API key.")
    LUMA_API_KEY: str = Field(default="", description="Luma AI API key.")

    # ── Social Media Publishing ──────────────────────────────────────────
    META_ACCESS_TOKEN: str = Field(
        default="", description="Meta Graph API long-lived access token."
    )
    META_IG_USER_ID: str = Field(
        default="", description="Instagram Business account user ID."
    )
    TIKTOK_ACCESS_TOKEN: str = Field(default="", description="TikTok API access token.")

    # ── Media / Storage Paths ────────────────────────────────────────────
    MEDIA_DIR: Path = Field(
        default=Path("media"),
        description="Root directory for generated media artifacts.",
    )

    @property
    def renders_dir(self) -> Path:
        """Directory for final rendered videos."""
        path = self.MEDIA_DIR / "renders"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def audio_dir(self) -> Path:
        """Directory for synthesised voiceover audio files."""
        path = self.MEDIA_DIR / "audio"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def broll_dir(self) -> Path:
        """Directory for downloaded B-roll clips."""
        path = self.MEDIA_DIR / "broll"
        path.mkdir(parents=True, exist_ok=True)
        return path


@lru_cache
def get_settings() -> Settings:
    """Return a cached singleton of the application settings."""
    return Settings()
