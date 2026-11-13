"""
Social Publisher Service — Auto-posting to Instagram Reels & TikTok.

Handles:
  1. Uploading the rendered video to Meta Graph API (IG Reels).
  2. Uploading the rendered video to TikTok Content Posting API.
  3. Recording the post metadata back to the SocialPost table.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

import requests
from tenacity import retry, stop_after_attempt, wait_exponential

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class SocialPublisherService:
    """Publish rendered videos to social platforms."""

    def __init__(
        self,
        meta_token: Optional[str] = None,
        meta_ig_user_id: Optional[str] = None,
        tiktok_token: Optional[str] = None,
    ):
        self._meta_token = meta_token or settings.META_ACCESS_TOKEN
        self._meta_ig_user_id = meta_ig_user_id or settings.META_IG_USER_ID
        self._tiktok_token = tiktok_token or settings.TIKTOK_ACCESS_TOKEN

    # ── Instagram Reels (Meta Graph API) ─────────────────────────────────

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=5, max=60),
        reraise=True,
    )
    def publish_instagram_reel(
        self,
        video_url: str,
        caption: str = "",
    ) -> Optional[str]:
        """
        Publish a video as an Instagram Reel via the Meta Graph API.

        NOTE: The caller must provide a public HTTPS object-storage URL.

        Returns the IG media creation ID on success.
        """
        if not self._meta_token or not self._meta_ig_user_id:
            logger.warning(
                "Meta credentials not configured — skipping IG Reels publish."
            )
            return None

        # Step 1: Create media container
        container_url = (
            f"https://graph.facebook.com/v19.0/{self._meta_ig_user_id}/media"
        )
        container_resp = requests.post(
            container_url,
            params={
                "media_type": "REELS",
                "video_url": video_url,
                "caption": caption,
                "access_token": self._meta_token,
            },
            timeout=60,
        )
        container_resp.raise_for_status()
        creation_id = container_resp.json()["id"]
        logger.info("IG Reels container created: %s", creation_id)

        # Step 2: Wait for processing, then publish
        self._wait_for_ig_processing(creation_id)

        publish_url = (
            f"https://graph.facebook.com/v19.0/{self._meta_ig_user_id}/media_publish"
        )
        publish_resp = requests.post(
            publish_url,
            params={
                "creation_id": creation_id,
                "access_token": self._meta_token,
            },
            timeout=60,
        )
        publish_resp.raise_for_status()
        media_id = publish_resp.json().get("id")
        logger.info("IG Reel published — media ID: %s", media_id)
        return media_id

    def _wait_for_ig_processing(
        self, creation_id: str, max_wait: int = 120, poll_interval: int = 5
    ) -> None:
        """Poll the container status until it's ready to publish."""
        status_url = f"https://graph.facebook.com/v19.0/{creation_id}"
        elapsed = 0
        while elapsed < max_wait:
            resp = requests.get(
                status_url,
                params={
                    "fields": "status_code",
                    "access_token": self._meta_token,
                },
                timeout=30,
            )
            status_code = resp.json().get("status_code")
            if status_code == "FINISHED":
                return
            if status_code == "ERROR":
                raise RuntimeError(f"IG media processing failed: {resp.json()}")
            time.sleep(poll_interval)
            elapsed += poll_interval
        raise TimeoutError("IG media processing timed out.")

    # ── TikTok Content Posting API ───────────────────────────────────────

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=5, max=60),
        reraise=True,
    )
    def publish_tiktok(
        self,
        video_path: Path,
        caption: str = "",
    ) -> Optional[str]:
        """
        Upload a video to TikTok via the Content Posting API.

        This is a simplified placeholder. The real TikTok API flow requires:
        1. Initiate upload → get upload URL
        2. Upload video bytes to that URL
        3. Publish the post

        Returns the TikTok publish ID on success.
        """
        if not self._tiktok_token:
            logger.warning(
                "TIKTOK_ACCESS_TOKEN not configured — skipping TikTok publish."
            )
            return None

        # Step 1: Init upload
        init_url = "https://open.tiktokapis.com/v2/post/publish/inbox/video/init/"
        headers = {
            "Authorization": f"Bearer {self._tiktok_token}",
            "Content-Type": "application/json",
        }
        init_payload = {
            "post_info": {
                "title": caption[:150],
                "privacy_level": "PUBLIC_TO_EVERYONE",
            },
            "source_info": {
                "source": "FILE_UPLOAD",
                "video_size": video_path.stat().st_size,
            },
        }

        init_resp = requests.post(init_url, json=init_payload, headers=headers, timeout=60)
        init_resp.raise_for_status()
        upload_url = init_resp.json().get("data", {}).get("upload_url")
        publish_id = init_resp.json().get("data", {}).get("publish_id")

        if not upload_url:
            raise ValueError("TikTok did not return an upload URL.")

        # Step 2: Upload video
        with open(video_path, "rb") as vf:
            upload_resp = requests.put(
                upload_url,
                data=vf,
                headers={
                    "Content-Type": "video/mp4",
                    "Content-Range": f"bytes 0-{video_path.stat().st_size - 1}/{video_path.stat().st_size}",
                },
                timeout=300,
            )
            upload_resp.raise_for_status()

        logger.info("TikTok video uploaded — publish ID: %s", publish_id)
        return publish_id
