"""
B-Roll Fetcher Service — Stock / generative video clip acquisition.

Supports multiple providers:
  1. Kling AI for coherent story video generation.
  2. Luma Ray 3.2 for coherent story video generation.
  3. Google Gemini Interactions API for coherent story video generation.
  4. Pexels API (free stock video).
  5. Pixabay API with isolated subject search.

Falls back to generating solid-colour placeholder clips when no API
keys are available, so the pipeline can run end-to-end locally.
"""

from __future__ import annotations

import base64
import logging
from pathlib import Path
import time
from typing import List, Optional

import PIL.Image
if not hasattr(PIL.Image, "ANTIALIAS"):
    PIL.Image.ANTIALIAS = getattr(PIL.Image, "LANCZOS", getattr(PIL.Image, "Resampling", None) and PIL.Image.Resampling.LANCZOS)

import requests
from tenacity import retry, stop_after_attempt, wait_exponential

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class GeminiVideoQuotaError(RuntimeError):
    """Raised when the Google project has no available Gemini video quota."""


class BRollFetcherService:
    """Download or generate short B-roll video clips for each visual prompt."""

    PEXELS_SEARCH_URL = "https://api.pexels.com/videos/search"
    PIXABAY_SEARCH_URL = "https://pixabay.com/api/videos/"
    KLING_API_URL = "https://api-singapore.klingai.com"
    LUMA_GENERATIONS_URL = "https://agents.lumalabs.ai/v1/generations"

    def __init__(
        self,
        pexels_key: Optional[str] = None,
        pixabay_key: Optional[str] = None,
        gemini_api_key: Optional[str] = None,
        google_flow_key: Optional[str] = None,
        luma_api_key: Optional[str] = None,
        kling_api_key: Optional[str] = None,
    ):
        settings_fresh = get_settings()
        self._pexels_key = pexels_key or settings_fresh.PEXELS_API_KEY
        self._pixabay_key = pixabay_key or settings_fresh.PIXABAY_API_KEY
        self._gemini_api_key = gemini_api_key or settings_fresh.GEMINI_API_KEY
        self._google_flow_key = google_flow_key or getattr(settings_fresh, "GOOGLE_FLOW_API_KEY", "")
        self._luma_api_key = (
            settings_fresh.LUMA_API_KEY if luma_api_key is None else luma_api_key
        )
        self._kling_api_key = (
            settings_fresh.KLING_API_KEY if kling_api_key is None else kling_api_key
        )

    def fetch_clips(
        self,
        prompts: List[str],
        output_dir: Path,
        clip_duration_secs: int = 5,
        topic: Optional[str] = None,
        scenes: Optional[List[dict]] = None,
    ) -> List[Path]:
        """
        For each prompt, try to download or generate a matching video clip.
        Priority:
          1. Kling AI if a Kling API key is configured
          2. Luma Ray 3.2 if a Luma API key is configured
          3. Gemini video generation if a Google API key is configured
          4. Pexels stock video API
          5. Pixabay stock video API with isolated subject search
          6. Dynamic procedural visual clip
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        clip_paths: List[Path] = []
        used_ids: set = set()

        # If a scene storyboard is provided, derive visual prompts directly from it for tighter alignment
        if scenes:
            # Prefer the explicit visual_search_query from each scene; fall back to the scene narration if missing
            prompts = [scene.get("visual_search_query") or scene.get("narration_text", "") for scene in scenes]
            logger.info("Using scene‑derived visual prompts for B‑roll: %s", prompts)
        # Extract primary anchor subject from topic to keep ALL clips cohesive and relevant
        anchor_subject = self._extract_anchor_subject(topic, prompts)
        logger.info("Cohesive visual anchor subject established: '%s'", anchor_subject)

        if self._kling_api_key:
            try:
                clip_path = output_dir / "clip_00.mp4"
                story_prompt = self._build_story_video_prompt(topic, scenes, prompts)
                logger.info("Attempting Kling video generation...")
                self._generate_with_kling_video(story_prompt, clip_path, self._kling_api_key)
                if clip_path.exists() and clip_path.stat().st_size > 10000:
                    return [clip_path]
                logger.warning("Kling video returned an empty or invalid video, falling back.")
            except Exception as exc:
                logger.warning("Kling video generation failed (%s) — falling back to next provider.", exc)

        if self._luma_api_key:
            try:
                clip_path = output_dir / "clip_00.mp4"
                story_prompt = self._build_story_video_prompt(topic, scenes, prompts)
                logger.info("Attempting Luma video generation...")
                self._generate_with_luma_video(story_prompt, clip_path, self._luma_api_key)
                if clip_path.exists() and clip_path.stat().st_size > 10000:
                    return [clip_path]
                logger.warning("Luma video returned an empty or invalid video, falling back.")
            except Exception as exc:
                logger.warning("Luma video generation failed (%s) — falling back to next provider.", exc)

        if self._google_flow_key:
            try:
                clip_path = output_dir / "clip_00.mp4"
                story_prompt = self._build_story_video_prompt(topic, scenes, prompts)
                logger.info("Attempting Google Flow video generation...")
                self._generate_with_google_flow(story_prompt, clip_path)
                if clip_path.exists() and clip_path.stat().st_size > 10000:
                    return [clip_path]
                logger.warning("Google video generation returned an empty or invalid video, falling back.")
            except Exception as exc:
                logger.warning("Google video generation failed (%s) — falling back to next provider.", exc)

        if self._gemini_api_key:
            try:
                clip_path = output_dir / "clip_00.mp4"
                story_prompt = self._build_story_video_prompt(topic, scenes, prompts)
                logger.info("Attempting Gemini video generation...")
                self._generate_with_gemini_video(story_prompt, clip_path, self._gemini_api_key)
                if clip_path.exists() and clip_path.stat().st_size > 10000:
                    return [clip_path]
                logger.warning("Gemini video generation returned an empty or invalid video, falling back.")
            except Exception as exc:
                logger.warning("Gemini video generation failed (%s) — falling back to next provider.", exc)

        for idx, prompt in enumerate(prompts):
            clip_path = output_dir / f"clip_{idx:02d}.mp4"
            prompt_low = prompt.lower()
            if any(term in prompt_low for term in ("gaming laptop", "gaming notebook", "laptop", "notebook")):
                logger.info("Using topic-specific procedural laptop visual for prompt: %s", prompt)
                clip_paths.append(
                    self._generate_placeholder_clip(
                        clip_path, prompt, duration_secs=clip_duration_secs
                    )
                )
                continue

            # 3. Try Pexels if configured
            if self._pexels_key:
                try:
                    downloaded = self._fetch_from_pexels(prompt, clip_path, anchor=anchor_subject)
                    clip_paths.append(downloaded)
                    continue
                except Exception as exc:
                    logger.warning(
                        "Pexels fetch failed for prompt %d: %s — trying next source.",
                        idx,
                        exc,
                    )

            # 4. Try Pixabay with an isolated scene query
            if self._pixabay_key:
                try:
                    downloaded = self._fetch_from_pixabay(
                        prompt, clip_path, used_ids=used_ids, clip_index=idx, anchor=anchor_subject
                    )
                    clip_paths.append(downloaded)
                    continue
                except Exception as exc:
                    logger.warning(
                        "Pixabay fetch failed for prompt %d: %s — falling back to procedural visual.",
                        idx,
                        exc,
                    )

            # 5. Fallback: generate a dynamic procedural clip
            placeholder = self._generate_placeholder_clip(
                clip_path, prompt, duration_secs=clip_duration_secs
            )
            clip_paths.append(placeholder)

        return clip_paths

    # ── Pexels integration ───────────────────────────────────────────────

    @retry(
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    def _fetch_from_pexels(self, query: str, output_path: Path, anchor: Optional[str] = None) -> Path:
        """Search Pexels for a video matching *query* and download the best file."""
        headers = {"Authorization": self._pexels_key}

        # Clean query: strip typical prompt filler words to get core visual keywords
        import re
        clean_q = re.sub(
            r"\b(cinematic|aerial|drone|shot|close-up|timelapse|time-lapse|slow-motion|slowmo|video|clip|footage|related to|of a|of|with|in|the|a|an)\b",
            " ",
            query,
            flags=re.IGNORECASE,
        )
        clean_q = " ".join(clean_q.split()[:4]).strip()
        # A topic-wide anchor must not be appended to scene-specific searches:
        # it can turn a correct clip query into a multi-subject or contradictory query.
        search_terms = [clean_q, query]

        data = None
        for q in search_terms:
            if not q:
                continue
            try:
                # 1st try: portrait orientation for 9:16 vertical video
                params = {"query": q, "per_page": 5, "orientation": "portrait", "size": "medium"}
                resp = requests.get(self.PEXELS_SEARCH_URL, headers=headers, params=params, timeout=20)
                if resp.status_code == 200:
                    d = resp.json()
                    if d.get("videos"):
                        data = d
                        logger.info("Pexels match found for query '%s' (%d videos)", q, len(d["videos"]))
                        break

                # 2nd try: standard orientation if portrait has 0 results
                params.pop("orientation", None)
                resp = requests.get(self.PEXELS_SEARCH_URL, headers=headers, params=params, timeout=20)
                if resp.status_code == 200:
                    d = resp.json()
                    if d.get("videos"):
                        data = d
                        break
            except Exception as e:
                logger.warning("Pexels search attempt error for '%s': %s", q, e)

        if not data or not data.get("videos"):
            raise ValueError(f"No Pexels results for query: {query}")

        videos = data.get("videos", [])
        videos = sorted(
            videos,
            key=lambda hit: self._stock_hit_relevance(hit, query, provider="pexels"),
            reverse=True,
        )
        if not self._stock_hit_is_exact_match(videos[0], clean_q or query, provider="pexels"):
            raise ValueError(f"No sufficiently relevant Pexels results for query: {query}")
        video_files = videos[0].get("video_files", [])

        # Prioritize portrait files (height > width) with HD or SD quality
        download_url = None
        portrait_files = [vf for vf in video_files if vf.get("height", 0) >= vf.get("width", 0)]
        candidate_pool = portrait_files if portrait_files else video_files

        for vf in candidate_pool:
            if vf.get("quality") in ("hd", "sd") and vf.get("link"):
                download_url = vf["link"]
                break

        if not download_url and video_files:
            download_url = video_files[0].get("link")

        if not download_url:
            raise ValueError("No downloadable video file found in Pexels results.")

        logger.info("Downloading Pexels clip from: %s", download_url)
        dl = requests.get(download_url, timeout=120)
        dl.raise_for_status()
        output_path.write_bytes(dl.content)
        logger.info("Saved clip to %s (%d bytes)", output_path, len(dl.content))
        return output_path

    # ── Pixabay integration ──────────────────────────────────────────────

    @retry(
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    def _fetch_from_pixabay(
        self,
        query: str,
        output_path: Path,
        used_ids: Optional[set] = None,
        clip_index: int = 0,
        anchor: str = "",
    ) -> Path:
        """Search Pixabay for a video matching one isolated subject query."""
        import re
        if used_ids is None:
            used_ids = set()

        # Clean query: strip typical prompt filler words to extract core search keywords
        clean_q = self._clean_pixabay_query(query)

        is_tr = any(c in query.lower() for c in "çğıöşü") or any(
            w in query.lower() for w in [
                "araba", "yapay", "robot", "para", "sehir", "deniz", "yemek",
                "spor", "basari", "surus", "kokpit", "hiz", "kedi", "köpek",
                "kaçıyor", "sarı", "siyah", "beyaz",
            ]
        )

        # Comprehensive Turkish-to-English visual mapping
        tr_to_en = {
            "sarı kedi": "yellow cat",
            "siyah kedi": "black cat",
            "beyaz kedi": "white cat",
            "siyah köpek": "black dog",
            "beyaz köpek": "white dog",
            "sarı köpek": "yellow dog",
            "sarı kedi kaçıyor": "running yellow cat",
            "siyah köpek koşuyor": "running black dog",
            "spor araba": "supercar driving fast",
            "araba kokpit": "luxury car interior",
            "araba": "sports car automotive",
            "otomobil motor": "car engine speed",
            "otomobil": "luxury automobile",
            "kokpit": "car dashboard interior",
            "otoban": "highway driving road",
            "surus": "night driving car",
            "hiz": "speed fast motion",
            "motor": "engine car mechanical",
            "yapay zeka": "artificial intelligence robot",
            "robot": "humanoid robot technology",
            "kodlama": "programming software code",
            "yazilim": "coding computer monitor",
            "teknoloji": "futuristic technology cyber",
            "siber": "cyberpunk neon city",
            "kripto": "cryptocurrency bitcoin gold",
            "borsa": "stock market trading chart",
            "para": "counting cash money",
            "altin": "gold luxury wealth",
            "finans": "finance business investment",
            "fitness": "gym workout bodybuilding",
            "antrenman": "crossfit athletic training",
            "spor": "athlete running workout",
            "motivasyon": "mountain summit sunrise epic",
            "basari": "success achievement businessman",
            "yemek": "chef cooking flames food",
            "lezzet": "gourmet delicious steak",
            "kahve": "espresso coffee pouring",
            "tatli": "chocolate dessert luxury",
            "deniz": "tropical ocean aerial",
            "plaj": "beach drone coastline",
            "manzara": "nature landscape majestic",
            "dag": "mountains aerial fog",
            "uzay": "galaxy space cosmos",
            "oyun": "gaming neon gamer setup",
            "saglik": "doctor hospital medicine",
            "tip": "medical laboratory science",
            "e-ticaret": "ecommerce shopping packages",
            "pazarlama": "digital marketing analytics",
        }

        search_candidates = []
        # If query is in Turkish, find matching English visual keywords and search English FIRST
        if is_tr:
            q_lower = query.lower()
            matched_en = None
            for k, v in tr_to_en.items():
                if k in q_lower:
                    matched_en = v
                    break

            if matched_en:
                clean_translated = self._clean_pixabay_query(matched_en)
                search_candidates.append({"q": clean_translated, "lang": "en"})
            else:
                # Real-time Google Translate for any Turkish topic (cats, dogs, animals, science, etc.)
                translated = self._translate_to_english(clean_q or query)
                if translated and translated.lower() != (clean_q or query).lower():
                    search_candidates.append({
                        "q": self._clean_pixabay_query(translated),
                        "lang": "en",
                    })
            if not search_candidates:
                search_candidates.append({"q": clean_q or query, "lang": "tr"})
        else:
            # Query is already in English
            search_candidates.append({"q": clean_q or query, "lang": "en"})

        data = None
        for cand in search_candidates:
            q_text = cand.get("q", "").strip()
            if not q_text:
                continue
            try:
                params = {
                    "key": self._pixabay_key,
                    "q": q_text,
                    "video_type": "all",
                    "per_page": 20,
                }
                if cand.get("lang"):
                    params["lang"] = cand["lang"]
                resp = requests.get(self.PIXABAY_SEARCH_URL, params=params, timeout=20)
                if resp.status_code == 200:
                    d = resp.json()
                    if d.get("hits"):
                        data = d
                        logger.info(
                            "Pixabay match found for query '%s' (lang=%s, %d hits)",
                            q_text,
                            params.get("lang"),
                            len(d["hits"]),
                        )
                        break
            except Exception as e:
                logger.warning("Pixabay search attempt error for '%s': %s", q_text, e)

        if not data or not data.get("hits"):
            raise ValueError(f"No Pixabay results for query: {query}")

        hits = data["hits"]

        # Re-rank API hits by visible metadata instead of trusting the first result.
        selected_hit = None
        ranked_hits = sorted(
            hits,
            key=lambda hit: (
                self._stock_hit_relevance(hit, query, provider="pixabay"),
                -hits.index(hit),
            ),
            reverse=True,
        )
        for hit in ranked_hits:
            h_id = hit.get("id")
            if (
                h_id
                and h_id not in used_ids
                and self._stock_hit_is_exact_match(hit, clean_q or query, provider="pixabay")
            ):
                selected_hit = hit
                used_ids.add(h_id)
                break

        if not selected_hit:
            raise ValueError(f"No sufficiently relevant unused Pixabay result for query: {query}")

        # Find best MP4 link from videos dictionary
        download_url = None
        video_dict = selected_hit.get("videos", {})
        for size_key in ("medium", "large", "small", "tiny"):
            v_obj = video_dict.get(size_key)
            if v_obj and v_obj.get("url"):
                download_url = v_obj["url"]
                break

        if not download_url:
            raise ValueError("No downloadable video file found in chosen Pixabay hit.")

        logger.info("Downloading Pixabay clip (ID %s) from: %s", selected_hit.get("id"), download_url)
        headers = {"User-Agent": "Mozilla/5.0"}
        dl = requests.get(download_url, headers=headers, timeout=120)
        dl.raise_for_status()
        output_path.write_bytes(dl.content)
        logger.info("Saved Pixabay clip to %s (%d bytes)", output_path, len(dl.content))
        return output_path

    # ── Placeholder generator ────────────────────────────────────────────

    @staticmethod
    def _generate_placeholder_clip(
        output_path: Path,
        prompt: str,
        duration_secs: int = 5,
        width: int = 1080,
        height: int = 1920,
        fps: int = 30,
    ) -> Path:
        """
        Generate a simple placeholder video (solid colour with text overlay)
        using MoviePy so the editing pipeline has clips to work with.
        """
        try:
            import numpy as np
            from PIL import Image, ImageDraw, ImageFont
            from moviepy.editor import VideoClip
            import math, hashlib

            # Deterministic color themes based on prompt hash
            h = int(hashlib.md5(prompt.encode()).hexdigest(), 16)
            themes = [
                ((20, 10, 45), (147, 51, 234), (236, 72, 153)),  # Purple - Pink Cyber
                ((10, 25, 47), (6, 182, 212), (59, 130, 246)),   # Cyan - Blue Tech
                ((30, 15, 10), (249, 115, 22), (234, 179, 8)),   # Sunset Fire
                ((15, 30, 20), (16, 185, 129), (101, 163, 13)),  # Emerald Nature
                ((20, 20, 25), (99, 102, 241), (168, 85, 247)),  # Deep Violet
            ]
            bg_dark, col1, col2 = themes[h % len(themes)]

            # Generate procedural dynamic visual frame for time t
            def make_frame(t):
                img = Image.new("RGB", (width, height), bg_dark)
                draw = ImageDraw.Draw(img)
                for y in range(height):
                    blend = y / max(height - 1, 1)
                    color = tuple(
                        int(bg_dark[channel] * 0.72 + col1[channel] * blend * 0.28)
                        for channel in range(3)
                    )
                    draw.line((0, y, width, y), fill=color)

                prompt_low = prompt.lower()
                if any(word in prompt_low for word in ("ram", "memory", "bellek", "computer", "bilgisayar", "motherboard")):
                    board_y = int(height * 0.30)
                    board_h = int(height * 0.34)
                    draw.rounded_rectangle(
                        (90, board_y, width - 90, board_y + board_h),
                        radius=36,
                        fill=(22, 42, 58),
                        outline=col1,
                        width=10,
                    )
                    chip_y = board_y + 150
                    chip_gap = 18
                    chip_w = (width - 180 - chip_gap * 5) // 4
                    for chip_index in range(4):
                        chip_x = 90 + chip_gap + chip_index * (chip_w + chip_gap)
                        draw.rounded_rectangle(
                            (chip_x, chip_y, chip_x + chip_w, chip_y + 390),
                            radius=18,
                            fill=(13, 24, 38),
                            outline=col2,
                            width=7,
                        )
                        for pin in range(10):
                            pin_y = chip_y + 55 + pin * 30
                            draw.line((chip_x + 20, pin_y, chip_x + chip_w - 20, pin_y), fill=col1, width=4)
                    for circuit_y in range(board_y + 70, board_y + board_h - 40, 95):
                        draw.line((120, circuit_y, width - 120, circuit_y), fill=col1, width=5)
                        draw.ellipse((width - 170, circuit_y - 18, width - 134, circuit_y + 18), fill=col2)
                if any(word in prompt_low for word in ("laptop", "notebook", "gaming pc", "monster")):
                    laptop_y = int(height * 0.30)
                    laptop_h = int(height * 0.34)
                    draw.rounded_rectangle(
                        (120, laptop_y, width - 120, laptop_y + laptop_h),
                        radius=38,
                        fill=(18, 22, 31),
                        outline=col1,
                        width=12,
                    )
                    draw.rounded_rectangle(
                        (170, laptop_y + 55, width - 170, laptop_y + laptop_h - 55),
                        radius=18,
                        fill=(9, 16, 28),
                        outline=(74, 222, 128),
                        width=5,
                    )
                    for glow_index in range(7):
                        glow_y = laptop_y + 130 + glow_index * 48
                        draw.line(
                            (210, glow_y, width - 210, glow_y),
                            fill=col2 if glow_index % 2 else col1,
                            width=7,
                        )
                    keyboard_y = laptop_y + laptop_h + 25
                    draw.polygon(
                        [
                            (170, keyboard_y),
                            (width - 170, keyboard_y),
                            (width - 70, keyboard_y + 190),
                            (70, keyboard_y + 190),
                        ],
                        fill=(28, 31, 43),
                        outline=col1,
                    )
                if any(word in prompt_low for word in ("cat", "kedi")) or any(
                    word in prompt_low for word in ("dog", "köpek", "puppy")
                ):
                    ground_y = int(height * 0.68)
                    draw.rectangle((0, ground_y, width, height), fill=(38, 91, 55))
                    animal_color = (
                        (245, 193, 75) if "yellow" in prompt_low or "sarı" in prompt_low
                        else (35, 35, 38) if "black" in prompt_low or "siyah" in prompt_low
                        else (210, 210, 205)
                    )
                    bob_x = int(width * 0.52 + math.sin(t * 2.2) * 35)
                    bob_y = int(height * 0.42 + math.sin(t * 3.0) * 12)
                    if any(word in prompt_low for word in ("cat", "kedi")):
                        draw.ellipse((bob_x - 210, ground_y - 280, bob_x + 210, ground_y), fill=animal_color)
                        draw.ellipse((bob_x - 150, bob_y - 80, bob_x + 150, bob_y + 220), fill=animal_color)
                        draw.polygon([(bob_x - 120, bob_y - 45), (bob_x - 170, bob_y - 190), (bob_x - 55, bob_y - 100)], fill=animal_color)
                        draw.polygon([(bob_x + 120, bob_y - 45), (bob_x + 170, bob_y - 190), (bob_x + 55, bob_y - 100)], fill=animal_color)
                    else:
                        draw.ellipse((bob_x - 260, ground_y - 250, bob_x + 260, ground_y), fill=animal_color)
                        draw.ellipse((bob_x - 155, bob_y - 70, bob_x + 155, bob_y + 210), fill=animal_color)
                        draw.polygon([(bob_x - 170, bob_y - 25), (bob_x - 250, bob_y + 150), (bob_x - 80, bob_y + 80)], fill=animal_color)
                        draw.polygon([(bob_x + 170, bob_y - 25), (bob_x + 250, bob_y + 150), (bob_x + 80, bob_y + 80)], fill=animal_color)
                        draw.ellipse((bob_x + 105, bob_y + 95, bob_x + 290, bob_y + 190), fill=animal_color)

                # Modern glass badge in lower-middle
                bw, bh = width - 140, 300
                bx = (width - bw) // 2
                by = int(height * 0.78 - bh // 2 + math.sin(t * 2) * 15)
                draw.rounded_rectangle([bx, by, bx + bw, by + bh], radius=24, fill=(12, 16, 34, 235), outline=(255, 255, 255), width=3)

                # Visual scene label
                title = "SCENE VISUAL"
                font_large = ImageFont.truetype("arial.ttf", 46)
                font_small = ImageFont.truetype("arial.ttf", 38)
                draw.text((bx + 35, by + 30), title, fill=(255, 255, 255), font=font_large)
                
                # Truncate prompt
                p_text = prompt[:70] + ("..." if len(prompt) > 70 else "")
                draw.text((bx + 35, by + 105), p_text, fill=(255, 255, 255), font=font_small)
                draw.text((bx + 35, by + 195), f"T = {t:.1f}s | 9:16 VERTICAL B-ROLL", fill=(210, 220, 235), font=font_small)

                return np.array(img)

            clip = VideoClip(make_frame, duration=duration_secs)
            clip.write_videofile(
                str(output_path),
                fps=fps,
                codec="libx264",
                audio=False,
                logger=None,
            )
            logger.info("Dynamic visual B-Roll clip rendered to %s", output_path)
            return output_path
        except Exception as exc:
            logger.warning("Dynamic B-roll generation failed (%s) — falling back to basic clip", exc)
            output_path.write_bytes(b"")
            return output_path

    @staticmethod
    def _stock_hit_relevance(hit: dict, query: str, provider: str = "") -> int:
        """Score a stock result against query terms found in provider metadata."""
        import re

        metadata_parts = []
        if provider == "pixabay":
            metadata_parts = [hit.get("tags", ""), hit.get("pageURL", "")]
        else:
            metadata_parts = [hit.get("url", ""), hit.get("image", "")]
            metadata_parts.extend(str(value) for value in hit.get("tags", []) if value)
            metadata_parts.extend([
                hit.get("user", {}).get("name", "") if isinstance(hit.get("user"), dict) else "",
            ])

        metadata = re.sub(r"[^a-z0-9çğıöşü]+", " ", " ".join(metadata_parts).lower())
        query_clean = re.sub(r"[^a-z0-9çğıöşü]+", " ", query.lower())
        stop_words = {
            "a", "an", "the", "of", "in", "on", "with", "video", "clip",
            "footage", "cinematic", "close", "up", "shot",
        }
        terms = [term for term in query_clean.split() if len(term) > 2 and term not in stop_words]
        if not terms:
            return 0
        metadata_words = set(metadata.split())
        if "cat" in terms and metadata_words.intersection({
            "cheetah", "leopard", "lion", "tiger", "panther", "cougar", "lynx",
        }):
            return -100
        if "dog" in terms and metadata_words.intersection({
            "wolf", "coyote", "jackal", "hyena", "fox",
        }):
            return -100
        matched = sum(term in metadata_words for term in terms)
        return matched * 10 + sum(1 for term in terms if term in metadata)

    @staticmethod
    def _stock_hit_is_exact_match(hit: dict, query: str, provider: str = "") -> bool:
        import re

        metadata_parts = []
        if provider == "pixabay":
            metadata_parts = [hit.get("tags", ""), hit.get("pageURL", "")]
        else:
            metadata_parts = [hit.get("url", ""), hit.get("image", "")]

        metadata_words = set(re.findall(r"[a-z0-9çğıöşü]+", " ".join(metadata_parts).lower()))
        stop_words = {
            "a", "an", "the", "of", "in", "on", "with", "video", "clip",
            "footage", "cinematic", "close", "up", "shot",
        }
        terms = [
            term for term in re.findall(r"[a-z0-9çğıöşü]+", query.lower())
            if len(term) > 2 and term not in stop_words
        ]
        matched_terms = [term for term in terms if term in metadata_words]
        required_matches = len(terms) if len(terms) <= 2 else max(1, (len(terms) + 1) // 2)
        if not terms or len(matched_terms) < required_matches:
            return False
        if "cat" in terms and metadata_words.intersection({
            "cheetah", "leopard", "lion", "tiger", "panther", "cougar", "lynx",
        }):
            return False
        if "dog" in terms and metadata_words.intersection({
            "wolf", "coyote", "jackal", "hyena", "fox",
        }):
            return False
        return True

    @staticmethod
    def _clean_pixabay_query(query: str) -> str:
        """Reduce stock-video queries to a short phrase about one subject."""
        import re

        query = re.split(
            r"\b(and|while|from|with|chasing|pursuing|followed by)\b",
            query,
            maxsplit=1,
            flags=re.IGNORECASE,
        )[0]
        words = re.sub(
            r"\b(cinematic|aerial|drone|shot|close-up|extreme|angle|view|timelapse|time-lapse|slow-motion|slowmo|video|clip|footage|related to|of a|of|in|on|at|by|the|a|an|to|through|hd|4k|towards|preparing|lighting|dramatic|gaze|background|foreground|air|portrait|wide|establishing|scene|photorealistic|ultra-detailed)\b",
            " ",
            query,
            flags=re.IGNORECASE,
        ).split()
        words = [re.sub(r"^[^\w-]+|[^\w-]+$", "", word) for word in words]
        words = [word for word in words if len(word) > 1]
        normalized = " ".join(words[:3])
        normalized_low = normalized.lower()
        ram_terms = ("ram", "memory", "bellek")
        if any(term == normalized_low or term in normalized_low.split() for term in ram_terms):
            normalized = "computer RAM memory hardware"
        return normalized[:100]

    @staticmethod
    def _build_story_video_prompt(
        topic: Optional[str], scenes: Optional[List[dict]], prompts: List[str]
    ) -> str:
        """Describe the complete storyboard in one request to preserve action continuity."""
        scene_lines = []
        for index, scene in enumerate(scenes or []):
            start = scene.get("start_time", 0)
            end = scene.get("end_time", "")
            narration = scene.get("narration_text", "")
            visual = scene.get("visual_description", "")
            scene_lines.append(
                f"Beat {index + 1} ({start}-{end} seconds): {visual or narration}. "
                f"Story action/narration: {narration}"
            )
        if not scene_lines:
            scene_lines = [
                f"Beat {index + 1}: {prompt}" for index, prompt in enumerate(prompts)
            ]

        premise = f"Story premise: {topic}\n" if topic else ""
        return (
            "Create one cinematic, continuous 9:16 vertical video (not a montage) "
            "that tells this story from beginning to end in chronological order.\n"
            f"{premise}"
            "CHARACTER AND ACTION CONTINUITY IS ESSENTIAL: use the exact same "
            "characters throughout; keep each character's species, fur/skin color, "
            "size, and distinctive features unchanged. Never recolor, replace, duplicate, "
            "or add characters. Actions must causally continue from one beat to the next; "
            "do not restart the scene or show unrelated stock-style shots. Show the "
            "specified interaction when the story calls for it. No text, captions, "
            "subtitles, logos, or watermarks.\n"
            "Storyboard:\n"
            + "\n".join(scene_lines)
        )

    @classmethod
    def _generate_with_kling_video(
        cls, prompt: str, output_path: Path, api_key: str
    ) -> Path:
        """Generate and download a 10-second vertical Kling 3.0 video."""
        if not api_key:
            raise ValueError("A Kling API key is required for Kling video generation.")
        if not prompt or len(prompt) > 3072:
            raise ValueError("Kling 3.0 prompts must contain between 1 and 3072 characters.")

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        response = requests.post(
            f"{cls.KLING_API_URL}/text-to-video/kling-3.0",
            headers=headers,
            json={
                "prompt": prompt,
                "settings": {
                    "multi_shot": False,
                    "audio": "off",
                    "resolution": "720p",
                    "aspect_ratio": "9:16",
                    "duration": 10,
                },
                "options": {"watermark_info": {"enabled": False}},
            },
            timeout=(15, 120),
        )
        if not response.ok:
            raise RuntimeError(
                f"Kling video generation request failed with HTTP {response.status_code}: "
                f"{response.text[:1000]}"
            )

        result = response.json()
        if result.get("code") not in (None, 0):
            raise RuntimeError(
                f"Kling rejected video generation ({result.get('code')}): "
                f"{result.get('message', 'No error message provided.')}"
            )
        generation = result.get("data")
        task_id = generation.get("id") if isinstance(generation, dict) else None
        if not task_id:
            raise RuntimeError("Kling did not return a video task ID.")

        deadline = time.monotonic() + 900
        status = generation.get("status")
        while status not in {"succeeded", "failed"}:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    f"Kling video task {task_id} did not finish within 15 minutes."
                )
            time.sleep(min(10, remaining))
            status_response = requests.get(
                f"{cls.KLING_API_URL}/tasks",
                headers=headers,
                params={"task_ids": task_id},
                timeout=(15, 60),
            )
            if not status_response.ok:
                raise RuntimeError(
                    f"Kling task status request failed with HTTP "
                    f"{status_response.status_code}: {status_response.text[:1000]}"
                )
            task_result = status_response.json()
            if task_result.get("code") not in (None, 0):
                raise RuntimeError(
                    f"Kling task query failed ({task_result.get('code')}): "
                    f"{task_result.get('message', 'No error message provided.')}"
                )
            tasks = task_result.get("data")
            if not isinstance(tasks, list) or not tasks or not isinstance(tasks[0], dict):
                raise RuntimeError("Kling returned no task status data.")
            generation = tasks[0]
            status = generation.get("status")

        if status == "failed":
            raise RuntimeError(
                "Kling video generation failed: "
                f"{generation.get('message') or 'No failure reason provided.'}"
            )

        outputs = generation.get("outputs")
        video_url = next(
            (
                output.get("url")
                for output in outputs or []
                if isinstance(output, dict)
                and output.get("type") == "video"
                and output.get("url")
            ),
            None,
        )
        if not video_url:
            raise RuntimeError("Kling completed the task but returned no video URL.")

        download = requests.get(video_url, timeout=(15, 120))
        if not download.ok:
            raise RuntimeError(
                f"Could not download the Kling video (HTTP {download.status_code})."
            )
        if len(download.content) <= 10000:
            raise RuntimeError("Kling returned an empty or invalid video.")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(download.content)
        logger.info(
            "Kling AI video saved to %s (%d bytes)", output_path, len(download.content)
        )
        return output_path

    @classmethod
    def _generate_with_luma_video(
        cls, prompt: str, output_path: Path, api_key: str
    ) -> Path:
        """Create one continuous Ray 3.2 video, poll its job, and download the MP4."""
        if not api_key:
            raise ValueError("A Luma API key is required for Luma video generation.")
        if not prompt or len(prompt) > 6000:
            raise ValueError("Luma video prompts must contain between 1 and 6000 characters.")

        headers = {"Authorization": f"Bearer {api_key}"}
        response = requests.post(
            cls.LUMA_GENERATIONS_URL,
            headers=headers,
            json={
                "model": "ray-3.2",
                "type": "video",
                "prompt": prompt,
                "aspect_ratio": "9:16",
                "video": {"resolution": "720p", "duration": "10s"},
            },
            timeout=(15, 120),
        )
        if not response.ok:
            raise RuntimeError(
                f"Luma video generation request failed with HTTP {response.status_code}: "
                f"{response.text[:1000]}"
            )

        generation = response.json()
        generation_id = generation.get("id")
        if not generation_id:
            raise RuntimeError("Luma did not return a generation ID.")

        deadline = time.monotonic() + 600
        state = generation.get("state")
        while state not in {"completed", "failed"}:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    f"Luma video generation {generation_id} did not finish within 10 minutes."
                )
            time.sleep(min(30, remaining))
            status_response = requests.get(
                f"{cls.LUMA_GENERATIONS_URL}/{generation_id}",
                headers=headers,
                timeout=(15, 60),
            )
            if not status_response.ok:
                raise RuntimeError(
                    f"Luma generation status request failed with HTTP "
                    f"{status_response.status_code}: {status_response.text[:1000]}"
                )
            generation = status_response.json()
            state = generation.get("state")

        if state == "failed":
            reason = generation.get("failure_reason") or "No failure reason provided."
            code = generation.get("failure_code")
            raise RuntimeError(f"Luma video generation failed ({code or 'unknown'}): {reason}")

        outputs = generation.get("output")
        video_url = (
            outputs[0].get("url")
            if isinstance(outputs, list) and outputs and isinstance(outputs[0], dict)
            else None
        )
        if not video_url:
            raise RuntimeError("Luma completed the generation but returned no video URL.")

        download = requests.get(video_url, timeout=(15, 120))
        if not download.ok:
            raise RuntimeError(
                f"Could not download the Luma video (HTTP {download.status_code})."
            )
        if len(download.content) <= 10000:
            raise RuntimeError("Luma returned an empty or invalid video.")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(download.content)
        logger.info(
            "Luma Ray 3.2 video saved to %s (%d bytes)",
            output_path,
            len(download.content),
        )
        return output_path

    def _generate_with_google_flow(
        self, prompt: str, output_path: Path, anchor: str = ""
    ) -> Path:
        """Use the Gemini video-generation API configured through GOOGLE_FLOW_API_KEY."""
        del anchor
        return self._generate_with_gemini_video(
            prompt, output_path, self._google_flow_key
        )

    @staticmethod
    def _generate_with_gemini_video(
        prompt: str, output_path: Path, api_key: str
    ) -> Path:
        """Generate one coherent video via Google's documented Gemini Interactions API."""
        if not api_key:
            raise ValueError("A Gemini API key is required for AI video generation.")

        logger.info("Requesting continuous AI video generation for the storyboard.")
        response = requests.post(
            "https://generativelanguage.googleapis.com/v1beta/interactions",
            headers={"x-goog-api-key": api_key},
            json={
                "model": "gemini-omni-1.1-flash",
                "input": prompt,
                "response_format": {
                    "type": "video",
                    "aspect_ratio": "9:16",
                    "resolution": "720p",
                },
            },
            timeout=(15, 600),
        )
        if not response.ok:
            if response.status_code == 429:
                try:
                    error_body = response.json()
                except ValueError:
                    error_body = {}
                error_info = error_body.get("error", {}) if isinstance(error_body, dict) else {}
                provider_message = str(error_info.get("message", response.text[:1000]))
                normalized_message = provider_message.lower()
                if "limit: 0" in normalized_message and "free tier" in normalized_message:
                    raise GeminiVideoQuotaError(
                        "Google Gemini video kotası bu proje için Free Tier'da 0. "
                        "Bu durum kodla tekrar deneyerek çözülemez: Google AI Studio'da "
                        "API anahtarının bağlı olduğu projede billing'i etkinleştirip "
                        "Paid Tier'a yükseltin, ardından yeniden deneyin. Yeni API anahtarı "
                        "tek başına kota sağlamaz. Kota: "
                        "https://aistudio.google.com/rate-limit?timeRange=last-28-days "
                        "| Billing: https://ai.google.dev/gemini-api/docs/billing"
                    )
            raise RuntimeError(
                f"Gemini video generation failed with HTTP {response.status_code}: "
                f"{response.text[:1000]}"
            )

        result = response.json()
        video_output = result.get("output_video")
        encoded_video = (
            video_output.get("data") if isinstance(video_output, dict) else None
        )
        if not encoded_video:
            for step in result.get("steps", []):
                if step.get("type") != "model_output":
                    continue
                for item in step.get("content", []):
                    if item.get("type") == "video" and item.get("mime_type") == "video/mp4":
                        encoded_video = item.get("data")
                        break
                if encoded_video:
                    break
        if not encoded_video:
            raise RuntimeError(
                "Gemini video generation returned no MP4 data. "
                f"Response status: {result.get('status', 'unknown')}."
            )

        try:
            video_bytes = base64.b64decode(encoded_video, validate=True)
        except (ValueError, TypeError) as exc:
            raise RuntimeError("Gemini returned invalid base64 video data.") from exc
        if len(video_bytes) <= 10000:
            raise RuntimeError("Gemini returned an empty or invalid video.")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(video_bytes)
        logger.info("AI-generated story video saved to %s (%d bytes)", output_path, len(video_bytes))
        return output_path

    # ── Instant Free Translation Helper ──────────────────────────────────

    # ── Subject Anchor Extractor ─────────────────────────────────────────

    @staticmethod
    def _extract_anchor_subject(topic: Optional[str], prompts: List[str]) -> str:
        """Extract cohesive core anchor noun (e.g. 'cat', 'car', 'coding', 'gym') so all scenes stay consistent."""
        if not topic:
            return ""
        t_low = topic.lower()
        mapping = {
            "kedi": "cat",
            "köpek": "dog",
            "araba": "car",
            "otomobil": "car",
            "yapay zeka": "ai robot",
            "robot": "robot",
            "para": "money",
            "zengin": "wealth",
            "fitness": "fitness gym",
            "spor": "athlete sports",
            "yemek": "cooking food",
            "kahve": "coffee",
            "deniz": "ocean",
            "kod": "programming code",
            "yazılım": "software coding",
            "uzay": "space galaxy",
        }
        for k, v in mapping.items():
            if k in t_low:
                return v

        for eng in ["cat", "dog", "car", "robot", "money", "gym", "coffee", "ocean", "space", "coding"]:
            if eng in t_low:
                return eng

        if prompts:
            first_w = prompts[0].split()[0].lower()
            if len(first_w) > 2 and first_w not in ["the", "a", "an", "and", "close", "slow"]:
                return first_w
        return ""
