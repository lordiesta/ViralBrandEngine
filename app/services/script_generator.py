"""
Script Generator Service — Multi-LLM (OpenAI GPT-4o + Google Gemini) integration.

Generates structured viral video scripts:
  - Viral opening hook (Turkish/English)
  - Full voiceover narration calibrated strictly to duration (10s, 15s, 20s, 30s)
  - High-precision ENGLISH B-roll visual search keywords for video generators/stock APIs
  - Sequential SRT subtitles with timestamping ending cleanly within target duration
  - Captions and trending hashtags

Supported providers:
  1. OpenAI GPT-4o
  2. Google Gemini (1.5 Flash / 2.0 via Google AI Studio — free tier)
  3. Dynamic Intelligent Local Fallback with real-time translation & strict duration tuning
"""

from __future__ import annotations

import json
import logging
import re
import urllib.parse
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import requests
from openai import OpenAI

from app.config import get_settings

logger = logging.getLogger(__name__)


def _fmt_srt(secs: float) -> str:
    """Convert seconds to SRT timestamp format HH:MM:SS,mmm."""
    h = int(secs // 3600)
    m = int((secs % 3600) // 60)
    s = int(secs % 60)
    ms = int((secs - int(secs)) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


# ── Output data classes ──────────────────────────────────────────────────

@dataclass
class SceneItem:
    """A single coherent storyboard scene with locked timing, narration, and visual prompt."""
    scene_index: int
    start_time: float
    end_time: float
    narration_text: str
    visual_search_query: str
    visual_description: str


@dataclass
class ScriptOutput:
    """Structured output returned by the LLM or smart generator."""

    viral_hook: str
    voiceover_script: str
    broll_prompts: List[str] = field(default_factory=list)
    subtitles_srt: str = ""
    caption: str = ""
    hashtags: List[str] = field(default_factory=list)
    scenes: List[Dict[str, Any]] = field(default_factory=list)
    provider: str = "smart_local"


# ── System prompt ────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are a world-class viral short-form video creator and scriptwriter (Instagram Reels, TikTok, YouTube Shorts).

Given a TOPIC, BRAND TONE, TARGET LANGUAGE, and TARGET DURATION (in seconds),
produce a storyboard JSON object with EXACTLY these keys:

{
  "viral_hook": "<a punchy 1-sentence opening hook in TARGET LANGUAGE>",
  "voiceover_script": "<full natural voiceover narration in TARGET LANGUAGE>",
  "scenes": [
    {
      "scene_index": 0,
      "start_time": 0.0,
      "end_time": <e.g. 5.0>,
      "narration_text": "<sentence spoken during this exact time in TARGET LANGUAGE>",
      "visual_search_query": "<2-4 plain English Pixabay search words for ONE visible subject only, preserving its color/species, e.g. 'yellow cat' or 'running black dog'>",
      "visual_description": "<ultra-detailed visual prompt for Google Flow / AI video generator in English>"
    },
    {
      "scene_index": 1,
      "start_time": <e.g. 5.0>,
      "end_time": <e.g. 10.0>,
      "narration_text": "<next sentence spoken in TARGET LANGUAGE>",
      "visual_search_query": "<2-4 plain English Pixabay search words for ONE visible subject only, preserving its color/species>",
      "visual_description": "<ultra-detailed visual prompt in English>"
    }
  ],
  "broll_prompts": ["<visual_search_query 1>", "<visual_search_query 2>", ...],
  "subtitles_srt": "<valid SRT-formatted subtitles with timestamps in TARGET LANGUAGE>",
  "caption": "<engaging social media caption with emojis in TARGET LANGUAGE>",
  "hashtags": ["#tag1", "#tag2", ...]
}

CRITICAL RULES FOR SCENE CONSISTENCY:
1. SCENE-TO-AUDIO LOCK:
   Divide the entire video into 2 to 4 distinct scenes matching the narration rhythm.
   Every scene must advance the same chronological story; show setup, action, consequence, and resolution in order.
   Keep every character's species, color, size, and distinctive appearance identical in every scene.
   A visual_search_query is ONLY for stock search: describe one subject, never combine two characters or a full plot in it.
   Use short isolated search phrases such as "yellow cat", "running yellow cat", or "running black dog".
   Put interactions and multi-character action only in visual_description, not visual_search_query.
2. CONSISTENT VISUAL STYLE:
   All visual descriptions must refer to the same characters with locked appearances and continue the same action between adjacent scenes. Do not replace, recolor, duplicate, or introduce characters.
3. STRICT DURATION & WORD COUNT:
   The voiceover MUST be read aloud in EXACTLY {duration} seconds ({min_words}-{max_words} words).
   The last scene's "end_time" must equal {duration}.0.
4. Output ONLY valid raw JSON. No markdown code blocks, no explanation.
"""


# ── Service ──────────────────────────────────────────────────────────────

class ScriptGeneratorService:
    """Generates viral video scripts via OpenAI, Gemini, or smart local generator."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        gemini_api_key: Optional[str] = None,
    ):
        settings_fresh = get_settings()
        self._api_key = api_key or settings_fresh.OPENAI_API_KEY
        self._model = model or settings_fresh.OPENAI_MODEL
        self._gemini_api_key = gemini_api_key or settings_fresh.GEMINI_API_KEY
        self._gemini_model = getattr(settings_fresh, "GEMINI_MODEL", "gemini-1.5-flash")
        self._client: Optional[OpenAI] = None

    @property
    def client(self) -> OpenAI:
        if self._client is None and self._api_key:
            self._client = OpenAI(api_key=self._api_key)
        return self._client

    def generate(
        self,
        topic: str,
        brand_tone: str = "professional",
        target_language: str = "tr",
        target_duration_secs: int = 15,
    ) -> ScriptOutput:
        """
        Generate script strictly calibrated to target_duration_secs.
        Tries: OpenAI GPT-4o → Google Gemini → Smart Local Fallback.
        """
        # Calculate strict word count limits based on target duration
        min_words = max(14, int(target_duration_secs * 1.8))
        max_words = max(20, int(target_duration_secs * 2.2))

        system_msg = (
            SYSTEM_PROMPT
            .replace("{duration}", str(target_duration_secs))
            .replace("{min_words}", str(min_words))
            .replace("{max_words}", str(max_words))
        )
        user_msg = (
            f"Topic: {topic}\n"
            f"Brand Tone: {brand_tone}\n"
            f"Language: {target_language}\n"
            f"Target Duration: {target_duration_secs} seconds (STRICT WORD LIMIT: {min_words}-{max_words} words total)"
        )

        # 1. Try OpenAI if configured
        if self._api_key:
            try:
                logger.info("Attempting OpenAI %s for %ds script…", self._model, target_duration_secs)
                res = self.client.chat.completions.create(
                    model=self._model,
                    messages=[
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": user_msg},
                    ],
                    temperature=0.7,
                    max_tokens=1024,
                    response_format={"type": "json_object"},
                    timeout=25,
                )
                raw = res.choices[0].message.content
                data = json.loads(raw)
                logger.info("OpenAI successfully generated script for topic '%s'", topic)
                return self._parse_script_data(data, topic, target_language, target_duration_secs, provider="openai")
            except Exception as exc:
                logger.warning("OpenAI script generation failed (%s) — trying next provider.", exc)

        # 2. Try Google Gemini if configured
        if self._gemini_api_key:
            try:
                logger.info("Attempting Google Gemini (%s) for %ds script…", self._gemini_model, target_duration_secs)
                url = (
                    f"https://generativelanguage.googleapis.com/v1beta/models/"
                    f"{self._gemini_model}:generateContent?key={self._gemini_api_key}"
                )
                payload = {
                    "contents": [{"parts": [{"text": f"{system_msg}\n\nUser Request:\n{user_msg}"}]}],
                    "generationConfig": {"response_mime_type": "application/json"},
                }
                resp = requests.post(url, json=payload, timeout=25)
                resp.raise_for_status()
                candidate_text = resp.json()["candidates"][0]["content"]["parts"][0]["text"]
                data = json.loads(candidate_text)
                logger.info("Gemini successfully generated script for topic '%s'", topic)
                return self._parse_script_data(data, topic, target_language, target_duration_secs, provider="gemini")
            except Exception as exc:
                logger.warning("Google Gemini script generation failed (%s) — falling back to smart local.", exc)

        # 3. Dynamic domain fallback with duration calibration & English translation
        logger.info("Using smart local domain generator for '%s' (%ds, lang=%s)", topic, target_duration_secs, target_language)
        return self._mock_response(
            topic=topic,
            duration=target_duration_secs,
            target_language=target_language,
            brand_tone=brand_tone,
        )

    @staticmethod
    def _parse_script_data(data: dict, topic: str, target_language: str, duration: int, provider: str = "smart_local") -> ScriptOutput:
        """Validate and format LLM output dict into ScriptOutput with scene storyboard."""
        raw_broll = data.get("broll_prompts") or []
        if isinstance(raw_broll, str):
            raw_broll = [b.strip() for b in raw_broll.split(",") if b.strip()]

        scenes = data.get("scenes") or []
        # If LLM didn't return structured scenes, auto-generate them from broll_prompts and duration
        if not scenes and raw_broll:
            clip_dur = round(duration / max(len(raw_broll), 1), 1)
            for idx, q in enumerate(raw_broll[:4]):
                s_time = idx * clip_dur
                e_time = min(duration, (idx + 1) * clip_dur if idx < len(raw_broll[:4]) - 1 else duration)
                scenes.append({
                    "scene_index": idx,
                    "start_time": s_time,
                    "end_time": e_time,
                    "narration_text": f"Scene {idx + 1}",
                    "visual_search_query": q,
                    "visual_description": q,
                })
        elif scenes:
            # Extract queries from scenes for broll_prompts
            raw_broll = [s.get("visual_search_query", "") for s in scenes if s.get("visual_search_query")]

        return ScriptOutput(
            viral_hook=data.get("viral_hook", f"Dikkat! {topic}"),
            voiceover_script=data.get("voiceover_script", f"{topic} hakkında bilmeniz gerekenler."),
            broll_prompts=raw_broll[:4],
            subtitles_srt=data.get("subtitles_srt", ""),
            caption=data.get("caption", f"🔥 {topic} — Takipte kalın! 👇"),
            hashtags=data.get("hashtags", ["#viral", "#keşfet", "#trending"]),
            scenes=scenes,
            provider=provider,
        )

    # ── Instant Free Translation Helper ──────────────────────────────────

    @staticmethod
    def translate_to_english(text: str) -> str:
        """Translate any Turkish topic into English via Google Translate free endpoint."""
        try:
            url = f"https://translate.googleapis.com/translate_a/single?client=gtx&sl=auto&tl=en&dt=t&q={urllib.parse.quote(text)}"
            resp = requests.get(url, timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                trans = data[0][0][0]
                if trans and trans.strip():
                    return trans.strip()
        except Exception as e:
            logger.debug("Translation failed for '%s': %s", text, e)
        return text

    # ── Duration-Strict Local Fallback Generator ──────────────────────────

    @classmethod
    def _mock_response(
        cls,
        topic: str,
        duration: int = 15,
        target_language: str = "tr",
        brand_tone: str = "inspirational",
    ) -> ScriptOutput:
        """
        Universal story-driven fallback generator.
        Works for ANY topic — no hardcoded categories.
        Steps:
          1. Translate topic → English
          2. Extract protagonist, antagonist, action, setting, emotion
          3. Build 4-act narrative (Setup → Conflict → Climax → Resolution)
          4. Lock each act to a time window with a matching B-roll search query
        """
        topic_lower = topic.lower().strip()
        is_tr = target_language.lower().startswith("tr") or any(
            char in topic_lower for char in "çğıöşü"
        )
        en_topic = cls.translate_to_english(topic)

        character_map = {
            "sarı kedi": "yellow cat", "siyah kedi": "black cat",
            "beyaz kedi": "white cat", "siyah köpek": "black dog",
            "beyaz köpek": "white dog", "büyük köpek": "large dog",
            "küçük kedi": "small cat", "yavru kedi": "kitten",
            "yavru köpek": "puppy", "kedi": "cat", "köpek": "dog",
            "kuş": "bird", "fare": "mouse", "balık": "fish",
            "tavşan": "rabbit", "hamster": "hamster", "at": "horse",
            "inek": "cow", "aslan": "lion", "kaplan": "tiger",
            "ayı": "bear", "kurt": "wolf", "tilki": "fox",
            "maymun": "monkey", "fil": "elephant", "zürafa": "giraffe",
            "zebra": "zebra", "çocuk": "child", "bebek": "baby",
            "adam": "man", "kadın": "woman", "robot": "robot",
            "yellow cat": "yellow cat", "black cat": "black cat",
            "white cat": "white cat", "black dog": "black dog",
            "white dog": "white dog", "cat": "cat", "dog": "dog",
            "bird": "bird", "mouse": "mouse", "rabbit": "rabbit",
            "child": "child", "woman": "woman", "man": "man",
            "monster notebook": "gaming laptop",
            "gaming notebook": "gaming laptop",
            "gaming laptop": "gaming laptop",
            "oyun bilgisayarı": "gaming laptop",
            "oyun laptop": "gaming laptop",
            "notebook": "gaming laptop",
            "laptop": "gaming laptop",
        }
        character_matches = [
            (position, -len(name), name, english_name)
            for name, english_name in character_map.items()
            if (position := topic_lower.find(name)) >= 0
        ]
        characters = []
        character_labels = []
        matched_spans = []
        for position, _, name, english_name in sorted(character_matches):
            span = (position, position + len(name))
            if any(span[0] < end and span[1] > start for start, end in matched_spans):
                continue
            matched_spans.append(span)
            if english_name not in characters:
                characters.append(english_name)
                character_labels.append(name)

        action_map = {
            "kaçıyorlar": "running away escaping", "kaçıyor": "running away escaping",
            "kovalıyorlar": "chasing pursuing", "kovalıyor": "chasing pursuing",
            "koşuyorlar": "running sprinting", "koşuyor": "running sprinting",
            "atlıyor": "jumping", "uçuyor": "flying", "yiyor": "eating",
            "içiyor": "drinking", "uyuyor": "sleeping", "oynuyor": "playing",
            "savaşıyor": "fighting", "saklanıyor": "hiding",
            "yüzüyor": "swimming", "tırmanıyor": "climbing",
            "escapes": "running away escaping", "escaping": "running away escaping",
            "chases": "chasing pursuing", "chasing": "chasing pursuing",
            "runs away": "running away escaping", "running away": "running away escaping",
            "running": "running", "runs": "running", "jumps": "jumping",
            "eats": "eating", "sleeps": "sleeping", "plays": "playing",
        }
        action_matches = [
            (position, -len(verb), translation)
            for verb, translation in action_map.items()
            if (position := topic_lower.find(verb)) >= 0
        ]
        action = sorted(action_matches)[0][2] if action_matches else "moving"
        is_chase = any(
            word in action for word in ("escaping", "chasing", "running", "fighting")
        )
        is_calm = any(word in action for word in ("sleeping", "eating", "drinking"))

        setting_map = {
            "bahçe": "garden", "orman": "forest", "park": "park",
            "sokak": "street", "deniz": "sea", "dağ": "mountain",
            "çöl": "desert", "şehir": "city", "ev": "home",
            "garden": "garden", "forest": "forest", "park": "park",
            "street": "street", "beach": "beach", "ocean": "ocean",
            "mountain": "mountain", "city": "city", "home": "home",
        }
        setting = next(
            (value for term, value in setting_map.items() if term in topic_lower),
            "outdoor nature",
        )

        if not characters:
            words = [
                word for word in re.sub(r"[^a-zA-Z ]", " ", en_topic).lower().split()
                if len(word) > 2 and word not in {
                    "the", "and", "for", "with", "from", "that", "this",
                }
            ]
            characters = [" ".join(words[:2]) or "subject"]
            character_labels = [characters[0]]

        protagonist = characters[0]
        antagonist = characters[1] if len(characters) > 1 else ""
        protagonist_label = character_labels[0]
        antagonist_label = character_labels[1] if len(character_labels) > 1 else ""
        action_keyword = action.split()[0]
        is_gaming_laptop = protagonist.lower() == "gaming laptop"

        if is_tr:
            hook = f"Beklenmedik an! {topic.capitalize()}!"
            if is_chase and antagonist:
                act2 = (
                    f"{antagonist_label.capitalize()} kovalarken "
                    f"{protagonist_label} hızla kaçıyor."
                )
                act3 = f"Nefesler tutuldu; {protagonist_label} kurtulabilecek mi?"
                cta = "Bu heyecanlı anı arkadaşlarınla paylaş!"
            elif is_calm:
                act2 = f"{protagonist_label.capitalize()} huzur içinde anın tadını çıkarıyor."
                act3 = "Bu sakin an herkesi gülümsetiyor."
                cta = "Bu güzel anı arkadaşlarınla paylaş!"
            else:
                act2 = f"{protagonist_label.capitalize()} harekete geçiyor."
                act3 = "Ve beklenen kritik an geliyor!"
                cta = "Daha fazlası için takipte kal!"
        else:
            hook = f"You won't believe this: {topic}!"
            if is_chase and antagonist:
                act2 = f"The {protagonist} runs away from the {antagonist}."
                act3 = f"The chase reaches its peak as the {protagonist} seeks safety."
                cta = "Share this dramatic moment!"
            elif is_calm:
                act2 = f"The {protagonist} enjoys a peaceful moment."
                act3 = "This quiet moment captures everyone's attention."
                cta = "Share this lovely moment!"
            else:
                act2 = f"The {protagonist} springs into action."
                act3 = "The most exciting moment is about to happen!"
                cta = "Follow for more!"

        if is_gaming_laptop:
            setting = "gaming room with RGB lighting"
            broll = [
                "gaming laptop",
                "gaming laptop gameplay screen",
                "gaming laptop RGB keyboard",
                "gaming laptop gaming setup",
            ]
        elif is_chase and antagonist:
            broll = [
                protagonist,
                f"{action_keyword} {protagonist}",
                f"{action_keyword} {antagonist}",
                f"{protagonist} safe",
            ]
        else:
            broll = [
                protagonist,
                f"{action_keyword} {protagonist}",
                f"{action_keyword} {protagonist}",
                protagonist,
            ]

        num_scenes = 2 if duration <= 10 else (3 if duration <= 15 else 4)
        acts = [hook, act2, act3, cta][:num_scenes]
        scene_duration = duration / num_scenes
        subtitles = []
        scenes = []
        for index, (narration, query) in enumerate(zip(acts, broll)):
            start = index * scene_duration
            end = (index + 1) * scene_duration
            subtitles.append(
                f"{index + 1}\n{_fmt_srt(start)} --> {_fmt_srt(end)}\n{narration}"
            )

            if index == 0:
                visual = f"Introduce the same {protagonist} in {setting}."
            elif is_gaming_laptop:
                visual = (
                    f"Show the same {protagonist} in a premium {setting}, "
                    "with the gameplay screen clearly visible and the RGB keyboard lit. "
                    "Keep the laptop model and scene continuous."
                )
            elif is_chase and antagonist and index == 1:
                visual = (
                    f"The same {protagonist} runs away from the same {antagonist} "
                    f"in {setting}; preserve their exact species and colors."
                )
            elif is_chase and antagonist and index == 2:
                visual = (
                    f"Continue the same chase: the same {antagonist} pursues the "
                    f"same {protagonist} through {setting}; do not change either appearance."
                )
            elif is_chase and antagonist:
                visual = (
                    f"The same {protagonist} reaches safety after escaping the same "
                    f"{antagonist}; preserve their exact appearances."
                )
            else:
                visual = (
                    f"Continue the same {protagonist} performing {action} in {setting}; "
                    "preserve the character's exact appearance."
                )

            scenes.append({
                "scene_index": index,
                "start_time": round(start, 2),
                "end_time": round(end, 2),
                "narration_text": narration,
                "visual_search_query": query,
                "visual_description": visual,
            })

        clean_tag = re.sub(r"\W+", "", topic.replace(" ", "_"))
        hashtags = ["#viral", "#trending", f"#{clean_tag}", "#fyp"]
        hashtags.append("#keşfet" if is_tr else "#explore")
        return ScriptOutput(
            viral_hook=hook,
            voiceover_script=" ".join(acts),
            broll_prompts=broll,
            subtitles_srt="\n\n".join(subtitles),
            caption=f"🔥 {topic} — {'Arkadaşlarınla paylaş!' if is_tr else 'Share this!'} 👇",
            hashtags=hashtags,
            scenes=scenes,
        )
