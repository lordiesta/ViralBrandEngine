"""
Voice Synthesizer Service — Multi-provider TTS with Turkish support.

Priority chain:
  1. ElevenLabs (eleven_multilingual_v2) — best quality, supports Turkish
  2. Google Cloud TTS (tr-TR WaveNet) — free 1M chars/month, real Turkish voice
  3. pyttsx3 (Windows SAPI5) with Turkish voice selection
  4. Harmonic tone WAV fallback (last resort)
"""

from __future__ import annotations

import logging
import struct
import wave
from pathlib import Path
from typing import Optional

import requests

from app.config import get_settings

logger = logging.getLogger(__name__)


class VoiceSynthesizerService:
    """Synthesises voiceover audio via ElevenLabs → Google TTS → SAPI5 → tone."""

    ELEVENLABS_TTS_URL = "https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
    GOOGLE_TTS_URL = "https://texttospeech.googleapis.com/v1/text:synthesize"

    def __init__(
        self,
        api_key: Optional[str] = None,
        voice_id: Optional[str] = None,
        google_api_key: Optional[str] = None,
    ):
        settings_fresh = get_settings()
        self._api_key = api_key or settings_fresh.ELEVENLABS_API_KEY
        self._voice_id = voice_id or settings_fresh.ELEVENLABS_VOICE_ID
        self._google_api_key = google_api_key or settings_fresh.GOOGLE_TTS_API_KEY

    def synthesize(
        self,
        text: str,
        output_path: Path,
        language: str = "tr",
        stability: float = 0.5,
        similarity_boost: float = 0.75,
    ) -> Path:
        """
        Convert *text* to speech and save the audio to *output_path*.
        Tries: ElevenLabs → Google Cloud TTS → SAPI5 → tone fallback.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # 1. Try ElevenLabs
        if self._api_key:
            result = self._try_elevenlabs(text, output_path, stability, similarity_boost)
            if result:
                return result

        # 2. Try Google Cloud TTS (best Turkish WaveNet support)
        if self._google_api_key:
            result = self._try_google_tts(text, output_path, language)
            if result:
                return result

        # 3. Try gTTS (Free Google TTS with fluent Turkish voice, zero API key needed)
        result = self._try_gtts(text, output_path, language)
        if result:
            return result

        # 4. Try SAPI5 (Windows native TTS) with Turkish voice preference
        result = self._try_sapi5(text, output_path, language)
        if result:
            return result

        # 5. Last resort: tone WAV
        return self._generate_tone_audio(output_path)

    # ── ElevenLabs ────────────────────────────────────────────────────────

    def _try_elevenlabs(
        self, text: str, output_path: Path, stability: float, similarity_boost: float
    ) -> Optional[Path]:
        url = self.ELEVENLABS_TTS_URL.format(voice_id=self._voice_id)
        headers = {
            "xi-api-key": self._api_key,
            "Content-Type": "application/json",
            "Accept": "audio/mpeg",
        }
        payload = {
            "text": text,
            "model_id": "eleven_multilingual_v2",
            "voice_settings": {
                "stability": stability,
                "similarity_boost": similarity_boost,
            },
        }
        try:
            logger.info("Requesting ElevenLabs TTS (%d chars)…", len(text))
            resp = requests.post(url, json=payload, headers=headers, timeout=30)
            resp.raise_for_status()
            output_path.write_bytes(resp.content)
            logger.info("ElevenLabs audio saved to %s (%d bytes)", output_path, len(resp.content))
            return output_path
        except Exception as exc:
            logger.warning("ElevenLabs failed (%s) — trying Google TTS.", exc)
            return None

    # ── Google Cloud TTS ──────────────────────────────────────────────────

    def _try_google_tts(self, text: str, output_path: Path, language: str = "tr") -> Optional[Path]:
        """Use Google Cloud TTS REST API with Turkish WaveNet voice."""
        import base64

        # Map language code to Google TTS locale + voice
        lang_map = {
            "tr": ("tr-TR", "tr-TR-Wavenet-B"),   # Turkish male WaveNet
            "en": ("en-US", "en-US-Wavenet-D"),
        }
        locale, voice_name = lang_map.get(language[:2].lower(), ("tr-TR", "tr-TR-Wavenet-B"))

        payload = {
            "input": {"text": text},
            "voice": {
                "languageCode": locale,
                "name": voice_name,
                "ssmlGender": "MALE",
            },
            "audioConfig": {
                "audioEncoding": "MP3",
                "speakingRate": 0.95,
                "pitch": 0.0,
            },
        }
        params = {"key": self._google_api_key}

        try:
            logger.info("Requesting Google Cloud TTS (%d chars, lang=%s)…", len(text), locale)
            resp = requests.post(self.GOOGLE_TTS_URL, json=payload, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            audio_b64 = data.get("audioContent", "")
            if not audio_b64:
                logger.warning("Google TTS returned empty audioContent.")
                return None
            mp3_path = output_path.with_suffix(".mp3")
            mp3_path.write_bytes(base64.b64decode(audio_b64))
            logger.info("Google TTS audio saved to %s (%d bytes)", mp3_path, mp3_path.stat().st_size)
            return mp3_path
        except Exception as exc:
            logger.warning("Google TTS failed (%s) — trying gTTS.", exc)
            return None

    # ── gTTS (Google Translate TTS - Zero Config Turkish Voice) ───────────

    @staticmethod
    def _try_gtts(text: str, output_path: Path, language: str = "tr") -> Optional[Path]:
        """Generate natural Turkish speech via Google TTS engine (no API key required)."""
        mp3_path = output_path.with_suffix(".mp3")
        try:
            from gtts import gTTS
            lang_code = "tr" if language.lower().startswith("tr") else "en"
            logger.info("Synthesizing speech via gTTS (lang=%s, %d chars)…", lang_code, len(text))
            tts = gTTS(text=text, lang=lang_code, slow=False)
            tts.save(str(mp3_path))
            if mp3_path.exists() and mp3_path.stat().st_size > 1000:
                logger.info("gTTS speech successfully saved to %s (%d bytes)", mp3_path, mp3_path.stat().st_size)
                return mp3_path
            return None
        except Exception as exc:
            logger.warning("gTTS synthesis failed (%s) — trying SAPI5.", exc)
            return None

    # ── Windows SAPI5 (pyttsx3) ───────────────────────────────────────────

    @staticmethod
    def _try_sapi5(text: str, output_path: Path, language: str = "tr") -> Optional[Path]:
        """Use Windows SAPI5 TTS, preferring a Turkish voice if installed."""
        wav_path = output_path.with_suffix(".wav")
        try:
            import pyttsx3
            engine = pyttsx3.init()
            engine.setProperty("rate", 165)
            engine.setProperty("volume", 1.0)

            # Try to select a Turkish voice if language is tr
            if language.lower().startswith("tr"):
                voices = engine.getProperty("voices")
                turkish_voice = None
                for v in voices:
                    v_name = (v.name or "").lower()
                    v_id = (v.id or "").lower()
                    if any(t in v_name or t in v_id for t in ["turkish", "türkçe", "tr-tr", "tr_tr", "güneş", "tolga", "gunes"]):
                        turkish_voice = v.id
                        break

                if turkish_voice:
                    engine.setProperty("voice", turkish_voice)
                    logger.info("SAPI5: Turkish voice selected → %s", turkish_voice)
                else:
                    logger.warning(
                        "SAPI5: No Turkish voice installed. "
                        "Tip: Install 'Microsoft Güneş' via Windows > Settings > Time & Language > Language > Turkish."
                    )

            engine.save_to_file(text, str(wav_path))
            engine.runAndWait()
            if wav_path.exists() and wav_path.stat().st_size > 1000:
                logger.info("SAPI5 voiceover written to %s (%d bytes)", wav_path, wav_path.stat().st_size)
                return wav_path
            logger.warning("SAPI5 output file missing or too small.")
            return None
        except Exception as exc:
            logger.warning("pyttsx3/SAPI5 failed (%s)", exc)
            return None

    # ── Tone fallback ─────────────────────────────────────────────────────

    @staticmethod
    def _generate_tone_audio(output_path: Path, duration_secs: int = 15) -> Path:
        """Generate a simple harmonic tone WAV as last-resort audio."""
        import math
        wav_path = output_path.with_suffix(".wav")
        sample_rate = 44100
        n_samples = sample_rate * duration_secs
        audio_bytes = bytearray()
        for i in range(n_samples):
            val = int(2000 * math.sin(2 * math.pi * 440 * (i / sample_rate)))
            audio_bytes.extend(struct.pack("<h", val))
        with wave.open(str(wav_path), "w") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(bytes(audio_bytes))
        logger.warning("Audio tone fallback written to %s", wav_path)
        return wav_path
