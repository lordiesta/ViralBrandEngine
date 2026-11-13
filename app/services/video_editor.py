"""
Video Editor Service — native FFmpeg rendering engine.

Responsibilities:
  1. Stitch B-roll clips to match voiceover audio duration (9:16 vertical).
  2. Overlay the voiceover audio track.
  3. Optionally burn subtitle overlays (CapCut / Reels style).
  4. Export the final rendered video.
"""

from __future__ import annotations

import logging
import math
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import PIL.Image
if not hasattr(PIL.Image, "ANTIALIAS"):
    PIL.Image.ANTIALIAS = getattr(PIL.Image, "LANCZOS", getattr(PIL.Image, "Resampling", None) and PIL.Image.Resampling.LANCZOS)

logger = logging.getLogger(__name__)


# ── SRT parser ───────────────────────────────────────────────────────────

@dataclass
class SubtitleCue:
    """A single subtitle cue parsed from SRT."""
    index: int
    start_secs: float
    end_secs: float
    text: str


def _parse_srt_timestamp(ts: str) -> float:
    """Convert ``HH:MM:SS,mmm`` to seconds."""
    ts = ts.strip().replace(",", ".")
    parts = ts.split(":")
    return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])


def parse_srt(srt_text: str) -> List[SubtitleCue]:
    """Parse a full SRT string into a list of ``SubtitleCue`` objects."""
    cues: List[SubtitleCue] = []
    blocks = re.split(r"\n\s*\n", srt_text.strip())
    for block in blocks:
        lines = block.strip().splitlines()
        if len(lines) < 3:
            continue
        idx = int(lines[0])
        timestamps = lines[1].split("-->")
        start = _parse_srt_timestamp(timestamps[0])
        end = _parse_srt_timestamp(timestamps[1])
        text = "\n".join(lines[2:])
        cues.append(SubtitleCue(index=idx, start_secs=start, end_secs=end, text=text))
    return cues


# ── Video Editor ─────────────────────────────────────────────────────────

class VideoEditorService:
    """
    Composes the final 9:16 video from B-roll clips, voiceover audio,
    and subtitle data.
    """

    # Target canvas size (9:16 portrait for Reels / TikTok)
    WIDTH = 1080
    HEIGHT = 1920
    FPS = 30

    MAX_CLIP_LEN = 4.5

    def render(
        self,
        broll_paths: List[Path],
        audio_path: Path,
        srt_text: str,
        output_path: Path,
        target_duration_secs: Optional[int] = None,
        scenes: Optional[List[dict]] = None,
        subtitles_enabled: bool = False,
    ) -> Path:
        """Render the final video with FFmpeg's native filters and encoders."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            return self._render_with_ffmpeg(
                broll_paths,
                audio_path,
                srt_text,
                output_path,
                target_duration_secs,
                scenes=scenes,
                subtitles_enabled=subtitles_enabled,
            )
        except Exception as exc:
            logger.error("Render failed: %s", exc, exc_info=True)
            raise

    def _render_with_ffmpeg(
        self,
        broll_paths: List[Path],
        audio_path: Path,
        srt_text: str,
        output_path: Path,
        target_duration_secs: Optional[int],
        scenes: Optional[List[dict]] = None,
        subtitles_enabled: bool = False,
    ) -> Path:
        try:
            import imageio_ffmpeg
            from moviepy.editor import AudioFileClip, VideoFileClip
        except ImportError as exc:
            raise RuntimeError(
                "FFmpeg rendering requires MoviePy and imageio-ffmpeg. "
                "Install the project requirements and retry."
            ) from exc

        ffmpeg_path = shutil.which("ffmpeg") or imageio_ffmpeg.get_ffmpeg_exe()
        audio_metadata = AudioFileClip(str(audio_path))
        try:
            audio_duration = float(audio_metadata.duration)
        finally:
            audio_metadata.close()

        duration = float(target_duration_secs or audio_duration)
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError(f"Invalid render duration: {duration}")
        audio_duration = min(audio_duration, duration)
        logger.info(
            "Rendering %.2fs video (audio %.2fs) with FFmpeg%s",
            duration,
            audio_duration,
            " and subtitles" if subtitles_enabled else "",
        )

        valid_paths = []
        clip_durations = []
        for path in broll_paths:
            path = Path(path)
            if not path.exists() or path.stat().st_size == 0:
                logger.warning("Skipping missing/empty clip: %s", path)
                continue
            try:
                clip = VideoFileClip(str(path), audio=False)
                try:
                    clip_duration = float(clip.duration)
                finally:
                    clip.close()
                if not math.isfinite(clip_duration) or clip_duration <= 0:
                    logger.warning("Skipping clip with invalid duration: %s", path)
                    continue
                valid_paths.append(path)
                clip_durations.append(clip_duration)
            except Exception as exc:
                logger.warning("Could not inspect clip %s: %s", path, exc)

        segments = self._plan_segments(
            valid_paths, clip_durations, scenes, duration
        )
        if not segments:
            logger.warning("No valid B-roll clips — using a solid background.")

        with tempfile.TemporaryDirectory(prefix="video-render-", dir=output_path.parent) as temp_dir:
            command = [
                ffmpeg_path,
                "-hide_banner",
                "-loglevel",
                "error",
                "-nostdin",
                "-y",
                "-i",
                str(audio_path),
            ]
            video_input_indexes = []

            if segments:
                for path, _ in segments:
                    command.extend(["-stream_loop", "-1", "-i", str(path)])
                    video_input_indexes.append(len(video_input_indexes) + 1)
            else:
                command.extend([
                    "-f", "lavfi", "-i",
                    f"color=c=0x141414:s={self.WIDTH}x{self.HEIGHT}:r={self.FPS}:d={duration}",
                ])
                video_input_indexes.append(1)

            subtitle_images = []
            if subtitles_enabled is True:
                subtitle_images = self._build_subtitle_images(
                    srt_text, duration, Path(temp_dir)
                )
            subtitle_input_indexes = []
            for image_path, _, _ in subtitle_images:
                command.extend([
                    "-loop", "1", "-framerate", str(self.FPS), "-i", str(image_path)
                ])
                subtitle_input_indexes.append(len(video_input_indexes) + 1 + len(subtitle_input_indexes))

            filters = []
            video_labels = []
            if segments:
                for segment_index, ((_, segment_duration), input_index) in enumerate(
                    zip(segments, video_input_indexes)
                ):
                    label = f"v{segment_index}"
                    filters.append(
                        f"[{input_index}:v]scale={self.WIDTH}:{self.HEIGHT}:"
                        f"force_original_aspect_ratio=increase,"
                        f"crop={self.WIDTH}:{self.HEIGHT},fps={self.FPS},"
                        f"trim=duration={segment_duration:.6f},"
                        f"setpts=PTS-STARTPTS,setsar=1[{label}]"
                    )
                    video_labels.append(f"[{label}]")
                filters.append(
                    f"{''.join(video_labels)}concat=n={len(video_labels)}:v=1:a=0[base]"
                )
            else:
                input_index = video_input_indexes[0]
                filters.append(
                    f"[{input_index}:v]fps={self.FPS},"
                    f"trim=duration={duration:.6f},setpts=PTS-STARTPTS,setsar=1[base]"
                )

            current_label = "base"
            for cue_index, ((_, start, end), input_index) in enumerate(
                zip(subtitle_images, subtitle_input_indexes)
            ):
                next_label = f"sub{cue_index}"
                filters.append(
                    f"[{current_label}][{input_index}:v]overlay="
                    f"x=(W-w)/2:y=H-440:"
                    f"enable='between(t\\,{start:.3f}\\,{end:.3f})'"
                    f"[{next_label}]"
                )
                current_label = next_label

            filters.extend([
                f"[0:a]atrim=duration={duration:.6f},asetpts=PTS-STARTPTS,"
                f"aresample=44100,aformat=sample_fmts=fltp:channel_layouts=mono,"
                f"apad=whole_dur={duration:.6f},atrim=duration={duration:.6f},"
                f"loudnorm=I=-16:LRA=11:TP=-1.5[aout]",
            ])
            command.extend([
                "-filter_complex", ";".join(filters),
                "-map", f"[{current_label}]",
                "-map", "[aout]",
                "-t", f"{duration:.6f}",
                "-r", str(self.FPS),
                "-c:v", "libx264",
                "-preset", "ultrafast",
                "-crf", "23",
                "-threads", "0",
                "-pix_fmt", "yuv420p",
                "-c:a", "aac",
                "-movflags", "+faststart",
                str(output_path),
            ])

            logger.info("Starting native FFmpeg render to %s", output_path)
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode:
                output_path.unlink(missing_ok=True)
                error = result.stderr.strip() or "FFmpeg exited without an error message."
                raise RuntimeError(f"FFmpeg render failed: {error[-4000:]}")

        logger.info("Render complete: %s", output_path)
        output_size = output_path.stat().st_size
        if output_size < 10_000:
            output_path.unlink(missing_ok=True)
            raise RuntimeError(f"Rendered video is too small: {output_size} bytes.")
        poster_path = output_path.with_suffix(".jpg")
        poster_result = subprocess.run(
            [
                ffmpeg_path,
                "-hide_banner",
                "-loglevel", "error",
                "-y",
                "-i", str(output_path),
                "-frames:v", "1",
                "-vf", "scale=540:-2",
                "-q:v", "2",
                str(poster_path),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if poster_result.returncode or not poster_path.exists() or poster_path.stat().st_size < 2_000:
            poster_path.unlink(missing_ok=True)
        try:
            rendered_clip = VideoFileClip(str(output_path), audio=False)
            try:
                frame = rendered_clip.get_frame(min(rendered_clip.duration / 2, 1.0))
                frame_variance = float(frame.std())
                frame_mean = float(frame.mean())
            finally:
                rendered_clip.close()
            if frame_variance < 2.0 or frame_mean < 2.0:
                output_path.unlink(missing_ok=True)
                raise RuntimeError(
                    f"Rendered video contains no usable visual content (variance={frame_variance:.2f})."
                )
        except RuntimeError:
            raise
        except Exception as exc:
            output_path.unlink(missing_ok=True)
            raise RuntimeError(f"Could not validate rendered video: {exc}") from exc
        return output_path

    def _plan_segments(
        self,
        paths: List[Path],
        clip_durations: List[float],
        scenes: Optional[List[dict]],
        target_duration: float,
    ) -> List[Tuple[Path, float]]:
        """Plan clip segments matching the storyboard or a uniform clip sequence."""
        if not paths:
            return []
        if len(paths) == 1:
            return [(paths[0], target_duration)]

        if scenes:
            scene_segments = []
            for index, scene in enumerate(scenes):
                start = float(scene.get("start_time", 0.0))
                end = float(scene.get("end_time", start + 4.0))
                scene_segments.append((paths[index % len(paths)], max(0.5, end - start)))
            if scene_segments:
                planned = []
                total = 0.0
                while total < target_duration:
                    for path, segment_duration in scene_segments:
                        remaining = target_duration - total
                        if remaining <= 0:
                            break
                        segment_duration = min(segment_duration, remaining)
                        planned.append((path, segment_duration))
                        total += segment_duration
                return planned

        planned = []
        total = 0.0
        while total < target_duration:
            for path, clip_duration in zip(paths, clip_durations):
                remaining = target_duration - total
                if remaining <= 0:
                    break
                segment_duration = min(self.MAX_CLIP_LEN, clip_duration, remaining)
                segment_duration = max(min(0.5, remaining), segment_duration)
                planned.append((path, segment_duration))
                total += segment_duration
        return planned

    def _build_subtitle_images(
        self, srt_text: str, max_duration: float, output_dir: Path
    ) -> List[Tuple[Path, float, float]]:
        """Render subtitle banners to transparent PNGs for FFmpeg overlays."""
        from PIL import Image, ImageDraw, ImageFont
        import os

        font_main = None
        for fp in [
            "C:/Windows/Fonts/arialbd.ttf",
            "C:/Windows/Fonts/segoeuib.ttf",
            "C:/Windows/Fonts/calibrib.ttf",
            "C:/Windows/Fonts/arial.ttf",
        ]:
            if os.path.exists(fp):
                try:
                    font_main = ImageFont.truetype(fp, size=44)
                    break
                except Exception:
                    pass

        if font_main is None:
            font_main = ImageFont.load_default()

        images = []
        for cue in parse_srt(srt_text):
            if cue.start_secs >= max_duration:
                break
            end = min(cue.end_secs, max_duration)
            if end <= cue.start_secs:
                continue

            sub_img = Image.new("RGBA", (self.WIDTH, 320), (0, 0, 0, 0))
            draw = ImageDraw.Draw(sub_img)
            lines = [line.strip() for line in cue.text.strip().upper().splitlines() if line.strip()]
            if not lines:
                continue
            formatted_text = "\n".join(lines)
            bbox = draw.multiline_textbbox(
                (0, 0), formatted_text, font=font_main, align="center", spacing=8
            )
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
            pill_w = min(self.WIDTH - 60, max(360, text_w + 120))
            pill_h = text_h + 90
            px0 = (self.WIDTH - pill_w) // 2
            py0 = (320 - pill_h) // 2
            draw.rounded_rectangle(
                [px0, py0, px0 + pill_w, py0 + pill_h],
                radius=16,
                fill=(0, 0, 0, 190),
                outline=(250, 204, 21),
                width=2,
            )
            draw.multiline_text(
                ((self.WIDTH - text_w) // 2, (320 - text_h) // 2),
                formatted_text,
                font=font_main,
                fill=(255, 255, 255),
                align="center",
                spacing=8,
            )
            image_path = output_dir / f"subtitle-{cue.index}.png"
            sub_img.save(image_path)
            images.append((image_path, cue.start_secs, end))

        return images
