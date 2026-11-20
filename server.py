"""VideoAnalyzer MCP: public Instagram/TikTok metadata and music recognition."""
from __future__ import annotations

import asyncio
import os
import subprocess
import tempfile
import wave
from pathlib import Path
from time import monotonic
from urllib.parse import urlsplit, urlunsplit

import imageio_ffmpeg
from fastmcp import FastMCP
from fastmcp.utilities.types import Image
import json
import logging
from shazamio import Shazam
from yt_dlp import YoutubeDL
from yt_dlp.utils import DownloadError

mcp = FastMCP("VideoAnalyzer")

ALLOWED_HOSTS = {
    "instagram.com", "www.instagram.com", "m.instagram.com",
    "tiktok.com", "www.tiktok.com", "m.tiktok.com",
    "vm.tiktok.com", "vt.tiktok.com",
}
MAX_AUDIO_BYTES = 25 * 1024 * 1024
SAMPLE_STEP_SECONDS = 10
SAMPLE_LENGTH_SECONDS = 12


def _validate_url(url: str) -> str:
    if not isinstance(url, str) or not (1 <= len(url) <= 2048):
        raise ValueError("Invalid video URL")
    parts = urlsplit(url.strip())
    if (
        parts.scheme != "https"
        or parts.hostname not in ALLOWED_HOSTS
        or parts.username is not None
        or parts.password is not None
        or parts.port is not None
    ):
        raise ValueError("Only HTTPS Instagram and TikTok video links are supported")
    if not parts.path or parts.path == "/":
        raise ValueError("Please provide a video URL, not a homepage")
    return urlunsplit(("https", parts.netloc, parts.path, parts.query, ""))


def _ydl_options() -> dict:
    return {
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
        "socket_timeout": 12,
        "retries": 1,
        "fragment_retries": 1,
        "extractor_retries": 1,
        "ignoreerrors": False,
    }


def _get_metadata(url: str) -> dict:
    with YoutubeDL({**_ydl_options(), "skip_download": True}) as ydl:
        info = ydl.extract_info(url, download=False)
    if not isinstance(info, dict):
        raise DownloadError("No metadata returned")
    return info


def _safe_error(exc: Exception) -> dict:
    # Do not return extractor exception strings containing signed media URLs.
    return {"status": "error", "error_type": type(exc).__name__,
            "message": "Request failed; check Render logs for diagnostics"}


@mcp.tool()
async def video_info(url: str) -> dict:
    """Read available public metadata; does not download the video."""
    try:
        clean_url = _validate_url(url)
        info = await asyncio.wait_for(
            asyncio.to_thread(_get_metadata, clean_url), timeout=45
        )
        return {
            "status": "ok",
            "platform": info.get("extractor_key") or info.get("extractor"),
            "video_id": info.get("id"),
            "title": info.get("title"),
            "description": info.get("description"),
            "creator": info.get("uploader"),
            "creator_id": info.get("uploader_id"),
            "duration_seconds": info.get("duration"),
            "upload_date": info.get("upload_date"),
            "view_count": info.get("view_count"),
            "like_count": info.get("like_count"),
            "track": info.get("track"),
            "artist": info.get("artist"),
            "location": info.get("location"),
            "tags": info.get("tags"),
            "webpage_url": info.get("webpage_url"),
            "note": "Metadata only; not a speech, frame or location analysis.",
        }
    except ValueError as exc:
        return {"status": "invalid_url", "error": str(exc)}
    except asyncio.TimeoutError:
        return {"status": "unavailable", "error": "Metadata lookup timed out"}
    except DownloadError:
        return {"status": "unavailable", "error": "Public metadata unavailable"}
    except Exception as exc:
        return _safe_error(exc)


@mcp.tool()
async def video_probe(url: str) -> dict:
    """Check whether metadata advertises audio and video streams."""
    try:
        clean_url = _validate_url(url)
        info = await asyncio.wait_for(
            asyncio.to_thread(_get_metadata, clean_url), timeout=45
        )
        formats = info.get("formats") or []
        if not formats and info.get("url"):
            formats = [info]
        has_video = any(f.get("vcodec") not in (None, "none") for f in formats)
        has_audio = any(f.get("acodec") not in (None, "none") for f in formats)
        return {
            "status": "ok",
            "video_id": info.get("id"),
            "duration": info.get("duration"),
            "video_stream_found": has_video,
            "audio_stream_found": has_audio,
            "format_count": len(formats),
            "note": "Streams detected; actual download is not guaranteed.",
        }
    except ValueError as exc:
        return {"status": "invalid_url", "error": str(exc)}
    except Exception as exc:
        return _safe_error(exc)


def _download_audio(url: str, folder: str) -> tuple[Path, dict]:
    start = monotonic()

    def progress(data: dict) -> None:
        if monotonic() - start > 55:
            raise RuntimeError("Download time limit exceeded")
        if data.get("status") == "downloading":
            received = data.get("downloaded_bytes") or 0
            total = data.get("total_bytes") or data.get("total_bytes_estimate") or 0
            if max(received, total) > MAX_AUDIO_BYTES:
                raise RuntimeError("Download exceeds 25 MB")

    opts = {
        **_ydl_options(),
        "format": "bestaudio[ext=m4a]/bestaudio/best[acodec!=none]",
        "outtmpl": str(Path(folder) / "source.%(ext)s"),
        "max_filesize": MAX_AUDIO_BYTES,
        "progress_hooks": [progress],
    }
    with YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)

    if not isinstance(info, dict):
        raise RuntimeError("No audio information returned")
    candidates = [p for p in Path(folder).iterdir() if p.is_file()]
    if len(candidates) != 1:
        raise RuntimeError("Downloaded file not found")
    path = candidates[0]
    if not 0 < path.stat().st_size <= MAX_AUDIO_BYTES:
        raise RuntimeError("Invalid audio file size")
    return path, info


def _to_wav(source: Path, target: Path, *, start: float | None = None,
            length: float | None = None) -> None:
    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-nostdin", "-y", "-v", "error"]
    if start is not None:
        cmd += ["-ss", str(start)]
    cmd += ["-i", str(source)]
    if length is not None:
        cmd += ["-t", str(length)]
    cmd += ["-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(target)]
    subprocess.run(cmd, check=True, timeout=40, capture_output=True)


def _duration_wav(path: Path) -> float:
    with wave.open(str(path), "rb") as handle:
        return handle.getnframes() / handle.getframerate()


def _clip_starts(duration: float, offset: int, count: int) -> list[int]:
    # A tail shorter than 6 seconds is not sent to Shazam.
    starts = []
    pos = offset
    while pos <= duration - 6 and len(starts) < count:
        starts.append(pos)
        pos += SAMPLE_STEP_SECONDS
    return starts


@mcp.tool()
async def audio_extract(url: str) -> dict:
    """Verify that the audio can be downloaded; delete it afterwards."""
    try:
        clean_url = _validate_url(url)
        with tempfile.TemporaryDirectory() as folder:
            file_path, info = await asyncio.wait_for(
                asyncio.to_thread(_download_audio, clean_url, folder), timeout=65
            )
            return {
                "status": "ok",
                "video_id": info.get("id"),
                "audio_downloaded": True,
                "audio_format": file_path.suffix.lstrip("."),
                "audio_size_bytes": file_path.stat().st_size,
                "duration_seconds": info.get("duration"),
                "note": "Audio verified; temporary file deleted.",
            }
    except ValueError as exc:
        return {"status": "invalid_url", "error": str(exc)}
    except Exception as exc:
        return _safe_error(exc)


async def _recognize_segment(shazam: Shazam, path: Path) -> dict | None:
    """Retry transient timeouts using a new request for the same segment."""
    for attempt in range(2):
        try:
            response = await asyncio.wait_for(shazam.recognize(str(path)), timeout=18)
            if isinstance(response, dict) and isinstance(response.get("track"), dict):
                return response["track"]
            return None
        except (asyncio.TimeoutError, TimeoutError):
            if attempt == 1:
                raise
            await asyncio.sleep(1)
    return None


@mcp.tool()
async def identify_music(url: str) -> dict:
    """Recognize one song from an Instagram or TikTok audio sample."""
    try:
        clean_url = _validate_url(url)
        with tempfile.TemporaryDirectory() as folder:
            source, _ = await asyncio.wait_for(
                asyncio.to_thread(_download_audio, clean_url, folder), timeout=65
            )
            wav = Path(folder) / "recognition.wav"
            await asyncio.to_thread(_to_wav, source, wav)
            # Shazam uses its own 12-second fingerprint window.
            async with Shazam() as shazam:
                track = await _recognize_segment(shazam, wav)
        if not track:
            return {"status": "not_recognized", "message": "No song recognized"}
        return {
            "status": "ok",
            "title": track.get("title"),
            "artist": track.get("subtitle"),
            "shazam_url": track.get("url"),
            "track_id": track.get("key"),
            "note": "One audio sample analyzed.",
        }
    except ValueError as exc:
        return {"status": "invalid_url", "error": str(exc)}
    except Exception as exc:
        return _safe_error(exc)


@mcp.tool()
async def identify_all_music(
    url: str, offset_seconds: int = 0, max_segments: int = 2
) -> dict:
    """Recognize songs in up to two segments. Call again using next_offset_seconds."""
    try:
        clean_url = _validate_url(url)
        offset = int(offset_seconds)
        count = int(max_segments)
        if offset < 0:
            raise ValueError("offset_seconds must be nonnegative")
        count = max(1, min(count, 2))
    except (ValueError, TypeError) as exc:
        return {"status": "invalid_input", "error": str(exc)}

    try:
        with tempfile.TemporaryDirectory() as folder:
            source, _ = await asyncio.wait_for(
                asyncio.to_thread(_download_audio, clean_url, folder), timeout=65
            )
            full_wav = Path(folder) / "full_audio.wav"
            await asyncio.to_thread(_to_wav, source, full_wav)
            duration = await asyncio.to_thread(_duration_wav, full_wav)
            if duration < 6:
                return {"status": "too_short", "duration_seconds": round(duration, 2)}
            if duration > 600:
                return {"status": "too_long", "duration_seconds": round(duration, 2),
                        "message": "Audio exceeds 10-minute limit"}

            starts = _clip_starts(duration, offset, count)
            if not starts:
                return {
                    "status": "complete", "duration_seconds": round(duration, 2),
                    "segments_analyzed": 0, "songs_found": 0, "songs": [],
                    "segment_errors": [], "failed_offsets_seconds": [],
                    "has_more": False, "next_offset_seconds": None,
                }

            songs: dict[str, dict] = {}
            errors: list[dict] = []
            async with Shazam() as shazam:
                for index, start in enumerate(starts):
                    clip = Path(folder) / f"clip_{index}.wav"
                    try:
                        await asyncio.to_thread(
                            _to_wav, full_wav, clip,
                            start=float(start), length=min(12.0, duration - start)
                        )
                        track = await _recognize_segment(shazam, clip)
                        if not track:
                            continue
                        key = str(track.get("key") or (
                            f"{track.get('title')}|{track.get('subtitle')}"
                        ))
                        if key not in songs:
                            songs[key] = {
                                "title": track.get("title"),
                                "artist": track.get("subtitle"),
                                "shazam_url": track.get("url"),
                                "track_id": track.get("key"),
                                "detected_at_seconds": start,
                                "timestamps_seconds": [],
                            }
                        songs[key]["timestamps_seconds"].append(start)
                    except Exception as exc:
                        errors.append({
                            "segment_start": start,
                            "error_type": type(exc).__name__,
                            "details": "Recognition or audio conversion failed",
                        })
                    finally:
                        clip.unlink(missing_ok=True)

            next_offset = starts[-1] + SAMPLE_STEP_SECONDS
            has_more = next_offset <= duration - 6
            return {
                "status": "partial_error" if errors else (
                    "ok" if songs else "not_recognized"
                ),
                "duration_seconds": round(duration, 2),
                "segments_analyzed": len(starts),
                "songs_found": len(songs),
                "songs": list(songs.values()),
                "segment_errors": errors,
                "failed_offsets_seconds": [e["segment_start"] for e in errors],
                "has_more": has_more,
                "next_offset_seconds": next_offset if has_more else None,
                "note": "Timestamps are sample positions, not exact song boundaries.",
            }
    except Exception as exc:
        return _safe_error(exc)



# -----------------------------------------------------------------------------
# Visual content tools: return actual JPEG frames as MCP Image content.
# ChatGPT can inspect images; this server does not pretend to run vision models.
# -----------------------------------------------------------------------------
MAX_VIDEO_BYTES = 65 * 1024 * 1024
MAX_FRAME_COUNT = 4
MAX_MEDIA_DURATION = 180.0


def _download_video(url: str, folder: str) -> tuple[Path, dict]:
    """Download a single playable video in a bounded temporary directory."""
    started = monotonic()

    def progress(data: dict) -> None:
        if monotonic() - started > 60:
            raise RuntimeError("Video download time limit exceeded")
        if data.get("status") == "downloading":
            downloaded = data.get("downloaded_bytes") or 0
            expected = data.get("total_bytes") or data.get("total_bytes_estimate") or 0
            if max(downloaded, expected) > MAX_VIDEO_BYTES:
                raise RuntimeError("Video exceeds 65 MB")

    options = {
        **_ydl_options(),
        "format": "best[height<=720][ext=mp4]/best[height<=720]/best",
        "outtmpl": str(Path(folder) / "video.%(ext)s"),
        "max_filesize": MAX_VIDEO_BYTES,
        "progress_hooks": [progress],
    }
    with YoutubeDL(options) as ydl:
        info = ydl.extract_info(url, download=True)
    if not isinstance(info, dict):
        raise RuntimeError("No video metadata returned")
    files = [p for p in Path(folder).iterdir() if p.is_file()]
    if len(files) != 1:
        raise RuntimeError("Could not find a single downloaded video")
    video = files[0]
    if not 0 < video.stat().st_size <= MAX_VIDEO_BYTES:
        raise RuntimeError("Invalid video size")
    return video, info


def _probe_media_duration(path: Path) -> float | None:
    """Read duration from FFmpeg stderr without depending on ffprobe binary."""
    import re
    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-nostdin", "-hide_banner", "-i", str(path)]
    try:
        completed = subprocess.run(cmd, capture_output=True, text=True, timeout=12)
        match = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", completed.stderr)
        if match:
            return int(match[1]) * 3600 + int(match[2]) * 60 + float(match[3])
    except (OSError, subprocess.TimeoutExpired):
        pass
    return None


def _frame_positions(duration: float | None, timestamps: list[float] | None,
                     max_frames: int) -> list[float]:
    if timestamps is not None:
        if not isinstance(timestamps, list) or not timestamps:
            raise ValueError("timestamps_seconds must be a nonempty array")
        if len(timestamps) > MAX_FRAME_COUNT:
            raise ValueError("At most 4 timestamps per call")
        points = [float(t) for t in timestamps]
        if any(t < 0 or t > MAX_MEDIA_DURATION for t in points):
            raise ValueError("Timestamp out of supported range 0..180 seconds")
        return points
    count = max(1, min(int(max_frames), MAX_FRAME_COUNT))
    if duration is None:
        return [float(x) for x in (0, 5, 10, 15)[:count]]
    bounded = max(0, min(duration, MAX_MEDIA_DURATION))
    return [round((i + 1) * bounded / (count + 1), 2) for i in range(count)]


def _jpeg_frame(video: Path, timestamp: float) -> bytes:
    """Bound one frame to about 960 px so it fits MCP responses."""
    cmd = [
        imageio_ffmpeg.get_ffmpeg_exe(), "-nostdin", "-hide_banner", "-loglevel", "error",
        "-ss", str(timestamp), "-i", str(video), "-frames:v", "1",
        "-vf", "scale=960:960:force_original_aspect_ratio=decrease",
        "-q:v", "5", "-f", "image2pipe", "-vcodec", "mjpeg", "pipe:1"
    ]
    p = subprocess.run(cmd, capture_output=True, timeout=20)
    if p.returncode or not p.stdout:
        raise RuntimeError("Could not extract frame")
    if len(p.stdout) > 1_500_000:
        raise RuntimeError("Frame exceeds image output limit")
    return p.stdout


def _extract_frames_payload(url: str, timestamps_seconds: list[float] | None,
                            max_frames: int, purpose: str) -> list:
    clean_url = _validate_url(url)
    with tempfile.TemporaryDirectory() as folder:
        video, info = _download_video(clean_url, folder)
        duration = info.get("duration")
        if duration is None:
            duration = _probe_media_duration(video)
        if duration is not None:
            duration = float(duration)
        if duration is not None and duration > MAX_MEDIA_DURATION:
            # Short videos are the main use case; explicit timestamp selection
            # can still inspect the first 180 seconds of longer public videos.
            duration = MAX_MEDIA_DURATION
        points = _frame_positions(duration, timestamps_seconds, max_frames)
        images = []
        failures = []
        for second in points:
            try:
                payload = _jpeg_frame(video, second)
                images.append((second, payload))
            except Exception:
                failures.append(second)
        if not images:
            return [{"status": "unavailable", "reason": "No usable video frames", "failed_timestamps_seconds": failures}]
        manifest = {
            "status": "ok", "video_id": info.get("id"),
            "duration_seconds": info.get("duration") or duration,
            "frames_returned": len(images),
            "timestamps_seconds": [p[0] for p in images],
            "failed_timestamps_seconds": failures,
            "instruction": purpose,
            "limitations": "Images are sampled, not continuous coverage; conclusions are visual inferences."
        }
        # Data, not paths: files disappear after the temporary directory closes.
        output: list = [json.dumps(manifest, ensure_ascii=False)]
        for ts, payload in images:
            output.append(f"Video frame at approximately {ts:.2f} seconds:")
            output.append(Image(data=payload, format="jpeg"))
        return output


async def _get_frames(url: str, timestamps_seconds: list[float] | None,
                      max_frames: int, purpose: str) -> list:
    try:
        return await asyncio.wait_for(
            asyncio.to_thread(_extract_frames_payload, url, timestamps_seconds, max_frames, purpose),
            timeout=110,
        )
    except ValueError as exc:
        return [{"status": "invalid_input", "message": str(exc)}]
    except asyncio.TimeoutError:
        return [{"status": "timeout", "message": "Frame extraction timed out"}]
    except Exception as exc:
        logging.exception("Frame extraction failed")
        return [_safe_error(exc)]


@mcp.tool()
async def extract_frames(url: str, timestamps_seconds: list[float] | None = None,
                         max_frames: int = 3) -> list:
    """Return real JPEG video frames to ChatGPT with sample timestamps (at most 4)."""
    return await _get_frames(
        url, timestamps_seconds, max_frames,
        "Describe exactly what is visible in each frame, noting uncertainty."
    )


@mcp.tool()
async def extract_text(url: str, timestamps_seconds: list[float] | None = None,
                       max_frames: int = 3) -> list:
    """Return video images for ChatGPT to read visible subtitles and screen text (visual OCR)."""
    return await _get_frames(
        url, timestamps_seconds, max_frames,
        "Read visible text verbatim from these frames. Label it by timestamp; "
        "mark illegible words as uncertain. This tool provides image evidence, not OCR text."
    )


@mcp.tool()
async def analyze_frames(url: str, timestamps_seconds: list[float] | None = None,
                         max_frames: int = 3) -> list:
    """Show actual video frames for ChatGPT to identify objects, actions and scenes."""
    return await _get_frames(
        url, timestamps_seconds, max_frames,
        "Describe visible objects, actions, logos and scene changes. "
        "Do not assert invisible details or identify people from appearance."
    )


@mcp.tool()
async def identify_location(url: str, timestamps_seconds: list[float] | None = None,
                            max_frames: int = 3) -> list:
    """Show video frames as evidence for a tentative location guess by ChatGPT."""
    return await _get_frames(
        url, timestamps_seconds, max_frames,
        "Look for geographic clues (landmarks, readable signs, languages, architecture). "
        "Suggest a location ONLY if supported by observable evidence; "
        "label uncertainty and never invent GPS coordinates."
    )


@mcp.tool()
async def extract_subtitles(url: str) -> dict:
    """Return uploader-provided subtitle metadata or captions when available."""
    try:
        info = await asyncio.wait_for(
            asyncio.to_thread(_get_metadata, _validate_url(url)), timeout=45
        )
        subtitles = info.get("subtitles") or {}
        auto = info.get("automatic_captions") or {}
        return {
            "status": "ok", "video_id": info.get("id"),
            "subtitle_languages": sorted(subtitles.keys())[:30],
            "automatic_caption_languages": sorted(auto.keys())[:30],
            "description": info.get("description"),
            "note": "Lists available subtitle languages; does not claim to transcribe spoken audio.",
        }
    except ValueError as exc:
        return {"status": "invalid_url", "message": str(exc)}
    except Exception as exc:
        return _safe_error(exc)


# Audio speech-to-text: optional API. Never pretend Shazam recognizes dialogue.
# Configured by GROQ_API_KEY on Render. Default works without this key,
# but speech transcription returns 'not_configured'.

def _speech_request(url: str, language: str | None) -> dict:
    token = os.environ.get("GROQ_API_KEY", "").strip()
    if not token:
        return {
            "status": "not_configured",
            "message": "Automatic speech transcription needs GROQ_API_KEY or another ASR engine.",
            "note": "No speech has been transcribed. The other tools remain available."
        }
    import httpx
    clean_url = _validate_url(url)
    with tempfile.TemporaryDirectory() as folder:
        src, info = _download_audio(clean_url, folder)
        mp3 = Path(folder) / "speech.mp3"
        cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-nostdin", "-y", "-v", "error",
               "-i", str(src), "-vn", "-ac", "1", "-ar", "16000", "-b:a", "48k",
               str(mp3)]
        subprocess.run(cmd, capture_output=True, check=True, timeout=50)
        if mp3.stat().st_size > 24 * 1024 * 1024:
            raise RuntimeError("Speech audio too large for single API request")
        params = {"model": "whisper-large-v3-turbo", "response_format": "verbose_json"}
        if language and language.lower() != "auto":
            if not (language.isascii() and language.replace("-", "").isalpha() and len(language) <= 8):
                raise ValueError("Provide an ISO language code such as ru, en or auto")
            params["language"] = language
        with httpx.Client(timeout=70, follow_redirects=False) as client:
            with mp3.open("rb") as fileobj:
                response = client.post(
                    "https://api.groq.com/openai/v1/audio/transcriptions",
                    headers={"Authorization": f"Bearer {token}"},
                    data=params,
                    files={"file": ("speech.mp3", fileobj, "audio/mpeg")},
                )
        if response.status_code != 200:
            return {"status": "provider_error", "http_status": response.status_code,
                    "message": "Speech provider rejected the request. Check API key, quota and model."}
        data = response.json()
        return {
            "status": "ok", "video_id": info.get("id"),
            "language": data.get("language"),
            "text": data.get("text"),
            "segments": [
                {"start": s.get("start"), "end": s.get("end"), "text": s.get("text")}
                for s in (data.get("segments") or [])[:120]
            ],
            "note": "Automatic speech recognition is fallible, especially over background music."
        }


@mcp.tool()
async def transcribe_speech(url: str, language: str = "auto") -> dict:
    """Transcribe speech with optional configured Groq ASR; reports not_configured otherwise."""
    try:
        return await asyncio.wait_for(
            asyncio.to_thread(_speech_request, url, language), timeout=115
        )
    except ValueError as exc:
        return {"status": "invalid_input", "message": str(exc)}
    except asyncio.TimeoutError:
        return {"status": "timeout", "message": "Speech request timed out"}
    except Exception as exc:
        logging.exception("Speech transcription error")
        return _safe_error(exc)


@mcp.tool()
async def full_analysis(url: str, max_frames: int = 3) -> list:
    """Return video metadata and sampled frames for a single ChatGPT visual analysis.

    For songs call identify_all_music separately; for speech call transcribe_speech.
    """
    try:
        clean_url = _validate_url(url)
        info = await asyncio.wait_for(asyncio.to_thread(_get_metadata, clean_url), timeout=45)
        metadata = {
            "status": "ok", "video_id": info.get("id"),
            "title": info.get("title"), "creator": info.get("uploader"),
            "description": info.get("description"), "duration_seconds": info.get("duration"),
            "track_metadata": info.get("track"), "artist_metadata": info.get("artist"),
            "location_metadata": info.get("location"),
            "subtitle_languages": sorted((info.get("subtitles") or {}).keys()),
            "steps_for_complete_report": [
                "Inspect the image frames below for visible text, locations and objects",
                "Call identify_all_music separately for music from the audio",
                "Call transcribe_speech if a speech recognition provider is configured",
            ],
            "note": "Do not claim speech or audio recognition from the images alone."
        }
        frames = await _get_frames(
            clean_url, None, max_frames,
            "Summarize visible scenes, transcribe visible text and assess geographic clues "
            "with uncertainty. Do not invent unseen events."
        )
        return [json.dumps(metadata, ensure_ascii=False)] + frames
    except ValueError as exc:
        return [{"status": "invalid_url", "message": str(exc)}]
    except Exception as exc:
        return [_safe_error(exc)]


@mcp.tool()
def capabilities() -> dict:
    """List functional features and explicitly distinguish configured vs unavailable ones."""
    return {
        "status": "ok", "features": {
            "metadata": "ready", "stream_probe": "ready", "audio_download": "ready",
            "single_song": "ready", "multiple_songs": "ready",
            "frame_extraction": "ready", "visual_text_via_chatgpt": "ready",
            "scene_objects_via_chatgpt": "ready", "location_clues_via_chatgpt": "ready",
            "uploader_subtitle_languages": "ready",
            "speech_to_text": "configured" if os.getenv("GROQ_API_KEY") else "needs_GROQ_API_KEY",
            "full_visual_report": "ready",
        },
        "limitations": [
            "Video/image extraction from Instagram and TikTok can be blocked",
            "Image understanding happens in ChatGPT, not a separate on-server vision model",
            "Location estimates are not proof of GPS location",
            "No continuous video coverage; images are sampled",
            "Music recognition is a non-official Shazam integration",
        ],
    }


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "10000"))
    print(f"Starting VideoAnalyzer MCP on port {port}", flush=True)
    mcp.run(transport="http", host="0.0.0.0", port=port)
