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


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "10000"))
    mcp.run(transport="http", host="0.0.0.0", port=port)
