"""VideoAnalyzer: first milestone - metadata for public Instagram/TikTok videos."""
from __future__ import annotations

import asyncio
import os
from urllib.parse import urlsplit, urlunsplit

from fastmcp import FastMCP
from yt_dlp import YoutubeDL
from yt_dlp.utils import DownloadError

mcp = FastMCP("VideoAnalyzer")

ALLOWED_HOSTS = {
    "instagram.com", "www.instagram.com", "m.instagram.com",
    "tiktok.com", "www.tiktok.com", "m.tiktok.com", "vm.tiktok.com", "vt.tiktok.com",
}


def _validate_url(url: str) -> str:
    if not isinstance(url, str) or len(url) > 2048:
        raise ValueError("Invalid video URL")
    parts = urlsplit(url.strip())
    if parts.scheme != "https" or parts.hostname not in ALLOWED_HOSTS or parts.username or parts.password or parts.port:
        raise ValueError("Only public HTTPS Instagram and TikTok links are supported")
    if not parts.path or parts.path == "/":
        raise ValueError("Please provide a link to a video, not the home page")
    return urlunsplit((parts.scheme, parts.netloc, parts.path, parts.query, ""))


def _extract(url: str) -> dict:
    options = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        "noplaylist": True,
        "socket_timeout": 15,
        "retries": 1,
        "extractor_retries": 1,
        "ignoreerrors": False,
    }
    with YoutubeDL(options) as ydl:
        data = ydl.extract_info(url, download=False)
    if not isinstance(data, dict):
        raise DownloadError("No metadata returned")
    # Only allow structured fields; never expose signed stream URLs or authentication data.
    return {
        "status": "ok",
        "platform": data.get("extractor_key") or data.get("extractor"),
        "video_id": data.get("id"),
        "title": data.get("title"),
        "description": data.get("description"),
        "creator": data.get("uploader"),
        "creator_id": data.get("uploader_id"),
        "duration_seconds": data.get("duration"),
        "upload_date": data.get("upload_date"),
        "view_count": data.get("view_count"),
        "like_count": data.get("like_count"),
        "track": data.get("track"),
        "artist": data.get("artist"),
        "location": data.get("location"),
        "tags": data.get("tags"),
        "webpage_url": data.get("webpage_url"),
        "note": "Metadata only. Music, speech, frames and geolocation analysis are not enabled yet.",
    }


@mcp.tool()
async def video_info(url: str) -> dict:
    """Fetch available public metadata for a TikTok or Instagram video URL.

    Does not log into accounts and does not download the video.
    """
    try:
        clean = _validate_url(url)
    except (ValueError, TypeError) as exc:
        return {"status": "invalid_url", "error": str(exc)}
    try:
        return await asyncio.wait_for(asyncio.to_thread(_extract, clean), timeout=45)
    except asyncio.TimeoutError:
        return {"status": "unavailable", "error": "Metadata request timed out"}
    except DownloadError:
        return {"status": "unavailable", "error": "Video metadata is unavailable. The video may be private, geo-restricted or blocked without login."}
    except Exception:
        return {"status": "error", "error": "Could not retrieve video metadata"}


def _probe_video(url: str) -> dict:
    options = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        "noplaylist": True,
        "socket_timeout": 15,
    }

    with YoutubeDL(options) as ydl:
        info = ydl.extract_info(url, download=False)

    if not isinstance(info, dict):
        return {"status": "unavailable"}

    formats = info.get("formats") or []

    has_video = any(
        f.get("vcodec") not in (None, "none")
        for f in formats
    )

    has_audio = any(
        f.get("acodec") not in (None, "none")
        for f in formats
    )

    return {
        "status": "ok",
        "video_id": info.get("id"),
        "duration": info.get("duration"),
        "video_stream_found": has_video,
        "audio_stream_found": has_audio,
        "format_count": len(formats),
        "note": "Streams detected, not downloaded or verified.",
    }


@mcp.tool()
async def video_probe(url: str) -> dict:
    """Check available video and audio streams."""
    try:
        clean_url = _validate_url(url)
        return await asyncio.wait_for(
            asyncio.to_thread(_probe_video, clean_url),
            timeout=45,
        )
    except Exception:
        return {
            "status": "unavailable",
            "error": "Could not inspect video streams",
        }

import tempfile
from pathlib import Path
from time import monotonic


def _download_audio_check(url: str) -> dict:
    max_bytes = 25 * 1024 * 1024
    deadline = monotonic() + 55

    def check_progress(data: dict) -> None:
        if monotonic() > deadline:
            raise RuntimeError("Audio download timed out")

        if data.get("status") == "downloading":
            received = data.get("downloaded_bytes") or 0
            total = (
                data.get("total_bytes")
                or data.get("total_bytes_estimate")
                or 0
            )
            if max(received, total) > max_bytes:
                raise RuntimeError("Audio exceeds 25 MB")

    with tempfile.TemporaryDirectory() as folder:
        options = {
            "quiet": True,
            "no_warnings": True,
            "noplaylist": True,
            "format": "bestaudio[ext=m4a]/bestaudio/best[acodec!=none]",
            "outtmpl": str(Path(folder) / "audio.%(ext)s"),
            "socket_timeout": 12,
            "retries": 1,
            "fragment_retries": 1,
            "extractor_retries": 1,
            "max_filesize": max_bytes,
            "progress_hooks": [check_progress],
        }

        with YoutubeDL(options) as ydl:
            info = ydl.extract_info(url, download=True)

        if not isinstance(info, dict):
            raise RuntimeError("No audio information")

        files = [
            p for p in Path(folder).iterdir()
            if p.is_file()
        ]

        if len(files) != 1:
            raise RuntimeError("Audio not downloaded")

        audio_file = files[0]
        size = audio_file.stat().st_size

        if size <= 0 or size > max_bytes:
            raise RuntimeError("Invalid audio size")

        return {
            "status": "ok",
            "video_id": info.get("id"),
            "audio_downloaded": True,
            "audio_format": audio_file.suffix.lstrip("."),
            "audio_size_bytes": size,
            "duration_seconds": info.get("duration"),
            "note": "Audio verified and temporary file deleted.",
        }


@mcp.tool()
async def audio_extract(url: str) -> dict:
    """Test audio downloading for Instagram or TikTok."""
    try:
        clean_url = _validate_url(url)
    except (ValueError, TypeError) as exc:
        return {
            "status": "invalid_url",
            "error": str(exc),
        }

    try:
        return await asyncio.to_thread(
            _download_audio_check, clean_url
        )
    except Exception as exc:
        return {
            "status": "unavailable",
            "error_type": type(exc).__name__,
            "error": "Audio download failed or exceeded limits",
        }

from shazamio import Shazam


def _download_for_shazam(url: str, folder: str) -> str:
    from pathlib import Path
    from time import monotonic

    limit = 25 * 1024 * 1024
    deadline = monotonic() + 55

    def progress(data):
        if monotonic() > deadline:
            raise RuntimeError("Download timeout")

        if data.get("status") == "downloading":
            received = data.get("downloaded_bytes") or 0
            total = (
                data.get("total_bytes")
                or data.get("total_bytes_estimate")
                or 0
            )
            if max(received, total) > limit:
                raise RuntimeError("File too large")

    options = {
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
        "format": "bestaudio[ext=m4a]/bestaudio/best[acodec!=none]",
        "outtmpl": str(Path(folder) / "music.%(ext)s"),
        "socket_timeout": 12,
        "retries": 1,
        "fragment_retries": 1,
        "max_filesize": limit,
        "progress_hooks": [progress],
    }

    with YoutubeDL(options) as ydl:
        ydl.extract_info(url, download=True)

    files = [p for p in Path(folder).iterdir() if p.is_file()]
    if len(files) != 1:
        raise RuntimeError("Audio file not found")

    audio = files[0]
    if not 0 < audio.stat().st_size <= limit:
        raise RuntimeError("Invalid audio size")

    return str(audio)


@mcp.tool()
async def identify_music(url: str) -> dict:
    """Recognize music in an Instagram or TikTok video."""
    import tempfile

    try:
        clean_url = _validate_url(url)
    except (ValueError, TypeError) as exc:
        return {
            "status": "invalid_url",
            "error": str(exc),
        }

    try:
        with tempfile.TemporaryDirectory() as folder:
            audio_path = await asyncio.wait_for(
                asyncio.to_thread(
                    _download_for_shazam,
                    clean_url,
                    folder,
                ),
                timeout=65,
            )

            
            import subprocess
            from pathlib import Path
            import imageio_ffmpeg

            wav_path = str(Path(folder) / "recognition.wav")

            def convert_audio():
                subprocess.run(
                    [
                        imageio_ffmpeg.get_ffmpeg_exe(),
                        "-y",
                        "-v", "error",
                        "-i", audio_path,
                        "-vn",
                        "-ac", "1",
                        "-ar", "16000",
                        "-c:a", "pcm_s16le",
                        wav_path,
                    ],
                    check=True,
                    timeout=35,
                    capture_output=True,
                )

            await asyncio.to_thread(convert_audio)

            async with Shazam() as shazam:
                result = await asyncio.wait_for(
                    shazam.recognize(wav_path),
                    timeout=35,
                )

        track = result.get("track") if isinstance(result, dict) else None

        if not track:
            return {
                "status": "not_recognized",
                "message": "Shazam did not identify a song",
            }

        return {
            "status": "ok",
            "title": track.get("title"),
            "artist": track.get("subtitle"),
            "shazam_url": track.get("url"),
            "track_id": track.get("key"),
            "note": "One audio segment analyzed",
        }

    except Exception as exc:
        return {
            "status": "error",
            "error_type": type(exc).__name__,
            "error_details": str(exc)[:1000],
            "message": "Audio processing or recognition failed",
    }
    
@mcp.tool()
async def identify_all_music(
    url: str,
    offset_seconds: int = 0,
    max_segments: int = 5,
) -> dict:
    """Recognize multiple songs with approximate timestamps."""
    import tempfile
    import subprocess
    import wave
    from pathlib import Path
    import imageio_ffmpeg

    try:
        clean_url = _validate_url(url)
    except (ValueError, TypeError) as exc:
        return {"status": "invalid_url", "error": str(exc)}

    offset_seconds = max(0, int(offset_seconds))
    max_segments = max(1, min(int(max_segments), 6))

    try:
        with tempfile.TemporaryDirectory() as folder:
            audio_path = await asyncio.wait_for(
                asyncio.to_thread(
                    _download_for_shazam, clean_url, folder
                ),
                timeout=65,
            )

            ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
            wav_path = str(Path(folder) / "full_audio.wav")

            def convert():
                subprocess.run(
                    [
                        ffmpeg, "-y", "-v", "error",
                        "-i", audio_path,
                        "-vn", "-ac", "1",
                        "-ar", "16000",
                        "-c:a", "pcm_s16le",
                        wav_path,
                    ],
                    capture_output=True,
                    check=True,
                    timeout=45,
                )

            await asyncio.to_thread(convert)

            with wave.open(wav_path, "rb") as wav:
                duration = wav.getnframes() / wav.getframerate()

            if duration < 6:
                return {
                    "status": "too_short",
                    "duration_seconds": round(duration, 2),
                }

            if duration > 600:
                return {
                    "status": "too_long",
                    "duration_seconds": round(duration, 2),
                    "message": "Audio exceeds 10 minutes",
                }

            starts = []
            position = float(offset_seconds)

            while (
                position <= duration - 6
                and len(starts) < max_segments
            ):
                starts.append(position)
                position += 10

            if not starts:
                return {
                    "status": "complete",
                    "songs": [],
                    "has_more": False,
                    "duration_seconds": round(duration, 2),
                }

            songs = {}
            errors = []

            async with Shazam() as shazam:
                for index, start in enumerate(starts):
                    clip_path = str(
                        Path(folder) / f"clip_{index}.wav"
                    )
                    clip_length = min(12, duration - start)

                    def make_clip():
                        subprocess.run(
                            [
                                ffmpeg, "-y", "-v", "error",
                                "-ss", str(start),
                                "-i", wav_path,
                                "-t", str(clip_length),
                                "-ac", "1",
                                "-ar", "16000",
                                "-c:a", "pcm_s16le",
                                clip_path,
                            ],
                            capture_output=True,
                            check=True,
                            timeout=20,
                        )

                    try:
                        await asyncio.to_thread(make_clip)

                        result = await asyncio.wait_for(
                            shazam.recognize(clip_path),
                            timeout=20,
                        )

                        track = (
                            result.get("track")
                            if isinstance(result, dict)
                            else None
                        )

                        if track:
                            key = str(
                                track.get("key")
                                or (
                                    str(track.get("title"))
                                    + "|"
                                    + str(track.get("subtitle"))
                                )
                            )

                            if key not in songs:
                                songs[key] = {
                                    "title": track.get("title"),
                                    "artist": track.get("subtitle"),
                                    "shazam_url": track.get("url"),
                                    "track_id": track.get("key"),
                                    "detected_at_seconds": round(
                                        start, 2
                                    ),
                                    "timestamps_seconds": [],
                                }

                            songs[key]["timestamps_seconds"].append(
                                round(start, 2)
                            )

                    except Exception as exc:
                        errors.append({
                            "segment_start": round(start, 2),
                            "error_type": type(exc).__name__,
                            "details": str(exc)[:200],
                        })
                    finally:
                        Path(clip_path).unlink(missing_ok=True)

            has_more = position <= duration - 6

            return {
                "status": (
                    "ok" if songs else
                    "partial_error" if errors else
                    "not_recognized"
                ),
                "duration_seconds": round(duration, 2),
                "segments_analyzed": len(starts),
                "songs_found": len(songs),
                "songs": list(songs.values()),
                "segment_errors": errors,
                "has_more": has_more,
                "next_offset_seconds": (
                    int(position) if has_more else None
                ),
                "note": (
                    "Timestamps are sample positions, "
                    "not exact song boundaries."
                ),
            }

    except Exception as exc:
        return {
            "status": "error",
            "error_type": type(exc).__name__,
            "details": str(exc)[:500],
        }


if __name__ == "__main__":
    mcp.run(transport="http", host="0.0.0.0", port=int(os.environ.get("PORT", "10000")))
