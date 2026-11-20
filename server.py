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

if __name__ == "__main__":
    mcp.run(transport="http", host="0.0.0.0", port=int(os.environ.get("PORT", "10000")))
