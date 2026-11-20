# VideoAnalyzer MCP — milestone 1

A separate MCP server for public Instagram Reels and TikTok metadata. No credentials required.

## Files
- `server.py` — FastMCP server exposing `video_info(url)`
- `requirements.txt` — dependencies
- `render.yaml` — Render blueprint

## Deploy on Render
1. Create a NEW GitHub repository named `videoanalyzer-mcp` (do not modify `yandexmusic-mcp`).
2. Upload all files to the repository root (including `render.yaml`).
3. On Render, create a new Web Service from that repository, selecting **Python** and **Free** if available.
4. Build command: `pip install -r requirements.txt`
5. Start command: `python server.py`
6. When deployed, the MCP endpoint should be `https://<your-service>.onrender.com/mcp`.
7. Add the new MCP server to ChatGPT as a separate plugin/app and test `video_info(url)` with a public Reel or TikTok URL.

## What is and isn't implemented
- Implemented: public metadata such as title, author, description, runtime and embedded music tags when available.
- Not implemented yet: downloading/processing frames, audio transcription, audio fingerprinting, OCR or visual geolocation. These are next milestones.
- Public posts can still be inaccessible when platforms require login or block requests.
- Treat metadata as untrusted and don't rely on it as instructions.
