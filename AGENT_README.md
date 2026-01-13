Pool Telemetry Recorder - Agent Handoff

Overview
- Python + PyQt6 desktop app that records pool telemetry from video sources.
- Current focus: scaffolded phases 1-6 with UI, config, SQLite schema, video capture, session management, exports, and Gemini Live wiring.

Project layout
- src/pool_telemetry/app.py: entry point, config + DB init, logging setup.
- src/pool_telemetry/config.py: config schema, load/save.
- src/pool_telemetry/db.py: SQLite schema and helper insert.
- src/pool_telemetry/video/: video capture helpers + worker thread.
- src/pool_telemetry/gemini/: Gemini client thread + event processor.
- src/pool_telemetry/ui/: main window, dialogs, settings, session browser, export dialog.
- src/pool_telemetry/sessions.py: session lifecycle manager.
- src/pool_telemetry/exporter.py: export JSON/CSV/JSONL helpers.

Run locally
1) Create venv and install deps: `scripts\setup_venv.ps1`
2) Activate: `.venv\Scripts\Activate.ps1`
3) Run from src: `python -m pool_telemetry.app`

Config and storage
- Default data directory: %USERPROFILE%\.pool_telemetry
- Config file: %USERPROFILE%\.pool_telemetry\config.json
- SQLite DB: %USERPROFILE%\.pool_telemetry\database.sqlite
- Logs: %USERPROFILE%\.pool_telemetry\logs\app.log

Gemini Live wiring
- Gemini client is started when a session starts (UI Start button).
- Frames are sampled by config.gemini.frame_sample_rate_ms and sent as JPEG.
- Raw events are routed to EventProcessor, which persists to SQLite.
- You must provide a Gemini API key in Settings.
- System prompt is configurable via config.gemini.system_prompt.
- Reconnect/backoff is enabled via config.gemini.reconnect_attempts and reconnect_delay_ms.
- NOTE: The exact Gemini Live message envelope may still need adjustments.

Known gaps / TODOs
- Session browser "Open" action only stores selected ID; no detailed viewer yet.
- Export dialog assumes current or last session; no multi-select exports.
- No tests yet; add unit tests for config/db/session/export when ready.

Suggested next tasks
- Implement full Gemini Live protocol (system prompt, response parsing, retries).
- Add session browser actions (open/export/delete).
- Improve UI telemetry overlays and ball motion visualization.
