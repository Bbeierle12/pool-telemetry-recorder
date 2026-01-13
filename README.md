# Pool Telemetry Recorder

Phase 1 foundation scaffold for the desktop telemetry recorder.

## Quick start
1. Create a virtual environment:
   - PowerShell: `scripts\setup_venv.ps1`
   - Manual: `python -m venv .venv` then `.venv\Scripts\pip install -r requirements.txt`
2. Activate: `.venv\Scripts\Activate.ps1`
3. Run `python -m pool_telemetry.app` from `src`.

## Notes
- Config is stored under the user data directory.
- SQLite database lives alongside the config data directory.
