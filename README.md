# Qualcomm Low-Powered CV 2026

This repository contains notebooks, preprocessing code, and experiment files for LPCVC Track-2.
Large model artifacts (ONNX/PyTorch checkpoints and related binary files) should be kept in Google Drive and downloaded on demand to keep the repository lightweight.

## Documentation Index

- Competition intro notes: competition-conext/introduction.md
- Track 1 notes: competition-conext/track1.md
- Track 2 notes: competition-conext/track2.md
- Track 3 notes: competition-conext/track3.md
- Qualcomm Workbench notes: competition-conext/workbench-docs.md
- Track 1 sample solution notes: competition-conext/track1-sample-solution.md
- Track 2 sample solution notes: competition-conext/track2-sample-solution.md
- Track 3 sample solution notes: competition-conext/track3-sample-solution.md
- Team workflow guide: competition-conext/team-usage-guide.md

## Why this setup

GitHub repositories become hard to manage when very large binaries are committed repeatedly.
Store heavy artifacts in Drive and pull them only when needed.

## Google Drive Links

- Simple solution folder:
  - https://drive.google.com/drive/folders/156w6N9ZWx2pfQB0SNzdohEzeJ7fzFmVC?usp=drive_link
- Model storage folder:
  - https://drive.google.com/drive/folders/1bzfwjChnI4NeEdUTlDsQbJU8qn0lyQ5b?usp=sharing

## Install gdown

```bash
pip install gdown
```

## AI Hub API Key Usage (Secure)

Do not store API keys in this repository.
Use environment variables locally.

PowerShell:

```powershell
$env:QAI_HUB_API_TOKEN = "<your_api_token_here>"
$env:QUALCOMM_WORKBENCH_API_KEY = $env:QAI_HUB_API_TOKEN
```

Python pattern:

```python
import os

api_token = os.getenv("QAI_HUB_API_TOKEN") or os.getenv("QUALCOMM_WORKBENCH_API_KEY")
if not api_token:
  raise RuntimeError("Missing AI Hub API token")
```

## Windows script (recommended)

Use this PowerShell script to download your Drive folder and preserve the same folder structure locally.

Script file: `download_models_from_drive.ps1`

### Run with full Drive folder URL

```powershell
powershell -ExecutionPolicy Bypass -File .\download_models_from_drive.ps1 `
  -DriveFolder "https://drive.google.com/drive/folders/1bzfwjChnI4NeEdUTlDsQbJU8qn0lyQ5b?usp=sharing" `
  -Destination "." `
  -InstallGdown
```

### Run with folder ID only

```powershell
powershell -ExecutionPolicy Bypass -File .\download_models_from_drive.ps1 `
  -DriveFolder "1bzfwjChnI4NeEdUTlDsQbJU8qn0lyQ5b" `
  -Destination "." `
  -InstallGdown
```

If Drive is structured as `Dadhichi-Track2/88` and `Dadhichi-Track2/89`, it will be recreated in your local destination.

## Download model artifacts with gdown

### Option 1: Download a full Drive folder

```bash
# Download the whole models folder into repository root
gdown --folder "https://drive.google.com/drive/folders/1bzfwjChnI4NeEdUTlDsQbJU8qn0lyQ5b" -O .
```

If your Drive folder already has subfolders like `Dadhichi-Track2/88%` and `Dadhichi-Track2/89%`, this command will recreate that structure locally.

### Option 2: Download a single file (if you know file ID)

```bash
# Replace FILE_ID and output path
gdown "https://drive.google.com/uc?id=FILE_ID" -O "Dadhichi-Track2/88%/lpcvc_final_unified.onnx"
```

## Files recommended to store in Drive (large/binary artifacts)

### Dadhichi-Track2/88%
- best_r2plus1d_qevd.pth - done - https://drive.google.com/file/d/1m9aK8JgjFa6ewDhLpIb5rAepUybs5veU/view?usp=sharing
- latest_checkpoint.pth - done - https://drive.google.com/file/d/1mbgYeDpvPdjFYlHuuT18oEBN79bWrl3f/view?usp=drive_link
- lpcvc_final_unified.onnx - done - https://drive.google.com/file/d/1tEObF3rGGO69y7DvEM3xeieUcuwLs6WH/view?usp=drive_link
- lpcvc_final_unified_fixed.onnx - done - https://drive.google.com/file/d/1AAq-jPIooA3k5mR1-EL8bs6zXaGCzT_V/view?usp=drive_link
- qualcomm_r2plus1d.onnx - done - https://drive.google.com/file/d/1phVY0DqCkBqSfTdZcvfVa8Kc3nE9zi0s/view?usp=drive_link
- qualcomm_r2plus1d.onnx.data - done - https://drive.google.com/file/d/1-7eo0o6zGjD4h4iEol3C_Y3oXZvnHnDg/view?usp=drive_link

### Dadhichi-Track2/89%
- best_r2plus1d_qevd.pth - done - https://drive.google.com/file/d/1txn4uzy8rdl-XtK6P1KTQkry61diOn1b/view?usp=drive_link
- latest_checkpoint.pth - done - https://drive.google.com/file/d/1X0xFXLdmqNXi-FkyJwH2-L9t9ZoGNrdx/view?usp=drive_link
- qualcomm_r2plus1d.onnx - done - https://drive.google.com/file/d/13QB9_7sFMzYg_AoGcqfbHCG9BthXBw0I/view?usp=drive_link
- calibration_inputs.npy - done - https://drive.google.com/file/d/12N6WmkBA2vg1rpoSe7PkaPOK7jy22Ny7/view?usp=drive_link

## Keep in GitHub (code and lightweight files)

These files are small and should stay in the repository:

- notebooks (`*.ipynb`)
- configs and mappings (`*.json`)
- docs (`README`, notes, links)
- small images/logs if needed

## Suggested workflow

1. Clone the repo.
2. Install dependencies (including `gdown`).
3. Download required model files from Drive.
4. Run notebooks/training/inference.
5. Do not commit large binary artifacts back to GitHub.

## Submission Guide (Team Standard)

For Track 1 and Track 2, use this flow:

1. Compile and validate model on Qualcomm AI Hub.
2. Share compile job/model permission with lowpowervision@gmail.com.
3. Submit official LPCVC form for your track with compile job ID.
4. Save form confirmation and job IDs in team logs.

If share step or form step is missing, the submission can be considered invalid.

For detailed team process, see: competition-conext/team-usage-guide.md

## Optional: avoid accidental commits of heavy files

Add patterns like these to `.gitignore` if needed:

```gitignore
*.onnx
*.onnx.data
*.pth
*.npy
```

If you need selective tracking, remove broad patterns and ignore exact files only.
