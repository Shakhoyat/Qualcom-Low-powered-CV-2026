# Team Usage Guide (LPCVC 2026)

This guide defines how our team should run experiments, compile on AI Hub, and submit safely.

## 1. Security First
- Never commit API keys or tokens to the repository.
- Store credentials in environment variables only.
- Rotate leaked keys immediately.

PowerShell setup per session:

```powershell
$env:QAI_HUB_API_TOKEN = "<your_api_token_here>"
# If your scripts expect another variable name, map it as well
$env:QUALCOMM_WORKBENCH_API_KEY = $env:QAI_HUB_API_TOKEN
```

Python usage pattern:

```python
import os

api_token = os.getenv("QAI_HUB_API_TOKEN") or os.getenv("QUALCOMM_WORKBENCH_API_KEY")
if not api_token:
    raise RuntimeError("Missing AI Hub API token. Set QAI_HUB_API_TOKEN in your shell.")
```

## 2. Repository Setup
1. Clone repository.
2. Create Python environment.
3. Install dependencies.
4. Pull large model files from Drive.

Use model pull script:

```powershell
powershell -ExecutionPolicy Bypass -File .\download_models_from_drive.ps1 -Destination "." -InstallGdown
```

## 3. Team Working Rules
- Keep only code and lightweight artifacts in Git.
- Store large binaries in Drive.
- Name experiments consistently: date_track_model_variant.
- Log every AI Hub run with model commit hash and job IDs.

## 4. Track 2 Daily Workflow
1. Prepare dataset and preprocessing config.
2. Train or fine-tune.
3. Export or compile model on AI Hub.
4. Profile and run inference.
5. Save results and update experiment log.

## 5. Submission Workflow (Track 1/2 pattern)
1. Ensure model is compiled and validated on AI Hub.
2. Share compile job/model permission with lowpowervision@gmail.com.
3. Open official track submission form and submit:
   - Team info.
   - Compile job ID.
   - Required metadata.
4. Keep submission confirmation and timestamp in team log.

Important:
- If sharing step or form step is missing, submission may not be evaluated.
- Use one form entry per model/job as required by track instructions.

## 6. Suggested Team Roles
- Training owner: model training and checkpoints.
- Deployment owner: ONNX export, AI Hub compile/profile jobs.
- Evaluation owner: metrics, latency checks, leaderboard tracking.
- Submission owner: permission sharing + final form submission.

## 7. Pre-Submission Checklist
- Model passes required latency validity gate for the track.
- Accuracy metric computed with track-compatible evaluation script.
- Compile job ID recorded and shared to organizer email.
- Submission form completed with correct job ID.
- Artifacts and scripts are reproducible from repository state.
