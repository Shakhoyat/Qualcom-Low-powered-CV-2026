# Qualcomm AI Hub Workbench Notes

Source: https://workbench.aihub.qualcomm.com/docs/

## What Workbench Provides
- Model conversion from source frameworks.
- Hardware-aware optimization for Qualcomm targets.
- On-device profiling and inference on cloud-provisioned physical devices.
- Deployment artifacts for supported runtimes.

## Typical Workflow
1. Optimize or compile model for target runtime.
2. Profile on target device.
3. Run inference for numerical and task-level checks.
4. Download artifacts and integrate into evaluation pipeline.

## Documentation Areas
- Getting started.
- Compile, profile, inference examples.
- CLI usage.
- Jobs and devices.
- API reference.

## Team Recommendation
- Standardize on one target device per track during iteration.
- Track compile job IDs and profile job IDs for each experiment.
- Keep an experiment log mapping model commit, dataset version, and job IDs.
