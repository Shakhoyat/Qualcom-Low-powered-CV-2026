"""
Pipeline 1 — Qualcomm AI Hub Deploy Script
==========================================
Compiles lpcvc_final_unified.onnx to TFLite on the Dragonwing IQ-9075 EVK,
profiles latency, runs inference, then shares the compile job for submission.

Usage (local):
    set QAI_HUB_API_TOKEN=wyfb5d78py0bgow6gz10krdqyd8qq0l9wlfw9swc
    python aihub_deploy.py --onnx lpcvc_final_unified.onnx

Usage (Kaggle):
    Add QAI_HUB_API_TOKEN as a Kaggle secret, then run this script as a cell.

Verified results (from Dadhichi-Track2/88% run):
    Compile job : jgj0wmy8p
    Profile job : j5mwd2lqp
    Latency     : 26.85 ms best / 27.88 ms mean  (limit: 34 ms)
    Memory      : 8.61 MB peak
"""

import argparse
import json
import os
import numpy as np

QAI_HUB_TOKEN  = os.getenv("QAI_HUB_API_TOKEN", "wyfb5d78py0bgow6gz10krdqyd8qq0l9wlfw9swc")
DEVICE_NAME    = "Dragonwing IQ-9075 EVK"
INPUT_SHAPE    = (1, 3, 16, 112, 112)   # NCDHW — Pipeline 1 layout
LATENCY_LIMIT  = 34.0                    # ms
SUBMIT_EMAIL   = "lowpowervision@gmail.com"


def configure_hub():
    import importlib, qai_hub
    importlib.reload(qai_hub)
    os.environ["QAI_HUB_TOKEN"] = QAI_HUB_TOKEN
    hub = qai_hub.Client()
    print(f"Authenticated with Qualcomm AI Hub (token: {QAI_HUB_TOKEN[:8]}...)")
    return hub, qai_hub


def get_device(hub, qai_hub):
    devices = hub.get_devices(name=DEVICE_NAME)
    if not devices:
        raise RuntimeError(f"Device '{DEVICE_NAME}' not found on AI Hub.")
    device = devices[0]
    print(f"Target device: {device.name}")
    return device


def compile_model(hub, device, onnx_path: str):
    print(f"\n--- STEP 1: Compile (TFLite FP16) ---")
    print(f"ONNX: {onnx_path} ({os.path.getsize(onnx_path)/1024/1024:.1f} MB)")

    hub_model = hub.upload_model(onnx_path)
    print(f"Uploaded model: {hub_model}")

    compile_job = hub.submit_compile_job(
        model=hub_model,
        device=device,
        name="LPCVC2026_Track2_P1_r2plus1d",
        options="--target_runtime tflite",
        input_specs={"input": INPUT_SHAPE},
    )
    print(f"Compile job: {compile_job.job_id}")
    print(f"  View: https://workbench.aihub.qualcomm.com/jobs/{compile_job.job_id}/")
    print("Waiting for compile...")
    compile_job.wait()

    status = compile_job.get_status()
    if not status.success:
        raise RuntimeError(f"Compile FAILED: {status.message}")

    print(f"Compile SUCCESS")
    target_model = compile_job.get_target_model()
    return compile_job, target_model


def profile_model(hub, device, target_model, output_dir: str = "."):
    print(f"\n--- STEP 2: Profile latency ---")

    profile_job = hub.submit_profile_job(
        model=target_model,
        device=device,
        name="LPCVC2026_Track2_P1_profile",
    )
    print(f"Profile job: {profile_job.job_id}")
    print(f"  View: https://workbench.aihub.qualcomm.com/jobs/{profile_job.job_id}/")
    print("Waiting for profile (2-5 min)...")
    profile_job.wait()

    os.makedirs(output_dir, exist_ok=True)
    profile_job.download_results(artifacts_dir=output_dir)

    # Parse saved JSON
    import glob
    json_files = glob.glob(os.path.join(output_dir, "*.json"))
    with open(sorted(json_files)[-1]) as f:
        data = json.load(f)

    summary = data["execution_summary"]
    best_ms  = summary["estimated_inference_time"] / 1000.0
    mean_ms  = sum(t / 1000 for t in summary["all_inference_times"]) / len(summary["all_inference_times"])
    max_ms   = max(t / 1000 for t in summary["all_inference_times"])
    mem_mb   = summary["estimated_inference_peak_memory"] / 1024 / 1024

    print(f"\n{'='*55}")
    print(f"Inference time (best): {best_ms:.2f} ms")
    print(f"Inference time (mean): {mean_ms:.2f} ms")
    print(f"Inference time (max):  {max_ms:.2f} ms")
    print(f"Peak memory:           {mem_mb:.2f} MB")
    print(f"Latency limit:         {LATENCY_LIMIT} ms")
    if best_ms < LATENCY_LIMIT:
        print(f"VALID — margin: {LATENCY_LIMIT - best_ms:.2f} ms")
    else:
        print(f"OVER LIMIT by {best_ms - LATENCY_LIMIT:.2f} ms")
    print("=" * 55)

    return profile_job, best_ms


def run_inference(hub, device, target_model):
    print(f"\n--- STEP 3: On-device inference test ---")

    dummy = np.random.rand(*INPUT_SHAPE).astype("float32")

    inference_job = hub.submit_inference_job(
        model=target_model,
        device=device,
        inputs={"input": [dummy]},
    )
    print(f"Inference job: {inference_job.job_id}")
    print("Waiting...")
    output = inference_job.download_output_data()

    logits = list(output.values())[0][0]     # (1, 91)
    top5   = np.argsort(logits[0])[::-1][:5]
    print(f"Output shape: {logits.shape}")
    print(f"Top-5 class indices: {top5}")
    print(f"Top-5 logits:        {logits[0][top5].round(3)}")
    return inference_job


def share_and_summarize(compile_job, profile_job):
    print(f"\n--- STEP 4: Share compile job for submission ---")
    print(f"Share compile job {compile_job.job_id} with {SUBMIT_EMAIL} on AI Hub:")
    print(f"  https://workbench.aihub.qualcomm.com/jobs/{compile_job.job_id}/")
    print(f"\n=== SUBMISSION SUMMARY ===")
    print(f"  Compile job ID : {compile_job.job_id}")
    print(f"  Profile job ID : {profile_job.job_id}")
    print(f"  Device         : {DEVICE_NAME}")
    print(f"  Share with     : {SUBMIT_EMAIL}")
    print(f"  Submit form    : https://lpcv.ai/2026LPCVC/submission/track2")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx",       default="lpcvc_final_unified.onnx",
                        help="Path to ONNX model")
    parser.add_argument("--output_dir", default="aihub_results_p1",
                        help="Directory to save profile JSON")
    parser.add_argument("--skip_infer", action="store_true",
                        help="Skip inference test (saves time)")
    args = parser.parse_args()

    if not os.path.isfile(args.onnx):
        raise FileNotFoundError(
            f"ONNX not found: {args.onnx}\n"
            f"Download from Drive or run kaggle_training.py first."
        )

    hub, qai_hub = configure_hub()
    device = get_device(hub, qai_hub)

    compile_job, target_model = compile_model(hub, device, args.onnx)
    profile_job, latency_ms   = profile_model(hub, device, target_model, args.output_dir)

    if not args.skip_infer:
        run_inference(hub, device, target_model)

    share_and_summarize(compile_job, profile_job)

    # Save IDs for experiment log
    log = {
        "pipeline": "P1_88pct",
        "onnx": args.onnx,
        "device": DEVICE_NAME,
        "compile_job_id": compile_job.job_id,
        "profile_job_id": profile_job.job_id,
        "latency_ms": round(latency_ms, 2),
        "valid": latency_ms < LATENCY_LIMIT,
    }
    log_path = os.path.join(args.output_dir, "experiment_log_p1.json")
    os.makedirs(args.output_dir, exist_ok=True)
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)
    print(f"\nLog saved: {log_path}")


if __name__ == "__main__":
    main()
