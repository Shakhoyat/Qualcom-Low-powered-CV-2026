"""
Pipeline 2 — Qualcomm AI Hub Deploy Script (NDHWC + INT8 QNN)
=============================================================
Compiles qualcomm_r2plus1d.onnx (NDHWC + baked norm) to QNN binary format
on the Dragonwing IQ-9075 EVK with INT8 quantization.

Key differences from Pipeline 1:
  - Input:  NDHWC (1, 16, 112, 112, 3)  [not NCDHW]
  - Target: qnn_context_binary            [not tflite]
  - Quant:  INT8 with calibration data   [not FP16]

Usage (local):
    set QAI_HUB_API_TOKEN=wyfb5d78py0bgow6gz10krdqyd8qq0l9wlfw9swc
    python aihub_deploy.py --onnx qualcomm_r2plus1d.onnx --calib calibration_inputs.npy

Usage (Kaggle cell):
    Add QAI_HUB_API_TOKEN as a Kaggle secret, run this file as a script cell.

Drive artifacts (pre-trained, skip training):
    ONNX  : https://drive.google.com/file/d/13QB9_7sFMzYg_AoGcqfbHCG9BthXBw0I/view
    Calib : https://drive.google.com/file/d/12N6WmkBA2vg1rpoSe7PkaPOK7jy22Ny7/view
"""

import argparse
import json
import os
import glob
import numpy as np

QAI_HUB_TOKEN  = os.getenv("QAI_HUB_API_TOKEN", "wyfb5d78py0bgow6gz10krdqyd8qq0l9wlfw9swc")
DEVICE_NAME    = "Dragonwing IQ-9075 EVK"
INPUT_SHAPE    = (1, 16, 112, 112, 3)   # NDHWC — Pipeline 2 layout
LATENCY_LIMIT  = 34.0                    # ms
SUBMIT_EMAIL   = "lowpowervision@gmail.com"


def configure_hub():
    import importlib, qai_hub
    importlib.reload(qai_hub)
    os.environ["QAI_HUB_TOKEN"] = QAI_HUB_TOKEN
    hub = qai_hub.Client()
    print(f"Authenticated with Qualcomm AI Hub (token: {QAI_HUB_TOKEN[:8]}...)")
    return hub, qai_hub


def get_device(hub):
    devices = hub.get_devices(name=DEVICE_NAME)
    if not devices:
        raise RuntimeError(f"Device '{DEVICE_NAME}' not found on AI Hub.")
    device = devices[0]
    print(f"Target device: {device.name}")
    return device


def load_calibration(calib_path: str) -> dict:
    """
    Load calibration .npy array and format as AI Hub calibration dataset.
    Array shape expected: (N, 16, 112, 112, 3) — NDHWC, float32, [0,1]
    """
    data = np.load(calib_path).astype("float32")
    print(f"Calibration data: {data.shape}  dtype={data.dtype}  "
          f"range=[{data.min():.3f}, {data.max():.3f}]")
    # Hub expects: {"input_name": [sample_1, sample_2, ...]}
    # Each sample must match INPUT_SHAPE = (1, 16, 112, 112, 3)
    samples = [data[i:i+1] for i in range(len(data))]
    return {"input": samples}


def compile_fp16(hub, device, onnx_path: str, hub_model):
    """FP16 compile — establishes latency baseline."""
    print("\n--- STEP 1a: FP16 Compile (QNN binary) ---")
    job = hub.submit_compile_job(
        model=hub_model,
        device=device,
        name="LPCVC2026_Track2_P2_fp16",
        options="--target_runtime qnn_context_binary",
        input_specs={"input": INPUT_SHAPE},
    )
    print(f"FP16 compile job: {job.job_id}")
    print(f"  View: https://workbench.aihub.qualcomm.com/jobs/{job.job_id}/")
    print("Waiting...")
    job.wait()
    if not job.get_status().success:
        print(f"FP16 compile failed — skipping FP16 baseline.")
        return None, None
    target = job.get_target_model()
    print("FP16 compile SUCCESS")
    return job, target


def compile_int8(hub, device, hub_model, calib_dataset: dict):
    """INT8 compile with calibration data — target for submission."""
    print("\n--- STEP 1b: INT8 Compile (QNN binary + quantize_full_type int8) ---")
    job = hub.submit_compile_job(
        model=hub_model,
        device=device,
        name="LPCVC2026_Track2_P2_int8",
        options="--target_runtime qnn_context_binary --quantize_full_type int8 --quantize_io",
        input_specs={"input": INPUT_SHAPE},
        calibration_data=calib_dataset,
    )
    print(f"INT8 compile job: {job.job_id}")
    print(f"  View: https://workbench.aihub.qualcomm.com/jobs/{job.job_id}/")
    print("Waiting (INT8 compile takes 5-10 min)...")
    job.wait()
    if not job.get_status().success:
        raise RuntimeError(f"INT8 compile FAILED: {job.get_status().message}")
    target = job.get_target_model()
    print("INT8 compile SUCCESS")
    return job, target


def profile_model(hub, device, target_model, label: str, output_dir: str):
    print(f"\n--- STEP 2: Profile {label} ---")
    job = hub.submit_profile_job(
        model=target_model,
        device=device,
        name=f"LPCVC2026_Track2_P2_profile_{label}",
    )
    print(f"Profile job: {job.job_id}")
    print(f"  View: https://workbench.aihub.qualcomm.com/jobs/{job.job_id}/")
    print("Waiting (2-5 min)...")
    job.wait()

    os.makedirs(output_dir, exist_ok=True)
    job.download_results(artifacts_dir=output_dir)

    json_files = glob.glob(os.path.join(output_dir, "*.json"))
    with open(sorted(json_files)[-1]) as f:
        data = json.load(f)

    s        = data["execution_summary"]
    best_ms  = s["estimated_inference_time"] / 1000.0
    times_ms = [t / 1000 for t in s["all_inference_times"]]
    mean_ms  = sum(times_ms) / len(times_ms)
    max_ms   = max(times_ms)
    mem_mb   = s["estimated_inference_peak_memory"] / 1024 / 1024

    print(f"\n{'='*55}")
    print(f"[{label}] Inference time (best): {best_ms:.2f} ms")
    print(f"[{label}] Inference time (mean): {mean_ms:.2f} ms")
    print(f"[{label}] Inference time (max):  {max_ms:.2f} ms")
    print(f"[{label}] Peak memory:           {mem_mb:.2f} MB")
    print(f"[{label}] Latency limit:         {LATENCY_LIMIT} ms")
    if best_ms < LATENCY_LIMIT:
        print(f"VALID — margin: {LATENCY_LIMIT - best_ms:.2f} ms")
    else:
        print(f"OVER LIMIT by {best_ms - LATENCY_LIMIT:.2f} ms")
    print("=" * 55)
    return job, best_ms


def run_inference(hub, device, target_model, label: str):
    print(f"\n--- STEP 3: On-device inference test ({label}) ---")
    dummy = np.random.rand(*INPUT_SHAPE).astype("float32")
    job = hub.submit_inference_job(
        model=target_model,
        device=device,
        inputs={"input": [dummy]},
    )
    print(f"Inference job: {job.job_id}")
    print("Waiting...")
    output = job.download_output_data()
    logits = list(output.values())[0][0]
    top5   = np.argsort(logits[0])[::-1][:5]
    print(f"Output shape: {logits.shape}")
    print(f"Top-5 class indices: {top5}")
    print(f"Top-5 logits:        {logits[0][top5].round(3)}")
    return job


def share_and_summarize(compile_job_fp16, profile_job_fp16,
                        compile_job_int8, profile_job_int8,
                        latency_fp16, latency_int8):
    print(f"\n--- STEP 4: Share compile job for submission ---")
    best_job = compile_job_int8 if compile_job_int8 else compile_job_fp16
    best_lat = latency_int8 if latency_int8 else latency_fp16

    print(f"Share compile job {best_job.job_id} with {SUBMIT_EMAIL}:")
    print(f"  https://workbench.aihub.qualcomm.com/jobs/{best_job.job_id}/")

    print(f"\n{'='*60}")
    print(f"PIPELINE 2 SUBMISSION SUMMARY")
    print(f"{'='*60}")
    if compile_job_fp16:
        print(f"  FP16 compile job : {compile_job_fp16.job_id}  ({latency_fp16:.2f} ms)")
    if compile_job_int8:
        print(f"  INT8 compile job : {compile_job_int8.job_id}  ({latency_int8:.2f} ms)  ← SUBMIT THIS")
    print(f"  Device           : {DEVICE_NAME}")
    print(f"  Input layout     : NDHWC (1, 16, 112, 112, 3)")
    print(f"  Quantization     : INT8 qnn_context_binary")
    print(f"  Share with       : {SUBMIT_EMAIL}")
    print(f"  Submit form      : https://lpcv.ai/2026LPCVC/submission/track2")
    print("=" * 60)

    return best_job, best_lat


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx",       default="qualcomm_r2plus1d.onnx",
                        help="NDHWC ONNX model path")
    parser.add_argument("--calib",      default="calibration_inputs.npy",
                        help="Calibration data .npy path (100 samples NDHWC)")
    parser.add_argument("--output_dir", default="aihub_results_p2",
                        help="Directory to save profile JSON")
    parser.add_argument("--fp16_only",  action="store_true",
                        help="Skip INT8 compile (FP16 only)")
    parser.add_argument("--skip_infer", action="store_true",
                        help="Skip inference verification")
    args = parser.parse_args()

    if not os.path.isfile(args.onnx):
        raise FileNotFoundError(
            f"ONNX not found: {args.onnx}\n"
            "Download from Drive:\n"
            "  gdown https://drive.google.com/uc?id=13QB9_7sFMzYg_AoGcqfbHCG9BthXBw0I -O qualcomm_r2plus1d.onnx"
        )

    hub, _ = configure_hub()
    device  = get_device(hub)

    print(f"\nUploading ONNX ({os.path.getsize(args.onnx)/1024/1024:.1f} MB)...")
    hub_model = hub.upload_model(args.onnx)
    print(f"Uploaded: {hub_model}")

    # FP16 baseline
    compile_job_fp16, target_fp16 = compile_fp16(hub, device, args.onnx, hub_model)
    latency_fp16 = None
    if target_fp16:
        pj_fp16, latency_fp16 = profile_model(hub, device, target_fp16, "FP16",
                                               os.path.join(args.output_dir, "fp16"))
        if not args.skip_infer:
            run_inference(hub, device, target_fp16, "FP16")
    else:
        pj_fp16 = None

    # INT8 quantized (target for submission)
    compile_job_int8 = target_int8 = None
    latency_int8 = None
    if not args.fp16_only:
        if not os.path.isfile(args.calib):
            print(f"WARNING: calibration file not found at {args.calib}")
            print("Download: gdown https://drive.google.com/uc?id=12N6WmkBA2vg1rpoSe7PkaPOK7jy22Ny7 -O calibration_inputs.npy")
            print("Skipping INT8 compile.")
        else:
            calib_dataset = load_calibration(args.calib)
            compile_job_int8, target_int8 = compile_int8(hub, device, hub_model, calib_dataset)
            pj_int8, latency_int8 = profile_model(hub, device, target_int8, "INT8",
                                                   os.path.join(args.output_dir, "int8"))
            if not args.skip_infer:
                run_inference(hub, device, target_int8, "INT8")

    best_job, best_lat = share_and_summarize(
        compile_job_fp16, pj_fp16,
        compile_job_int8, pj_int8 if compile_job_int8 else None,
        latency_fp16, latency_int8,
    )

    log = {
        "pipeline": "P2_89pct",
        "onnx": args.onnx,
        "input_layout": "NDHWC",
        "device": DEVICE_NAME,
        "fp16_compile_job_id": compile_job_fp16.job_id if compile_job_fp16 else None,
        "fp16_latency_ms": round(latency_fp16, 2) if latency_fp16 else None,
        "int8_compile_job_id": compile_job_int8.job_id if compile_job_int8 else None,
        "int8_latency_ms": round(latency_int8, 2) if latency_int8 else None,
        "submit_job_id": best_job.job_id,
        "submit_latency_ms": round(best_lat, 2),
        "valid": best_lat < LATENCY_LIMIT,
    }
    os.makedirs(args.output_dir, exist_ok=True)
    log_path = os.path.join(args.output_dir, "experiment_log_p2.json")
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)
    print(f"\nLog saved: {log_path}")


if __name__ == "__main__":
    main()
