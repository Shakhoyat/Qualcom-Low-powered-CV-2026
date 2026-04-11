"""
Pipeline 2 prompts — single-pass prefill-guided JSON generation.

Key differences from prompts.py (Pipeline 1):
  - Criteria ordered GLOBAL → LOCAL for sequential attention scan
  - Single forward pass (no two-step reasoning→JSON split)
  - PREFILL_JSON_START hardcoded as assistant prefill to force JSON mode
  - Evidence budget: ≤30 tokens per criterion (~150 characters)

Do NOT change after fine-tuning starts.
"""

SYSTEM_PROMPT_P2 = (
    "You are a forensic image analyst. "
    "When shown an image, you evaluate 8 physical and visual criteria strictly in the order given, "
    "scanning from global scene structure down to fine local details. "
    "You output ONLY a single valid JSON object — no preamble, no markdown, no extra text."
)

# Criteria in global-to-local order (required for sequential attention scan)
CRITERIA_P2 = [
    "Perspective & Spatial Relationships",
    "Physical & Common Sense Logic",
    "Lighting & Shadows Consistency",
    "Human & Biological Structure Integrity",
    "Material & Object Details",
    "Texture & Resolution",
    "Edges & Boundaries",
    "Text & Symbols",
]

VALID_LABELS = {"AI-Generated", "Real"}

# This single prompt asks for the full JSON in one shot.
# The assistant prefill (PREFILL_JSON_START) forces immediate JSON generation.
PROMPT_P2 = (
    "Examine this image forensically across 8 criteria (in order listed below). "
    "For each criterion: assign score=1 if suspicious/anomalous, score=0 if natural. "
    "Keep each evidence string under 30 words — be specific and concrete.\n\n"
    "Criteria (evaluate in this exact order):\n"
    "1. Perspective & Spatial Relationships — relative sizes, vanishing points, depth plausibility.\n"
    "2. Physical & Common Sense Logic — impossible objects, physics violations, implausible combinations.\n"
    "3. Lighting & Shadows Consistency — light source direction, cast shadows, highlights.\n"
    "4. Human & Biological Structure Integrity — finger count, eye symmetry, skin, hair coherence.\n"
    "5. Material & Object Details — glass/metal/fabric realism, specularity, surface properties.\n"
    "6. Texture & Resolution — uniform smoothness, repetitive patterns, unnatural sharpness.\n"
    "7. Edges & Boundaries — halos, unnatural blending, overly sharp or blurred object outlines.\n"
    "8. Text & Symbols — any text, logos, numbers — legibility and correctness.\n\n"
    "Respond with ONLY the JSON object below, filled in completely:"
)

# Injected as the start of the assistant's response during both training and inference.
# Forces the model into immediate JSON generation mode (Prefill-Guided Thinking).
PREFILL_JSON_START = '{"overall_likelihood": "'


def build_full_json_template(label: str, criteria_data: list[dict]) -> str:
    """
    Build the complete target JSON string for training supervision.
    label: "AI-Generated" or "Real"
    criteria_data: list of dicts with keys criterion, score, evidence — in CRITERIA_P2 order.
    """
    import json as _json
    obj = {
        "overall_likelihood": label,
        "per_criterion": [
            {
                "criterion": item["criterion"],
                "score": item["score"],
                "evidence": item["evidence"][:200],  # hard cap
            }
            for item in criteria_data
        ],
    }
    return _json.dumps(obj, ensure_ascii=False)


def validate_output_p2(output: dict) -> bool:
    """Return True if output matches Pipeline 2 JSON schema exactly."""
    if not isinstance(output, dict):
        return False
    if output.get("overall_likelihood") not in VALID_LABELS:
        return False
    criteria = output.get("per_criterion", [])
    if len(criteria) != 8:
        return False
    for i, item in enumerate(criteria):
        if item.get("criterion") != CRITERIA_P2[i]:
            return False
        if item.get("score") not in (0, 1):
            return False
        if not isinstance(item.get("evidence"), str) or len(item["evidence"]) < 5:
            return False
    return True
