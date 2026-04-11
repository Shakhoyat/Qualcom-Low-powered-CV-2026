"""
Shared prompt templates for Track 3 — AI Generated Image Detection.
Import this everywhere: training, inference, annotation, evaluation.
These prompts are the core of the pipeline — do not change after fine-tuning starts.
"""

SYSTEM_PROMPT = (
    "You are a forensic image analyst specializing in detecting AI-generated content. "
    "You examine images with scientific precision, looking for subtle artifacts that differentiate "
    "AI-generated images from authentic photographs."
)

PROMPT_1 = (
    "Carefully examine this image and analyze it across these 8 forensic criteria. "
    "For each criterion, think step by step about what you observe.\n\n"
    "1. **Lighting & Shadows Consistency**: Are light sources consistent? Do shadows fall correctly? "
    "Look for impossible highlights, missing cast shadows, or inconsistent shadow directions.\n\n"
    "2. **Edges & Boundaries**: Examine object boundaries carefully. AI images often show "
    "unnatural smoothing, halos, or blending artifacts at edges between objects.\n\n"
    "3. **Texture & Resolution**: Is texture uniformly sharp or strangely uniform? "
    "AI images often have either over-smooth textures or repetitive texture patterns.\n\n"
    "4. **Perspective & Spatial Relationships**: Do objects maintain correct relative sizes? "
    "Are parallel lines converging correctly? Does depth look physically plausible?\n\n"
    "5. **Physical & Common Sense Logic**: Do all elements make physical sense together? "
    "Look for impossible object combinations or physically implausible configurations.\n\n"
    "6. **Text & Symbols**: Examine any text, logos, signs, numbers, or symbols closely. "
    "AI models frequently corrupt text into illegible or nonsensical characters.\n\n"
    "7. **Human & Biological Structure Integrity**: Check hands (finger count/shape), "
    "eyes (symmetry/reflections), ears, teeth, hair strand coherence, skin texture.\n\n"
    "8. **Material & Object Details**: Are material properties (glass, metal, fabric, skin) "
    "rendered with realistic light interaction? Look for plastic-looking skin or "
    "unrealistic material specularity.\n\n"
    "Analyze each criterion and provide your detailed forensic observations."
)

PROMPT_2 = (
    "Based on your forensic analysis above, provide your final assessment "
    "in the following exact JSON format. Use only the criterion names as specified.\n\n"
    '{\n'
    '  "overall_likelihood": "<AI-Generated or Real>",\n'
    '  "per_criterion": [\n'
    '    {\n'
    '      "criterion": "Lighting & Shadows Consistency",\n'
    '      "score": <0 for natural, 1 for suspicious>,\n'
    '      "evidence": "<your specific observation>"\n'
    '    },\n'
    '    {\n'
    '      "criterion": "Edges & Boundaries",\n'
    '      "score": <0 or 1>,\n'
    '      "evidence": "<specific observation>"\n'
    '    },\n'
    '    {\n'
    '      "criterion": "Texture & Resolution",\n'
    '      "score": <0 or 1>,\n'
    '      "evidence": "<specific observation>"\n'
    '    },\n'
    '    {\n'
    '      "criterion": "Perspective & Spatial Relationships",\n'
    '      "score": <0 or 1>,\n'
    '      "evidence": "<specific observation>"\n'
    '    },\n'
    '    {\n'
    '      "criterion": "Physical & Common Sense Logic",\n'
    '      "score": <0 or 1>,\n'
    '      "evidence": "<specific observation>"\n'
    '    },\n'
    '    {\n'
    '      "criterion": "Text & Symbols",\n'
    '      "score": <0 or 1>,\n'
    '      "evidence": "<specific observation>"\n'
    '    },\n'
    '    {\n'
    '      "criterion": "Human & Biological Structure Integrity",\n'
    '      "score": <0 or 1>,\n'
    '      "evidence": "<specific observation>"\n'
    '    },\n'
    '    {\n'
    '      "criterion": "Material & Object Details",\n'
    '      "score": <0 or 1>,\n'
    '      "evidence": "<specific observation>"\n'
    '    }\n'
    '  ]\n'
    '}\n\n'
    "Respond with ONLY the JSON. No markdown fences, no explanation."
)

CRITERIA = [
    "Lighting & Shadows Consistency",
    "Edges & Boundaries",
    "Texture & Resolution",
    "Perspective & Spatial Relationships",
    "Physical & Common Sense Logic",
    "Text & Symbols",
    "Human & Biological Structure Integrity",
    "Material & Object Details",
]

VALID_LABELS = {"AI-Generated", "Real"}


def validate_output(output: dict) -> bool:
    """Return True if output matches the required JSON schema exactly."""
    if not isinstance(output, dict):
        return False
    if output.get("overall_likelihood") not in VALID_LABELS:
        return False
    criteria = output.get("per_criterion", [])
    if len(criteria) != 8:
        return False
    for i, item in enumerate(criteria):
        if item.get("criterion") != CRITERIA[i]:
            return False
        if item.get("score") not in (0, 1):
            return False
        if not isinstance(item.get("evidence"), str) or len(item["evidence"]) < 5:
            return False
    return True
