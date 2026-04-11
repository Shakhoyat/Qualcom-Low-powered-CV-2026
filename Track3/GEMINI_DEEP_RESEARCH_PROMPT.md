# Gemini Deep Research Prompt
### Use this in Gemini 1.5 Pro / 2.0 Flash Thinking / Deep Research mode

---

Copy everything between the `---PROMPT START---` and `---PROMPT END---` markers and paste into Gemini.

---PROMPT START---

## Research Task: Win the LPCVC 2026 Track 3 Competition

I am competing in the **2026 IEEE Low-Power Computer Vision Challenge (LPCVC) Track 3: AI Generated Image Detection**.

I need you to conduct deep research across recent academic papers (2023–2025), GitHub repositories, and technical blogs to help me build the best possible pipeline for this competition. I will use your findings to build a second competing pipeline independently of my first approach.

---

### Competition Context

**Task:** Given an input image, determine if it is AI-generated or real (authentic photograph), AND provide a structured forensic explanation across exactly 8 criteria.

**Output format (exact JSON schema required):**
```json
{
  "overall_likelihood": "AI-Generated" | "Real",
  "per_criterion": [
    {"criterion": "Lighting & Shadows Consistency", "score": 0 | 1, "evidence": "..."},
    {"criterion": "Edges & Boundaries", "score": 0 | 1, "evidence": "..."},
    {"criterion": "Texture & Resolution", "score": 0 | 1, "evidence": "..."},
    {"criterion": "Perspective & Spatial Relationships", "score": 0 | 1, "evidence": "..."},
    {"criterion": "Physical & Common Sense Logic", "score": 0 | 1, "evidence": "..."},
    {"criterion": "Text & Symbols", "score": 0 | 1, "evidence": "..."},
    {"criterion": "Human & Biological Structure Integrity", "score": 0 | 1, "evidence": "..."},
    {"criterion": "Material & Object Details", "score": 0 | 1, "evidence": "..."}
  ]
}
```

**Hardware constraint:** Must run on **Snapdragon 8 Gen 5** (Qualcomm mobile SoC) within a strict latency budget.  
**Submission format:** Quantized model compiled to Qualcomm QNN binary format.  
**Base model used by organizers:** Qwen2-VL-2B-Instruct (2 billion parameter VLM).  
**Scoring:** Stage 1 = latency validity gate. Stage 2 = detection accuracy + explanation quality score.

My first pipeline uses LoRA fine-tuning of Qwen2-VL-2B. I need you to research **alternative architectures and approaches** that might outperform this.

---

### Research Questions — Answer All of These

#### A. SOTA AI-Generated Image Detection (papers & methods)

1. What are the best-performing methods for AI-generated image detection published in 2023–2025? Include paper titles, authors, arXiv IDs, and key accuracy numbers on standard benchmarks (GenImage, ArtiFact, CIFAKE).

2. Which detection approaches are **model-agnostic** (work across SD, DALL-E, Midjourney, Firefly) versus **generator-specific**? What is the accuracy gap?

3. What are the best **frequency-domain methods** (FFT, DCT, wavelet-based) for detecting AI images? Papers like DIRE, UnivFD — are there better 2024-2025 alternatives?

4. What is the current SOTA on the **GenImage benchmark** specifically? What method achieves the highest cross-generator generalization?

5. Are there methods that use **semantic anomaly detection** — embedding an image and checking if it falls inside the real-image distribution? How do these compare to supervised classifiers?

#### B. Vision-Language Models for Structured Forensic Analysis

6. What is the best way to fine-tune a **2B-parameter VLM** (Qwen2-VL, InternVL2, LLaVA) for a structured classification + explanation task? Compare LoRA, QLoRA, DoRA, and full fine-tuning for this model size.

7. Are there papers that combine a **fast binary detector** with a **VLM explanation generator**? For example: run a lightweight CLIP-based detector first, then condition the VLM prompt on the detector score. Does this improve accuracy?

8. What is the best prompting strategy for getting VLMs to produce **reliable binary classifications** (AI-Generated vs Real) with high confidence? Research: chain-of-thought, few-shot, self-consistency voting, structured output forcing.

9. What **training data strategies** maximize detection accuracy of fine-tuned VLMs on unseen AI generators? Data augmentation? Contrastive learning? Label smoothing?

10. For the **explanation quality** component (generating per-criterion evidence): what methods produce the most specific, grounded evidence text rather than generic statements? Are there papers on "grounded VLM captioning"?

#### C. Edge Deployment & Quantization (Snapdragon 8 Gen 5 specific)

11. What is the accuracy drop (in percentage points) when quantizing Qwen2-VL-2B from FP16 to:
    - W8A8 (INT8 weights, INT8 activations)
    - W4A8 (INT4 weights, INT8 activations)
    - W4A16 (INT4 weights, FP16 activations)
    Which scheme gives the best accuracy/latency tradeoff on Snapdragon?

12. What are the best **quantization-aware training (QAT)** techniques for VLMs specifically? Is QAT with AIMET better than post-training quantization (PTQ) with GPTQ/AWQ for a 2B model?

13. Are there **sub-2B models** that achieve better accuracy than Qwen2-VL-2B for this specific task (image classification + structured text generation) and that would fit in the same or smaller latency budget? Consider: moondream2 (1.8B), SmolVLM-2 (2.2B), MiniCPM-V-2 (2.4B).

14. What **alternative architectures** would be faster on Snapdragon 8 Gen 5 for this task? For example: a two-head model with a ViT encoder for detection + a lightweight LLM for explanation. Would this be faster than a single VLM pass?

#### D. Dataset Strategy

15. What is the best **dataset combination** for training a generalizable AI-image detector in 2025? Rank these by expected contribution: CIFAKE, GenImage, ArtiFact, WildFake, RAISE, DiffusionForensics, LSDA. Are there newer 2024-2025 datasets I should use?

16. For **synthetic annotation** (using GPT-4o or Claude to generate per-criterion evidence labels): what prompt engineering produces the highest-quality forensic annotations? Are there papers on using LLMs to annotate forensic datasets?

17. What is the minimum training set size needed to meaningfully fine-tune a 2B VLM for this task? Are there papers on few-shot or low-data fine-tuning for AI detection?

#### E. Competition-Specific Strategy

18. Given that the competition evaluates **both detection AND explanation quality**: is it better to optimize for detection accuracy and hope explanations improve with it, or should I train with explicit explanation supervision?

19. Are there published approaches where a model performs **worse at detection** but **better at explanation quality** due to being more cautious/verbose? What is the tradeoff?

20. What is the best strategy for handling the **8 criteria independently** vs. jointly? Should I train the model to reason about all 8 simultaneously, or sequentially (chain-of-thought per criterion)?

21. For the **latency constraint** on Snapdragon 8 Gen 5: given the two-step prompt pipeline (analysis → JSON output), what is the approximate token budget per step that keeps total inference under typical mobile latency limits (~5-10 seconds)?

22. Based on all of the above: propose your best **alternative pipeline** to my LoRA-fine-tuned Qwen2-VL-2B approach. The pipeline must:
    - Fit within Snapdragon 8 Gen 5 latency budget
    - Produce the exact JSON schema above
    - Maximize both detection accuracy and explanation quality
    - Be feasible to implement in 2-3 weeks with Kaggle GPU access

---

### Output Format I Need

For each research question above, provide:

1. **Answer** (2-5 sentences with specific numbers/claims)
2. **Key papers** (title + arXiv ID or venue + year)
3. **Recommended action** for my pipeline (one concrete step I should take)

At the end, provide:

**Section F: Recommended Alternative Pipeline**  
A step-by-step description of the best pipeline you found that differs from my LoRA approach. Include:
- Model choice and why
- Dataset and preprocessing
- Training approach
- Quantization strategy
- Expected accuracy estimate
- Kaggle notebook structure (what cells to run in what order)

**Section G: Key Papers to Read First**  
Top 5 most important papers ranked by expected impact on my competition score, with arXiv links and a 2-sentence summary of why each matters.

---

### Context About My Current Setup

- **Hardware:** Kaggle T4 x2 (32GB VRAM total), Windows 11 local machine
- **Budget:** Kaggle free tier GPU hours (30h/week), small OpenAI API budget for annotation
- **Deadline:** April 30, 2026 (approximately 3 weeks remaining from now)
- **First pipeline:** Qwen2-VL-2B + LoRA (r=64) fine-tuning → AIMET quantization → QNN compile
- **I need:** A **fundamentally different** second pipeline — different model, different training approach, or different architecture — that I can run independently and potentially ensemble or compare against

Be as specific as possible. I need paper citations, model names, dataset download commands, and concrete implementation guidance. Generic advice is not useful.

---PROMPT END---

---

## Tips for Using This Prompt

1. **Use Gemini 1.5 Pro with "Deep Research" mode** enabled (the research globe icon) — this allows Gemini to browse the web and retrieve recent papers.

2. **Follow-up prompts to use after you get the initial response:**

   - "For the alternative pipeline you proposed in Section F, write me the exact Kaggle notebook cell structure with pip installs, model loading code, and training loop."
   - "Find me the exact HuggingFace dataset IDs for GenImage and ArtiFact, and the gdown/wget commands to download them in Kaggle."
   - "Compare Qwen2-VL-2B vs MiniCPM-V-2 vs InternVL2-2B for this specific task — which has better visual reasoning on anomaly detection?"
   - "What does the compute_score formula look like for explanation quality — how do the organizers likely score the evidence strings?"
   - "Give me the best system prompt + few-shot examples for getting Qwen2-VL to reliably produce valid JSON on the first try without need for retries."

3. **If Gemini can't access a paper:** ask it to describe the method in detail based on what it knows from training data.

4. **Save the full Gemini response** as `Track3/GEMINI_RESEARCH_OUTPUT.md` in this repo for team reference.
