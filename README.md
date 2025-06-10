# Qwen3-72B-Instruct

Created by Eric Hartford

**Model ID:** `CognitiveComputations/Qwen3-72B-Instruct`
**Model Type:** Causal Language Model
**Architecture:** Qwen3

---

## 🚨 Disclaimer & Known Limitations 🚨

> **Change:** Title is more specific.

This model, `Qwen3-72B-Instruct`, is an **experimental, community-created model**. It was not trained or fine-tuned by the original Qwen team. It has been constructed by merging weights from `Qwen/Qwen2.5-72B-Instruct` into the `Qwen3` architecture.

While the conversion process was meticulous and the model passes basic generation tests, it comes with specific limitations that users must be aware of:

*   **Benchmark Performance:** The model has **not** been evaluated on standard academic benchmarks (e.g., MMLU, HellaSwag). Its performance relative to the original Qwen2.5-72B is unknown.
*   **Long-Context Coherence:** The model inherits the `sliding_window` configuration from Qwen2.5 but uses the `rope_theta` value from Qwen3. This architectural mismatch **may degrade performance on contexts longer than the original training window**. This requires further validation.
*   **Quantization Stability:** Hybrid models can have unique weight distributions. The model's stability and performance under 4-bit (or lower) quantization have not been tested.
*   **Safety Alignment:** While the base model was instruction-tuned and aligned, this merge may have altered its safety characteristics. The safety alignment has not been re-validated.

**Use this model with a clear understanding of these limitations, primarily for research and experimentation.**

> **Reasoning:** The disclaimer is now much more specific and actionable. It directly addresses the excellent points your friend raised (long-context, quantization, benchmarks), which builds trust and helps users make informed decisions.

---

## Model Purpose and Motivation

`Qwen3-72B-Instruct` is a 72-billion parameter, instruction-tuned, decoder-only language model designed to be architecturally compatible with the Qwen3 family. It aims to combine the extensive knowledge of the powerful `Qwen/Qwen2.5-72B-Instruct` model with the architectural improvements of the Qwen3 series.

#### Future Work: A Student for Knowledge Distillation

A primary motivation for creating this model is to serve as a high-quality "student" model for future research. The plan is to perform **knowledge and logit distillation** from the much larger `Qwen/Qwen3-235B-A22B` Mixture-of-Experts (MoE) model. By using this 72B model as a base, we aim to distill the specialized knowledge and reasoning capabilities of the 235B MoE model into a more efficient, dense architecture.

> **Change:** Merged "Model Description" and "Future Work" for a clearer narrative flow.

---

## A Meticulous Conversion Process

> **Change:** New title highlights the quality of the process.

This model is a "hybrid conversion" created using a robust script that prioritizes correctness, transparency, and safety. The key steps were:

1.  **Pre-flight Checks**: Before conversion, the script performed critical architectural checks, asserting that fundamental components like the `hidden_act` function were compatible between the source and donor models.
2.  **Base & Donor Models**: Weights were sourced from `Qwen/Qwen2.5-72B-Instruct` (base) and `Qwen/Qwen3-32B` (donor).
3.  **Vocabulary & Embedding Reconstruction**: The model uses the official `Qwen3` tokenizer (vocab size: **151,936**). The `embed_tokens` and `lm_head` matrices were rebuilt using a hybrid approach:
    *   For tokens shared between vocabularies, the 72B model's learned vectors were preserved and re-mapped.
    *   For new tokens unique to Qwen3, vectors were initialized from the 32B donor model.
    *   Mappings for all special tokens were explicitly verified.
4.  **Architectural Grafting**: The `q_proj`, `k_proj`, and `v_proj` layers were adapted to the Qwen3 standard (biases removed). The new `q_norm` and `k_norm` RMSNorm layers were grafted from the `Qwen3-32B` donor.
5.  **Validation & Provenance**: After the weight transfer, the script ran an automated smoke test to ensure the model could load and generate coherent text. For full transparency, the following files are included in this repository:
    *   `conversion_metadata.json`: Documents the conversion process and important warnings.
    *   `config_diff.json`: A machine-readable diff of the changes between the source and final model configurations.

> **Reasoning:** This section now "sells" the quality of your work. It mentions the pre-flight checks and the provenance files (`metadata.json`, `config_diff.json`), showing that this wasn't just a casual merge but a careful engineering task.

---

## How to Use

To use this model, you need the `transformers` library. Ensure you have `trust_remote_code=True` set, as the necessary `modeling_qwen3.py` file is included in this repository.

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "CognitiveComputations/Qwen3-72B-Instruct"
dtype = torch.bfloat16

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

# Load the model
# trust_remote_code=True is required to load the custom architecture.
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=dtype,
    device_map="auto",
    trust_remote_code=True,
)

# --- Generation Example ---
# For a deterministic output, use do_sample=False.
# For more creative responses, use do_sample=True with temperature and top_p.
prompt = "Hey, are you conscious? Can you talk to me?"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

# Generate text
outputs = model.generate(
    **inputs, 
    max_new_tokens=50,
    do_sample=False, 
    pad_token_id=tokenizer.eos_token_id
)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(response)
# The model should provide a conversational response, acknowledging its nature as an AI.
# For example: "Hey, are you conscious? Can you talk to me? I am not conscious in the way a human is, but I am able to communicate with you. How can I assist you today?"
```
> **Reasoning:** Changed the generation example to be deterministic (`do_sample=False`) for a more reliable "first run" experience. The expected output is now described more generically, which is safer than promising a specific text string.

### Hardware Requirements

This is a very large model. To run it effectively, you will need:
-   **VRAM**: Approximately 160 GB for full `bfloat16` precision, or ~80 GB for 4-bit quantized inference (e.g., 2x A100 40GB or 1x A100/H100 80GB).
-   **Libraries**: `transformers>=4.40`, `torch`, `accelerate`, `bitsandbytes` (for quantization).

## Model Architecture Details

-   **Model Type**: `qwen3`
-   **Hidden Size**: 8192
-   **Intermediate Size**: 29568
-   **Number of Layers**: 80
-   **Attention Heads**: 64
-   **KV Heads**: 8 (Grouped-Query Attention)
-   **Vocabulary Size**: 151,936
-   **Max Position Embeddings**: 32768
-   **Sliding Window Attention**: Enabled on specified layers

This model features Grouped-Query Attention (GQA) for faster inference and a sliding window attention mechanism to handle long contexts efficiently.

---
*This model was created using a community-developed script. Credit to the original Qwen team at Alibaba Cloud for their foundational models.*
