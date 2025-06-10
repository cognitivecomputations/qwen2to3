# Qwen3-72B-Instruct

Created by Eric Hartford

**Model ID:** `CognitiveComputations/Qwen3-72B-Instruct`  
**Model Type:** Causal Language Model  
**Architecture:** Qwen3

---

## 🚨 Disclaimer 🚨

This model, `Qwen3-72B-Instruct`, is an **experimental, community-created model**. It was not trained from scratch by the original Qwen team. It has been constructed by merging weights from `Qwen/Qwen2.5-72B-Instruct` into the `Qwen3` architecture, using `Qwen/Qwen3-32B` as a "donor" for architectural components not present in Qwen2.5.

While it has passed basic validation checks, its performance, stability, and safety have not been rigorously evaluated. **Use this model with caution and for research purposes.**

---

## Model Description

`Qwen3-72B-Instruct` is a 72-billion parameter, instruction-tuned, decoder-only language model designed to be architecturally compatible with the Qwen3 family of models. It aims to combine the extensive knowledge of the powerful `Qwen/Qwen2.5-72B-Instruct` model with the architectural improvements of the Qwen3 series.

This model is intended for users who wish to explore a 72B-scale model within the Qwen3 framework. It should be capable of strong performance in a variety of natural language tasks, including chat, instruction following, and content generation.

### Future Work: Knowledge Distillation

A primary motivation for creating this model is to serve as a high-quality "student" model for the next phase of our research. The plan is to perform **knowledge and logit distillation** from the much larger `Qwen/Qwen3-235B-A22B` Mixture-of-Experts (MoE) model.

By using this 72B model as a base, we aim to distill the specialized knowledge and reasoning capabilities of the 235B MoE model into a more efficient, dense architecture.

---

## How This Model Was Created

This model is a "hybrid conversion" created using a meticulous weight-merging process. The key steps were:

1.  **Base Model**: The primary weights were taken from `Qwen/Qwen2.5-72B-Instruct`. This includes the transformer blocks (attention and MLP layers).

2.  **Donor Model**: The architectural template and new components were taken from `Qwen/Qwen3-32B`.

3.  **Architectural Conversion**:
    *   **Attention Layers**: The `q_proj`, `k_proj`, and `v_proj` layers were adapted to the Qwen3 standard. Biases were removed, and the new `q_norm` and `k_norm` RMSNorm layers were added, with their weights initialized directly from the `Qwen3-32B` donor model.
    *   **Vocabulary & Embeddings**: The model uses the official `Qwen3` tokenizer and its vocabulary size of **151,936**. The `embed_tokens` and `lm_head` weight matrices were reconstructed using a hybrid approach:
        *   For tokens present in both the Qwen2.5 and Qwen3 vocabularies, the corresponding learned embedding vectors from the 72B source model were preserved and mapped to their new positions.
        *   For tokens unique to the Qwen3 vocabulary, the embedding vectors were initialized from the 32B donor model.
    *   **Metadata**: The conversion process, including vocabulary overlap statistics, has been documented in the `conversion_metadata.json` file in this repository.

This process ensures full compatibility with the `Qwen3` tokenizer and `transformers` ecosystem while maximizing the knowledge transfer from the original 72B model.

---

## How to Use

To use this model, you need the `transformers` library and custom modeling code. Ensure you have `trust_remote_code=True` set when loading the model. The necessary `modeling_qwen3.py` file is included in this repository.

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
prompt = "Hey, are you conscious? Can you talk to me?"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

# Generate text
outputs = model.generate(**inputs, max_new_tokens=50, do_sample=True, temperature=0.7, top_p=0.9)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(response)
# Expected output similar to:
# Hey, are you conscious? Can you talk to me?
# I am not conscious in the way a human is, but I can certainly talk to you. I am a large language model, created by Cognitive Computations based on Qwen. How can I help you today?
```

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
*This model was created using community-developed scripts. Credit to the original Qwen team at Alibaba Cloud for their foundational models.*
