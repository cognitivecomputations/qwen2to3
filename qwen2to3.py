# file: convert_qwen2.5_to_qwen3_definitive.py

import torch
import os
import json
from datetime import datetime
from tqdm import tqdm
from transformers import AutoTokenizer, AutoConfig
from collections import Counter

# Import the custom modeling files provided
from modeling_qwen2 import Qwen2ForCausalLM
from modeling_qwen3 import Qwen3ForCausalLM, Qwen3Config

# --- Helper Functions (Definitive Version) ---

def create_vocab_mapping(s_tok, t_tok):
    s_vocab = s_tok.get_vocab()
    t_vocab = t_tok.get_vocab()
    s_tok_to_id = {t: i for t, i in s_vocab.items()}
    mapping = {t_id: s_tok_to_id.get(t, -1) for t, t_id in t_vocab.items()}
    matches = sum(1 for v in mapping.values() if v != -1)
    print(f"Vocabulary overlap: {matches}/{len(t_vocab)} tokens ({matches/len(t_vocab)*100:.1f}%) will be transferred.")
    return mapping

def verify_special_tokens(s_tok, t_tok, mapping):
    print("\nVerifying special token mappings...")
    for name, token in t_tok.special_tokens_map.items():
        if token in t_tok.get_vocab():
            t_id = t_tok.convert_tokens_to_ids(token)
            s_id = mapping.get(t_id, -1)
            status = f"Mapped (T: {t_id} -> S: {s_id})" if s_id != -1 else "NOT FOUND in source (will use donor)"
            print(f"  ✓ {name} ('{token}'): {status}")

def report_missing_high_freq_tokens(s_tokenizer, t_tokenizer, vocab_mapping, top_n=10):
    """Reports if any very common tokens from a sample text were not mapped."""
    sample_text = (
        "The quick brown fox jumps over the lazy dog. Hello, world! "
        "你好世界！1234567890,.:;()[]{}'\"`~-+=_ "
        "This is a test of the emergency broadcast system."
    )
    # Get frequencies from the *target* tokenizer's perspective on a sample
    token_freq = Counter(t_tokenizer.tokenize(sample_text))
    missing_tokens = []
    
    for token, freq in token_freq.most_common():
        if token in t_tokenizer.get_vocab():
            t_id = t_tokenizer.convert_tokens_to_ids(token)
            if vocab_mapping.get(t_id, -1) == -1:
                missing_tokens.append((token, freq))
    
    if missing_tokens:
        print("\n  ⚠️ Found high-frequency sample tokens that were not in the source vocab (initialized from donor):")
        for token, freq in missing_tokens[:top_n]:
            print(f"    - Token: '{token}' (Sample Freq: {freq})")

def create_hybrid_matrix(s_matrix, d_matrix, mapping, shape):
    hybrid = torch.zeros(shape, dtype=s_matrix.dtype, device='cpu')
    for t_id, s_id in mapping.items():
        hybrid[t_id] = s_matrix[s_id].to('cpu') if s_id != -1 else d_matrix[t_id].to('cpu')
    return hybrid.to(s_matrix.device)

def save_config_diff(s_conf, t_conf, path):
    s_dict, t_dict = s_conf.to_dict(), t_conf.to_dict()
    diff = {'changed': {}, 'added': {}, 'removed': {}}
    for k in set(s_dict.keys()) | set(t_dict.keys()):
        if s_dict.get(k) != t_dict.get(k):
            if k in s_dict and k in t_dict:
                diff['changed'][k] = {'from': s_dict[k], 'to': t_dict[k]}
            elif k in t_dict:
                diff['added'][k] = t_dict[k]
            else:
                diff['removed'][k] = s_dict[k]
    with open(os.path.join(path, "config_diff.json"), "w") as f: json.dump(diff, f, indent=2)

def validate_model(path):
    print("\n[Step 6/6] Validating final model (smoke test)...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        model = Qwen3ForCausalLM.from_pretrained(path, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True)
        model.eval()
        prompt = "The theory of relativity states that"
        print(f"\nValidation Prompt: '{prompt}'")
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=25, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Generated Response: '{response}'")
        assert len(response) > len(prompt), "Model did not generate new tokens."
        print("\n  ✓ Validation successful: Model loads and generates coherent text.")
    except Exception as e:
        print(f"\n  ✗ Validation FAILED: {e}")

# --- Main Conversion Logic ---
def convert_qwen2_to_qwen3_definitive():
    source_model_id, donor_model_id = "Qwen/Qwen2.5-72B-Instruct", "Qwen/Qwen3-32B"
    target_model_path = "./Qwen3-72B"
    print("Starting DEFINITIVE conversion process (v4.0)...")

    # --- 1. Pre-flight Checks ---
    print("\n[Step 1/6] Running pre-flight architectural checks...")
    s_config = AutoConfig.from_pretrained(source_model_id, trust_remote_code=True)
    d_config = AutoConfig.from_pretrained(donor_model_id, trust_remote_code=True)
    assert s_config.hidden_act == d_config.hidden_act, f"FATAL: Hidden activation mismatch! Source: {s_config.hidden_act}, Donor: {d_config.hidden_act}."
    print("  ✓ Hidden activation functions match.")
    if s_config.rope_theta != d_config.rope_theta:
        print(f"  ✓ RoPE Theta: Using donor value {d_config.rope_theta} (source was {s_config.rope_theta})")
    
    # --- 2. Load Models & Tokenizers ---
    print("\n[Step 2/6] Loading source and donor models & tokenizers...")
    dtype = torch.bfloat16
    s_model = Qwen2ForCausalLM.from_pretrained(source_model_id, torch_dtype=dtype, device_map="auto", trust_remote_code=True)
    d_model = Qwen3ForCausalLM.from_pretrained(donor_model_id, torch_dtype=dtype, device_map="auto", trust_remote_code=True)
    s_tokenizer = AutoTokenizer.from_pretrained(source_model_id, trust_remote_code=True)
    t_tokenizer = AutoTokenizer.from_pretrained(donor_model_id, trust_remote_code=True)

    # --- 3. Create Target Config & Initialize ---
    print("\n[Step 3/6] Creating target Qwen3 72B config & initializing model...")
    t_config = Qwen3Config(hidden_size=s_config.hidden_size, intermediate_size=s_config.intermediate_size, num_hidden_layers=s_config.num_hidden_layers, num_attention_heads=s_config.num_attention_heads, num_key_value_heads=s_config.num_key_value_heads, max_position_embeddings=s_config.max_position_embeddings, max_window_layers=s_config.max_window_layers, sliding_window=s_config.sliding_window, attention_bias=d_config.attention_bias, hidden_act=d_config.hidden_act, initializer_range=d_config.initializer_range, rms_norm_eps=d_config.rms_norm_eps, rope_theta=d_config.rope_theta, vocab_size=d_config.vocab_size, tie_word_embeddings=True, model_type="qwen3")
    with torch.device("meta"): t_model = Qwen3ForCausalLM(t_config)

    # --- 4. Convert and Transfer Weights ---
    print("\n[Step 4/6] Converting weights (memory-safe)...")
    # CRITICAL FIX: Gather state dicts on CPU to prevent OOM errors.
    s_state_dict = {k: v.to('cpu', dtype=dtype) for k, v in tqdm(s_model.state_dict().items(), desc="Source state dict to CPU")}
    d_state_dict = {k: v.to('cpu', dtype=dtype) for k, v in tqdm(d_model.state_dict().items(), desc="Donor state dict to CPU")}
    
    vocab_mapping = create_vocab_mapping(s_tokenizer, t_tokenizer)
    verify_special_tokens(s_tokenizer, t_tokenizer, vocab_mapping)
    report_missing_high_freq_tokens(s_tokenizer, t_tokenizer, vocab_mapping)
    
    new_state_dict = {}
    for key in tqdm(t_model.state_dict().keys(), desc="Transferring weights"):
        if "q_norm" in key or "k_norm" in key: new_state_dict[key] = d_state_dict[key].clone()
        elif "model.embed_tokens.weight" in key: new_state_dict[key] = create_hybrid_matrix(s_state_dict[key], d_state_dict[key], vocab_mapping, (t_config.vocab_size, t_config.hidden_size))
        elif "lm_head.weight" in key: new_state_dict[key] = create_hybrid_matrix(s_state_dict[key], d_state_dict[key], vocab_mapping, (t_config.vocab_size, t_config.hidden_size))
        elif key in s_state_dict: new_state_dict[key] = s_state_dict[key].clone()
        else: print(f"  ⚠️ Unhandled key: {key} (not in source, skipping)") # IMPORTANT: Warn about unhandled keys

    t_model.load_state_dict(new_state_dict, strict=True, assign=True)
    t_model = t_model.to(dtype)

    # --- 5. Save Final Model & Metadata ---
    print("\n[Step 5/6] Saving final model and supporting files...")
    if not os.path.exists(target_model_path): os.makedirs(target_model_path)
    t_model.save_pretrained(target_model_path, safe_serialization=True)
    t_tokenizer.save_pretrained(target_model_path)
    save_config_diff(s_config, t_config, target_model_path)
    metadata = {"conversion_date_utc": datetime.utcnow().isoformat(), "source_model": source_model_id, "donor_model": donor_model_id,
                "warnings": ["This is a community-created model merge. Its behavior may be unpredictable.", "Sliding window config inherited from Qwen2.5 with Qwen3 RoPE theta - long context behavior MUST be validated.", "Post-conversion evaluation is highly recommended for numerical stability, quantization, and safety alignment."]}
    with open(os.path.join(target_model_path, "conversion_metadata.json"), "w") as f: json.dump(metadata, f, indent=2)
    print(f"✅ Model saved to: {target_model_path}")
    
    # --- 6. Final Validation ---
    del s_model, d_model, s_state_dict, d_state_dict, new_state_dict, t_model
    torch.cuda.empty_cache()
    validate_model(target_model_path)

if __name__ == "__main__":
    convert_qwen2_to_qwen3_definitive()
