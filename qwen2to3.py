import torch
import os
import json
from datetime import datetime
from tqdm import tqdm
from transformers import AutoTokenizer, AutoConfig

# Import the custom modeling files provided
from modeling_qwen2 import Qwen2ForCausalLM
from modeling_qwen3 import Qwen3ForCausalLM, Qwen3Config

# --- Helper Functions (with enhancements) ---

def create_vocab_mapping(source_tokenizer, target_tokenizer):
    """Creates a mapping from target vocab IDs to source vocab IDs."""
    source_vocab = source_tokenizer.get_vocab()
    target_vocab = target_tokenizer.get_vocab()
    source_token_to_id = {token: idx for token, idx in source_vocab.items()}
    
    mapping = {}
    for token, target_idx in target_vocab.items():
        mapping[target_idx] = source_token_to_id.get(token, -1)
        
    matches = sum(1 for v in mapping.values() if v != -1)
    print(f"Vocabulary overlap: {matches}/{len(target_vocab)} tokens ({matches/len(target_vocab)*100:.1f}%) will be transferred.")
    return mapping

def verify_special_tokens(source_tokenizer, target_tokenizer, vocab_mapping):
    """Verifies that special tokens are correctly mapped."""
    print("\nVerifying special token mappings...")
    # Use the official special_tokens_map to be robust
    for token_name, token_str in target_tokenizer.special_tokens_map.items():
        if token_str in target_tokenizer.get_vocab():
            target_id = target_tokenizer.convert_tokens_to_ids(token_str)
            source_id = vocab_mapping.get(target_id, -1)
            if source_id != -1:
                print(f"  ✓ {token_name} ('{token_str}'): Mapped correctly (Target ID: {target_id} -> Source ID: {source_id})")
            else:
                print(f"  ⚠️ {token_name} ('{token_str}'): Not found in source vocab. Will be initialized from donor.")
        else:
             print(f"  - {token_name} ('{token_str}'): Not in target vocabulary.")


def create_hybrid_weight_matrix(source_matrix, donor_matrix, vocab_mapping, target_shape):
    """
    Creates a new weight matrix on CPU to conserve VRAM, then moves it to the target device.
    This is a memory-efficient implementation of the hybrid transfer logic.
    """
    print(f"    Creating new matrix with target shape: {target_shape}")
    # Force creation on CPU to avoid spiking VRAM. This is a more efficient approach than chunking.
    hybrid_matrix = torch.zeros(target_shape, dtype=source_matrix.dtype, device='cpu')
    
    transferred_from_source = 0
    used_from_donor = 0
    
    for target_idx, source_idx in vocab_mapping.items():
        if source_idx != -1:
            hybrid_matrix[target_idx] = source_matrix[source_idx].to('cpu')
            transferred_from_source += 1
        else:
            hybrid_matrix[target_idx] = donor_matrix[target_idx].to('cpu')
            used_from_donor += 1
            
    print(f"    Transferred {transferred_from_source} vectors from source model.")
    print(f"    Initialized {used_from_donor} new vectors from donor model.")
    # Move the final, consolidated matrix to the source device in one go.
    return hybrid_matrix.to(source_matrix.device)

def validate_model(model_path):
    """Loads the newly saved model and performs a quick generation test."""
    print("\n[Step 6/6] Validating the final converted model...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = Qwen3ForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True)
        model.eval()

        prompt = "The capital of France is"
        print(f"Validation Prompt: '{prompt}'")
        
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=10, do_sample=False)
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Generated Response: '{response}'")
        
        if len(response) > len(prompt):
             print("  ✓ Validation successful: Model generated output.")
        else:
             print("  ✗ Validation failed: Model did not generate new tokens.")

    except Exception as e:
        print(f"\nAn error occurred during validation: {e}")
        print("  ✗ Validation failed.")


# --- Main Conversion Logic ---

def convert_qwen2_to_qwen3_final():
    source_model_id = "Qwen/Qwen2.5-72B-Instruct"
    donor_model_id = "Qwen/Qwen3-32B"
    target_model_path = "./Qwen3-72B-Instruct"

    print(f"Starting FINAL conversion process...")
    
    # --- 1. Load Models & Tokenizers ---
    print("\n[Step 1/6] Loading source and donor models & tokenizers...")
    dtype = torch.bfloat16
    source_model = Qwen2ForCausalLM.from_pretrained(source_model_id, torch_dtype=dtype, device_map="auto", trust_remote_code=True)
    donor_model = Qwen3ForCausalLM.from_pretrained(donor_model_id, torch_dtype=dtype, device_map="auto", trust_remote_code=True)
    source_tokenizer = AutoTokenizer.from_pretrained(source_model_id, trust_remote_code=True)
    target_tokenizer = AutoTokenizer.from_pretrained(donor_model_id, trust_remote_code=True)
    
    # --- 2. Create Target Config ---
    print("\n[Step 2/6] Creating target Qwen3 72B configuration...")
    source_config = source_model.config
    donor_config = donor_model.config
    target_config = Qwen3Config(
        hidden_size=source_config.hidden_size, intermediate_size=source_config.intermediate_size,
        num_hidden_layers=source_config.num_hidden_layers, num_attention_heads=source_config.num_attention_heads,
        num_key_value_heads=source_config.num_key_value_heads, max_position_embeddings=source_config.max_position_embeddings,
        max_window_layers=source_config.max_window_layers, sliding_window=source_config.sliding_window,
        attention_bias=donor_config.attention_bias, hidden_act=donor_config.hidden_act,
        initializer_range=donor_config.initializer_range, rms_norm_eps=donor_config.rms_norm_eps,
        rope_theta=donor_config.rope_theta, vocab_size=donor_config.vocab_size, 
        tie_word_embeddings=True, model_type="qwen3",
    )

    # --- 3. Initialize Target Model Shell ---
    print("\n[Step 3/6] Initializing empty target Qwen3-72B-Instruct model...")
    with torch.device("meta"):
        target_model = Qwen3ForCausalLM(target_config)

    # --- 4. Convert and Transfer Weights ---
    print("\n[Step 4/6] Converting and transferring weights with alignment...")
    source_state_dict = source_model.state_dict()
    donor_state_dict = donor_model.state_dict()
    target_state_dict = target_model.state_dict()
    new_state_dict = {}

    vocab_mapping = create_vocab_mapping(source_tokenizer, target_tokenizer)
    verify_special_tokens(source_tokenizer, target_tokenizer, vocab_mapping)
    print("") # Newline for cleaner output

    for key in tqdm(target_state_dict.keys(), desc="Transferring weights"):
        param = target_state_dict[key]
        if "self_attn.q_norm.weight" in key or "self_attn.k_norm.weight" in key:
            new_state_dict[key] = donor_state_dict[key].clone()
        elif "model.embed_tokens.weight" in key:
            new_state_dict[key] = create_hybrid_weight_matrix(
                source_matrix=source_state_dict[key], donor_matrix=donor_state_dict["model.embed_tokens.weight"],
                vocab_mapping=vocab_mapping, target_shape=(target_config.vocab_size, target_config.hidden_size)
            )
        elif "lm_head.weight" in key:
            new_state_dict[key] = create_hybrid_weight_matrix(
                source_matrix=source_state_dict[key], donor_matrix=donor_state_dict["lm_head.weight"],
                vocab_mapping=vocab_mapping, target_shape=(target_config.vocab_size, target_config.hidden_size)
            )
        elif key in source_state_dict and source_state_dict[key].shape == param.shape:
            new_state_dict[key] = source_state_dict[key].clone()
        else:
            print(f"!! Unhandled or Mismatched Key: '{key}'")

    target_model.load_state_dict(new_state_dict, strict=True, assign=True)
    target_model = target_model.to(dtype)
    print("Weight conversion complete.")

    # --- 5. Save Final Model & Metadata ---
    print("\n[Step 5/6] Saving the final model, tokenizer, and metadata...")
    if not os.path.exists(target_model_path):
        os.makedirs(target_model_path)
    
    target_model.save_pretrained(target_model_path, safe_serialization=True)
    target_tokenizer.save_pretrained(target_model_path)
    
    # Save metadata for reproducibility
    conversion_metadata = {
        "conversion_date_utc": datetime.utcnow().isoformat(),
        "source_model": source_model_id,
        "donor_model": donor_model_id,
        "target_model_path": target_model_path,
        "vocab_overlap_stats": {
            "total_target_tokens": len(target_tokenizer),
            "transferred_from_source": sum(1 for v in vocab_mapping.values() if v != -1),
            "initialized_from_donor": sum(1 for v in vocab_mapping.values() if v == -1)
        },
        "script_version": "2.0_final"
    }
    with open(os.path.join(target_model_path, "conversion_metadata.json"), "w") as f:
        json.dump(conversion_metadata, f, indent=2)

    print(f"✅ Model saved to: {target_model_path}")
    
    # --- 6. Final Validation ---
    # We release the memory of the large models before loading the new one for validation
    del source_model, donor_model, source_state_dict, donor_state_dict, new_state_dict, target_model
    torch.cuda.empty_cache()
    
    validate_model(target_model_path)


if __name__ == "__main__":
    convert_qwen2_to_qwen3_final()
