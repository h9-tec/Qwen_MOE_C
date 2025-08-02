#!/usr/bin/env python3
"""
Convert Qwen3 MoE PyTorch weights to binary format for C inference
"""

import torch
import struct
import numpy as np
import json
from pathlib import Path
from safetensors.torch import load_file
from huggingface_hub import snapshot_download


def convert_qwen3_to_binary(repo_id, output_path):
    """
    Convert Qwen3 MoE model weights to binary format for C inference
    """
    print(f"Downloading model from {repo_id}...")
    
    # Download model
    local_dir = Path(repo_id).parts[-1]
    repo_dir = snapshot_download(repo_id=repo_id, local_dir=local_dir)
    
    # Load weights
    index_path = Path(repo_dir) / "model.safetensors.index.json"
    with open(index_path, "r") as f:
        index = json.load(f)

    weights_dict = {}
    for filename in set(index["weight_map"].values()):
        shard_path = Path(repo_dir) / filename
        shard = load_file(shard_path)
        weights_dict.update(shard)
    
    # Load config
    config_path = Path(repo_dir) / "config.json"
    with open(config_path, "r") as f:
        config_json = json.load(f)
    
    # Create C-compatible config
    config = {
        'dim': config_json['hidden_size'],                    # 2048
        'hidden_dim': 0,                                      # Not used for MoE layers
        'n_layers': config_json['num_hidden_layers'],         # 48
        'n_heads': config_json['num_attention_heads'],        # 32
        'n_kv_heads': config_json['num_key_value_heads'],     # 4
        'vocab_size': config_json['vocab_size'],              # 151936
        'seq_len': config_json['max_position_embeddings'],    # 262144
        'head_dim': config_json['hidden_size'] // config_json['num_attention_heads'],  # 128
        'qk_norm': 1 if config_json.get('qk_norm', False) else 0,
        'num_experts': config_json['num_experts'],            # 128
        'num_experts_per_tok': config_json['num_experts_per_tok'],  # 8
        'moe_intermediate_size': config_json['moe_intermediate_size'],  # 768
        'rope_theta': float(config_json.get('rope_theta', 10000000.0))
    }
    
    print("Config:")
    for k, v in config.items():
        print(f"  {k}: {v}")
    
    with open(output_path, 'wb') as f:
        # Write config as binary (13 integers + 1 float)
        f.write(struct.pack('i', config['dim']))
        f.write(struct.pack('i', config['hidden_dim']))
        f.write(struct.pack('i', config['n_layers']))
        f.write(struct.pack('i', config['n_heads']))
        f.write(struct.pack('i', config['n_kv_heads']))
        f.write(struct.pack('i', config['vocab_size']))
        f.write(struct.pack('i', config['seq_len']))
        f.write(struct.pack('i', config['head_dim']))
        f.write(struct.pack('i', config['qk_norm']))
        f.write(struct.pack('i', config['num_experts']))
        f.write(struct.pack('i', config['num_experts_per_tok']))
        f.write(struct.pack('i', config['moe_intermediate_size']))
        f.write(struct.pack('f', config['rope_theta']))
        
        # Write weights in the order expected by the C code
        print("Writing weights...")
        
        # 1. Token embeddings
        print("  Token embeddings...")
        write_tensor(f, weights_dict["model.embed_tokens.weight"])
        
        # 2. RMSNorm weights (attention)
        print("  Attention RMSNorm weights...")
        for l in range(config['n_layers']):
            write_tensor(f, weights_dict[f"model.layers.{l}.input_layernorm.weight"])
        
        # 3. RMSNorm weights (FFN)
        print("  FFN RMSNorm weights...")
        for l in range(config['n_layers']):
            write_tensor(f, weights_dict[f"model.layers.{l}.post_attention_layernorm.weight"])
        
        # 4. QK norm weights (if enabled)
        if config['qk_norm']:
            print("  QK norm weights...")
            for l in range(config['n_layers']):
                write_tensor(f, weights_dict[f"model.layers.{l}.self_attn.q_norm.weight"])
            for l in range(config['n_layers']):
                write_tensor(f, weights_dict[f"model.layers.{l}.self_attn.k_norm.weight"])
        
        # 5. Attention weights
        print("  Attention Q weights...")
        for l in range(config['n_layers']):
            write_tensor(f, weights_dict[f"model.layers.{l}.self_attn.q_proj.weight"])
        
        print("  Attention K weights...")
        for l in range(config['n_layers']):
            write_tensor(f, weights_dict[f"model.layers.{l}.self_attn.k_proj.weight"])
        
        print("  Attention V weights...")
        for l in range(config['n_layers']):
            write_tensor(f, weights_dict[f"model.layers.{l}.self_attn.v_proj.weight"])
        
        print("  Attention O weights...")
        for l in range(config['n_layers']):
            write_tensor(f, weights_dict[f"model.layers.{l}.self_attn.o_proj.weight"])
        
        # 6. MoE gating weights
        print("  MoE gating weights...")
        for l in range(config['n_layers']):
            write_tensor(f, weights_dict[f"model.layers.{l}.mlp.gate.weight"])
        
        # 7. Expert weights
        print("  Expert weights...")
        # Write all gate_proj weights
        for l in range(config['n_layers']):
            for e in range(config['num_experts']):
                write_tensor(f, weights_dict[f"model.layers.{l}.mlp.experts.{e}.gate_proj.weight"])
        
        # Write all down_proj weights
        for l in range(config['n_layers']):
            for e in range(config['num_experts']):
                write_tensor(f, weights_dict[f"model.layers.{l}.mlp.experts.{e}.down_proj.weight"])
        
        # Write all up_proj weights
        for l in range(config['n_layers']):
            for e in range(config['num_experts']):
                write_tensor(f, weights_dict[f"model.layers.{l}.mlp.experts.{e}.up_proj.weight"])
        
        # 8. Final norm and output
        print("  Final norm and output...")
        write_tensor(f, weights_dict["model.norm.weight"])
        
        # Check if model uses weight tying
        if "lm_head.weight" in weights_dict:
            write_tensor(f, weights_dict["lm_head.weight"])
        else:
            # Reuse embedding weights
            write_tensor(f, weights_dict["model.embed_tokens.weight"])
    
    print(f"Successfully converted weights to {output_path}")
    

def write_tensor(f, tensor):
    """Write tensor to file in float32 format"""
    if tensor.dtype != torch.float32:
        tensor = tensor.float()
    tensor_np = tensor.cpu().numpy()
    f.write(tensor_np.tobytes())


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 3:
        print("Usage: python convert_qwen3_weights.py <repo_id> <output_path>")
        print("Example: python convert_qwen3_weights.py Qwen/Qwen3-Coder-30B-A3B-Instruct qwen3_moe.bin")
        sys.exit(1)
    
    repo_id = sys.argv[1]
    output_path = sys.argv[2]
    
    convert_qwen3_to_binary(repo_id, output_path)