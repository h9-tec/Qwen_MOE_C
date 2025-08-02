# Qwen_MOE_C

A pure C implementation for inference of Qwen3 Mixture-of-Experts models, inspired by Andrej Karpathy's llama2.c approach. This implementation supports the full Qwen3-30B-A3B model architecture with 128 experts and sparse activation.

## Features

- **Pure C Implementation**: Single file, no external dependencies except libc and libm
- **Memory Efficient**: Uses memory mapping to avoid loading entire model into RAM
- **Optimized Performance**: 
  - OpenMP parallelization for attention heads
  - SIMD-friendly matrix multiplication
  - Efficient sparse MoE computation (only activates 8 out of 128 experts)
- **Complete Architecture Support**:
  - Grouped Query Attention (GQA) with 32 query heads, 4 key/value heads
  - RMSNorm with QK normalization
  - RoPE (Rotary Position Embedding)
  - SwiGLU activation in experts
- **Robust Error Handling**: Comprehensive bounds checking and validation

## Architecture Details

The implementation supports Qwen3-30B-A3B models with the following specifications:

- **Model Size**: 30B parameters (with 3B active per token)
- **Layers**: 48 transformer blocks
- **Dimensions**: 2048 hidden size, 128 head dimension
- **Attention**: 32 query heads, 4 key/value heads (8:1 ratio)
- **MoE**: 128 experts per layer, 8 experts activated per token
- **Expert Size**: 768 intermediate dimension per expert
- **Context**: Up to 262,144 tokens
- **Vocabulary**: 151,936 tokens

## Building

### Requirements

- GCC or Clang compiler
- OpenMP (optional, for parallel processing)
- libm (math library)

### Quick Build

```bash
make
```

### Build Options

```bash
make                    # Optimized build with OpenMP (recommended)
make debug             # Debug build with symbols
make no-openmp         # Build without OpenMP (single-threaded)
make portable          # Portable build for older systems
make static            # Static build (all libraries linked)
```

### Manual Compilation

```bash
# With OpenMP for multi-threading
gcc -O3 -fopenmp -march=native qwen_moe.c -lm -o qwen3_moe

# Without OpenMP
gcc -O3 qwen_moe.c -lm -o qwen3_moe
```

## Converting Model Weights

First, convert the PyTorch weights to binary format:

### Install Python Dependencies

```bash
make install-deps
# or manually:
pip install torch safetensors huggingface_hub
```

### Convert Weights

```bash
python convert_qwen3_weights.py Qwen/Qwen3-Coder-30B-A3B-Instruct qwen3_moe.bin
```

This will download the model from Hugging Face and convert it to the binary format expected by the C code.

### Supported Models

- `Qwen/Qwen3-Coder-30B-A3B-Instruct` (Qwen3 Coder Flash)
- `Qwen/Qwen3-30B-A3B-Thinking-2507` (Thinking model)
- `Qwen/Qwen3-235B-A22B-Instruct-2507` (Large instruct model)
- `Qwen/Qwen3-30B-A3B` (Original instruct/thinking hybrid)

## Usage

```bash
./qwen3_moe <model.bin> [temperature] [max_tokens]
```

### Parameters

- `model.bin`: Path to the converted binary model file
- `temperature`: Sampling temperature (0.0 for greedy, 0.8 recommended, default: 0.8)
- `max_tokens`: Maximum tokens to generate (default: 100)

### Examples

```bash
# Greedy decoding (deterministic)
./qwen3_moe qwen3_moe.bin 0.0 50

# Creative sampling
./qwen3_moe qwen3_moe.bin 1.0 200

# Balanced sampling (recommended)
./qwen3_moe qwen3_moe.bin 0.8 100
```

## Performance

### Memory Requirements

- **Model weights**: ~30 GB (float32)
- **Runtime memory**: ~2-4 GB (activation buffers, KV cache)
- **Total**: ~32-34 GB RAM minimum

### Speed Optimizations

The implementation includes several performance optimizations:

1. **Memory Mapping**: Weights are memory-mapped, not loaded into RAM
2. **SIMD Optimization**: Matrix multiplication uses 4-way unrolling for better vectorization
3. **Sparse MoE**: Only computes active experts (8 out of 128)
4. **OpenMP Parallelization**: Attention heads processed in parallel
5. **Cache-Friendly Access**: Sequential memory access patterns where possible

### Expected Performance

On a modern CPU (e.g., Intel i9 or AMD Ryzen 9):
- **Single-threaded**: ~1-2 tokens/second
- **Multi-threaded**: ~3-8 tokens/second (depending on core count)

GPU acceleration is not currently implemented but would provide significant speedup.

## Architecture Implementation Details

### RoPE (Rotary Position Embedding)

```c
// Applied per-head with proper frequency calculation
void rope(float* q, float* k, int pos, int n_heads, int n_kv_heads, int head_dim, float theta_base)
```

### Mixture of Experts (MoE)

```c
// Sparse computation: only processes top-8 experts
1. Compute gating scores for all 128 experts
2. Select top-8 using efficient selection algorithm
3. Apply softmax to selected experts
4. Compute SwiGLU: gate_proj → silu → × up_proj → down_proj
5. Weighted combination based on gating scores
```

### Grouped Query Attention

```c
// 32 query heads, 4 key/value heads (8:1 ratio)
// Keys and values are repeated across query head groups
```

## Limitations and TODOs

### Current Limitations

1. **No Tokenizer**: Currently uses dummy tokens, needs proper Qwen3 tokenizer integration
2. **No Chat Templates**: Raw token generation only
3. **CPU Only**: No GPU acceleration
4. **Basic Sampling**: Only supports temperature sampling, no top-p/top-k

### Future Improvements

1. **Tokenizer Integration**: Add Qwen3 tokenizer for text input/output
2. **Quantization**: Add int8/int4 quantization for smaller memory footprint
3. **CUDA Support**: GPU acceleration for faster inference
4. **Advanced Sampling**: Top-p, top-k, and other sampling strategies
5. **Beam Search**: Multi-sequence generation
6. **KV Cache Optimization**: Better memory management for long sequences

## Implementation Notes

### Binary Format

The binary format stores weights in the following order:

1. Config (13 integers + 1 float)
2. Token embeddings
3. RMSNorm weights (attention + FFN)
4. QK norm weights (if enabled)
5. Attention weights (Q, K, V, O)
6. MoE gating weights
7. Expert weights (gate_proj, down_proj, up_proj for all experts)
8. Final norm and output weights

### Error Handling

The implementation includes comprehensive error checking:

- Configuration validation
- Memory allocation failures
- File operation errors
- Bounds checking for array accesses
- Numerical stability checks

## Contributing

This implementation follows the simplicity principle of llama2.c. Contributions should:

1. Maintain single-file architecture
2. Avoid external dependencies
3. Include comprehensive error handling
4. Add performance optimizations where possible
5. Follow existing code style

## License

This code is released under the MIT License, following the same permissive approach as llama2.c.

## Acknowledgments

- **Sebastian Raschka** for the comprehensive Qwen3 MoE implementation and educational materials in his [LLMs-from-scratch repository](https://github.com/rasbt/LLMs-from-scratch/blob/main/ch05/11_qwen3/standalone-qwen3-moe.ipynb), which served as the reference for understanding the Qwen3 architecture details
- **Andrej Karpathy** for the llama2.c approach and inspiration for clean, educational C implementations
- **Qwen team** for the model architecture and pre-trained weights
- The broader LLM open-source community for advancing accessible AI implementations

## References

- [Qwen3 Technical Report](https://arxiv.org/abs/2505.09388)
- [Sebastian Raschka's LLMs-from-scratch Repository](https://github.com/rasbt/LLMs-from-scratch) - Educational LLM implementations
- [Qwen3 MoE Standalone Notebook](https://github.com/rasbt/LLMs-from-scratch/blob/main/ch05/11_qwen3/standalone-qwen3-moe.ipynb) by Sebastian Raschka
- [llama2.c](https://github.com/karpathy/llama2.c) by Andrej Karpathy
- [Build a Large Language Model (From Scratch)](http://mng.bz/orYv) by Sebastian Raschka