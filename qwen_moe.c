/* Inference for Qwen3 MoE Transformer model in pure C 
 * Based on Andrej Karpathy's llama2.c approach
 * Supports Qwen3-30B-A3B models with Mixture of Experts
 */

// Compiler optimization hints
#ifdef __GNUC__
    #pragma GCC optimize("O3")
    #pragma GCC target("avx2,fma")
#endif

#include <stdio.h>
 #include <stdlib.h>
 #include <ctype.h>
 #include <time.h>
 #include <math.h>
 #include <string.h>
 #include <fcntl.h>
 #include <stdint.h>
 
 #ifdef _WIN32
     #include <windows.h>
     #include <io.h>
     #define mmap(addr, len, prot, flags, fd, offset) \
         ((char*)MapViewOfFile(CreateFileMapping((HANDLE)_get_osfhandle(fd), NULL, PAGE_READONLY, 0, 0, NULL), \
                               FILE_MAP_READ, 0, 0, (len)))
     #define munmap(addr, len) UnmapViewOfFile(addr)
     #define MAP_FAILED NULL
     #define MAP_PRIVATE 0
     #define PROT_READ 0
 #else
     #include <unistd.h>
     #include <sys/mman.h>
 #endif
 
 // ----------------------------------------------------------------------------
 // Transformer model structures
 
 typedef struct {
     int dim;                // transformer dimension (2048)
     int hidden_dim;         // for standard ffn layers (not used in MoE layers)
     int n_layers;          // number of layers (48)
     int n_heads;           // number of query heads (32)
     int n_kv_heads;        // number of key/value heads (4)
     int vocab_size;        // vocabulary size (151936)
     int seq_len;           // max sequence length (262144)
     int head_dim;          // dimension per head (128)
     int qk_norm;           // whether to use QK normalization (1)
     int num_experts;       // number of experts per layer (128)
     int num_experts_per_tok; // experts activated per token (8)
     int moe_intermediate_size; // expert hidden dimension (768)
     float rope_theta;      // RoPE theta base (10000000.0)
 } Config;
 
 typedef struct {
     // token embedding table
     float* token_embedding_table;    // (vocab_size, dim)
     // weights for rmsnorms
     float* rms_att_weight; // (layer, dim) rmsnorm weights
     float* rms_ffn_weight; // (layer, dim)
     // QK norms (if enabled)
     float* q_norm_weight;  // (layer, head_dim)
     float* k_norm_weight;  // (layer, head_dim)
     // weights for attention
     float* wq; // (layer, dim, dim)
     float* wk; // (layer, dim, n_kv_heads * head_dim)
     float* wv; // (layer, dim, n_kv_heads * head_dim)
     float* wo; // (layer, dim, dim)
     // MoE gating weights
     float* moe_gate; // (layer, dim, num_experts)
     // Expert weights - stored as contiguous arrays
     float* expert_w1; // (layer, num_experts, moe_intermediate_size, dim)
     float* expert_w2; // (layer, num_experts, dim, moe_intermediate_size)  
     float* expert_w3; // (layer, num_experts, moe_intermediate_size, dim)
     // final rmsnorm
     float* rms_final_weight; // (dim,)
     // output projection
     float* wcls; // (dim, vocab_size)
 } TransformerWeights;
 
 typedef struct {
     // current wave of activations
     float *x;      // activation at current time stamp (dim,)
     float *xb;     // same, but inside a residual branch (dim,)
     float *xb2;    // additional buffer (dim,)
     float *q;      // query (dim,)
     float *k;      // key (dim,) 
     float *v;      // value (dim,)
     float *att;    // attention scores (n_heads, seq_len)
     float *logits; // output logits (vocab_size,)
     // kv cache
     float* key_cache;   // (layer, seq_len, n_kv_heads * head_dim)
     float* value_cache; // (layer, seq_len, n_kv_heads * head_dim)
         // MoE specific buffers
    float* gate_scores;     // (num_experts,)
    float* expert_outputs;  // (num_experts_per_tok, dim)
    float* moe_buffer;      // (moe_intermediate_size,)
    float* moe_temp_buffer; // (moe_intermediate_size,) - temporary buffer for SwiGLU
    int* topk_indices;      // (num_experts_per_tok,)
    float* topk_weights;    // (num_experts_per_tok,)
 } RunState;
 
 typedef struct {
     Config config;
     TransformerWeights weights;
     RunState state;
     // memory mapping
     int fd;
     float* data;
     ssize_t file_size;
 } Transformer;
 
 // ----------------------------------------------------------------------------
 // Memory allocation and initialization
 
 void malloc_run_state(RunState* s, Config* p) {
     int kv_dim = (p->n_kv_heads * p->head_dim);
     
     s->x = calloc(p->dim, sizeof(float));
     s->xb = calloc(p->dim, sizeof(float));
     s->xb2 = calloc(p->dim, sizeof(float));
     s->q = calloc(p->dim, sizeof(float));
     s->k = calloc(kv_dim, sizeof(float));
     s->v = calloc(kv_dim, sizeof(float));
     s->att = calloc(p->n_heads * p->seq_len, sizeof(float));
     s->logits = calloc(p->vocab_size, sizeof(float));
     
     s->key_cache = calloc(p->n_layers * p->seq_len * kv_dim, sizeof(float));
     s->value_cache = calloc(p->n_layers * p->seq_len * kv_dim, sizeof(float));
     
         // MoE specific
    s->gate_scores = calloc(p->num_experts, sizeof(float));
    s->expert_outputs = calloc(p->num_experts_per_tok * p->dim, sizeof(float));
    s->moe_buffer = calloc(p->moe_intermediate_size, sizeof(float));
    s->moe_temp_buffer = calloc(p->moe_intermediate_size, sizeof(float));
    s->topk_indices = calloc(p->num_experts_per_tok, sizeof(int));
    s->topk_weights = calloc(p->num_experts_per_tok, sizeof(float));
    
    // Check allocations
    if (!s->x || !s->xb || !s->xb2 || !s->q || !s->k || !s->v || !s->att || 
        !s->logits || !s->key_cache || !s->value_cache || !s->gate_scores ||
        !s->expert_outputs || !s->moe_buffer || !s->moe_temp_buffer || 
        !s->topk_indices || !s->topk_weights) {
        fprintf(stderr, "malloc failed!\n");
        exit(EXIT_FAILURE);
    }
 }
 
 void free_run_state(RunState* s) {
     free(s->x);
     free(s->xb);
     free(s->xb2);
     free(s->q);
     free(s->k);
     free(s->v);
     free(s->att);
     free(s->logits);
     free(s->key_cache);
     free(s->value_cache);
         free(s->gate_scores);
    free(s->expert_outputs);
    free(s->moe_buffer);
    free(s->moe_temp_buffer);
    free(s->topk_indices);
    free(s->topk_weights);
 }
 
 // ----------------------------------------------------------------------------
 // Neural network blocks
 
 void rmsnorm(float* o, float* x, float* weight, int size) {
     // Calculate sum of squares
     float ss = 0.0f;
     for (int j = 0; j < size; j++) {
         ss += x[j] * x[j];
     }
     ss /= size;
     ss += 1e-6f;  // Qwen3 uses 1e-6 epsilon
     ss = 1.0f / sqrtf(ss);
     // Normalize and scale
     for (int j = 0; j < size; j++) {
         o[j] = weight[j] * (ss * x[j]);
     }
 }
 
 void softmax(float* x, int size) {
     // Find max for numerical stability
     float max_val = x[0];
     for (int i = 1; i < size; i++) {
         if (x[i] > max_val) {
             max_val = x[i];
         }
     }
     // Exp and sum
     float sum = 0.0f;
     for (int i = 0; i < size; i++) {
         x[i] = expf(x[i] - max_val);
         sum += x[i];
     }
     // Normalize
     for (int i = 0; i < size; i++) {
         x[i] /= sum;
     }
 }
 
 void matmul(float* xout, float* x, float* w, int n, int d) {
    // W (d,n) @ x (n,) -> xout (d,)
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < d; i++) {
        float val = 0.0f;
        // Add restrict hint and unroll hint for better vectorization
        float* __restrict__ w_row = w + i * n;
        float* __restrict__ x_vec = x;
        
        // Process in chunks of 4 for better SIMD utilization
        int n_vec = n & ~3;  // Round down to multiple of 4
        for (int j = 0; j < n_vec; j += 4) {
            val += w_row[j] * x_vec[j] + 
                   w_row[j+1] * x_vec[j+1] + 
                   w_row[j+2] * x_vec[j+2] + 
                   w_row[j+3] * x_vec[j+3];
        }
        
        // Handle remaining elements
        for (int j = n_vec; j < n; j++) {
            val += w_row[j] * x_vec[j];
        }
        
        xout[i] = val;
    }
}
 
 // SiLU activation function (used in SwiGLU)
 float silu(float x) {
     return x / (1.0f + expf(-x));
 }
 
 // Apply RoPE to q and k vectors per head
void rope(float* q, float* k, int pos, int n_heads, int n_kv_heads, int head_dim, float theta_base) {
    // Apply RoPE to each query head
    for (int h = 0; h < n_heads; h++) {
        float* head_q = q + h * head_dim;
        
        for (int i = 0; i < head_dim; i += 2) {
            float freq = 1.0f / powf(theta_base, (float)i / (float)head_dim);
            float val = pos * freq;
            float fcr = cosf(val);
            float fci = sinf(val);
            
            float v0 = head_q[i];
            float v1 = head_q[i + 1];
            head_q[i] = v0 * fcr - v1 * fci;
            head_q[i + 1] = v0 * fci + v1 * fcr;
        }
    }
    
    // Apply RoPE to each key head
    if (k != NULL) {
        for (int h = 0; h < n_kv_heads; h++) {
            float* head_k = k + h * head_dim;
            
            for (int i = 0; i < head_dim; i += 2) {
                float freq = 1.0f / powf(theta_base, (float)i / (float)head_dim);
                float val = pos * freq;
                float fcr = cosf(val);
                float fci = sinf(val);
                
                float v0 = head_k[i];
                float v1 = head_k[i + 1];
                head_k[i] = v0 * fcr - v1 * fci;
                head_k[i + 1] = v0 * fci + v1 * fcr;
            }
        }
    }
}
 
 // Find top-k values and their indices using heap-based approach
void topk(float* values, int n, int k, int* indices, float* topk_values) {
    // For small k (which is typical for MoE), use simple selection
    // Initialize with first k elements
    for (int i = 0; i < k; i++) {
        indices[i] = i;
        topk_values[i] = values[i];
    }
    
    // Sort initial k elements descending
    for (int i = 0; i < k - 1; i++) {
        for (int j = i + 1; j < k; j++) {
            if (topk_values[j] > topk_values[i]) {
                float temp_val = topk_values[i];
                int temp_idx = indices[i];
                topk_values[i] = topk_values[j];
                indices[i] = indices[j];
                topk_values[j] = temp_val;
                indices[j] = temp_idx;
            }
        }
    }
    
    // Check remaining elements
    for (int i = k; i < n; i++) {
        // If current value is larger than smallest in top-k
        if (values[i] > topk_values[k-1]) {
            // Find insertion position
            int pos = k - 1;
            while (pos > 0 && values[i] > topk_values[pos-1]) {
                pos--;
            }
            
            // Shift and insert
            for (int j = k - 1; j > pos; j--) {
                topk_values[j] = topk_values[j-1];
                indices[j] = indices[j-1];
            }
            topk_values[pos] = values[i];
            indices[pos] = i;
        }
    }
}
 
 // ----------------------------------------------------------------------------
 // Forward pass
 
 float* forward(Transformer* transformer, int token, int pos) {
    Config* p = &transformer->config;
    TransformerWeights* w = &transformer->weights;
    RunState* s = &transformer->state;
    float *x = s->x;
    int dim = p->dim;
    int kv_dim = p->n_kv_heads * p->head_dim;
    int head_size = p->head_dim;
    
    // Bounds checking
    if (token < 0 || token >= p->vocab_size) {
        fprintf(stderr, "Error: Token %d out of bounds [0, %d)\n", token, p->vocab_size);
        exit(EXIT_FAILURE);
    }
    if (pos < 0 || pos >= p->seq_len) {
        fprintf(stderr, "Error: Position %d out of bounds [0, %d)\n", pos, p->seq_len);
        exit(EXIT_FAILURE);
    }
    
    // Copy token embedding into x
    float* content_row = w->token_embedding_table + token * dim;
    memcpy(x, content_row, dim * sizeof(float));
     
     // Forward all layers
     for (int l = 0; l < p->n_layers; l++) {
         // Attention RMSNorm
         rmsnorm(s->xb, x, w->rms_att_weight + l * dim, dim);
         
         // QKV projections
         matmul(s->q, s->xb, w->wq + l * dim * dim, dim, dim);
         matmul(s->k, s->xb, w->wk + l * dim * kv_dim, dim, kv_dim);
         matmul(s->v, s->xb, w->wv + l * dim * kv_dim, dim, kv_dim);
         
         // Apply QK normalization if enabled
         if (p->qk_norm) {
             // Normalize Q
             for (int h = 0; h < p->n_heads; h++) {
                 rmsnorm(s->q + h * head_size, s->q + h * head_size, 
                        w->q_norm_weight + l * head_size, head_size);
             }
             // Normalize K
             for (int h = 0; h < p->n_kv_heads; h++) {
                 rmsnorm(s->k + h * head_size, s->k + h * head_size,
                        w->k_norm_weight + l * head_size, head_size);
             }
         }
         
                 // Apply RoPE
        rope(s->q, s->k, pos, p->n_heads, p->n_kv_heads, head_size, p->rope_theta);
         
         // Cache K and V
         int loff = l * p->seq_len * kv_dim;
         float* key_cache_row = s->key_cache + loff + pos * kv_dim;
         float* value_cache_row = s->value_cache + loff + pos * kv_dim;
         memcpy(key_cache_row, s->k, kv_dim * sizeof(float));
         memcpy(value_cache_row, s->v, kv_dim * sizeof(float));
         
         // Multihead attention
         memset(s->xb, 0, dim * sizeof(float));
         int kv_mul = p->n_heads / p->n_kv_heads;
         
         #pragma omp parallel for
         for (int h = 0; h < p->n_heads; h++) {
             float* q = s->q + h * head_size;
             float* att = s->att + h * p->seq_len;
             
             // Compute attention scores
             for (int t = 0; t <= pos; t++) {
                 float* k = s->key_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
                 float score = 0.0f;
                 for (int i = 0; i < head_size; i++) {
                     score += q[i] * k[i];
                 }
                 score /= sqrtf(head_size);
                 att[t] = score;
             }
             
             // Softmax
             softmax(att, pos + 1);
             
             // Weighted sum of values
             float* xb = s->xb + h * head_size;
             for (int t = 0; t <= pos; t++) {
                 float* v = s->value_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
                 float a = att[t];
                 for (int i = 0; i < head_size; i++) {
                     xb[i] += a * v[i];
                 }
             }
         }
         
         // Output projection
         matmul(s->xb2, s->xb, w->wo + l * dim * dim, dim, dim);
         
         // Residual connection
         for (int i = 0; i < dim; i++) {
             x[i] += s->xb2[i];
         }
         
         // FFN RMSNorm
         rmsnorm(s->xb, x, w->rms_ffn_weight + l * dim, dim);
         
         // MoE layer
         // Compute gating scores
         matmul(s->gate_scores, s->xb, w->moe_gate + l * dim * p->num_experts, 
                dim, p->num_experts);
         
         // Find top-k experts
         topk(s->gate_scores, p->num_experts, p->num_experts_per_tok, 
              s->topk_indices, s->topk_weights);
         
         // Softmax over selected experts
         softmax(s->topk_weights, p->num_experts_per_tok);
         
         // Clear expert outputs
         memset(s->expert_outputs, 0, p->num_experts_per_tok * dim * sizeof(float));
         
                 // Process each selected expert
        for (int i = 0; i < p->num_experts_per_tok; i++) {
            int expert_id = s->topk_indices[i];
            float expert_weight = s->topk_weights[i];
            
            // Get expert weight offsets - weights are stored contiguously for all experts
            size_t expert_idx = (size_t)l * p->num_experts + expert_id;
            size_t w1_size = p->moe_intermediate_size * dim;
            size_t w2_size = dim * p->moe_intermediate_size;
            size_t w3_size = p->moe_intermediate_size * dim;
            
            float* w1 = w->expert_w1 + expert_idx * w1_size;  // gate_proj: dim -> intermediate
            float* w2 = w->expert_w2 + expert_idx * w2_size;  // down_proj: intermediate -> dim
            float* w3 = w->expert_w3 + expert_idx * w3_size;  // up_proj: dim -> intermediate
            
            // Compute SwiGLU: w2(silu(w1(x)) * w3(x))
            float* gate_output = s->moe_buffer;                    // w1(x) output
            float* up_output = s->moe_temp_buffer;                 // w3(x) output
            float* expert_output = s->expert_outputs + i * dim;   // final output
            
            // w1(x) -> gate_output (gate projection)
            matmul(gate_output, s->xb, w1, dim, p->moe_intermediate_size);
            
            // w3(x) -> up_output (up projection)
            matmul(up_output, s->xb, w3, dim, p->moe_intermediate_size);
            
            // SwiGLU: silu(w1(x)) * w3(x)
            for (int j = 0; j < p->moe_intermediate_size; j++) {
                gate_output[j] = silu(gate_output[j]) * up_output[j];
            }
            
            // w2(silu(w1(x)) * w3(x)) -> expert_output (down projection)
            matmul(expert_output, gate_output, w2, p->moe_intermediate_size, dim);
            
            // Weight by gating score
            for (int j = 0; j < dim; j++) {
                expert_output[j] *= expert_weight;
            }
        }
         
         // Sum expert outputs
         memset(s->xb2, 0, dim * sizeof(float));
         for (int i = 0; i < p->num_experts_per_tok; i++) {
             float* expert_out = s->expert_outputs + i * dim;
             for (int j = 0; j < dim; j++) {
                 s->xb2[j] += expert_out[j];
             }
         }
         
         // Residual connection
         for (int i = 0; i < dim; i++) {
             x[i] += s->xb2[i];
         }
     }
     
     // Final RMSNorm
     rmsnorm(x, x, w->rms_final_weight, dim);
     
     // Classifier
     matmul(s->logits, x, w->wcls, dim, p->vocab_size);
     return s->logits;
 }
 
 // ----------------------------------------------------------------------------
 // Model loading
 
 void read_checkpoint(char* checkpoint, Config* config, TransformerWeights* weights,
                    int* fd, float** data, ssize_t* file_size) {
    FILE *file = fopen(checkpoint, "rb");
    if (!file) { 
        fprintf(stderr, "Error: Couldn't open file %s\n", checkpoint); 
        exit(EXIT_FAILURE); 
    }
    
    // Read config fields individually to avoid struct padding issues
    if (fread(&config->dim, sizeof(int), 1, file) != 1) {
        fprintf(stderr, "Error: Failed to read config.dim\n");
        fclose(file);
        exit(EXIT_FAILURE);
    }
    if (fread(&config->hidden_dim, sizeof(int), 1, file) != 1) {
        fprintf(stderr, "Error: Failed to read config.hidden_dim\n");
        fclose(file);
        exit(EXIT_FAILURE);
    }
    if (fread(&config->n_layers, sizeof(int), 1, file) != 1) {
        fprintf(stderr, "Error: Failed to read config.n_layers\n");
        fclose(file);
        exit(EXIT_FAILURE);
    }
    if (fread(&config->n_heads, sizeof(int), 1, file) != 1) {
        fprintf(stderr, "Error: Failed to read config.n_heads\n");
        fclose(file);
        exit(EXIT_FAILURE);
    }
    if (fread(&config->n_kv_heads, sizeof(int), 1, file) != 1) {
        fprintf(stderr, "Error: Failed to read config.n_kv_heads\n");
        fclose(file);
        exit(EXIT_FAILURE);
    }
    if (fread(&config->vocab_size, sizeof(int), 1, file) != 1) {
        fprintf(stderr, "Error: Failed to read config.vocab_size\n");
        fclose(file);
        exit(EXIT_FAILURE);
    }
    if (fread(&config->seq_len, sizeof(int), 1, file) != 1) {
        fprintf(stderr, "Error: Failed to read config.seq_len\n");
        fclose(file);
        exit(EXIT_FAILURE);
    }
    if (fread(&config->head_dim, sizeof(int), 1, file) != 1) {
        fprintf(stderr, "Error: Failed to read config.head_dim\n");
        fclose(file);
        exit(EXIT_FAILURE);
    }
    if (fread(&config->qk_norm, sizeof(int), 1, file) != 1) {
        fprintf(stderr, "Error: Failed to read config.qk_norm\n");
        fclose(file);
        exit(EXIT_FAILURE);
    }
    if (fread(&config->num_experts, sizeof(int), 1, file) != 1) {
        fprintf(stderr, "Error: Failed to read config.num_experts\n");
        fclose(file);
        exit(EXIT_FAILURE);
    }
    if (fread(&config->num_experts_per_tok, sizeof(int), 1, file) != 1) {
        fprintf(stderr, "Error: Failed to read config.num_experts_per_tok\n");
        fclose(file);
        exit(EXIT_FAILURE);
    }
    if (fread(&config->moe_intermediate_size, sizeof(int), 1, file) != 1) {
        fprintf(stderr, "Error: Failed to read config.moe_intermediate_size\n");
        fclose(file);
        exit(EXIT_FAILURE);
    }
    if (fread(&config->rope_theta, sizeof(float), 1, file) != 1) {
        fprintf(stderr, "Error: Failed to read config.rope_theta\n");
        fclose(file);
        exit(EXIT_FAILURE);
    }
    
    // Validate config parameters
    if (config->dim <= 0 || config->dim > 8192) {
        fprintf(stderr, "Error: Invalid dim %d\n", config->dim);
        fclose(file);
        exit(EXIT_FAILURE);
    }
    if (config->n_layers <= 0 || config->n_layers > 200) {
        fprintf(stderr, "Error: Invalid n_layers %d\n", config->n_layers);
        fclose(file);
        exit(EXIT_FAILURE);
    }
    if (config->n_heads <= 0 || config->n_heads > 256) {
        fprintf(stderr, "Error: Invalid n_heads %d\n", config->n_heads);
        fclose(file);
        exit(EXIT_FAILURE);
    }
    if (config->n_kv_heads <= 0 || config->n_kv_heads > config->n_heads) {
        fprintf(stderr, "Error: Invalid n_kv_heads %d\n", config->n_kv_heads);
        fclose(file);
        exit(EXIT_FAILURE);
    }
    if (config->head_dim <= 0 || config->head_dim > 512) {
        fprintf(stderr, "Error: Invalid head_dim %d\n", config->head_dim);
        fclose(file);
        exit(EXIT_FAILURE);
    }
    if (config->num_experts <= 0 || config->num_experts > 1024) {
        fprintf(stderr, "Error: Invalid num_experts %d\n", config->num_experts);
        fclose(file);
        exit(EXIT_FAILURE);
    }
    if (config->num_experts_per_tok <= 0 || config->num_experts_per_tok > config->num_experts) {
        fprintf(stderr, "Error: Invalid num_experts_per_tok %d\n", config->num_experts_per_tok);
        fclose(file);
        exit(EXIT_FAILURE);
    }
    if (config->n_heads % config->n_kv_heads != 0) {
        fprintf(stderr, "Error: n_heads (%d) must be divisible by n_kv_heads (%d)\n", 
                config->n_heads, config->n_kv_heads);
        fclose(file);
        exit(EXIT_FAILURE);
    }
    
    // Get current position (config size)
    long config_size = ftell(file);
    
    // Get file size
    fseek(file, 0, SEEK_END);
    *file_size = ftell(file);
    fclose(file);
    
    // Memory map
    *fd = open(checkpoint, O_RDONLY);
    if (*fd == -1) { 
        fprintf(stderr, "open failed!\n"); 
        exit(EXIT_FAILURE); 
    }
    
    *data = (float*)mmap(NULL, *file_size, PROT_READ, MAP_PRIVATE, *fd, 0);
    if (*data == MAP_FAILED) { 
        fprintf(stderr, "mmap failed!\n"); 
        exit(EXIT_FAILURE); 
    }
    
    // Map weights in the exact order they were written by the Python script
    float* ptr = (float*)((char*)*data + config_size);
    
    // 1. Token embeddings
    weights->token_embedding_table = ptr;
    ptr += config->vocab_size * config->dim;
    
    // 2. RMSNorm weights (attention)
    weights->rms_att_weight = ptr;
    ptr += config->n_layers * config->dim;
    
    // 3. RMSNorm weights (FFN)
    weights->rms_ffn_weight = ptr;
    ptr += config->n_layers * config->dim;
    
    // 4. QK norm weights (if enabled)
    if (config->qk_norm) {
        weights->q_norm_weight = ptr;
        ptr += config->n_layers * config->head_dim;
        weights->k_norm_weight = ptr;
        ptr += config->n_layers * config->head_dim;
    } else {
        weights->q_norm_weight = NULL;
        weights->k_norm_weight = NULL;
    }
    
    // 5. Attention weights
    weights->wq = ptr;
    ptr += config->n_layers * config->dim * config->dim;
    weights->wk = ptr;
    ptr += config->n_layers * config->dim * config->n_kv_heads * config->head_dim;
    weights->wv = ptr;
    ptr += config->n_layers * config->dim * config->n_kv_heads * config->head_dim;
    weights->wo = ptr;
    ptr += config->n_layers * config->dim * config->dim;
    
    // 6. MoE gating weights
    weights->moe_gate = ptr;
    ptr += config->n_layers * config->dim * config->num_experts;
    
    // 7. Expert weights (gate_proj for all experts, then down_proj, then up_proj)
    size_t total_experts = (size_t)config->n_layers * config->num_experts;
    
    weights->expert_w1 = ptr;  // All gate_proj weights
    ptr += total_experts * config->moe_intermediate_size * config->dim;
    
    weights->expert_w2 = ptr;  // All down_proj weights  
    ptr += total_experts * config->dim * config->moe_intermediate_size;
    
    weights->expert_w3 = ptr;  // All up_proj weights
    ptr += total_experts * config->moe_intermediate_size * config->dim;
    
    // 8. Final norm and output
    weights->rms_final_weight = ptr;
    ptr += config->dim;
    weights->wcls = ptr;
}
 
 void build_transformer(Transformer *t, char* checkpoint_path) {
     read_checkpoint(checkpoint_path, &t->config, &t->weights, 
                    &t->fd, &t->data, &t->file_size);
     malloc_run_state(&t->state, &t->config);
 }
 
 void free_transformer(Transformer* t) {
     if (t->data != MAP_FAILED) { 
         munmap(t->data, t->file_size); 
     }
     if (t->fd != -1) { 
         close(t->fd); 
     }
     free_run_state(&t->state);
 }
 
 // ----------------------------------------------------------------------------
 // Sampling
 
 int sample_argmax(float* probabilities, int n) {
    int max_i = 0;
    float max_p = probabilities[0];
    for (int i = 1; i < n; i++) {
        if (probabilities[i] > max_p) {
            max_i = i;
            max_p = probabilities[i];
        }
    }
    return max_i;
}

int sample_temperature(float* logits, int n, float temperature) {
    // Apply temperature
    if (temperature == 0.0f) {
        return sample_argmax(logits, n);
    }
    
    // Find max for numerical stability
    float max_logit = logits[0];
    for (int i = 1; i < n; i++) {
        if (logits[i] > max_logit) {
            max_logit = logits[i];
        }
    }
    
    // Apply temperature and softmax
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        logits[i] = expf((logits[i] - max_logit) / temperature);
        sum += logits[i];
    }
    
    // Normalize to probabilities
    for (int i = 0; i < n; i++) {
        logits[i] /= sum;
    }
    
    // Sample from the distribution
    float r = (float)rand() / RAND_MAX;
    float cumsum = 0.0f;
    for (int i = 0; i < n; i++) {
        cumsum += logits[i];
        if (r < cumsum) {
            return i;
        }
    }
    return n - 1; // fallback
}
 
 // ----------------------------------------------------------------------------
 // Generation
 
 void generate(Transformer *transformer, int* prompt_tokens, int num_prompt_tokens, 
              int steps, float temperature) {
    int next;
    int token = prompt_tokens[0];
    int pos = 0;
    
    printf("\nGenerating with temperature %.2f...\n", temperature);
    
    while (pos < steps) {
        // Forward pass
        float* logits = forward(transformer, token, pos);
        
        // Get next token
        if (pos < num_prompt_tokens - 1) {
            next = prompt_tokens[pos + 1];
        } else {
            // Temperature-based sampling
            if (temperature == 0.0f) {
                next = sample_argmax(logits, transformer->config.vocab_size);
            } else {
                // Create a copy since temperature sampling modifies logits
                float* logits_copy = malloc(transformer->config.vocab_size * sizeof(float));
                if (!logits_copy) {
                    fprintf(stderr, "Error: Failed to allocate memory for logits copy\n");
                    exit(EXIT_FAILURE);
                }
                memcpy(logits_copy, logits, transformer->config.vocab_size * sizeof(float));
                next = sample_temperature(logits_copy, transformer->config.vocab_size, temperature);
                free(logits_copy);
            }
        }
        
        pos++;
        
        // Check for EOS token (you'd need to define this based on tokenizer)
        if (next == 151643) {  // Qwen3 <|im_end|> token
            break;
        }
        
        // Print token (you'd need proper decoding here)
        if (pos >= num_prompt_tokens) {
            printf("Token %d: %d\n", pos, next);
        }
        
        token = next;
    }
    
    printf("\nGenerated %d tokens\n", pos);
}
 
 // ----------------------------------------------------------------------------
 // Main
 
 int main(int argc, char *argv[]) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <checkpoint.bin> [temperature] [max_tokens]\n", argv[0]);
        fprintf(stderr, "  checkpoint.bin: Path to the binary model file\n");
        fprintf(stderr, "  temperature: Sampling temperature (0.0 for greedy, default 0.8)\n");
        fprintf(stderr, "  max_tokens: Maximum tokens to generate (default 100)\n");
        return 1;
    }
    
    char* checkpoint_path = argv[1];
    float temperature = (argc > 2) ? atof(argv[2]) : 0.8f;
    int max_tokens = (argc > 3) ? atoi(argv[3]) : 100;
    
    // Validate parameters
    if (temperature < 0.0f || temperature > 2.0f) {
        fprintf(stderr, "Warning: Temperature %.2f is outside recommended range [0.0, 2.0]\n", temperature);
    }
    if (max_tokens <= 0 || max_tokens > 4096) {
        fprintf(stderr, "Error: max_tokens must be between 1 and 4096\n");
        return 1;
    }
    
    // Seed random number generator
    srand(time(NULL));
    
    // Build transformer
    Transformer transformer;
    build_transformer(&transformer, checkpoint_path);
    
    printf("Model loaded successfully!\n");
    printf("Config: dim=%d, n_layers=%d, n_heads=%d, num_experts=%d\n",
           transformer.config.dim, transformer.config.n_layers,
           transformer.config.n_heads, transformer.config.num_experts);
    printf("Vocab size: %d, Seq len: %d, Head dim: %d\n",
           transformer.config.vocab_size, transformer.config.seq_len, transformer.config.head_dim);
    
    // Example generation with dummy tokens
    // In practice, you'd tokenize the input prompt
    int prompt_tokens[] = {151644, 882, 271};  // Example tokens (<|im_start|>user\n)
    int num_tokens = 3;
    
    generate(&transformer, prompt_tokens, num_tokens, max_tokens, temperature);
    
    // Cleanup
    free_transformer(&transformer);
    
    return 0;
}