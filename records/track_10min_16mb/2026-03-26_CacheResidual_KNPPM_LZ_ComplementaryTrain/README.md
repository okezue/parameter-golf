# TurboSerif Golf + Cache-Residual Compressor

## Core Thesis

**Use a geometry-aware, scale-folded codec only where dot products matter most, then spend the saved bytes where BPB improves most.** Stop treating the transformer as the whole compressor. The transformer becomes the residual specialist for tokens the online memory system cannot cheaply predict.

## Architecture

Built on the PR #549 stack (LeakyReLU(0.5)^2 + Legal TTT + Parallel Muon + GPTQ-lite int6 + EMA + XSA4 + Partial RoPE + SmearGate + BigramHash + Value Embedding).

---

## Part I: TurboSerif Codec (Artifact Compression)

### Geometry-Aware Q/K Weight Compression

For deep-layer Q/K projections, standard GPTQ-lite int6 minimizes **weight reconstruction MSE**. But Q/K only participates through `Q @ K^T` — so the right target is **score distortion**, not weight MSE.

TurboSerif represents each Q/K weight group as:

```
w_{r,g} ≈ 2^(E_r - Δ_{r,g}) * c_{z_{r,g}}
```

where:
- `c_z` is a vector from a **shared spherical codebook** (K=256, learned via spherical k-means)
- `E_r` is a **row base exponent** (pow2, from serif-217)
- `Δ_{r,g}` is a **per-group exponent delta** (4-bit)
- `z_{r,g}` is the **codebook index** (8-bit)

### Byte Savings

For a 512x512 Q/K matrix with G=32:

| Codec | Bytes | Compression |
|-------|-------|-------------|
| fp32 | 1,048,576 | 1x |
| int6 (GPTQ-lite) | ~198,000 | 5.3x |
| **TurboSerif** | **~21,000 + codebook** | **~42x** |

Applied to 8 Q/K matrices (4 deep layers): **~1.4 MB saved** vs int6, reallocatable to capacity elsewhere.

### Asymmetric Application

- **TurboSerif**: Q/K projections in deepest 4 layers only
- **GPTQ-lite int6**: V, output projection, MLP weights
- **FP32 passthrough**: control tensors
- **FP16 passthrough**: small tensors

This asymmetry matches the geometry: inner-product preservation matters for Q/K, not for V/O/MLP.

### Serif-217 Primitives Used

From `serif-217/src/common/`:
- **Power-of-two exponents** (`pack_int4_pow2.py`): row base exponent + shift deltas
- **Fold-aligned normalization** (`fold_aligned`): right-shift INT4 values by exponent differences
- **Late scale application** concept: don't fully reconstruct before the consuming op

---

## Part II: Bytewise BPB Allocator

Don't ask "is TurboSerif better than int6 everywhere?" Ask:

```
Which codec should each tensor use to maximize BPB improvement per byte spent?
```

The allocator evaluates candidate codecs per tensor under the 16MB budget:
- fp16, int8, int6 (GPTQ-lite), TurboSerif
- Estimates utility = ΔBpb / Δbytes
- Greedy knapsack allocation

---

## Part III: Cache-Residual Compressor (Eval-time)

### 6. Deployment-Matched Complementary Training

```
L_t = -log((1 - α_t) * p_neural(y_t) + α_t * p_cache(y_t))
```

GPU-resident bigram surrogate cache. α_t = confidence-gated mixing. Trains the transformer as a residual compressor.

### 7. Online Modified Kneser-Ney N-gram Cache

Online approximate KN with hashed count tables, KN smoothing with D1/D2/D3+ discounts.

### 8. LZ77 Suffix-Match Expert

Rolling-hash longest-match over scored tokens. Captures repeated HTML, URLs, boilerplate.

### 9. AdaHedge Expert Mixing

Three experts (neural, KN/PPM, LZ) mixed with online regret minimization.

### 10. LoRA-Only Frozen-Base TTT

The TurboSerif-compressed base stays frozen. Tiny BF16 LoRA adapters (rank=8) attached to last 4 blocks handle TTT adaptation:
- No gradient through compressed base
- Adapters correct quantization-induced local mismatch
- ~8K trainable params vs ~25M frozen — fast, stable TTT

---

## Key Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| TS_ENABLED | 1 | Enable TurboSerif Q/K codec |
| TS_CODEBOOK_K | 256 | Codebook size for directional codes |
| TS_GROUP_SIZE | 32 | Weight group size for encoding |
| TS_DEEP_LAYERS | 4 | Number of deep layers to apply TurboSerif |
| LORA_TTT | 1 | LoRA-only TTT (frozen compressed base) |
| LORA_RANK | 8 | LoRA adapter rank |
| CACHE_EVAL | 1 | Enable cache-augmented evaluation |
| COMP_TRAIN | 1 | Enable complementary training |
| COMP_ALPHA_BASE | 0.3 | Surrogate cache mixing weight |
| TTT_ENABLED | 1 | Enable test-time training |

## Run Command

```bash
TS_ENABLED=1 TTT_ENABLED=1 LORA_TTT=1 \
BIGRAM_VOCAB_SIZE=3072 CACHE_EVAL=1 COMP_TRAIN=1 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Ablation Order

1. Strong accepted baseline (PR #549 stack)
2. Replace deep Q/K int6 with serif-style pow2 grouped int6
3. Add fold-aligned row exponents
4. Replace grouped values with shared directional codebook + exponent folding → **TurboSerif**
5. Add bytewise BPB allocator
6. Add complementary training
7. Add KN/PPM + LZ + AdaHedge cache eval
8. Add LoRA-only frozen-base TTT

## Expected Impact

| Method | BPB Impact | Bytes Impact |
|--------|-----------|--------------|
| TurboSerif Q/K codec | -0.001 to -0.003 | -1.4 MB artifact |
| Byte savings → more capacity | -0.002 to -0.005 | neutral |
| Complementary training | -0.002 to -0.005 | none |
| KN/PPM + LZ cache | -0.003 to -0.008 | none (eval-time) |
| LoRA TTT | -0.001 to -0.002 | none (eval-time) |
| **Combined** | **-0.009 to -0.023** | **-1.4 MB** |
