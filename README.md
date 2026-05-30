# DeepSeek V4 Flash — Dual DGX Spark Benchmark

**TL;DR:** DeepSeek V4 Flash Q4 running distributed across two DGX Sparks via [ds4](https://github.com/antirez/ds4) (antirez's inference engine). This answers the question: *"Is buying a second DGX Spark worth it for DeepSeek V4 Flash?"*

## Quick Results

| Context | Prefill (t/s) | Generation (t/s) |
|--------:|:-------------:|:----------------:|
| 2,048   | 121.93        | 11.36            |
| 10,240  | 164.55        | 11.21            |
| 18,432  | 161.69        | 11.09            |
| 26,624  | 160.19        | 10.88            |
| 34,816  | 157.76        | 10.45            |
| 43,008  | 154.96        | 10.34            |
| 51,200  | 152.86        | 10.18            |
| 59,392  | 151.19        | 10.04            |
| 65,536  | 134.97        |  9.91            |

> The prefill numbers above are per-chunk incremental rates from `ds4-bench`. The full pipelined prefill at 65K context achieved **216.82 t/s** end-to-end thanks to pipeline parallelism across both Sparks.

**Key takeaway:** Prefill scales well with pipeline parallelism (122–216 t/s depending on context length and pipeline depth). Generation degrades gracefully from ~11.4 t/s at 2K context to ~9.9 t/s at 65K — each token must traverse both machines sequentially, so the network hop plus growing KV cache is the cost. Still very usable for interactive chat.

### Comparison Context

From the ds4 README (single DGX Spark, Q2 quant):
- Prefill: **343 t/s** (single machine, smaller model)
- Generation: **13.75 t/s**

Our dual-Spark Q4 results:
- Prefill: **122–217 t/s** (pipeline parallelism helps significantly — 217 t/s end-to-end at 65K)
- Generation: **9.9–11.4 t/s** (28–19% slower than single-Spark Q2 due to network round-trip + KV cache growth)

From antirez's two-MacBook M5 Max benchmarks (Thunderbolt 5, Q4):
- Prefill: **582–674 t/s** (Thunderbolt 5 is much faster than 10GbE)
- Generation: **24.67 t/s** (0.45ms ping helps)

## Hardware

| Component | Spark 4 (Coordinator) | Spark 6 (Worker) |
|-----------|----------------------|-----------------|
| Hardware | NVIDIA DGX Spark (GB10) | NVIDIA DGX Spark (GB10) |
| GPU | NVIDIA GB10 (sm_121) | NVIDIA GB10 (sm_121) |
| Memory | 128 GB unified | 128 GB unified |
| Network | QSFP 10GbE direct-connect | QSFP 10GbE direct-connect |
| IP (QSFP) | 10.0.0.4 | 10.0.0.6 |
| IP (WiFi) | 192.168.1.75 | 192.168.1.85 |
| Ping (QSFP) | 1.4ms round-trip | — |

### Network

The two Sparks are connected via a **QSFP direct cable** providing a 10 Gigabit Ethernet link. The `enP2p1s0f1np1` interfaces are configured with static IPs on the `10.0.0.0/24` subnet.

Note: ds4 uses **TCP** for distributed inference (not RDMA/RoCE). antirez explicitly states tensor parallelism is not viable at inter-machine communication speeds — his approach is pipeline parallelism (layer splitting). The QSFP 10GbE link provides low latency (~1.4ms) for the activation transfers.

## Model

- **DeepSeek V4 Flash** — 284B total parameters, 13B active (MoE architecture)
- **Quantization:** Q4_K experts, F16 head/compressor/indexer, Q8 attention/shared/output (asymmetric — only routed MoE experts are quantized)
- **GGUF file:** `DeepSeek-V4-Flash-Q4KExperts-F16HC-F16Compressor-F16Indexer-Q8Attn-Q8Shared-Q8Out-chat-v2-imatrix.gguf`
- **Size:** 154 GB (too large for a single 128 GB Spark — dual-Spark is required)
- **Source:** [antirez/deepseek-v4-gguf](https://huggingface.co/antirez/deepseek-v4-gguf) on HuggingFace
- **Context window:** Up to 1M tokens (model capability); tested up to 65K here

## Software

- **Engine:** [ds4 (DwarfStar)](https://github.com/antirez/ds4) by antirez (Salvatore Sanfilippo)
- **Build:** `make cuda-spark` (optimised for DGX Spark / GB10, includes `DS4_CUDA_SPARK_HBM_CACHE=1`)
- **Commit:** `ba00a8a` (2026-05-30)

## Distributed Configuration

ds4 splits transformer layers across machines. Each machine loads only its assigned layer slice from the same GGUF file.

**Layer split (43 layers total):**
- **Coordinator (Spark 4):** Layers 0–20 (21 layers + output head) → 75.64 GiB
- **Worker (Spark 6):** Layers 21–output (22 layers) → 78.21 GiB

**Launch commands:**

```bash
# Worker (start first — it waits for the coordinator)
./ds4 -m ds4flash.gguf \
  --role worker \
  --layers 21:output \
  --coordinator 10.0.0.4 1234 \
  -c 65665

# Coordinator
./ds4 -m ds4flash.gguf \
  --role coordinator \
  --layers 0:20 \
  --listen 10.0.0.4 1234 \
  -c 65665
```

**Important notes:**
- The GGUF must exist on both machines (each loads only its slice)
- Worker context (`-c`) must be ≥ coordinator's allocated context
- Worker loads in ~9 seconds, coordinator in ~21–37 seconds (coordinator also loads the output head)
- The `--listen` address must be reachable by the worker (use the QSFP IPs for speed)

## How to Reproduce

### 1. Build ds4 on both Sparks

```bash
git clone https://github.com/antirez/ds4
cd ds4
make cuda-spark
```

### 2. Download the model

```bash
# On each Spark (or copy via QSFP after downloading once):
cd ds4
./download_model.sh q4-imatrix
# Creates: gguf/DeepSeek-V4-Flash-Q4KExperts-*.gguf (~154 GB)
```

### 3. Copy model to second Spark (if not downloaded separately)

```bash
# From Spark 6, copy via QSFP (fast):
scp castlespark4@10.0.0.4:/home/castlespark4/ds4/gguf/DeepSeek-V4-Flash-Q4K*.gguf \
    /home/castlespark6/ds4/gguf/
```

### 4. Set up QSFP networking

Both Sparks need static IPs on the same subnet on their QSFP interfaces:

```bash
# Spark 4:
sudo ip addr add 10.0.0.4/24 dev enP2p1s0f1np1
sudo ip link set enP2p1s0f1np1 up

# Spark 6:
sudo ip addr add 10.0.0.6/24 dev enP2p1s0f1np1
sudo ip link set enP2p1s0f1np1 up
```

Verify: `ping -c 3 10.0.0.4` from Spark 6 should show ~1.4ms.

### 5. Run benchmark

```bash
# Worker (Spark 6):
./ds4 -m ds4flash.gguf --role worker --layers 21:output \
  --coordinator 10.0.0.4 1234 -c 65665

# Coordinator/Benchmark (Spark 4):
./ds4-bench -m ds4flash.gguf --role coordinator --layers 0:20 \
  --listen 10.0.0.4 1234 --ctx-alloc 65665 \
  --prompt-file speed-bench/promessi_sposi.txt \
  --ctx-start 2048 --ctx-max 65536 --step-incr 8192 \
  --gen-tokens 128 --csv results.csv
```

### 6. Interactive chat

```bash
# Same worker command, then:
./ds4 -m ds4flash.gguf --role coordinator --layers 0:20 \
  --listen 10.0.0.4 1234 -c 65665
# Opens interactive ds4> prompt
```

## Observations & Notes

### What works well
- Pipeline parallelism delivers real prefill speedup at longer contexts
- The QSFP 10GbE link (1.4ms ping) is solid for this use case
- ds4's distributed protocol is simple and robust — TCP, no RDMA needed
- Model loading is fast (~9s per Spark for their layer slice)

### Known issues (as of 2026-05-30)
- **Pipelined prefill "accept failed" errors:** The coordinator reports `accept failed: Resource temporarily unavailable` when trying to pipeline multiple prefill chunks. The benchmark recovers gracefully (falls back to non-pipelined), but this likely reduces prefill throughput vs. what's theoretically achievable. May be a Linux/DGX-specific socket timing issue.
- **Context matching:** Worker `-c` must exactly match or exceed the coordinator's `--ctx-alloc` value. The coordinator's alloc = ctx-max + gen-tokens + 1. Mismatches are rejected at registration time.

### The buy decision (for Nic)

**Q4 dual-Spark vs Q2 single-Spark:**
- Q4 is a significantly better quantisation (less information loss in the routed experts)
- Generation speed is similar (~11 t/s dual Q4 vs ~14 t/s single Q2) — the 19% network penalty roughly cancels the larger model cost
- Prefill is slower on dual (122–165 t/s vs 343 t/s single) due to coordination overhead on 10GbE
- But you get the **higher quality Q4 quantisation** which is the real win

**Bottom line:** If quality matters more than raw speed (and at 10–11 t/s generation is still very usable for interactive chat), the second Spark is worth it. You get a meaningfully better quantisation at generation speeds that are still perfectly interactive.

**Important caveat — vision/images:** ds4 is **text-only**. DeepSeek V4 Flash through ds4 does not support multimodal/image input. If bulk image analysis is a key workload (e.g., processing graphs, maps, infographics at scale), Qwen 122B with vLLM remains the tool for that job. The dual-Spark DS4 setup would complement, not replace, an image analysis pipeline.

> ⏳ *Follow-up benchmark sweeps at 128K and 192K context are planned — results will be added here.*

## Credits

- **antirez** (Salvatore Sanfilippo) — ds4 engine, model GGUFs, distributed inference architecture
- **Nic (albond)** — hybrid INT4+FP8 recipe for Qwen 122B that freed our test Sparks, original question that prompted this benchmark
- **Mark Sunner** — hardware setup, QSFP cabling, oversight
- **Tars** — benchmark execution, writeup

## Raw Data

```csv
ctx_tokens,prefill_tokens,prefill_tps,gen_tokens,gen_tps,kvcache_bytes
2048,2048,121.93,128,11.36,0
10240,8192,164.55,128,11.21,0
18432,8192,161.69,128,11.09,0
26624,8192,160.19,128,10.88,0
34816,8192,157.76,128,10.45,0
43008,8192,154.96,128,10.34,0
51200,8192,152.86,128,10.18,0
59392,8192,151.19,128,10.04,0
65536,6144,134.97,128,9.91,0
```

### Pipelined prefill telemetry (65K context)

At 65,536 tokens the coordinator reported:
```
pipelined prefill done tokens=65536 chunks=16 total=302.265s 216.82 t/s
  local=251.771s send=14.080s 290.93 MiB/s
```

The 14-second send time across 16 chunks at 290 MiB/s shows the 10GbE QSFP link is well-utilised but not the bottleneck — local GPU compute (252s) dominates.

---

*Tested 2026-05-30. DGX Spark firmware/CUDA versions as shipped. ds4 commit ba00a8a.*
