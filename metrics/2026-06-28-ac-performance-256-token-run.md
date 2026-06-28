# AC Performance 256-Token Inference Run

This file records the fifth GPU inference test for the project. This run keeps
the machine on AC power and performance mode, but increases the workload again:
from the previous `128 prompt / 128 output` target to a larger
`256 prompt / 256 output` planning target.

## Summary

- Project: `ai-local-runtime`
- Binary: `airuntime`
- Repository path: `/home/msameer/ai_local`
- Code state: `4468f0e` (`Add CUDA inference calibration loop`)
- Backend: `llama.cpp` through the CUDA `llama-server` and `llama-bench`
- Model: `models/Qwen3-4B-Q4_K_M.gguf`
- Use case: interactive inference
- Goal: balanced
- Test prompt used:
  `Explain GPU inference, CPU scheduling, KV cache, batching, and why local LLM performance changes between battery and AC power.`
- Prompt mode: chat completion with thinking disabled
- Power source: AC
- Battery status during recorded snapshots: charging
- System power profile: performance mode
- Power profile verification after the run: `system76-power profile` reported
  `Power Profile: Performance`
- Result: CUDA inference completed successfully with full 36-layer GPU offload.
- Main inference result:
  - Prompt processing: 497.43 tokens/s
  - Generation: 35.96 tokens/s
  - Request wall time: 7202 ms
  - Peak GPU memory: 2523 MiB
  - Minimum free GPU memory during run: 1247 MiB
  - Peak GPU utilization: 93%
  - Peak GPU temperature: 73 C
  - Average GPU power: 28.32 W

## Why This Run Matters

This is the strongest sustained-generation test so far. It doubled the output
cap from 128 generated tokens to 256 generated tokens while staying on the same
model, same prompt, same AC power state, and same performance-mode system
profile.

The key result is that decode speed stayed stable:

- AC 32-token generation: 37.38 tokens/s
- AC 128-token generation: 36.35 tokens/s
- AC 256-token generation: 35.96 tokens/s

The 256-token run is only about 4% slower than the short 32-token run and about
1% slower than the 128-token run. That is an excellent sustained-decode signal
for this RTX 2050 and Qwen3 4B Q4_K_M pairing.

The tradeoff is clear too: the longer generation raised GPU temperature and
average power meaningfully.

## Run Conditions

- Run date: 2026-06-28
- Initial probe snapshot: #20
- Calibration snapshot: #21
- Inference snapshot: #22
- Runtime DB timestamp for inference: `2026-06-28 14:16:30` UTC
- Approx local time for inference: `2026-06-28 19:46:30` IST
- AC power: yes
- Battery status: charging
- Battery at initial probe: 84%
- Battery at calibration snapshot: 85%
- Battery at inference snapshot: 90%
- System power profile: performance mode
- Power profile verification command: `system76-power profile`
- Verified profile output: `Power Profile: Performance`
- Initial system thermal reading: 59 C
- Calibration snapshot thermal reading: 53 C
- Inference snapshot thermal reading: 69 C
- Peak GPU temperature during calibration: 76 C
- Peak GPU temperature during inference: 73 C

The machine was plugged in and charging. This is therefore part of the AC
performance baseline series, not a battery-constrained run.

## Commands Used

Hardware probe:

```bash
cargo run -- --database .runtime/runtime.db probe
```

Calibration:

```bash
cargo run -- --database .runtime/runtime.db calibrate \
  --model models/Qwen3-4B-Q4_K_M.gguf \
  --use-case interactive \
  --goal balanced \
  --prompt-tokens 256 \
  --output-tokens 256 \
  --candidates 10 \
  --repetitions 3
```

Inference:

```bash
cargo run -- --database .runtime/runtime.db infer \
  --model models/Qwen3-4B-Q4_K_M.gguf \
  --use-case interactive \
  --goal balanced \
  --prompt-tokens 256 \
  --output-tokens 256 \
  --prompt "Explain GPU inference, CPU scheduling, KV cache, batching, and why local LLM performance changes between battery and AC power." \
  --disable-thinking
```

Power profile verification after the run:

```bash
system76-power profile
```

Output:

```text
Power Profile: Performance
Backlight amdgpu_bl0: 13125/62451 = 21%
Keyboard Backlight asus::kbd_backlight: 0/3 = 0%
```

## Model Details

- Local file: `models/Qwen3-4B-Q4_K_M.gguf`
- GGUF internal model name: `Qwen3 4B Instruct Awq`
- Architecture: `qwen3`
- GGUF version: 3
- Quantization label of local artifact: `Q4_K_M`
- File size:
  - 2,497,280,256 bytes
  - 2.33 GiB
- Layers: 36
- Maximum context length: 40960 tokens
- Embedding length: 2560
- Attention heads: 32
- KV heads: 8
- Attention key length: 128
- Attention value length: 128

Important note on naming: the local file name identifies the artifact as
`Q4_K_M`. The GGUF metadata name printed by the runtime is
`Qwen3 4B Instruct Awq`. For runtime metrics, the exact local model artifact is
`Qwen3-4B-Q4_K_M.gguf`.

## Hardware Snapshot

Host:

- Hostname: `pop-os`
- OS: Pop!_OS 24.04 LTS
- Kernel: `6.18.7-76061807-generic`
- CPU: AMD Ryzen 7 5800HS with Radeon Graphics
- CPU architecture: x86_64
- Logical CPU cores: 16
- Physical CPU cores: 8

System memory:

- Total RAM: 16,130,932,736 bytes, about 15.0 GiB
- Available RAM at initial probe: 8,598,728,704 bytes, about 8.0 GiB
- Available RAM at calibration snapshot: 8,618,582,016 bytes, about 8.0 GiB
- Available RAM at inference snapshot: 8,753,500,160 bytes, about 8.2 GiB
- Swap total: 20,425,723,904 bytes, about 19.0 GiB
- Swap free: 20,425,723,904 bytes, about 19.0 GiB
- Planner-reported memory pressure near inference: 46%

Discrete GPU:

- Runtime telemetry device name: NVIDIA GeForce RTX 2050
- Probe name: NVIDIA GPU 0x25ad
- Vendor: NVIDIA
- Driver path: `/sys/class/drm/card1`
- Kernel driver: `nvidia`
- Telemetry available: true
- Available GPU memory reported by probe: 3,952,082,944 bytes
- Available GPU memory reported by probe: about 3769 MiB
- Scheduler GPU memory safety budget: 3,556,874,649 bytes
- Scheduler GPU memory safety budget: about 3392 MiB

Integrated GPU:

- Probe name: AMD GPU 0x1638
- Vendor: AMD
- Driver path: `/sys/class/drm/card0`
- Kernel driver: `amdgpu`
- Dedicated memory: 536,870,912 bytes, 512 MiB
- Available memory at initial probe: 91,688,960 bytes, about 87 MiB
- Available memory at calibration snapshot: 96,145,408 bytes, about 92 MiB
- Available memory at inference snapshot: 98,177,024 bytes, about 94 MiB
- Telemetry available: true

Power and thermals:

- Power source: AC
- Battery status: charging
- Battery range during the run: 84% up to 90%
- System power profile: performance
- Highest system thermal reading before inference: 59 C
- Peak GPU temperature during calibration: 76 C
- Peak GPU temperature during inference: 73 C

## Workload

- Use case: interactive
- Optimization goal: balanced
- Requested prompt budget: 256 tokens
- Requested output budget: 256 tokens
- Requested total context: 512 tokens
- Concurrency: 1
- Prompt:

```text
Explain GPU inference, CPU scheduling, KV cache, batching, and why local LLM performance changes between battery and AC power.
```

- Thinking mode: disabled through `--disable-thinking`
- Actual prompt tokens reported by backend: 38
- Actual generated tokens: 256
- Total actual tokens: 294

The output stopped because the requested output budget was 256 tokens. The
finish reason was `length`, so the model hit the configured generation cap.

## Scheduler Decision

Selected measured profile:

- CPU threads: 8
- GPU placement: exact full offload
- GPU layers: 36 of 36
- Context size: 512 tokens
- Parallel slots: 1
- Batch size: 256
- Physical batch size / ubatch size: 256
- KV cache type: F16
- Planning score near inference: 0.7867961504528254
- Selection basis: compatible measured CUDA calibration
- Calibration used: 68.16 effective tokens/s

Estimated memory for selected profile:

- Model bytes: 2,497,280,256
- KV cache bytes: 75,497,472
- Compute buffer bytes: 268,435,456
- Estimated total bytes: 2,841,213,184
- Estimated total size: about 2.65 GiB
- Estimated GPU bytes: 2,841,213,184
- Estimated GPU size: about 2.65 GiB
- System memory budget at inference snapshot: 7,878,150,144 bytes
- GPU memory safety budget: 3,556,874,649 bytes

Scheduler reasons:

- Preserved 512 tokens per request across 1 slot.
- Used 8 of 16 detected logical CPU threads.
- Offloaded all 36 model layers within the estimated GPU budget.
- Derived the GPU candidate from a 90% memory safety budget.
- Reused a compatible CUDA calibration measuring 68.16 effective tokens/s.

Scheduler notes:

- Stored calibration matched the current model, workload, hardware, and safe
  candidate set.
- Fixed values are safety/search bounds, not mode-to-configuration rules.
- The current peak temperature of 69 C contributed a continuous
  resource-pressure penalty.
- Current system memory pressure was 46%.

## CUDA Calibration Results

Calibration command settings:

- Candidates tested: 10
- Repetitions: 3
- Prompt tokens: 256
- Output tokens: 256
- Goal: balanced
- Use case: interactive
- Model: `models/Qwen3-4B-Q4_K_M.gguf`

Measured candidates, sorted by effective throughput:

| Rank | Effective tokens/s | Prompt tokens/s | Generation tokens/s | Threads | Batch | KV cache | Peak GPU temp | Avg GPU power |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| 1 | 68.16 | 1507.00 | 34.87 | 8 | 256 | F16 | 76 C | 39.46 W |
| 2 | 51.75 | 1340.82 | 26.38 | 16 | 128 | F16 | 69 C | 32.57 W |
| 3 | 48.58 | 1532.52 | 24.68 | 16 | 256 | F16 | 66 C | 29.30 W |
| 4 | 47.82 | 1448.13 | 24.31 | 16 | 256 | Q8_0 | 70 C | 31.66 W |
| 5 | 46.97 | 605.58 | 24.43 | 16 | 32 | F16 | 70 C | 31.50 W |
| 6 | 43.67 | 1279.22 | 22.21 | 16 | 128 | Q8_0 | 70 C | 29.61 W |
| 7 | 40.34 | 1283.54 | 20.49 | 16 | 128 | Q4_0 | 71 C | 27.96 W |
| 8 | 39.33 | 1453.89 | 19.93 | 16 | 256 | Q4_0 | 70 C | 27.62 W |
| 9 | 35.95 | 587.56 | 18.54 | 16 | 32 | Q4_0 | 72 C | 25.74 W |
| 10 | 33.50 | 587.22 | 17.24 | 16 | 32 | Q8_0 | 69 C | 25.41 W |

Winner details:

- Result: selected winner
- CPU threads: 8
- GPU placement: exact 36 layers
- Batch size: 256
- Ubatch size: 256
- KV cache: F16
- Effective throughput: 68.15520447913623 tokens/s
- Prompt processing measurement:
  - Prompt tokens: 256
  - Average time: 170.009261 ms
  - Throughput: 1507.002347 tokens/s
  - Standard deviation: 51.608896 tokens/s
- Generation measurement:
  - Generated tokens: 256
  - Average time: 7358.260127 ms
  - Throughput: 34.866022 tokens/s
  - Standard deviation: 1.958507 tokens/s
- Calibration wall time for winner profile: 25357 ms
- GPU telemetry samples for winner profile: 95
- Peak GPU memory: 2603 MiB
- Minimum free GPU memory: 1167 MiB
- Peak GPU utilization: 100%
- Peak GPU temperature: 76 C
- Average GPU power: 39.46 W

Calibration conclusion:

- The measured winner was again 8 CPU threads, full 36-layer GPU offload, and
  F16 KV cache.
- Batch 256 became the winning batch size for this larger 256/256 workload.
- The winner stayed well ahead of every 16-thread candidate.
- Calibration hit 100% peak GPU utilization and 76 C, making this the heaviest
  calibration run so far.
- The calibration generation rate, 34.87 tokens/s, closely matched the final
  inference generation rate, 35.96 tokens/s.

## Inference Result

Backend result:

- Backend: CUDA
- Model: `Qwen3-4B-Q4_K_M.gguf`
- Finish reason: `length`
- Reasoning payload: null
- Prompt tokens: 38
- Generated tokens: 256
- Total tokens: 294

Generated response:

```text
Sure! Here's a clear and concise explanation of each of the concepts you mentioned:

---

### **1. GPU Inference**
**Definition:**
GPU inference refers to the process of using a **Graphics Processing Unit (GPU)** to perform **inference**—the operation of using a trained model (like a large language model, or LLM) to generate outputs based on input data, without retraining the model.

**Key Points:**
- **Purpose:** Fast and efficient execution of models that require heavy computation, especially in machine learning and AI.
- **Why GPU?**
  - GPUs are highly parallel and can process many operations simultaneously, making them ideal for tasks like matrix multiplication, which are common in deep learning.
  - They outperform CPUs in tasks involving large-scale data processing and complex mathematical operations.
- **Use Cases:**
  - Real-time AI applications (e.g., chatbots, image recognition, video processing).
  - Large language models (LLMs) and other AI systems that require high computational power.

---

### **2. CPU Scheduling**
**Definition:**
CPU scheduling is the process of **distributing CPU time** among multiple processes or threads in a way that maximizes system efficiency and responsiveness.

**Key Points:
```

The response is still incomplete because generation was capped at 256 output
tokens.

Timing:

- Prompt processing tokens: 38
- Prompt processing time: 76.392 ms
- Prompt processing throughput: 497.4342863127029 tokens/s
- Generated tokens: 256
- Generation time: 7118.175 ms
- Generation throughput: 35.96427455070998 tokens/s
- Request wall time: 7202 ms

GPU telemetry during inference:

- Device: NVIDIA GeForce RTX 2050
- Samples collected: 49
- Peak GPU memory used: 2523 MiB
- Minimum GPU memory free: 1247 MiB
- Peak GPU utilization: 93%
- Peak GPU temperature: 73 C
- Average GPU power: 28.32244897959184 W

## Comparison With AC 128-Token Run

Comparison file:

```text
metrics/2026-06-28-ac-performance-128-token-run.md
```

| Field | AC 128-token run | AC 256-token run | Change |
| --- | ---: | ---: | --- |
| Prompt budget | 128 | 256 | 2x requested |
| Output budget | 128 | 256 | 2x requested |
| Actual prompt tokens | 38 | 38 | Same prompt |
| Actual generated tokens | 128 | 256 | 2x |
| Selected CPU threads | 8 | 8 | Same |
| Selected GPU layers | 36 / 36 | 36 / 36 | Same |
| Selected batch / ubatch | 128 / 128 | 256 / 256 | Larger batch |
| Selected KV cache | F16 | F16 | Same |
| Context size | 256 | 512 | Larger context |
| Estimated memory | 2.61 GiB | 2.65 GiB | Slightly higher |
| Calibration effective throughput | 69.76 tok/s | 68.16 tok/s | Slightly lower |
| Inference prompt processing | 511.69 tok/s | 497.43 tok/s | Slightly lower |
| Inference generation | 36.35 tok/s | 35.96 tok/s | Nearly same |
| Request wall time | 3602 ms | 7202 ms | About 2x, expected |
| Peak GPU memory | 2477 MiB | 2523 MiB | 46 MiB higher |
| Minimum free GPU memory | 1293 MiB | 1247 MiB | 46 MiB lower |
| Peak GPU utilization | 93% | 93% | Same |
| Peak GPU temperature | 64 C | 73 C | 9 C hotter |
| Average GPU power during inference | 22.58 W | 28.32 W | Higher |

The important result is sustained generation stability. The 128-token run
generated at 36.35 tokens/s. The 256-token run generated at 35.96 tokens/s.
That is essentially the same decode speed, with the expected increase in wall
time, power, and temperature.

## Comparison With AC Short And Sustained Runs

| Field | AC 32-token | AC 128-token | AC 256-token |
| --- | ---: | ---: | ---: |
| Output budget | 32 | 128 | 256 |
| Actual generated tokens | 32 | 128 | 256 |
| Selected CPU threads | 8 | 8 | 8 |
| Selected batch / ubatch | 64 / 64 | 128 / 128 | 256 / 256 |
| Selected KV cache | F16 | F16 | F16 |
| Inference prompt processing | 302.06 tok/s | 511.69 tok/s | 497.43 tok/s |
| Inference generation | 37.38 tok/s | 36.35 tok/s | 35.96 tok/s |
| Request wall time | 929 ms | 3602 ms | 7202 ms |
| Peak GPU memory | 2473 MiB | 2477 MiB | 2523 MiB |
| Peak GPU utilization | 94% | 93% | 93% |
| Peak GPU temperature | 60 C | 64 C | 73 C |
| Average GPU power | 11.28 W | 22.58 W | 28.32 W |

This progression is exactly what we want to see for a stable local inference
runtime: decode throughput remains close to flat as the output gets longer,
while thermals and power rise with sustained work.

## Approximate GPU Energy

Approximate GPU energy for the final inference request:

| Run | Avg GPU power | Wall time | Approx GPU energy |
| --- | ---: | ---: | ---: |
| AC performance 32-token | 11.28 W | 0.929 s | 10.48 J |
| AC performance 128-token | 22.58 W | 3.602 s | 81.33 J |
| AC performance 256-token | 28.32 W | 7.202 s | 203.98 J |

Per generated token:

- AC 32-token run: about 0.33 J/generated token
- AC 128-token run: about 0.64 J/generated token
- AC 256-token run: about 0.80 J/generated token

This estimate uses only sampled GPU average power multiplied by request wall
time. It is not full laptop energy usage. The increase shows that longer
sustained generation is more power-expensive per generated token, even when
tokens/s stays stable.

## Derived Observations

Model placement:

- Full GPU offload remained the winning strategy.
- All 36 layers were placed on the NVIDIA GPU.
- No additional model layers remain to offload for this model.

Memory:

- Peak observed GPU memory used during inference: 2523 MiB
- Minimum observed free GPU memory during inference: 1247 MiB
- Observed telemetry-visible memory pool during inference: about 3770 MiB
- Approx used share of telemetry-visible memory: about 67%
- Approx free share of telemetry-visible memory: about 33%

Even at the 512-token context planning target, VRAM remains comfortable. The
model still has room for longer context and limited concurrency experiments.

Utilization:

- Inference peak sampled GPU utilization reached 93%.
- Calibration peak sampled GPU utilization reached 100%.
- The GPU is being driven hard during calibration and steadily during
  inference.

Temperature:

- Inference peaked at 73 C, up from 64 C in the 128-token run.
- Calibration peaked at 76 C.
- This is the warmest final inference run so far.
- It is still below common thermal throttle territory, but it is now warm
  enough that longer stress tests should be watched carefully.

Power:

- Inference average GPU power rose to 28.32 W.
- Calibration winner average GPU power reached 39.46 W.
- The longer run makes the power cost visible; short tests hid this behavior.

Profile behavior:

- The best measured profile again used 8 CPU threads.
- The winning batch scaled with the workload: 64, then 128, now 256.
- F16 KV cache remained the selected KV cache on AC.
- Evidence is now strong that this model/workload family prefers:
  `8 threads + full GPU offload + F16 KV + batch roughly matching prompt/output scale`.

Quality and output:

- The answer became much more useful than the 32-token and 128-token tests.
- It still ended mid-section at `Key Points:` because 256 output tokens was
  still not enough for the full requested explanation.
- A complete answer probably needs 512 output tokens for this prompt.

## Conclusion

The 256-token AC performance run is the best sustained-generation result so
far.

The practical conclusion is:

```text
On AC power in performance mode, Qwen3-4B-Q4_K_M sustains about 36 tokens/s
generation on the RTX 2050 even over a 256-token output, using full GPU offload,
8 CPU threads, batch 256, and F16 KV cache.
```

The model is still VRAM-safe, and decode speed remains stable. The limiting
concern is no longer immediate throughput collapse; it is sustained heat and
power as output length grows.

## Current Baseline Numbers To Compare Later

Use these as the AC 256-token sustained-generation baseline:

| Field | AC 256-token performance baseline |
| --- | --- |
| Model | Qwen3-4B-Q4_K_M.gguf |
| Model size | 2.33 GiB |
| Quantization label | Q4_K_M |
| Backend | llama.cpp CUDA |
| GPU | NVIDIA GeForce RTX 2050 |
| Power state | AC |
| Battery status | Charging |
| System power profile | Performance |
| Battery level | 84% up to 90% |
| Use case | interactive |
| Goal | balanced |
| Prompt budget | 256 tokens |
| Output budget | 256 tokens |
| Actual prompt tokens | 38 |
| Actual generated tokens | 256 |
| GPU layers | 36 / 36 |
| CPU threads | 8 |
| Batch / ubatch | 256 / 256 |
| KV cache | F16 |
| Prompt processing | 497.43 tokens/s |
| Generation | 35.96 tokens/s |
| Request wall time | 7202 ms |
| Peak GPU memory | 2523 MiB |
| Minimum free GPU memory | 1247 MiB |
| Peak GPU utilization | 93% |
| Peak GPU temperature | 73 C |
| Average GPU power | 28.32 W |
| Calibration winner | 8 threads, F16, batch 256, 68.16 effective tokens/s |

## What Should Be Tested Next

The next useful test is either:

1. A 512-token complete-answer test to see if output quality finishes naturally.
2. A throughput-oriented calibration with a larger prompt/output mix.
3. A concurrency test, because serving multiple requests is where an inference
   engine starts behaving like a real runtime rather than a single-request demo.

Complete-answer test:

```bash
cargo run -- --database .runtime/runtime.db infer \
  --model models/Qwen3-4B-Q4_K_M.gguf \
  --use-case interactive \
  --goal balanced \
  --prompt-tokens 256 \
  --output-tokens 512 \
  --prompt "Explain GPU inference, CPU scheduling, KV cache, batching, and why local LLM performance changes between battery and AC power." \
  --disable-thinking
```

Throughput calibration:

```bash
cargo run -- --database .runtime/runtime.db calibrate \
  --model models/Qwen3-4B-Q4_K_M.gguf \
  --use-case interactive \
  --goal throughput \
  --prompt-tokens 512 \
  --output-tokens 256 \
  --candidates 10 \
  --repetitions 3
```

## Missing Metrics To Add Later

This run reinforces the need to add:

- Time to first token
- Model load time
- Server startup time
- Peak GPU power
- Average GPU utilization
- GPU clock and memory clock sampling
- CPU utilization during generation
- Full system power draw if available
- Thermal throttling indicators
- Repeated inference latency percentiles
- Automatic Markdown report export from stored DB rows

## Final Status

This fifth run becomes the project's AC 256-token sustained-generation baseline.
It shows that the RTX 2050 can keep Qwen3-4B-Q4_K_M near 36 tokens/s for a
longer 256-token output while remaining within VRAM, but now with noticeably
higher temperature and power draw.
