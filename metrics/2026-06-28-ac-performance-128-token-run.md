# AC Performance 128-Token Inference Run

This file records the fourth GPU inference test for the project. This run keeps
the machine on AC power and performance mode, but increases the workload from
the earlier short `64 prompt / 32 output` test to a larger
`128 prompt / 128 output` planning target.

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
  - Prompt processing: 511.69 tokens/s
  - Generation: 36.35 tokens/s
  - Request wall time: 3602 ms
  - Peak GPU memory: 2477 MiB
  - Minimum free GPU memory during run: 1293 MiB
  - Peak GPU utilization: 93%
  - Peak GPU temperature: 64 C
  - Average GPU power: 22.58 W

## Why This Run Matters

The earlier AC performance run used only 32 generated tokens. That was useful
for a quick latency baseline, but too short to show sustained decode behavior.

This run generated 128 tokens. It is a better stress test for:

- Sustained generation throughput
- GPU temperature rise over a longer request
- Average GPU power under a longer decode
- Whether the measured scheduler profile remains stable for a larger context
- Whether batch 128 is useful for this model on the RTX 2050

The key result is that generation throughput stayed strong:

- Short AC run generation: 37.38 tokens/s over 32 generated tokens
- Longer AC run generation: 36.35 tokens/s over 128 generated tokens

That is only about 3% lower while generating 4 times as many tokens. This is a
good sign: the GPU is sustaining decode speed instead of collapsing after the
first tiny request.

## Run Conditions

- Run date: 2026-06-28
- Initial probe snapshot: #15
- Calibration snapshot: #16
- Inference snapshot: #17
- Runtime DB timestamp for inference: `2026-06-28 13:59:35` UTC
- Approx local time for inference: `2026-06-28 19:29:35` IST
- AC power: yes
- Battery status: charging
- Battery at initial probe: 67%
- Battery at calibration snapshot: 67%
- Battery at inference snapshot: 74%
- System power profile: performance mode
- Power profile verification command: `system76-power profile`
- Verified profile output: `Power Profile: Performance`
- Initial system thermal reading: 61 C
- Calibration snapshot thermal reading: 61 C
- Inference snapshot thermal reading: 59 C
- Peak GPU temperature during calibration: 76 C
- Peak GPU temperature during inference: 64 C

The machine was plugged in and charging. This run is therefore a stronger
performance baseline than either battery run.

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
  --prompt-tokens 128 \
  --output-tokens 128 \
  --candidates 10 \
  --repetitions 3
```

Inference:

```bash
cargo run -- --database .runtime/runtime.db infer \
  --model models/Qwen3-4B-Q4_K_M.gguf \
  --use-case interactive \
  --goal balanced \
  --prompt-tokens 128 \
  --output-tokens 128 \
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
- Available RAM at initial probe: 8,813,166,592 bytes, about 8.2 GiB
- Available RAM at calibration snapshot: 8,800,002,048 bytes, about 8.2 GiB
- Available RAM at inference snapshot: 8,904,142,848 bytes, about 8.3 GiB
- Swap total: 20,425,723,904 bytes, about 19.0 GiB
- Swap free: 20,425,723,904 bytes, about 19.0 GiB
- Planner-reported memory pressure near inference: 45%

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
- Available memory at initial probe: 90,976,256 bytes, about 87 MiB
- Available memory at calibration snapshot: 55,717,888 bytes, about 53 MiB
- Available memory at inference snapshot: 66,375,680 bytes, about 63 MiB
- Telemetry available: true

Power and thermals:

- Power source: AC
- Battery status: charging
- Battery range during the run: 67% up to 74%
- System power profile: performance
- Highest system thermal reading before inference: 61 C
- Peak GPU temperature during calibration: 76 C
- Peak GPU temperature during inference: 64 C

## Workload

- Use case: interactive
- Optimization goal: balanced
- Requested prompt budget: 128 tokens
- Requested output budget: 128 tokens
- Requested total context: 256 tokens
- Concurrency: 1
- Prompt:

```text
Explain GPU inference, CPU scheduling, KV cache, batching, and why local LLM performance changes between battery and AC power.
```

- Thinking mode: disabled through `--disable-thinking`
- Actual prompt tokens reported by backend: 38
- Actual generated tokens: 128
- Total actual tokens: 166

The output stopped because the requested output budget was 128 tokens. The
finish reason was `length`, so the model hit the configured generation cap.

## Scheduler Decision

Selected measured profile:

- CPU threads: 8
- GPU placement: exact full offload
- GPU layers: 36 of 36
- Context size: 256 tokens
- Parallel slots: 1
- Batch size: 128
- Physical batch size / ubatch size: 128
- KV cache type: F16
- Planning score near inference: 0.806900092234647
- Selection basis: compatible measured CUDA calibration
- Calibration used: 69.76 effective tokens/s

Estimated memory for selected profile:

- Model bytes: 2,497,280,256
- KV cache bytes: 37,748,736
- Compute buffer bytes: 268,435,456
- Estimated total bytes: 2,803,464,448
- Estimated total size: about 2.61 GiB
- Estimated GPU bytes: 2,803,464,448
- Estimated GPU size: about 2.61 GiB
- System memory budget at inference snapshot: 8,013,728,563 bytes
- GPU memory safety budget: 3,556,874,649 bytes

Scheduler reasons:

- Preserved 256 tokens per request across 1 slot.
- Used 8 of 16 detected logical CPU threads.
- Offloaded all 36 model layers within the estimated GPU budget.
- Derived the GPU candidate from a 90% memory safety budget.
- Reused a compatible CUDA calibration measuring 69.76 effective tokens/s.

Scheduler notes:

- Stored calibration matched the current model, workload, hardware, and safe
  candidate set.
- Fixed values are safety/search bounds, not mode-to-configuration rules.
- The current peak temperature of 59 C contributed a continuous
  resource-pressure penalty.
- Current system memory pressure was 45%.

## CUDA Calibration Results

Calibration command settings:

- Candidates tested: 10
- Repetitions: 3
- Prompt tokens: 128
- Output tokens: 128
- Goal: balanced
- Use case: interactive
- Model: `models/Qwen3-4B-Q4_K_M.gguf`

Measured candidates, sorted by effective throughput:

| Rank | Effective tokens/s | Prompt tokens/s | Generation tokens/s | Threads | Batch | KV cache | Peak GPU temp | Avg GPU power |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| 1 | 69.76 | 1286.39 | 35.85 | 8 | 128 | F16 | 76 C | 37.80 W |
| 2 | 44.09 | 589.12 | 22.90 | 16 | 32 | Q4_0 | 73 C | 30.11 W |
| 3 | 43.97 | 1224.45 | 22.38 | 16 | 128 | Q4_0 | 73 C | 29.67 W |
| 4 | 42.97 | 939.44 | 21.99 | 16 | 64 | Q8_0 | 72 C | 29.02 W |
| 5 | 42.53 | 940.16 | 21.76 | 16 | 64 | Q4_0 | 73 C | 29.15 W |
| 6 | 42.00 | 584.93 | 21.78 | 16 | 32 | Q8_0 | 72 C | 29.48 W |
| 7 | 40.16 | 1307.51 | 20.39 | 16 | 128 | F16 | 66 C | 25.16 W |
| 8 | 40.11 | 980.59 | 20.47 | 16 | 64 | F16 | 69 C | 26.94 W |
| 9 | 39.83 | 1196.27 | 20.25 | 16 | 128 | Q8_0 | 71 C | 27.33 W |
| 10 | 38.12 | 602.08 | 19.68 | 16 | 32 | F16 | 71 C | 26.81 W |

Winner details:

- Result: selected winner
- CPU threads: 8
- GPU placement: exact 36 layers
- Batch size: 128
- Ubatch size: 128
- KV cache: F16
- Effective throughput: 69.76056693975701 tokens/s
- Prompt processing measurement:
  - Prompt tokens: 128
  - Average time: 99.741257 ms
  - Throughput: 1286.39218 tokens/s
  - Standard deviation: 75.673496 tokens/s
- Generation measurement:
  - Generated tokens: 128
  - Average time: 3570.972253 ms
  - Throughput: 35.852415 tokens/s
  - Standard deviation: 0.647125 tokens/s
- Calibration wall time for winner profile: 13515 ms
- GPU telemetry samples for winner profile: 55
- Peak GPU memory: 2527 MiB
- Minimum free GPU memory: 1243 MiB
- Peak GPU utilization: 96%
- Peak GPU temperature: 76 C
- Average GPU power: 37.80 W

Calibration conclusion:

- The measured winner was 8 CPU threads, full 36-layer GPU offload, batch 128,
  and F16 KV cache.
- The 8-thread winner was far ahead of every 16-thread candidate.
- Batch 128 became the winning batch size for this 128/128 workload.
- Calibration became much hotter and more power-hungry than the final inference
  request.
- The best generation measurement was 35.85 tokens/s, which closely matches the
  final inference result of 36.35 tokens/s.

## Inference Result

Backend result:

- Backend: CUDA
- Model: `Qwen3-4B-Q4_K_M.gguf`
- Finish reason: `length`
- Reasoning payload: null
- Prompt tokens: 38
- Generated tokens: 128
- Total tokens: 166

Generated response:

```text
Certainly! Here's a detailed explanation of each of the concepts you mentioned, including **GPU inference**, **CPU scheduling**, **KV cache**, **batching**, and the reason why **local LLM performance can change between battery and AC power**:

---

### 1. **GPU Inference**
**GPU Inference** refers to the process of using a **Graphics Processing Unit (GPU)** to perform **inference** on a **Large Language Model (LLM)**. Unlike **CPU inference**, which is more general-purpose, GPU inference is optimized for **parallel processing**, making it highly efficient for the **computationally intensive tasks
```

The response is incomplete because generation was capped at 128 output tokens.

Timing:

- Prompt processing tokens: 38
- Prompt processing time: 74.263 ms
- Prompt processing throughput: 511.6949221011809 tokens/s
- Generated tokens: 128
- Generation time: 3520.961 ms
- Generation throughput: 36.35371138731727 tokens/s
- Request wall time: 3602 ms

GPU telemetry during inference:

- Device: NVIDIA GeForce RTX 2050
- Samples collected: 32
- Peak GPU memory used: 2477 MiB
- Minimum GPU memory free: 1293 MiB
- Peak GPU utilization: 93%
- Peak GPU temperature: 64 C
- Average GPU power: 22.5815625 W

## Comparison With Short AC Performance Run

Comparison file:

```text
metrics/2026-06-28-ac-performance-gpu-run.md
```

| Field | AC short run | AC 128-token run | Change |
| --- | ---: | ---: | --- |
| Prompt budget | 64 | 128 | 2x requested |
| Output budget | 32 | 128 | 4x requested |
| Actual prompt tokens | 20 | 38 | 1.9x |
| Actual generated tokens | 32 | 128 | 4x |
| Selected CPU threads | 8 | 8 | Same |
| Selected GPU layers | 36 / 36 | 36 / 36 | Same |
| Selected batch / ubatch | 64 / 64 | 128 / 128 | Larger batch |
| Selected KV cache | F16 | F16 | Same |
| Context size | 96 | 256 | Larger context |
| Estimated memory | 2.59 GiB | 2.61 GiB | Slightly higher |
| Calibration effective throughput | 102.20 tok/s | 69.76 tok/s | Lower due to larger decode |
| Inference prompt processing | 302.06 tok/s | 511.69 tok/s | Higher |
| Inference generation | 37.38 tok/s | 36.35 tok/s | Slightly lower |
| Request wall time | 929 ms | 3602 ms | Longer, expected |
| Peak GPU memory | 2473 MiB | 2477 MiB | Almost same |
| Minimum free GPU memory | 1297 MiB | 1293 MiB | Almost same |
| Peak GPU utilization | 94% | 93% | Almost same |
| Peak GPU temperature | 60 C | 64 C | Hotter |
| Average GPU power during inference | 11.28 W | 22.58 W | Higher |

The important result is generation stability. The short run generated 32 tokens
at 37.38 tokens/s. The longer run generated 128 tokens at 36.35 tokens/s. That
is very close, so the model is sustaining decode speed over a longer response.

## Comparison With Earlier Battery Runs

| Field | Balanced battery | Performance battery | AC short | AC 128-token |
| --- | ---: | ---: | ---: | ---: |
| Power source | Battery | Battery | AC | AC |
| System profile | Balanced / standard | Performance | Performance | Performance |
| Output budget | 32 | 32 | 32 | 128 |
| Actual generated tokens | 32 | 32 | 32 | 128 |
| Selected CPU threads | 16 | 8 | 8 | 8 |
| Selected batch / ubatch | 64 / 64 | 64 / 64 | 64 / 64 | 128 / 128 |
| Selected KV cache | F16 | Q8_0 | F16 | F16 |
| Inference generation | 22.70 tok/s | 27.47 tok/s | 37.38 tok/s | 36.35 tok/s |
| Request wall time | 1506 ms | 1309 ms | 929 ms | 3602 ms |
| Peak GPU memory | 2473 MiB | 2459 MiB | 2473 MiB | 2477 MiB |
| Peak GPU utilization | 91% | 93% | 94% | 93% |
| Peak GPU temperature | 55 C | 58 C | 60 C | 64 C |
| Average GPU power | 12.26 W | 12.59 W | 11.28 W | 22.58 W |

This table is not a strict A/B comparison because the 128-token test has a
larger prompt and output budget. It is useful because it shows sustained AC
decode performance remains near the short-run result.

## Approximate GPU Energy

Approximate GPU energy for the final inference request:

| Run | Avg GPU power | Wall time | Approx GPU energy |
| --- | ---: | ---: | ---: |
| Balanced battery 32-token | 12.26 W | 1.506 s | 18.46 J |
| Performance battery 32-token | 12.59 W | 1.309 s | 16.48 J |
| AC performance 32-token | 11.28 W | 0.929 s | 10.48 J |
| AC performance 128-token | 22.58 W | 3.602 s | 81.33 J |

Per generated token, the 128-token run is more informative:

- Approx GPU energy per generated token: 81.33 J / 128 = about 0.64 J/token
- Approx GPU energy per total token: 81.33 J / 166 = about 0.49 J/token

This estimate uses only sampled GPU average power multiplied by request wall
time. It is not full laptop energy usage.

## Derived Observations

Model placement:

- Full GPU offload remained the winning strategy.
- All 36 layers were placed on the NVIDIA GPU.
- There are still no model layers left to move from CPU to GPU.

Memory:

- Peak observed GPU memory used during inference: 2477 MiB
- Minimum observed free GPU memory during inference: 1293 MiB
- Observed telemetry-visible memory pool during inference: about 3770 MiB
- Approx used share of telemetry-visible memory: about 66%
- Approx free share of telemetry-visible memory: about 34%

Even with a larger context and 128 generated tokens, VRAM remained comfortable.
The model still leaves enough room for longer context and concurrency
experiments.

Utilization:

- Inference peak sampled GPU utilization reached 93%.
- Calibration peak sampled GPU utilization reached 96%.
- Utilization stayed high but not dramatically different from the short AC run.

Temperature:

- Inference peaked at 64 C, compared with 60 C in the short AC run.
- Calibration peaked at 76 C, the hottest measurement so far in our records.
- The sustained workload raises temperature clearly, but the inference peak is
  still reasonable for this laptop GPU.

Power:

- Inference average GPU power rose to 22.58 W, much higher than the short AC
  run's 11.28 W.
- Calibration winner average GPU power reached 37.80 W.
- Longer decode work is much more power demanding than the tiny 32-token test.

Profile behavior:

- The best measured profile again used 8 CPU threads.
- The winning batch increased from 64 to 128 when the workload increased.
- F16 KV cache remained the selected KV cache on AC.
- The evidence for this model now strongly favors 8 CPU threads on AC.

Quality and output:

- The model produced a more useful answer than the 32-token tests.
- The response still ended mid-sentence because `--output-tokens 128` was a hard
  cap.
- For a complete explanatory answer, the next inference test should use 256
  output tokens.

## Conclusion

The 128-token AC performance run is a strong result.

The practical conclusion is:

```text
On AC power in performance mode, Qwen3-4B-Q4_K_M sustains roughly 36 tokens/s
generation on the RTX 2050 for a longer 128-token response, using full GPU
offload, 8 CPU threads, batch 128, and F16 KV cache.
```

Compared with the 32-token AC test, generation speed stayed nearly the same
while the request became much longer. That is exactly what we wanted to check.
The main cost was higher temperature and much higher GPU power.

## Current Baseline Numbers To Compare Later

Use these as the AC 128-token sustained-generation baseline:

| Field | AC 128-token performance baseline |
| --- | --- |
| Model | Qwen3-4B-Q4_K_M.gguf |
| Model size | 2.33 GiB |
| Quantization label | Q4_K_M |
| Backend | llama.cpp CUDA |
| GPU | NVIDIA GeForce RTX 2050 |
| Power state | AC |
| Battery status | Charging |
| System power profile | Performance |
| Battery level | 67% up to 74% |
| Use case | interactive |
| Goal | balanced |
| Prompt budget | 128 tokens |
| Output budget | 128 tokens |
| Actual prompt tokens | 38 |
| Actual generated tokens | 128 |
| GPU layers | 36 / 36 |
| CPU threads | 8 |
| Batch / ubatch | 128 / 128 |
| KV cache | F16 |
| Prompt processing | 511.69 tokens/s |
| Generation | 36.35 tokens/s |
| Request wall time | 3602 ms |
| Peak GPU memory | 2477 MiB |
| Minimum free GPU memory | 1293 MiB |
| Peak GPU utilization | 93% |
| Peak GPU temperature | 64 C |
| Average GPU power | 22.58 W |
| Calibration winner | 8 threads, F16, batch 128, 69.76 effective tokens/s |

## What Should Be Tested Next

The next useful test is a complete-answer 256-token generation:

```bash
cargo run -- --database .runtime/runtime.db infer \
  --model models/Qwen3-4B-Q4_K_M.gguf \
  --use-case interactive \
  --goal balanced \
  --prompt-tokens 128 \
  --output-tokens 256 \
  --prompt "Explain GPU inference, CPU scheduling, KV cache, batching, and why local LLM performance changes between battery and AC power." \
  --disable-thinking
```

After that, run a throughput-oriented calibration:

```bash
cargo run -- --database .runtime/runtime.db calibrate \
  --model models/Qwen3-4B-Q4_K_M.gguf \
  --use-case interactive \
  --goal throughput \
  --prompt-tokens 512 \
  --output-tokens 128 \
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
- Repeated inference latency percentiles
- Automatic Markdown report export from stored DB rows

## Final Status

This fourth run becomes the project's AC sustained-generation baseline. It
shows that the RTX 2050 can keep Qwen3-4B-Q4_K_M near 36 tokens/s for a longer
128-token output while remaining within VRAM and reasonable inference
temperature.
