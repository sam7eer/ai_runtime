# AC Performance GPU Inference Run

This file records the third GPU inference test for the project. Unlike the
first two runs, this one was done on AC power while the system power profile was
set to performance mode.

## Summary

- Project: `ai-local-runtime`
- Binary: `airuntime`
- Repository path: `/home/msameer/ai_local`
- Code state: `4468f0e` (`Add CUDA inference calibration loop`)
- Backend: `llama.cpp` through the CUDA `llama-server` and `llama-bench`
- Model: `models/Qwen3-4B-Q4_K_M.gguf`
- Use case: interactive inference
- Goal: balanced
- Test prompt used: `Explain GPU inference in two sentences.`
- Prompt mode: chat completion with thinking disabled
- Power source: AC
- Battery status during recorded snapshots: charging
- System power profile: performance mode
- Power profile verification after the run: `system76-power profile` reported
  `Power Profile: Performance`
- Result: CUDA inference completed successfully with full 36-layer GPU offload.
- Main inference result:
  - Prompt processing: 302.06 tokens/s
  - Generation: 37.38 tokens/s
  - Request wall time: 929 ms
  - Peak GPU memory: 2473 MiB
  - Minimum free GPU memory during run: 1297 MiB
  - Peak GPU utilization: 94%
  - Peak GPU temperature: 60 C
  - Average GPU power: 11.28 W

## Why This Run Matters

This is the first plugged-in performance-mode result. It gives us the clearest
answer so far about the RTX 2050's useful local inference behavior with this
Qwen3 4B Q4_K_M model.

The AC run produced the best actual one-shot inference result so far:

- Fastest prompt processing: 302.06 tokens/s
- Fastest generation: 37.38 tokens/s
- Lowest request wall time: 929 ms
- Highest inference GPU utilization sample: 94%

The most important result is decode speed. Compared with the first balanced
battery run, AC performance mode improved generation from 22.70 tokens/s to
37.38 tokens/s, about a 65% increase.

## Run Conditions

- Run date: 2026-06-28
- Initial probe snapshot: #10
- Calibration snapshot: #11
- Inference snapshot: #12
- Runtime DB timestamp for inference: `2026-06-28 13:22:02` UTC
- Approx local time for inference: `2026-06-28 18:52:02` IST
- AC power: yes
- Battery status: charging
- Battery at initial probe: 32%
- Battery at calibration snapshot: 32%
- Battery at inference snapshot: 36%
- System power profile: performance mode
- Power profile verification command: `system76-power profile`
- Verified profile output: `Power Profile: Performance`
- Initial system thermal reading: 61 C
- Calibration snapshot thermal reading: 57 C
- Inference snapshot thermal reading: 60 C
- Peak GPU temperature during calibration: 69 C
- Peak GPU temperature during inference: 60 C

The machine was on AC power, so this run is more representative of performance
headroom than the battery-only runs. It is still not a final maximum because it
uses a short prompt, a 32-token output cap, and only one final inference request.

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
  --prompt-tokens 64 \
  --output-tokens 32 \
  --candidates 10 \
  --repetitions 3
```

Inference:

```bash
cargo run -- --database .runtime/runtime.db infer \
  --model models/Qwen3-4B-Q4_K_M.gguf \
  --use-case interactive \
  --goal balanced \
  --prompt-tokens 64 \
  --output-tokens 32 \
  --prompt "Explain GPU inference in two sentences." \
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
- Available RAM at initial probe: 8,895,401,984 bytes, about 8.3 GiB
- Available RAM at calibration snapshot: 8,942,866,432 bytes, about 8.3 GiB
- Available RAM at inference snapshot: 8,918,073,344 bytes, about 8.3 GiB
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
- Available memory at initial probe: 66,629,632 bytes, about 64 MiB
- Available memory at calibration snapshot: 87,465,984 bytes, about 83 MiB
- Available memory at inference snapshot: 88,244,224 bytes, about 84 MiB
- Telemetry available: true

Power and thermals:

- Power source: AC
- Battery status: charging
- Battery range during the run: 32% up to 36%
- System power profile: performance
- Highest system thermal reading before inference: 61 C
- Peak GPU temperature during calibration: 69 C
- Peak GPU temperature during inference: 60 C

## Workload

- Use case: interactive
- Optimization goal: balanced
- Requested prompt budget: 64 tokens
- Requested output budget: 32 tokens
- Requested total context: 96 tokens
- Concurrency: 1
- Prompt:

```text
Explain GPU inference in two sentences.
```

- Thinking mode: disabled through `--disable-thinking`
- Actual prompt tokens reported by backend: 20
- Actual generated tokens: 32
- Total actual tokens: 52

The output stopped because the requested output budget was 32 tokens. The finish
reason was `length`, so the model hit the configured generation cap.

## Scheduler Decision

Selected measured profile:

- CPU threads: 8
- GPU placement: exact full offload
- GPU layers: 36 of 36
- Context size: 96 tokens
- Parallel slots: 1
- Batch size: 64
- Physical batch size / ubatch size: 64
- KV cache type: F16
- Planning score near inference: 0.8027354811244192
- Selection basis: compatible measured CUDA calibration
- Calibration used: 102.20 effective tokens/s

Estimated memory for selected profile:

- Model bytes: 2,497,280,256
- KV cache bytes: 14,155,776
- Compute buffer bytes: 268,435,456
- Estimated total bytes: 2,779,871,488
- Estimated total size: about 2.59 GiB
- Estimated GPU bytes: 2,779,871,488
- Estimated GPU size: about 2.59 GiB
- System memory budget at inference snapshot: 8,026,266,009 bytes
- GPU memory safety budget: 3,556,874,649 bytes

Scheduler reasons:

- Preserved 96 tokens per request across 1 slot.
- Used 8 of 16 detected logical CPU threads.
- Offloaded all 36 model layers within the estimated GPU budget.
- Derived the GPU candidate from a 90% memory safety budget.
- Reused a compatible CUDA calibration measuring 102.20 effective tokens/s.

Scheduler notes:

- Stored calibration matched the current model, workload, hardware, and safe
  candidate set.
- Fixed values are safety/search bounds, not mode-to-configuration rules.
- The current peak temperature of 60 C contributed a continuous
  resource-pressure penalty.
- Current system memory pressure was 45%.

## CUDA Calibration Results

Calibration command settings:

- Candidates tested: 10
- Repetitions: 3
- Prompt tokens: 64
- Output tokens: 32
- Goal: balanced
- Use case: interactive
- Model: `models/Qwen3-4B-Q4_K_M.gguf`

Measured candidates, sorted by effective throughput:

| Rank | Effective tokens/s | Prompt tokens/s | Generation tokens/s | Threads | GPU layers | Batch | KV cache | Peak GPU temp | Avg GPU power |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| 1 | 102.20 | 940.69 | 36.73 | 8 | 36 | 64 | F16 | 69 C | 28.60 W |
| 2 | 98.44 | 605.25 | 36.81 | 8 | 36 | 32 | F16 | 69 C | 29.39 W |
| 3 | 79.97 | 944.26 | 28.25 | 16 | 36 | 64 | F16 | 59 C | 21.13 W |
| 4 | 77.37 | 901.07 | 27.36 | 16 | 36 | 64 | Q8_0 | 64 C | 25.55 W |
| 5 | 75.90 | 898.79 | 26.81 | 16 | 36 | 64 | Q4_0 | 66 C | 25.41 W |
| 6 | 71.67 | 584.70 | 26.01 | 16 | 36 | 32 | Q4_0 | 67 C | 26.15 W |
| 7 | 70.88 | 587.24 | 25.69 | 16 | 36 | 32 | Q8_0 | 65 C | 25.35 W |
| 8 | 69.42 | 606.18 | 25.05 | 16 | 36 | 32 | F16 | 61 C | 24.54 W |
| 9 | 45.25 | 390.39 | 16.35 | 16 | 27 | 64 | F16 | 65 C | 17.93 W |
| 10 | 43.58 | 218.31 | 16.76 | 16 | 27 | 32 | F16 | 64 C | 16.05 W |

Winner details:

- Result: selected winner
- CPU threads: 8
- GPU placement: exact 36 layers
- Batch size: 64
- Ubatch size: 64
- KV cache: F16
- Effective throughput: 102.19609522916052 tokens/s
- Prompt processing measurement:
  - Prompt tokens: 64
  - Average time: 68.269599 ms
  - Throughput: 940.685613 tokens/s
  - Standard deviation: 66.086289 tokens/s
- Generation measurement:
  - Generated tokens: 32
  - Average time: 871.367667 ms
  - Throughput: 36.725253 tokens/s
  - Standard deviation: 0.274595 tokens/s
- Calibration wall time for winner profile: 4795 ms
- GPU telemetry samples for winner profile: 22
- Peak GPU memory: 2487 MiB
- Minimum free GPU memory: 1283 MiB
- Peak GPU utilization: 99%
- Peak GPU temperature: 69 C
- Average GPU power: 28.60 W

Calibration conclusion:

- AC power produced the strongest calibration result so far.
- The selected profile used 8 CPU threads, full 36-layer GPU offload, batch 64,
  and F16 KV cache.
- The winner measured 102.20 effective tokens/s, far above both battery
  calibrations.
- Partial GPU offload, 27 layers, was much slower than full 36-layer offload.
- The calibration phase drove GPU power and temperature much harder than the
  one-shot inference phase.

## Inference Result

Backend result:

- Backend: CUDA
- Model: `Qwen3-4B-Q4_K_M.gguf`
- Finish reason: `length`
- Reasoning payload: null
- Prompt tokens: 20
- Generated tokens: 32
- Total tokens: 52

Generated response:

```text
GPU inference refers to the use of a Graphics Processing Unit (GPU) to
accelerate the processing of machine learning models, particularly in tasks like
image recognition or natural language
```

The response is incomplete because generation was capped at 32 output tokens.

Timing:

- Prompt processing tokens: 20
- Prompt processing time: 66.211 ms
- Prompt processing throughput: 302.06461162042564 tokens/s
- Generated tokens: 32
- Generation time: 856.086 ms
- Generation throughput: 37.37942216085767 tokens/s
- Request wall time: 929 ms

GPU telemetry during inference:

- Device: NVIDIA GeForce RTX 2050
- Samples collected: 20
- Peak GPU memory used: 2473 MiB
- Minimum GPU memory free: 1297 MiB
- Peak GPU utilization: 94%
- Peak GPU temperature: 60 C
- Average GPU power: 11.2835 W

## Three-Way Comparison

Comparison files:

```text
metrics/2026-06-21-first-battery-gpu-run.md
metrics/2026-06-28-high-performance-battery-gpu-run.md
metrics/2026-06-28-ac-performance-gpu-run.md
```

| Field | Balanced battery | Performance battery | AC performance |
| --- | ---: | ---: | ---: |
| Power source | Battery | Battery | AC |
| System profile | Balanced / standard | Performance | Performance |
| Calibration candidates | 2 | 10 | 10 |
| Calibration repetitions | 1 | 3 | 3 |
| Selected CPU threads | 16 | 8 | 8 |
| Selected GPU layers | 36 / 36 | 36 / 36 | 36 / 36 |
| Selected batch / ubatch | 64 / 64 | 64 / 64 | 64 / 64 |
| Selected KV cache | F16 | Q8_0 | F16 |
| Calibration effective throughput | 65.97 tok/s | 33.68 tok/s | 102.20 tok/s |
| Inference prompt processing | 231.81 tok/s | 161.90 tok/s | 302.06 tok/s |
| Inference generation | 22.70 tok/s | 27.47 tok/s | 37.38 tok/s |
| Request wall time | 1506 ms | 1309 ms | 929 ms |
| Peak GPU memory | 2473 MiB | 2459 MiB | 2473 MiB |
| Minimum free GPU memory | 1297 MiB | 1311 MiB | 1297 MiB |
| Peak GPU utilization | 91% | 93% | 94% |
| Peak GPU temperature | 55 C | 58 C | 60 C |
| Average GPU power during inference | 12.26 W | 12.59 W | 11.28 W |

Performance changes from the first balanced battery baseline:

- Prompt processing improved from 231.81 to 302.06 tokens/s, about 30% higher.
- Generation improved from 22.70 to 37.38 tokens/s, about 65% higher.
- Request wall time dropped from 1506 ms to 929 ms, about 38% lower.
- Peak GPU utilization increased from 91% to 94%.
- Peak GPU temperature increased from 55 C to 60 C.
- Average GPU power during the final inference request dropped from 12.26 W to
  11.28 W, even though the request ran faster.

Performance changes from the high-performance battery run:

- Prompt processing improved from 161.90 to 302.06 tokens/s, about 87% higher.
- Generation improved from 27.47 to 37.38 tokens/s, about 36% higher.
- Request wall time dropped from 1309 ms to 929 ms, about 29% lower.
- Peak GPU utilization increased from 93% to 94%.
- Peak GPU temperature increased from 58 C to 60 C.
- Average GPU power during inference dropped from 12.59 W to 11.28 W.

Approximate GPU energy for the final inference request:

| Run | Avg GPU power | Wall time | Approx GPU energy |
| --- | ---: | ---: | ---: |
| Balanced battery | 12.26 W | 1.506 s | 18.46 J |
| Performance battery | 12.59 W | 1.309 s | 16.48 J |
| AC performance | 11.28 W | 0.929 s | 10.48 J |

This estimate uses only sampled GPU average power multiplied by request wall
time. It is not full laptop energy usage. Still, the direction is useful: the
AC performance run finished faster and also had the lowest approximate GPU
energy for this short request.

## Derived Observations

Model placement:

- The model was fully offloaded to the NVIDIA GPU.
- All 36 layers were placed on the GPU.
- Full offload remained the winning strategy.
- Partial 27-layer offload was much slower in the AC calibration.

Memory headroom:

- Peak observed GPU memory used during inference: 2473 MiB
- Minimum observed free GPU memory during inference: 1297 MiB
- Observed telemetry-visible memory pool during inference: about 3770 MiB
- Approx used share of telemetry-visible memory: about 66%
- Approx free share of telemetry-visible memory: about 34%

This confirms that Qwen3 4B Q4_K_M comfortably fits on the RTX 2050 with room
left for larger context, larger batches, or limited concurrency experiments.

Utilization:

- Inference peak sampled GPU utilization reached 94%.
- Calibration peak sampled GPU utilization reached 99%.
- AC power drove much better tokens/s without needing much higher inference
  average power in the final short request.

Temperature:

- Inference peaked at 60 C.
- Calibration peaked at 69 C.
- The GPU is warmer on AC, but still not near an obvious thermal danger point.

Power:

- Final inference average GPU power was 11.28 W.
- Calibration winner average GPU power was 28.60 W.
- This tells us calibration is a heavier sustained workload than the one-shot
  final request.

Prompt vs generation:

- AC performance improved both prompt processing and generation.
- Generation is the key win because decode speed dominates perceived chat speed
  for short interactive prompts.

Profile behavior:

- The best AC profile used 8 CPU threads, not 16.
- This matches the performance battery run's thread count, but AC switched the
  winning KV cache back to F16.
- The evidence so far says 8 CPU threads is a better measured profile for this
  model/workload than 16 threads.

## Conclusion

The AC performance run is the best result so far.

The practical conclusion is:

```text
For Qwen3-4B-Q4_K_M on this RTX 2050 laptop, full GPU offload with 8 CPU
threads, batch 64, and F16 KV cache is currently the best measured short-prompt
profile. AC power materially improves decode speed and request latency.
```

This still is not the final engine ceiling. The test used a short prompt, a
32-token output cap, one final inference request, and no concurrent load. The
next serious performance step is to test longer prompts, longer generations,
and concurrency while collecting better clocks, peak power, and time-to-first
token metrics.

## Current Baseline Numbers To Compare Later

Use these as the AC performance baseline:

| Field | AC performance baseline |
| --- | --- |
| Model | Qwen3-4B-Q4_K_M.gguf |
| Model size | 2.33 GiB |
| Quantization label | Q4_K_M |
| Backend | llama.cpp CUDA |
| GPU | NVIDIA GeForce RTX 2050 |
| Power state | AC |
| Battery status | Charging |
| System power profile | Performance |
| Battery level | 32% up to 36% |
| Use case | interactive |
| Goal | balanced |
| Prompt budget | 64 tokens |
| Output budget | 32 tokens |
| Actual prompt tokens | 20 |
| Actual generated tokens | 32 |
| GPU layers | 36 / 36 |
| CPU threads | 8 |
| Batch / ubatch | 64 / 64 |
| KV cache | F16 |
| Prompt processing | 302.06 tokens/s |
| Generation | 37.38 tokens/s |
| Request wall time | 929 ms |
| Peak GPU memory | 2473 MiB |
| Minimum free GPU memory | 1297 MiB |
| Peak GPU utilization | 94% |
| Peak GPU temperature | 60 C |
| Average GPU power | 11.28 W |
| Calibration winner | 8 threads, F16, batch 64, 102.20 effective tokens/s |

## What Should Be Tested Next

Run a longer generation test on AC power:

```bash
cargo run -- --database .runtime/runtime.db infer \
  --model models/Qwen3-4B-Q4_K_M.gguf \
  --use-case interactive \
  --goal balanced \
  --prompt-tokens 128 \
  --output-tokens 128 \
  --prompt "Explain GPU inference, CPU scheduling, KV cache, and batching in a local LLM runtime." \
  --disable-thinking
```

Then run a throughput-oriented calibration:

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

The AC run reinforces the need to add:

- System power profile capture inside `probe`
- AC/battery charging state in benchmark summaries
- Peak GPU power, not only average GPU power
- Average GPU utilization, not only peak utilization
- GPU SM clock and memory clock sampling
- Time to first token
- Model load time
- Server startup time
- Repeated inference latency percentiles
- Concurrent request throughput
- Full laptop power draw if available

## Final Status

This third run becomes the project's AC performance baseline. It is the best
measured result so far for the current model, prompt, and scheduler path.
