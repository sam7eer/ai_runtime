# High-Performance Battery GPU Inference Run

This file records the second battery-only GPU inference test for the project.
The purpose of this run was to compare the first balanced/standard battery
baseline against a high-performance battery run with the same model and the
same short interactive workload.

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
- Power source: battery, discharging
- Charging state: not charging
- System power profile: high performance / performance mode
- Power profile verification after the run: `system76-power profile` reported
  `Power Profile: Performance`
- Result: CUDA inference completed successfully with full 36-layer GPU offload.
- Main inference result:
  - Prompt processing: 161.90 tokens/s
  - Generation: 27.47 tokens/s
  - Request wall time: 1309 ms
  - Peak GPU memory: 2459 MiB
  - Minimum free GPU memory during run: 1311 MiB
  - Peak GPU utilization: 93%
  - Peak GPU temperature: 58 C
  - Average GPU power: 12.59 W

## Why This Run Matters

This run is the first direct follow-up to the original balanced battery test.
It keeps the same model and user-facing workload, but changes the system power
profile to performance mode while still staying on battery.

The most important result is that the actual one-shot inference improved in
decode speed:

- Balanced battery generation baseline: 22.70 tokens/s
- Performance battery generation result: 27.47 tokens/s
- Change: about 21% higher generation throughput

The wall-clock request also improved:

- Balanced battery request wall time: 1506 ms
- Performance battery request wall time: 1309 ms
- Change: about 13% lower wall time

However, the calibration run became hotter and selected a different execution
profile. This means the result is useful, but it is not yet a final performance
ceiling.

## Run Conditions

- Run date: 2026-06-28
- Initial probe snapshot: #7
- Calibration snapshot: #8
- Inference snapshot: #9
- Runtime DB timestamp for inference: `2026-06-28 12:26:31` UTC
- Approx local time for inference: `2026-06-28 17:56:31` IST
- AC power: no
- Battery status: discharging
- Battery at initial probe: 57%
- Battery at calibration snapshot: 57%
- Battery at inference snapshot: 48%
- System power profile: performance mode
- Power profile verification command: `system76-power profile`
- Verified profile output: `Power Profile: Performance`
- Initial system thermal reading: 56 C
- Calibration snapshot thermal reading: 57 C
- Inference snapshot thermal reading: 58 C
- Peak GPU temperature during calibration: 66 C
- Peak GPU temperature during inference: 58 C

The machine was still on battery, so the GPU and CPU may still have been limited
by firmware, laptop thermal policy, battery discharge limits, or NVIDIA power
management. Performance mode is better than balanced mode for this test, but it
does not make the battery run equivalent to a plugged-in AC-power run.

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
- Available RAM at initial probe: 8,820,711,424 bytes, about 8.2 GiB
- Available RAM at calibration snapshot: 9,392,795,648 bytes, about 8.7 GiB
- Available RAM at inference snapshot: 9,537,892,352 bytes, about 8.9 GiB
- Swap total: 20,425,723,904 bytes, about 19.0 GiB
- Swap free: 20,425,723,904 bytes, about 19.0 GiB
- Planner-reported memory pressure near inference: 41%

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
- Available memory at initial probe: 108,486,656 bytes, about 103 MiB
- Available memory at calibration snapshot: 51,339,264 bytes, about 49 MiB
- Available memory at inference snapshot: 52,117,504 bytes, about 50 MiB
- Telemetry available: true

Power and thermals:

- Power source: battery
- Charging state: not charging
- Battery status: discharging
- Battery range during the run: 57% down to 48%
- System power profile: performance
- Highest system thermal reading before inference: 58 C
- Peak GPU temperature during calibration: 66 C
- Peak GPU temperature during inference: 58 C

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
- KV cache type: Q8_0
- Planning score near inference: 0.7251858615018392
- Selection basis: compatible measured CUDA calibration
- Calibration used: 33.68 effective tokens/s

Estimated memory for selected profile:

- Model bytes: 2,497,280,256
- KV cache bytes: 7,520,256
- Compute buffer bytes: 268,435,456
- Estimated total bytes: 2,773,235,968
- Estimated total size: about 2.58 GiB
- Estimated GPU bytes: 2,773,235,968
- Estimated GPU size: about 2.58 GiB
- System memory budget at inference snapshot: 8,584,103,116 bytes
- GPU memory safety budget: 3,556,874,649 bytes

Scheduler reasons:

- Preserved 96 tokens per request across 1 slot.
- Used 8 of 16 detected logical CPU threads.
- Offloaded all 36 model layers within the estimated GPU budget.
- Derived the GPU candidate from a 90% memory safety budget.
- Reused a compatible CUDA calibration measuring 33.68 effective tokens/s.

Scheduler notes:

- Stored calibration matched the current model, workload, hardware, and safe
  candidate set.
- Fixed values are safety/search bounds, not mode-to-configuration rules.
- The host was on battery; balanced and efficiency rankings include a
  resource-pressure penalty without replacing the requested goal.
- The current peak temperature of 58 C contributed a continuous
  resource-pressure penalty.
- Current system memory pressure was 41%.

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

| Rank | Effective tokens/s | Prompt tokens/s | Generation tokens/s | Threads | Batch | KV cache | Peak GPU temp | Avg GPU power |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| 1 | 33.68 | 135.50 | 13.46 | 8 | 64 | Q8_0 | 66 C | 18.75 W |
| 2 | 33.60 | 86.36 | 15.12 | 8 | 32 | Q8_0 | 66 C | 18.84 W |
| 3 | 33.45 | 142.09 | 13.23 | 8 | 64 | F16 | 65 C | 18.70 W |
| 4 | 31.06 | 88.94 | 13.49 | 8 | 32 | F16 | 65 C | 16.50 W |
| 5 | 14.15 | 789.26 | 4.78 | 16 | 64 | F16 | 57 C | 12.01 W |
| 6 | 10.47 | 135.95 | 3.68 | 16 | 64 | Q4_0 | 63 C | 16.63 W |
| 7 | 10.30 | 135.57 | 3.62 | 16 | 64 | Q8_0 | 61 C | 16.20 W |
| 8 | 10.08 | 88.62 | 3.64 | 16 | 32 | F16 | 59 C | 15.66 W |
| 9 | 10.03 | 86.61 | 3.62 | 16 | 32 | Q4_0 | 63 C | 16.63 W |
| 10 | 10.01 | 86.22 | 3.62 | 16 | 32 | Q8_0 | 62 C | 16.41 W |

Winner details:

- Result: selected winner
- CPU threads: 8
- GPU placement: exact 36 layers
- Batch size: 64
- Ubatch size: 64
- KV cache: Q8_0
- Effective throughput: 33.67882655513164 tokens/s
- Prompt processing measurement:
  - Prompt tokens: 64
  - Average time: 472.374891 ms
  - Throughput: 135.500643 tokens/s
  - Standard deviation: 1.744088 tokens/s
- Generation measurement:
  - Generated tokens: 32
  - Average time: 4592.140806 ms
  - Throughput: 13.455933 tokens/s
  - Standard deviation: 14.224826 tokens/s
- Calibration wall time for winner profile: 18388 ms
- GPU telemetry samples for winner profile: 57
- Peak GPU memory: 2473 MiB
- Minimum free GPU memory: 1297 MiB
- Peak GPU utilization: 100%
- Peak GPU temperature: 66 C
- Average GPU power: 18.75 W

Calibration conclusion:

- The measured winner changed from the first run's 16-thread F16 profile to an
  8-thread Q8_0 profile.
- The broader calibration explored more candidates than the first run.
- The calibration phase was much hotter than the final one-shot inference.
- Peak utilization reached 100% during calibration for all tested profiles.
- Calibration generation measurements were noisy, especially for some 8-thread
  candidates, so the result should be used as a practical profile hint rather
  than a final hardware limit.

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
accelerate the execution of machine learning models, particularly in tasks like
image recognition, natural language
```

The response is incomplete because generation was capped at 32 output tokens.

Timing:

- Prompt processing tokens: 20
- Prompt processing time: 123.534 ms
- Prompt processing throughput: 161.8987485226739 tokens/s
- Generated tokens: 32
- Generation time: 1165.082 ms
- Generation throughput: 27.465877938205207 tokens/s
- Request wall time: 1309 ms

GPU telemetry during inference:

- Device: NVIDIA GeForce RTX 2050
- Samples collected: 22
- Peak GPU memory used: 2459 MiB
- Minimum GPU memory free: 1311 MiB
- Peak GPU utilization: 93%
- Peak GPU temperature: 58 C
- Average GPU power: 12.594545454545456 W

## Comparison With First Battery Baseline

The first baseline was recorded in:

```text
metrics/2026-06-21-first-battery-gpu-run.md
```

Comparison table:

| Field | Balanced battery baseline | Performance battery run | Change |
| --- | ---: | ---: | --- |
| System power profile | Balanced / standard | Performance | Changed |
| Battery range | 80% to 78% | 57% to 48% | Lower battery in second run |
| Calibration candidates | 2 | 10 | Broader second search |
| Calibration repetitions | 1 | 3 | More reliable second search |
| Selected CPU threads | 16 | 8 | Lower thread count selected |
| Selected KV cache | F16 | Q8_0 | Different KV cache |
| Selected batch / ubatch | 64 / 64 | 64 / 64 | Same |
| GPU layers | 36 / 36 | 36 / 36 | Same full offload |
| Estimated memory | 2.59 GiB | 2.58 GiB | Nearly same |
| Calibration effective throughput | 65.97 tokens/s | 33.68 tokens/s | Lower in second calibration |
| Inference prompt processing | 231.81 tokens/s | 161.90 tokens/s | Lower by about 30% |
| Inference generation | 22.70 tokens/s | 27.47 tokens/s | Higher by about 21% |
| Request wall time | 1506 ms | 1309 ms | Lower by about 13% |
| Peak GPU memory | 2473 MiB | 2459 MiB | 14 MiB lower |
| Minimum free GPU memory | 1297 MiB | 1311 MiB | 14 MiB higher |
| Peak GPU utilization | 91% | 93% | 2 percentage points higher |
| Peak GPU temperature | 55 C | 58 C | 3 C hotter |
| Average GPU power | 12.26 W | 12.59 W | 0.33 W higher |

Important comparison caveat:

This is not a perfectly controlled A/B test. The first run used only 2
calibration candidates and 1 repetition. The second run used 10 candidates and
3 repetitions, started at a much lower battery level, had higher system
temperature, and had higher system memory pressure. The comparison is still
valuable, but the numbers should be interpreted as practical observed behavior,
not as a strict hardware law.

## Derived Observations

Model placement:

- The model was fully offloaded to the NVIDIA GPU.
- All 36 layers were placed on the GPU.
- There are still no remaining model layers to move from CPU to GPU for this
  model.

Memory headroom:

- Peak observed GPU memory used during inference: 2459 MiB
- Minimum observed free GPU memory during inference: 1311 MiB
- Observed telemetry-visible memory pool during inference: about 3770 MiB
- Approx used share of telemetry-visible memory: about 65%
- Approx free share of telemetry-visible memory: about 35%

This means the model still fits comfortably. Remaining VRAM can be used later
for larger context, more parallel slots, different KV cache choices, larger
batches, or possibly a larger model/quantization combination.

Utilization:

- Inference peak sampled GPU utilization reached 93%.
- Calibration peak sampled GPU utilization reached 100%.
- This indicates the GPU can be driven hard, but utilization percentage alone
  does not prove maximum tokens/s.

Temperature:

- Inference peaked at 58 C, only 3 C hotter than the first baseline.
- Calibration peaked at 66 C, much hotter than the first calibration.
- Performance mode and the broader calibration pushed the GPU harder.

Power:

- Inference average GPU power was 12.59 W, close to the first baseline's
  12.26 W.
- Calibration average GPU power for the winning profile was 18.75 W, much
  higher than the one-shot inference average.
- The battery dropped from 57% to 48% across this run, so performance-mode
  battery testing is noticeably more expensive.

Prompt vs generation:

- Prompt processing became slower than the first baseline.
- Generation became faster than the first baseline.
- For chat inference, generation speed is usually more important to perceived
  response speed once the prompt is short.

Profile behavior:

- The scheduler selected 8 threads instead of 16 after measuring the broader
  candidate set.
- In this run, several 16-thread candidates had poor generation throughput.
- That suggests CPU thread count is not simply "more is always better" for this
  laptop on battery.

## Conclusion

The high-performance battery run improved actual generation throughput and
reduced request wall time compared with the first balanced battery baseline.
That is the main positive result.

The result is not a final peak-performance claim. The broader calibration was
noisier, hotter, and selected a different profile. The second run also happened
at a much lower battery level and higher temperature. The correct conclusion is:

```text
Performance mode on battery improved the real one-shot generation result for
this short prompt, but we still need an AC-power test before claiming the GPU's
true ceiling.
```

## Current Baseline Numbers To Compare Later

Use these as the performance-mode battery baseline:

| Field | Performance battery baseline |
| --- | --- |
| Model | Qwen3-4B-Q4_K_M.gguf |
| Model size | 2.33 GiB |
| Quantization label | Q4_K_M |
| Backend | llama.cpp CUDA |
| GPU | NVIDIA GeForce RTX 2050 |
| Power state | Battery, discharging |
| Charging state | Not charging |
| System power profile | Performance |
| Battery level | 57% down to 48% |
| Use case | interactive |
| Goal | balanced |
| Prompt budget | 64 tokens |
| Output budget | 32 tokens |
| Actual prompt tokens | 20 |
| Actual generated tokens | 32 |
| GPU layers | 36 / 36 |
| CPU threads | 8 |
| Batch / ubatch | 64 / 64 |
| KV cache | Q8_0 |
| Prompt processing | 161.90 tokens/s |
| Generation | 27.47 tokens/s |
| Request wall time | 1309 ms |
| Peak GPU memory | 2459 MiB |
| Minimum free GPU memory | 1311 MiB |
| Peak GPU utilization | 93% |
| Peak GPU temperature | 58 C |
| Average GPU power | 12.59 W |
| Calibration winner | 8 threads, Q8_0, batch 64, 33.68 effective tokens/s |

## What Should Be Tested Next

The next useful comparison is AC power with the same high-performance profile:

```bash
cargo run -- --database .runtime/runtime.db probe
```

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

For a more realistic throughput test after AC power is available:

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

The current runtime already records useful telemetry, but this second run makes
some missing fields more obvious:

- System power profile should be captured automatically.
- Battery discharge rate should be captured during the run.
- Peak GPU power should be recorded, not only average GPU power.
- Average GPU utilization should be recorded, not only peak utilization.
- GPU SM clock and memory clock should be sampled.
- CPU utilization should be sampled.
- Time to load model should be separated from request time.
- Time to first token should be measured.
- Calibration should print and store KV cache type in the human-facing summary.
- Calibration should report variance more clearly.
- Repeated inference runs should produce latency percentiles.

## Final Status

This second run becomes the project's high-performance battery baseline. It
shows a better real generation result than the balanced battery run, while also
showing that calibration behavior can shift significantly when the search space
is larger and the laptop is hotter/lower on battery.
