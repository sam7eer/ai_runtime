# First Battery GPU Inference Run

This file records the first successful end-to-end GPU inference test for the
project. The run was done while the laptop was on battery, so this should be
treated as a baseline, not as the maximum performance ceiling.

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
- Power state: battery, discharging
- System power profile: balanced mode, standard performance, battery mode
- Result: CUDA inference completed successfully with full 36-layer GPU offload.
- Main inference result:
  - Prompt processing: 231.81 tokens/s
  - Generation: 22.70 tokens/s
  - Request wall time: 1506 ms
  - Peak GPU memory: 2473 MiB
  - Minimum free GPU memory during run: 1297 MiB
  - Peak GPU utilization: 91%
  - Peak GPU temperature: 55 C
  - Average GPU power: 12.26 W

## Why This Run Matters

This was the first real proof that the current runtime can:

1. Detect the host hardware.
2. Inspect a GGUF model.
3. Plan a model-aware execution profile.
4. Benchmark CUDA candidates.
5. Persist a compatible calibration result.
6. Reuse that calibration for later recommendations.
7. Start a CUDA-backed `llama-server`.
8. Send a prompt to the model.
9. Collect GPU telemetry during inference.
10. Shut the backend down after the one-shot request.

At this stage the project is still an inference control plane and scheduler
around `llama.cpp`, not a custom tensor-kernel inference engine. This run proves
that the outer runtime loop is alive.

## Run Conditions

- Run date: 2026-06-21
- Runtime DB timestamp for inference: `2026-06-21 10:31:42` UTC
- Approx local time: `2026-06-21 16:01:42` IST
- Battery state before first probe: 80%, discharging
- Battery state during calibration snapshot: 79%, discharging
- Battery state during inference snapshot: 78%, discharging
- AC power: no
- System power profile: balanced mode
- Performance profile: standard performance
- Battery mode: enabled
- Initial system thermal reading: 48 C
- System thermal reading before inference: 53 C

Because this was on battery, the GPU and CPU may have been power-limited by the
laptop firmware, the balanced/standard OS performance profile, battery mode, or
NVIDIA power management. These metrics are therefore a battery baseline.

## Commands Used

Build and tests:

```bash
cargo build
cargo test
```

Backend check:

```bash
cargo run -- backend-status
```

Hardware probe:

```bash
cargo run -- --database .runtime/runtime.db probe
```

Model inspection:

```bash
cargo run -- inspect-model \
  --model models/Qwen3-4B-Q4_K_M.gguf
```

Initial recommendation:

```bash
cargo run -- --database .runtime/runtime.db recommend \
  --model models/Qwen3-4B-Q4_K_M.gguf \
  --use-case interactive \
  --goal balanced \
  --prompt-tokens 64 \
  --output-tokens 32 \
  --show-candidates
```

Calibration:

```bash
cargo run -- --database .runtime/runtime.db calibrate \
  --model models/Qwen3-4B-Q4_K_M.gguf \
  --use-case interactive \
  --goal balanced \
  --prompt-tokens 64 \
  --output-tokens 32 \
  --candidates 2 \
  --repetitions 1
```

Recommendation after calibration:

```bash
cargo run -- --database .runtime/runtime.db recommend \
  --model models/Qwen3-4B-Q4_K_M.gguf \
  --use-case interactive \
  --goal balanced \
  --prompt-tokens 64 \
  --output-tokens 32 \
  --show-candidates
```

Successful inference:

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

Note: one earlier inference attempt was manually interrupted with `Ctrl+C`; it
is not counted as the successful baseline.

## Build And Test State

- `cargo build`: passed
- `cargo test`: passed
- Unit tests run: 14
- Unit tests passed: 14
- Unit tests failed: 0

The project was therefore in a clean, working Rust state before the benchmark
and inference test.

## Backend State

Backend discovery reported:

- `llama-server`: `/home/msameer/.local/share/llama.cpp/b9631-cuda/llama-server`
- Server backend: CUDA
- `llama-bench`: `/home/msameer/.local/share/llama.cpp/b9631-cuda/llama-bench`
- Bench backend: CUDA
- Ready for serving: true
- Ready for benchmarking: true

This means the runtime was using the CUDA build of `llama.cpp`, not a CPU-only
backend.

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

Important note on naming: the local file name clearly identifies the artifact as
`Q4_K_M`. The GGUF metadata name printed by the runtime is
`Qwen3 4B Instruct Awq`. For our runtime records, the actual local artifact used
for this test is `Qwen3-4B-Q4_K_M.gguf`.

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

- Total RAM: 16,130,928,640 bytes, about 15.0 GiB
- Available RAM near first probe: 11,009,400,832 bytes, about 10.3 GiB
- Available RAM near calibration: 11,030,253,568 bytes, about 10.3 GiB
- Available RAM near inference: 10,972,577,792 bytes, about 10.2 GiB
- Swap total: 20,425,723,904 bytes, about 19.0 GiB
- Swap free: 20,425,703,424 bytes, about 19.0 GiB

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
- Available memory: 169,254,912 bytes, about 161 MiB
- Telemetry available: true

Power and thermals:

- Power source: battery
- Battery status: discharging
- Battery range during the run: 80% down to 78%
- Highest system thermal reading before inference: 53 C
- Peak GPU temperature during inference: 55 C

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

The output stopped because the requested output budget was only 32 tokens.
The finish reason was `length`, meaning the backend hit the generation limit.

## Scheduler Decision

Selected measured profile:

- CPU threads: 16
- GPU placement: exact full offload
- GPU layers: 36 of 36
- Context size: 96 tokens
- Parallel slots: 1
- Batch size: 64
- Physical batch size / ubatch size: 64
- KV cache type: F16
- Planning score near inference: 0.8447546389862162
- Selection basis: compatible measured CUDA calibration
- Calibration used: 65.97 effective tokens/s

Estimated memory for selected profile:

- Model bytes: 2,497,280,256
- KV cache bytes: 14,155,776
- Compute buffer bytes: 268,435,456
- Estimated total bytes: 2,779,871,488
- Estimated total size: about 2.59 GiB
- Estimated GPU bytes: 2,779,871,488
- Estimated GPU size: about 2.59 GiB
- System memory budget at inference snapshot: 9,875,320,012 bytes
- GPU memory safety budget: 3,556,874,649 bytes

Scheduler reasons:

- Preserved 96 tokens per request across 1 slot.
- Used all 16 detected logical CPU threads.
- Offloaded all 36 model layers within the estimated GPU budget.
- Derived the GPU candidate from a 90% memory safety budget.
- Reused a compatible CUDA calibration measuring 65.97 effective tokens/s.

## CUDA Calibration Results

Calibration command settings:

- Candidates tested: 2
- Repetitions: 1
- Prompt tokens: 64
- Output tokens: 32
- Goal: balanced
- Use case: interactive

Candidate 1:

- Result: selected winner
- CPU threads: 16
- GPU placement: exact 36 layers
- Batch size: 64
- Ubatch size: 64
- KV cache: F16
- Effective throughput: 65.97080418750666 tokens/s
- Prompt processing measurement:
  - Prompt tokens: 64
  - Average time: 86.856396 ms
  - Throughput: 736.848441 tokens/s
- Generation measurement:
  - Generated tokens: 32
  - Average time: 1368.332785 ms
  - Throughput: 23.386124 tokens/s
- Calibration wall time: 4138 ms
- GPU telemetry samples: 18
- Peak GPU memory: 2487 MiB
- Minimum free GPU memory: 1283 MiB
- Peak GPU utilization: 91%
- Peak GPU temperature: 52 C
- Average GPU power: 12.57 W

Candidate 2:

- Result: slower than candidate 1
- CPU threads: 16
- GPU placement: exact 36 layers
- Batch size: 32
- Ubatch size: 32
- KV cache: F16
- Effective throughput: 32.92287792738993 tokens/s
- Prompt processing measurement:
  - Prompt tokens: 64
  - Average time: 725.431289 ms
  - Throughput: 88.22338 tokens/s
- Generation measurement:
  - Generated tokens: 32
  - Average time: 2190.474143 ms
  - Throughput: 14.608709 tokens/s
- Calibration wall time: 6897 ms
- GPU telemetry samples: 27
- Peak GPU memory: 2471 MiB
- Minimum free GPU memory: 1299 MiB
- Peak GPU utilization: 100%
- Peak GPU temperature: 53 C
- Average GPU power: 15.59 W

Calibration conclusion:

- Batch 64 was much better than batch 32 for this prompt/output mix.
- The stored measured profile replaced the analytical baseline for matching
  future recommendations.
- This calibration is useful, but not exhaustive. It used only 2 candidates and
  1 repetition, so it should not be treated as a final optimum.

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
image recognition, natural language
```

The response is incomplete because generation was capped at 32 output tokens.

Timing:

- Prompt processing tokens: 20
- Prompt processing time: 86.277 ms
- Prompt processing throughput: 231.81149089560367 tokens/s
- Generated tokens: 32
- Generation time: 1409.814 ms
- Generation throughput: 22.69802966916203 tokens/s
- Request wall time: 1506 ms

GPU telemetry during inference:

- Device: NVIDIA GeForce RTX 2050
- Samples collected: 26
- Peak GPU memory used: 2473 MiB
- Minimum GPU memory free: 1297 MiB
- Peak GPU utilization: 91%
- Peak GPU temperature: 55 C
- Average GPU power: 12.255769230769232 W

## Derived Observations

Model placement:

- The model was fully offloaded to the NVIDIA GPU.
- All 36 layers were placed on the GPU.
- There are no remaining model layers to move from CPU to GPU for this model.

Memory headroom:

- Peak observed GPU memory used: 2473 MiB
- Minimum observed free GPU memory: 1297 MiB
- Observed telemetry-visible memory pool during inference: about 3770 MiB
- Approx used share of telemetry-visible memory: about 66%
- Approx free share of telemetry-visible memory: about 34%

This means the model fits comfortably. Remaining VRAM can be used later for
larger context, more parallel slots, different KV cache choices, larger batches,
or possibly a larger model/quantization combination.

Utilization:

- Peak sampled GPU utilization reached 91%.
- This is high, but it does not prove that the GPU reached maximum possible
  tokens/s.
- Telemetry was sampled periodically, so short 100% windows can be missed.
- GPU utilization percentage alone is not the same thing as inference
  efficiency.

Temperature:

- Peak GPU temperature was only 55 C.
- This is not thermally alarming.
- There appears to be thermal headroom for a plugged-in performance test.

Power:

- Average GPU power was 12.26 W during the measured inference window.
- This average includes non-steady-state parts of the request.
- Since the machine was on battery, power management may have reduced sustained
  clocks or power draw.

Prompt vs generation:

- Prompt processing was much faster than generation.
- Prompt processing: 231.81 tokens/s in the actual inference run.
- Generation: 22.70 tokens/s in the actual inference run.
- This is normal for autoregressive LLM inference: decoding one token at a time
  is usually the slower part.

## Conclusion

The first battery test was successful.

The runtime can now run the complete loop:

```text
probe hardware -> inspect model -> plan profile -> calibrate CUDA candidate ->
reuse measured profile -> run inference -> collect telemetry
```

For this specific run, the scheduler selected the right broad strategy: full
GPU offload of all 36 layers. The model fits in VRAM with meaningful headroom,
and the GPU stayed cool.

However, this is not the maximum performance conclusion. The test was done on
battery, with only 2 calibration candidates, 1 calibration repetition, a short
64-token prompt budget, and a 32-token output cap. There is still room to test
larger batches, longer workloads, plugged-in power, repeated benchmarks,
throughput-focused goals, and concurrent requests.

## Current Baseline Numbers To Compare Later

Use these as the baseline when testing on AC power:

| Field | Battery baseline |
| --- | --- |
| Model | Qwen3-4B-Q4_K_M.gguf |
| Model size | 2.33 GiB |
| Quantization label | Q4_K_M |
| Backend | llama.cpp CUDA |
| GPU | NVIDIA GeForce RTX 2050 |
| Power state | Battery, discharging |
| System power profile | Balanced mode, standard performance, battery mode |
| Battery level | 78-80% |
| Use case | interactive |
| Goal | balanced |
| Prompt budget | 64 tokens |
| Output budget | 32 tokens |
| Actual prompt tokens | 20 |
| Actual generated tokens | 32 |
| GPU layers | 36 / 36 |
| CPU threads | 16 |
| Batch / ubatch | 64 / 64 |
| KV cache | F16 |
| Prompt processing | 231.81 tokens/s |
| Generation | 22.70 tokens/s |
| Request wall time | 1506 ms |
| Peak GPU memory | 2473 MiB |
| Minimum free GPU memory | 1297 MiB |
| Peak GPU utilization | 91% |
| Peak GPU temperature | 55 C |
| Average GPU power | 12.26 W |
| Calibration winner | batch 64, 65.97 effective tokens/s |

## What Should Be Tested Next

For a stronger plugged-in benchmark, run the same workload again on AC power:

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

Then run inference again:

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

For a more throughput-oriented test:

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

The current runtime already records useful GPU telemetry, but for a stronger
inference engine benchmark we should eventually add:

- Peak GPU power, not only average GPU power
- Average GPU utilization, not only peak utilization
- GPU SM clock during inference
- GPU memory clock during inference
- CPU utilization during inference
- RAM pressure during inference
- Tokens/s separated by prefill and decode for every request
- Time to load model
- Time to first token
- Server startup time
- Backend warmup behavior
- Multiple repeated inference runs
- Concurrent request throughput
- Latency percentiles for repeated requests
- Per-profile comparison history in a human-readable report

## Final Status

This first battery run should be considered the project's first successful CUDA
inference baseline. The important result is not just the generated text, but the
fact that the runtime selected a profile, measured it, reused it, executed it,
and collected telemetry end to end.
