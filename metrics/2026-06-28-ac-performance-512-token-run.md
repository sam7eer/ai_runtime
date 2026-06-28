# AC Performance 512-Token Inference Run

This file records the sixth GPU inference test for the project. This run keeps
the machine on AC power and performance mode, increases the planning target to
`512 prompt / 512 output`, and uses a longer prompt than the earlier tests.

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
  `You are helping design a local AI inference runtime for a laptop with an NVIDIA RTX 2050 GPU, an AMD Ryzen 7 5800HS CPU, and 16 GB of system memory. Explain GPU inference, CPU scheduling, KV cache, batching, quantized GGUF models, full GPU layer offload, power modes, thermal limits, and why local LLM performance changes between battery mode and AC performance mode. Also explain how these observations should guide the design of a scheduler for a future high-performance inference engine.`
- Prompt mode: chat completion with thinking disabled
- Power source: AC
- Battery status during recorded snapshots: charging
- System power profile: performance mode
- Power profile verification after the run: `system76-power profile` reported
  `Power Profile: Performance`
- Result: CUDA inference completed successfully with full 36-layer GPU offload.
- Main inference result:
  - Prompt processing: 1091.88 tokens/s
  - Generation: 20.45 tokens/s
  - Request wall time: 25153 ms
  - Peak GPU memory: 2639 MiB
  - Minimum free GPU memory during run: 1131 MiB
  - Peak GPU utilization: 93%
  - Peak GPU temperature: 66 C
  - Average GPU power: 24.04 W

## Why This Run Matters

This was the first run with both a larger prompt budget and a 512-token output
budget. It is also the first test where sustained generation speed dropped
sharply compared with the 128-token and 256-token AC runs.

Key result:

- AC 128-token generation: 36.35 tokens/s
- AC 256-token generation: 35.96 tokens/s
- AC 512-token generation: 20.45 tokens/s

The 512-token result is still successful, but it shows that the workload crossed
into a different performance regime. The scheduler selected a different profile:
`16 CPU threads`, `batch 512`, and `F16` KV cache. Earlier sustained AC winners
used `8 CPU threads`.

This matters for the scheduler: the current top-candidate search can miss
profiles that we should still test. For this run, calibration tested only five
candidates and all of the tested candidates used 16 CPU threads. The previous
evidence strongly favored 8 CPU threads, so a future scheduler should keep
measured winners from nearby workloads in the candidate set.

## Run Conditions

- Run date: 2026-06-28
- Initial probe snapshot: #23
- Calibration snapshot: #24
- Inference snapshot: #25
- Runtime DB timestamp for inference: `2026-06-28 16:33:22` UTC
- Approx local time for inference: `2026-06-28 22:03:22` IST
- AC power: yes
- Battery status: charging
- Battery at initial probe: 87%
- Battery at calibration snapshot: 87%
- Battery at inference snapshot: 91%
- System power profile: performance mode
- Power profile verification command: `system76-power profile`
- Verified profile output: `Power Profile: Performance`
- Initial system thermal reading: 57 C
- Calibration snapshot thermal reading: 58 C
- Inference snapshot thermal reading: 63 C
- Peak GPU temperature during calibration: 72 C
- Peak GPU temperature during inference: 66 C

The machine was plugged in and charging. This is part of the AC performance
baseline series, not a battery-constrained run.

## Commands Used

Calibration command:

```bash
cargo run -- --database .runtime/runtime.db calibrate \
  --model models/Qwen3-4B-Q4_K_M.gguf \
  --use-case interactive \
  --goal balanced \
  --prompt-tokens 512 \
  --output-tokens 512 \
  --candidates 5 \
  --repetitions 2
```

Inference command:

```bash
cargo run -- --database .runtime/runtime.db infer \
  --model models/Qwen3-4B-Q4_K_M.gguf \
  --use-case interactive \
  --goal balanced \
  --prompt-tokens 512 \
  --output-tokens 512 \
  --prompt "You are helping design a local AI inference runtime for a laptop with an NVIDIA RTX 2050 GPU, an AMD Ryzen 7 5800HS CPU, and 16 GB of system memory. Explain GPU inference, CPU scheduling, KV cache, batching, quantized GGUF models, full GPU layer offload, power modes, thermal limits, and why local LLM performance changes between battery mode and AC performance mode. Also explain how these observations should guide the design of a scheduler for a future high-performance inference engine." \
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
- Available RAM at initial probe: 8,472,059,904 bytes, about 7.9 GiB
- Available RAM at calibration snapshot: 8,511,844,352 bytes, about 7.9 GiB
- Available RAM at inference snapshot: 8,466,399,232 bytes, about 7.9 GiB
- Swap total: 20,425,723,904 bytes, about 19.0 GiB
- Swap free: 20,425,723,904 bytes, about 19.0 GiB
- Planner-reported memory pressure near inference: 48%

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
- Available memory at initial probe: 76,890,112 bytes, about 73 MiB
- Available memory at calibration snapshot: 58,744,832 bytes, about 56 MiB
- Available memory at inference snapshot: 50,929,664 bytes, about 49 MiB
- Telemetry available: true

Power and thermals:

- Power source: AC
- Battery status: charging
- Battery range during the run: 87% up to 91%
- System power profile: performance
- Highest system thermal reading before inference: 58 C
- Peak GPU temperature during calibration: 72 C
- Peak GPU temperature during inference: 66 C

## Workload

- Use case: interactive
- Optimization goal: balanced
- Requested prompt budget: 512 tokens
- Requested output budget: 512 tokens
- Requested total context: 1024 tokens
- Concurrency: 1
- Actual prompt tokens reported by backend: 123
- Actual generated tokens: 512
- Total actual tokens: 635

The output stopped because the requested output budget was 512 tokens. The
finish reason was `length`, so even 512 tokens did not complete the answer.

## Scheduler Decision

Selected measured profile:

- CPU threads: 16
- GPU placement: exact full offload
- GPU layers: 36 of 36
- Context size: 1024 tokens
- Parallel slots: 1
- Batch size: 512
- Physical batch size / ubatch size: 512
- KV cache type: F16
- Planning score near inference: 0.9172254794127416
- Selection basis: compatible measured CUDA calibration
- Calibration used: 50.97 effective tokens/s

Estimated memory for selected profile:

- Model bytes: 2,497,280,256
- KV cache bytes: 150,994,944
- Compute buffer bytes: 268,435,456
- Estimated total bytes: 2,916,710,656
- Estimated total size: about 2.72 GiB
- Estimated GPU bytes: 2,916,710,656
- Estimated GPU size: about 2.72 GiB
- System memory budget at inference snapshot: 7,619,759,308 bytes
- GPU memory safety budget: 3,556,874,649 bytes

Scheduler reasons:

- Preserved 1024 tokens per request across 1 slot.
- Used all 16 detected logical CPU threads.
- Offloaded all 36 model layers within the estimated GPU budget.
- Derived the GPU candidate from a 90% memory safety budget.
- Reused a compatible CUDA calibration measuring 50.97 effective tokens/s.

## CUDA Calibration Results

Calibration command settings:

- Candidates tested: 5
- Repetitions: 2
- Prompt tokens: 512
- Output tokens: 512
- Goal: balanced
- Use case: interactive
- Model: `models/Qwen3-4B-Q4_K_M.gguf`

Measured candidates, sorted by effective throughput:

| Rank | Effective tokens/s | Prompt tokens/s | Generation tokens/s | Threads | Batch | KV cache | Peak GPU temp | Avg GPU power |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| 1 | 50.97 | 1692.80 | 25.88 | 16 | 512 | F16 | 68 C | 30.99 W |
| 2 | 47.55 | 1534.61 | 24.15 | 16 | 256 | F16 | 70 C | 31.39 W |
| 3 | 38.79 | 593.46 | 20.05 | 16 | 32 | F16 | 71 C | 28.07 W |
| 4 | 38.51 | 1556.19 | 19.50 | 16 | 512 | Q8_0 | 72 C | 27.60 W |
| 5 | 30.48 | 1438.74 | 15.40 | 16 | 256 | Q8_0 | 72 C | 23.44 W |

Winner details:

- Result: selected winner
- CPU threads: 16
- GPU placement: exact 36 layers
- Batch size: 512
- Ubatch size: 512
- KV cache: F16
- Effective throughput: 50.97112912463239 tokens/s
- Prompt processing measurement:
  - Prompt tokens: 512
  - Average time: 302.491063 ms
  - Throughput: 1692.802385 tokens/s
  - Standard deviation: 25.389066 tokens/s
- Generation measurement:
  - Generated tokens: 512
  - Average time: 19788.812964 ms
  - Throughput: 25.875121 tokens/s
  - Standard deviation: 0.314918 tokens/s
- Calibration wall time for winner profile: 42505 ms
- GPU telemetry samples for winner profile: 180
- Peak GPU memory: 2787 MiB
- Minimum free GPU memory: 983 MiB
- Peak GPU utilization: 100%
- Peak GPU temperature: 68 C
- Average GPU power: 30.99 W

Calibration conclusion:

- The measured winner changed to 16 CPU threads, batch 512, and F16 KV cache.
- This differs from the 128-token and 256-token AC runs, where 8 CPU threads
  won.
- The calibration did not test any 8-thread candidate in this 5-candidate pass.
- Because previous runs strongly favored 8 threads, this result should be
  treated as the best measured profile from the tested set, not proof that 16
  threads is globally optimal for 512/512.

## Inference Result

Backend result:

- Backend: CUDA
- Model: `Qwen3-4B-Q4_K_M.gguf`
- Finish reason: `length`
- Reasoning payload: null
- Prompt tokens: 123
- Generated tokens: 512
- Total tokens: 635

Generated response summary:

- The model started with a structured explanation of local inference runtime
  design.
- It covered GPU inference and began CPU scheduling.
- The answer did not reach KV cache, batching, scheduler design, or final
  conclusions before the 512-token cap.
- The response contained several hardware inaccuracies. This is a quality issue,
  not a scheduler or GPU execution issue.

Notable hallucinations in the generated answer:

- It described the RTX 2050 as having 16 GB of VRAM. The runtime telemetry shows
  about 3.7 GiB available in the visible GPU memory pool.
- It described the Ryzen 7 5800HS as having 16 cores with performance and
  efficiency cores. The probe reports 8 physical cores and 16 logical threads.
- It gave specific hardware details that were not provided by our runtime to the
  model as verified facts.

This means future answer-quality tests should either pass verified probe data
directly into the prompt or add a grounding layer that prevents the model from
inventing hardware specifications.

Timing:

- Prompt processing tokens: 123
- Prompt processing time: 112.650 ms
- Prompt processing throughput: 1091.877496671105 tokens/s
- Generated tokens: 512
- Generation time: 25032.887 ms
- Generation throughput: 20.4530943634268 tokens/s
- Request wall time: 25153 ms

GPU telemetry during inference:

- Device: NVIDIA GeForce RTX 2050
- Samples collected: 117
- Peak GPU memory used: 2639 MiB
- Minimum GPU memory free: 1131 MiB
- Peak GPU utilization: 93%
- Peak GPU temperature: 66 C
- Average GPU power: 24.036752136752135 W

## Comparison With AC 256-Token Run

Comparison file:

```text
metrics/2026-06-28-ac-performance-256-token-run.md
```

| Field | AC 256-token run | AC 512-token run | Change |
| --- | ---: | ---: | --- |
| Prompt budget | 256 | 512 | 2x requested |
| Output budget | 256 | 512 | 2x requested |
| Actual prompt tokens | 38 | 123 | Larger prompt |
| Actual generated tokens | 256 | 512 | 2x |
| Selected CPU threads | 8 | 16 | Changed |
| Selected GPU layers | 36 / 36 | 36 / 36 | Same |
| Selected batch / ubatch | 256 / 256 | 512 / 512 | Larger batch |
| Selected KV cache | F16 | F16 | Same |
| Context size | 512 | 1024 | Larger context |
| Estimated memory | 2.65 GiB | 2.72 GiB | Higher |
| Calibration effective throughput | 68.16 tok/s | 50.97 tok/s | Lower |
| Inference prompt processing | 497.43 tok/s | 1091.88 tok/s | Higher |
| Inference generation | 35.96 tok/s | 20.45 tok/s | Much lower |
| Request wall time | 7202 ms | 25153 ms | Much longer |
| Peak GPU memory | 2523 MiB | 2639 MiB | 116 MiB higher |
| Minimum free GPU memory | 1247 MiB | 1131 MiB | 116 MiB lower |
| Peak GPU utilization | 93% | 93% | Same |
| Peak GPU temperature | 73 C | 66 C | Cooler |
| Average GPU power during inference | 28.32 W | 24.04 W | Lower |

The 512-token run did not preserve the near-36 tokens/s generation rate seen in
the 128-token and 256-token AC tests. The most likely contributors are:

- Larger context and output length.
- A different selected profile: 16 threads instead of 8.
- A limited calibration search that tested only 5 candidates.
- The top-5 analytical candidates did not include the 8-thread profiles that
  performed well in nearby workloads.

## AC Run Progression

| Field | AC 32-token | AC 128-token | AC 256-token | AC 512-token |
| --- | ---: | ---: | ---: | ---: |
| Output budget | 32 | 128 | 256 | 512 |
| Actual prompt tokens | 20 | 38 | 38 | 123 |
| Actual generated tokens | 32 | 128 | 256 | 512 |
| Selected CPU threads | 8 | 8 | 8 | 16 |
| Selected batch / ubatch | 64 / 64 | 128 / 128 | 256 / 256 | 512 / 512 |
| Selected KV cache | F16 | F16 | F16 | F16 |
| Inference prompt processing | 302.06 tok/s | 511.69 tok/s | 497.43 tok/s | 1091.88 tok/s |
| Inference generation | 37.38 tok/s | 36.35 tok/s | 35.96 tok/s | 20.45 tok/s |
| Request wall time | 929 ms | 3602 ms | 7202 ms | 25153 ms |
| Peak GPU memory | 2473 MiB | 2477 MiB | 2523 MiB | 2639 MiB |
| Peak GPU utilization | 94% | 93% | 93% | 93% |
| Peak GPU temperature | 60 C | 64 C | 73 C | 66 C |
| Average GPU power | 11.28 W | 22.58 W | 28.32 W | 24.04 W |

The first three AC runs showed stable decode near 36 tokens/s. The 512-token run
breaks that trend. This is useful because it identifies the next scheduler
problem: broaden candidate search and carry forward known-good profiles across
neighboring workloads.

## Approximate GPU Energy

Approximate GPU energy for the final inference request:

| Run | Avg GPU power | Wall time | Approx GPU energy |
| --- | ---: | ---: | ---: |
| AC performance 32-token | 11.28 W | 0.929 s | 10.48 J |
| AC performance 128-token | 22.58 W | 3.602 s | 81.33 J |
| AC performance 256-token | 28.32 W | 7.202 s | 203.98 J |
| AC performance 512-token | 24.04 W | 25.153 s | 604.75 J |

Per generated token:

- AC 32-token run: about 0.33 J/generated token
- AC 128-token run: about 0.64 J/generated token
- AC 256-token run: about 0.80 J/generated token
- AC 512-token run: about 1.18 J/generated token

This estimate uses only sampled GPU average power multiplied by request wall
time. It is not full laptop energy usage.

## Derived Observations

Model placement:

- Full GPU offload remained the selected strategy.
- All 36 layers were placed on the NVIDIA GPU.
- No additional model layers remain to offload for this model.

Memory:

- Peak observed GPU memory used during inference: 2639 MiB
- Minimum observed free GPU memory during inference: 1131 MiB
- Observed telemetry-visible memory pool during inference: about 3770 MiB
- Approx used share of telemetry-visible memory: about 70%
- Approx free share of telemetry-visible memory: about 30%

The model still fits within VRAM, but the 1024-token planning context and batch
512 move memory usage closer to the safety margin.

Utilization:

- Inference peak sampled GPU utilization reached 93%.
- Calibration peak sampled GPU utilization reached 100%.
- Utilization stayed high, but high utilization did not mean high decode speed.

Temperature:

- Inference peaked at 66 C, lower than the 256-token run's 73 C.
- Calibration peaked at 72 C.
- The run was long but not thermally dangerous.

Power:

- Inference average GPU power was 24.04 W, lower than the 256-token run's
  28.32 W.
- Because wall time was much longer, total approximate GPU energy was much
  higher.

Profile behavior:

- The selected profile changed from 8 threads to 16 threads.
- Batch scaled to 512.
- F16 KV cache remained the selected KV cache.
- The generation rate fell to 20.45 tokens/s, so the selected profile should
  not be accepted as globally optimal yet.

Quality behavior:

- The answer became longer and more structured.
- The answer still stopped by length.
- The answer hallucinated hardware details.
- Future inference-engine work should separate performance tests from
  truthfulness tests, and pass verified system facts into the prompt when asking
  the model to reason about local hardware.

## Conclusion

The 512-token run succeeded technically, but it is not the best performance
profile we have seen.

The practical conclusion is:

```text
The runtime can run a 512-token output on AC power with full GPU offload, but
generation drops to about 20.45 tokens/s under the selected 16-thread, batch 512
profile. The next scheduler improvement should broaden calibration so known
8-thread profiles are tested for nearby larger workloads.
```

This run is important because it reveals the next engineering task: the
scheduler should not depend only on the current analytical top-N list. It should
also include measured winners from neighboring workloads, such as 8-thread
profiles from the 128-token and 256-token tests.

## Current Baseline Numbers To Compare Later

Use these as the AC 512-token long-prompt baseline:

| Field | AC 512-token performance baseline |
| --- | --- |
| Model | Qwen3-4B-Q4_K_M.gguf |
| Model size | 2.33 GiB |
| Quantization label | Q4_K_M |
| Backend | llama.cpp CUDA |
| GPU | NVIDIA GeForce RTX 2050 |
| Power state | AC |
| Battery status | Charging |
| System power profile | Performance |
| Battery level | 87% up to 91% |
| Use case | interactive |
| Goal | balanced |
| Prompt budget | 512 tokens |
| Output budget | 512 tokens |
| Actual prompt tokens | 123 |
| Actual generated tokens | 512 |
| GPU layers | 36 / 36 |
| CPU threads | 16 |
| Batch / ubatch | 512 / 512 |
| KV cache | F16 |
| Prompt processing | 1091.88 tokens/s |
| Generation | 20.45 tokens/s |
| Request wall time | 25153 ms |
| Peak GPU memory | 2639 MiB |
| Minimum free GPU memory | 1131 MiB |
| Peak GPU utilization | 93% |
| Peak GPU temperature | 66 C |
| Average GPU power | 24.04 W |
| Calibration winner | 16 threads, F16, batch 512, 50.97 effective tokens/s |

## What Should Be Tested Next

The next test should not simply push output length higher. First, test whether
the 8-thread family is still better for this larger workload.

Ideal next benchmark direction:

- Add CLI override flags for candidate profile testing, or
- Broaden calibration so it always includes previous measured winners from
  nearby workloads, then rerun 512/512.

Until that code exists, run a wider calibration:

```bash
cargo run -- --database .runtime/runtime.db calibrate \
  --model models/Qwen3-4B-Q4_K_M.gguf \
  --use-case interactive \
  --goal balanced \
  --prompt-tokens 512 \
  --output-tokens 512 \
  --candidates 15 \
  --repetitions 2
```

Then rerun the same inference and compare whether generation stays near
20 tokens/s or moves back toward the 35 tokens/s seen in 128/256 runs.

## Missing Metrics To Add Later

This run reinforces the need to add:

- CLI profile override flags for direct A/B tests
- Neighbor-workload measured profile carry-forward
- Time to first token
- Model load time
- Server startup time
- Peak GPU power
- Average GPU utilization
- GPU clock and memory clock sampling
- CPU utilization during generation
- Full system power draw if available
- Thermal throttling indicators
- Quality notes in benchmark reports
- Automatic Markdown report export from stored DB rows

## Final Status

This sixth run becomes the project's AC 512-token long-prompt baseline. It
shows that the runtime can complete the workload safely within VRAM and thermal
limits, but it also shows a major decode-speed drop and a truthfulness problem
in the generated answer.
