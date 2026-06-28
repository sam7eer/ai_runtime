# AC Performance 512-Token Wide Calibration Run

This file records the seventh GPU inference test for the project. It repeats
the `512 prompt / 512 output` AC performance workload, but widens calibration
from the earlier 5-candidate run to a requested 15-candidate run.

The reason for this test was simple: the previous 512-token run selected a
16-thread profile and generation dropped to about 20 tokens/s. Earlier AC runs
strongly suggested that 8-thread profiles were better, so this run checked
whether a broader calibration would rediscover the 8-thread family.

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
- Battery status during recorded snapshots: full / AC
- System power profile: performance mode
- Power profile verification after the run: `system76-power profile` reported
  `Power Profile: Performance`
- Result: CUDA inference completed successfully with full 36-layer GPU offload.
- Main inference result:
  - Prompt processing: 1066.26 tokens/s
  - Generation: 35.53 tokens/s
  - Request wall time: 14536 ms
  - Peak GPU memory: 2639 MiB
  - Minimum free GPU memory during run: 1131 MiB
  - Peak GPU utilization: 93%
  - Peak GPU temperature: 71 C
  - Average GPU power: 34.25 W

## Why This Run Matters

This run confirms that the earlier 512-token slowdown was mainly a calibration
candidate-selection issue, not a hard limit of the model or GPU.

Previous 512-token run:

- Calibration requested: 5 candidates
- Selected profile: 16 CPU threads, batch 512, F16 KV
- Generation: 20.45 tokens/s

This wider 512-token run:

- Calibration requested: 15 candidates
- Measured compatible candidates: 10
- Selected profile: 8 CPU threads, batch 512, F16 KV
- Generation: 35.53 tokens/s

That is a large recovery. The measured 512-token decode speed moved back near
the 128-token and 256-token AC runs.

## Run Conditions

- Run date: 2026-06-28
- Initial probe snapshot: #26
- Calibration snapshot: #27
- Inference snapshot: #28
- Runtime DB timestamp for inference: `2026-06-28 17:45:42` UTC
- Approx local time for inference: `2026-06-28 23:15:42` IST
- AC power: yes
- Battery status: full
- Battery at initial probe: 100%
- Battery at calibration snapshot: 100%
- Battery at inference snapshot: 100%
- System power profile: performance mode
- Power profile verification command: `system76-power profile`
- Verified profile output: `Power Profile: Performance`
- Initial system thermal reading: 57 C
- Calibration snapshot thermal reading: 59 C
- Inference snapshot thermal reading: 64 C
- Peak GPU temperature during calibration: 76 C
- Peak GPU temperature during inference: 71 C

The machine was on AC power with the battery full. This is part of the AC
performance baseline series.

## Commands Used

Hardware probe:

```bash
cargo run -- --database .runtime/runtime.db probe
```

Calibration command:

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
- Available RAM at initial probe: 8,231,915,520 bytes, about 7.7 GiB
- Available RAM at calibration snapshot: 8,226,471,936 bytes, about 7.7 GiB
- Available RAM at inference snapshot: 8,380,461,056 bytes, about 7.8 GiB
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
- Available memory at initial probe: 75,804,672 bytes, about 72 MiB
- Available memory at calibration snapshot: 68,009,984 bytes, about 65 MiB
- Available memory at inference snapshot: 64,806,912 bytes, about 62 MiB
- Telemetry available: true

Power and thermals:

- Power source: AC
- Battery status: full
- Battery level during the run: 100%
- System power profile: performance
- Highest system thermal reading before inference: 59 C
- Peak GPU temperature during calibration: 76 C
- Peak GPU temperature during inference: 71 C

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
finish reason was `length`, so the answer still did not complete naturally.

## Scheduler Decision

Selected measured profile:

- CPU threads: 8
- GPU placement: exact full offload
- GPU layers: 36 of 36
- Context size: 1024 tokens
- Parallel slots: 1
- Batch size: 512
- Physical batch size / ubatch size: 512
- KV cache type: F16
- Planning score near inference: 0.8161604973826497
- Selection basis: compatible measured CUDA calibration
- Calibration used: 64.08 effective tokens/s

Estimated memory for selected profile:

- Model bytes: 2,497,280,256
- KV cache bytes: 150,994,944
- Compute buffer bytes: 268,435,456
- Estimated total bytes: 2,916,710,656
- Estimated total size: about 2.72 GiB
- Estimated GPU bytes: 2,916,710,656
- Estimated GPU size: about 2.72 GiB
- System memory budget at inference snapshot: 7,542,414,950 bytes
- GPU memory safety budget: 3,556,874,649 bytes

Scheduler reasons:

- Preserved 1024 tokens per request across 1 slot.
- Used 8 of 16 detected logical CPU threads.
- Offloaded all 36 model layers within the estimated GPU budget.
- Derived the GPU candidate from a 90% memory safety budget.
- Reused a compatible CUDA calibration measuring 64.08 effective tokens/s.

## CUDA Calibration Results

Calibration command settings:

- Candidates requested: 15
- Compatible candidates measured: 10
- Repetitions: 2
- Prompt tokens: 512
- Output tokens: 512
- Goal: balanced
- Use case: interactive
- Model: `models/Qwen3-4B-Q4_K_M.gguf`

Measured candidates, sorted by effective throughput:

| Rank | Effective tokens/s | Prompt tokens/s | Generation tokens/s | Threads | Batch | KV cache | Peak GPU temp | Avg GPU power |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| 1 | 64.08 | 1639.82 | 32.68 | 8 | 512 | F16 | 76 C | 38.67 W |
| 2 | 47.79 | 1687.67 | 24.24 | 16 | 512 | F16 | 68 C | 29.97 W |
| 3 | 44.96 | 1527.75 | 22.81 | 16 | 256 | F16 | 71 C | 30.34 W |
| 4 | 42.53 | 1556.21 | 21.56 | 16 | 512 | Q8_0 | 72 C | 29.86 W |
| 5 | 41.28 | 596.33 | 21.38 | 16 | 32 | F16 | 72 C | 29.82 W |
| 6 | 36.25 | 1440.25 | 18.36 | 16 | 256 | Q8_0 | 72 C | 26.33 W |
| 7 | 35.63 | 569.76 | 18.39 | 16 | 32 | Q4_0 | 72 C | 27.39 W |
| 8 | 34.76 | 1445.38 | 17.59 | 16 | 256 | Q4_0 | 71 C | 26.31 W |
| 9 | 33.54 | 1547.89 | 16.96 | 16 | 512 | Q4_0 | 72 C | 25.15 W |
| 10 | 28.12 | 571.72 | 14.42 | 16 | 32 | Q8_0 | 70 C | 21.42 W |

Winner details:

- Result: selected winner
- CPU threads: 8
- GPU placement: exact 36 layers
- Batch size: 512
- Ubatch size: 512
- KV cache: F16
- Effective throughput: 64.08453528623448 tokens/s
- Prompt processing measurement:
  - Prompt tokens: 512
  - Average time: 312.309610 ms
  - Throughput: 1639.8206 tokens/s
  - Standard deviation: 37.193448 tokens/s
- Generation measurement:
  - Generated tokens: 512
  - Average time: 15732.456485 ms
  - Throughput: 32.680855 tokens/s
  - Standard deviation: 2.98879 tokens/s
- Calibration wall time for winner profile: 35069 ms
- GPU telemetry samples for winner profile: 119
- Peak GPU memory: 2787 MiB
- Minimum free GPU memory: 983 MiB
- Peak GPU utilization: 100%
- Peak GPU temperature: 76 C
- Average GPU power: 38.67 W

Calibration conclusion:

- The wider calibration found an 8-thread profile for 512/512.
- That 8-thread profile beat the previous 16-thread winner by a large margin.
- The result confirms that top-N analytical candidate selection was too narrow
  in the earlier 512-token run.
- Calibration reached 76 C and 38.67 W average GPU power for the winning
  profile, so this is a serious sustained GPU workload.

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

- The model produced a longer structured explanation.
- It covered GPU inference and CPU scheduling.
- It started section 3 but still hit the 512-token limit before completing the
  full requested explanation.
- It again hallucinated hardware details, such as RTX 2050 memory and CPU core
  layout. This is an answer-grounding issue, not a scheduler-measurement issue.

Timing:

- Prompt processing tokens: 123
- Prompt processing time: 115.357 ms
- Prompt processing throughput: 1066.2551904089046 tokens/s
- Generated tokens: 512
- Generation time: 14412.303 ms
- Generation throughput: 35.52520370963614 tokens/s
- Request wall time: 14536 ms

GPU telemetry during inference:

- Device: NVIDIA GeForce RTX 2050
- Samples collected: 77
- Peak GPU memory used: 2639 MiB
- Minimum GPU memory free: 1131 MiB
- Peak GPU utilization: 93%
- Peak GPU temperature: 71 C
- Average GPU power: 34.249480519480514 W

## Comparison With Previous 512-Token Run

Comparison file:

```text
metrics/2026-06-28-ac-performance-512-token-run.md
```

| Field | Previous 512-token run | Wide-calibration 512-token run | Change |
| --- | ---: | ---: | --- |
| Calibration candidates requested | 5 | 15 | Wider search |
| Compatible candidates measured | 5 | 10 | More coverage |
| Actual prompt tokens | 123 | 123 | Same |
| Actual generated tokens | 512 | 512 | Same |
| Selected CPU threads | 16 | 8 | Better profile found |
| Selected GPU layers | 36 / 36 | 36 / 36 | Same |
| Selected batch / ubatch | 512 / 512 | 512 / 512 | Same |
| Selected KV cache | F16 | F16 | Same |
| Context size | 1024 | 1024 | Same |
| Estimated memory | 2.72 GiB | 2.72 GiB | Same |
| Calibration effective throughput | 50.97 tok/s | 64.08 tok/s | Higher |
| Calibration generation | 25.88 tok/s | 32.68 tok/s | Higher |
| Inference prompt processing | 1091.88 tok/s | 1066.26 tok/s | Similar |
| Inference generation | 20.45 tok/s | 35.53 tok/s | Much higher |
| Request wall time | 25153 ms | 14536 ms | Much lower |
| Peak GPU memory | 2639 MiB | 2639 MiB | Same |
| Minimum free GPU memory | 1131 MiB | 1131 MiB | Same |
| Peak GPU utilization | 93% | 93% | Same |
| Peak GPU temperature | 66 C | 71 C | Hotter |
| Average GPU power during inference | 24.04 W | 34.25 W | Higher |

This confirms the earlier performance drop was not a hard context-length limit.
The scheduler needed to test the 8-thread profile family for the larger
workload.

## AC Run Progression

| Field | AC 32-token | AC 128-token | AC 256-token | AC 512-token narrow | AC 512-token wide |
| --- | ---: | ---: | ---: | ---: | ---: |
| Actual prompt tokens | 20 | 38 | 38 | 123 | 123 |
| Actual generated tokens | 32 | 128 | 256 | 512 | 512 |
| Selected CPU threads | 8 | 8 | 8 | 16 | 8 |
| Selected batch / ubatch | 64 / 64 | 128 / 128 | 256 / 256 | 512 / 512 | 512 / 512 |
| Selected KV cache | F16 | F16 | F16 | F16 | F16 |
| Inference generation | 37.38 tok/s | 36.35 tok/s | 35.96 tok/s | 20.45 tok/s | 35.53 tok/s |
| Request wall time | 929 ms | 3602 ms | 7202 ms | 25153 ms | 14536 ms |
| Peak GPU memory | 2473 MiB | 2477 MiB | 2523 MiB | 2639 MiB | 2639 MiB |
| Peak GPU utilization | 94% | 93% | 93% | 93% | 93% |
| Peak GPU temperature | 60 C | 64 C | 73 C | 66 C | 71 C |
| Average GPU power | 11.28 W | 22.58 W | 28.32 W | 24.04 W | 34.25 W |

The wide 512-token run restores the stable decode trend. Generation is now back
near the 128-token and 256-token runs.

## Approximate GPU Energy

Approximate GPU energy for the final inference request:

| Run | Avg GPU power | Wall time | Approx GPU energy |
| --- | ---: | ---: | ---: |
| AC 512-token narrow | 24.04 W | 25.153 s | 604.75 J |
| AC 512-token wide | 34.25 W | 14.536 s | 497.84 J |

Even though the wider-calibrated run used more average GPU power, it finished
much faster. The approximate GPU energy for the completed request is therefore
lower.

Per generated token:

- Narrow 512-token run: about 1.18 J/generated token
- Wide 512-token run: about 0.97 J/generated token

This estimate uses only sampled GPU average power multiplied by request wall
time. It is not full laptop energy usage.

## Derived Observations

Scheduler behavior:

- Wider calibration fixed the 512-token performance drop.
- The measured winner returned to 8 CPU threads.
- Candidate selection needs to include nearby measured winners, not just the
  current analytical top-N list.
- The scheduler should probably always keep at least one candidate from each
  successful thread family, especially 8-thread and 16-thread profiles.

Model placement:

- Full GPU offload remained the selected strategy.
- All 36 layers were placed on the NVIDIA GPU.
- No model layers remain to offload.

Memory:

- Peak observed GPU memory used during inference: 2639 MiB
- Minimum observed free GPU memory during inference: 1131 MiB
- Observed telemetry-visible memory pool during inference: about 3770 MiB
- Approx used share of telemetry-visible memory: about 70%
- Approx free share of telemetry-visible memory: about 30%

The workload remains VRAM-safe, but calibration is now using nearly 2.8 GiB peak
GPU memory and leaving less than 1 GiB free during the heaviest profile.

Temperature:

- Inference peaked at 71 C.
- Calibration peaked at 76 C.
- This is warm but still acceptable for a short controlled AC test.

Power:

- Inference average GPU power rose to 34.25 W.
- Calibration winner average GPU power was 38.67 W.
- Higher power directly bought back throughput in this run.

Quality behavior:

- The generated explanation still hallucinated hardware details.
- The runtime metrics are valid, but answer text must be grounded with verified
  probe data before we trust model-written hardware explanations.

## Conclusion

The wide 512-token calibration run is a strong success.

The practical conclusion is:

```text
The 512-token workload can sustain about 35.5 generated tokens/s on AC power
when calibration includes the 8-thread profile family. The earlier 20.45 tok/s
result was a narrow-candidate calibration miss, not a GPU limit.
```

This directly gives us the next engineering direction: improve calibration
candidate generation so previous measured winners from nearby workloads are
always included.

## Current Baseline Numbers To Compare Later

Use these as the corrected AC 512-token wide-calibration baseline:

| Field | AC 512-token wide-calibration baseline |
| --- | --- |
| Model | Qwen3-4B-Q4_K_M.gguf |
| Model size | 2.33 GiB |
| Quantization label | Q4_K_M |
| Backend | llama.cpp CUDA |
| GPU | NVIDIA GeForce RTX 2050 |
| Power state | AC |
| Battery status | Full |
| System power profile | Performance |
| Battery level | 100% |
| Use case | interactive |
| Goal | balanced |
| Prompt budget | 512 tokens |
| Output budget | 512 tokens |
| Actual prompt tokens | 123 |
| Actual generated tokens | 512 |
| GPU layers | 36 / 36 |
| CPU threads | 8 |
| Batch / ubatch | 512 / 512 |
| KV cache | F16 |
| Prompt processing | 1066.26 tokens/s |
| Generation | 35.53 tokens/s |
| Request wall time | 14536 ms |
| Peak GPU memory | 2639 MiB |
| Minimum free GPU memory | 1131 MiB |
| Peak GPU utilization | 93% |
| Peak GPU temperature | 71 C |
| Average GPU power | 34.25 W |
| Calibration winner | 8 threads, F16, batch 512, 64.08 effective tokens/s |

## What Should Be Tested Next

The next best project work is code, not another bigger manual test:

1. Add calibration candidate carry-forward from nearby measured workloads.
2. Add direct profile override flags for exact A/B tests.
3. Add grounded prompt generation so model-written explanations use real probe
   facts instead of inventing hardware specs.

After that, rerun the 512/512 test and verify that the scheduler finds the
8-thread profile without needing an unusually wide manual calibration.

## Missing Metrics To Add Later

This run reinforces the need to add:

- Neighbor-workload measured profile carry-forward
- CLI profile override flags
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

This seventh run becomes the corrected AC 512-token baseline. It proves that
the project is learning exactly the right thing from calibration: the scheduler
needs a better candidate policy, and the runtime metrics are already good enough
to reveal that.
