# AI Local Runtime

An experimental hardware-aware control plane for local AI inference engines.

The first backend is `llama.cpp`. This project does not replace its tensor
kernels or model support. It probes the host, records telemetry, recommends an
execution profile, and will supervise inference processes through a stable API.

## Current milestone

- Linux CPU, memory, accelerator, thermal, and battery probing
- Direct GGUF model metadata inspection
- SQLite snapshot history
- Model- and workload-aware candidate generation
- Memory feasibility checks for weights, KV cache, and compute buffers
- Explicit `latency`, `throughput`, `efficiency`, and `balanced` objectives
- `llama-server` and `llama-bench` discovery
- Safe `llama-server` command construction

## Build

```bash
cargo build
cargo test
```

## Use

```bash
cargo run -- probe
cargo run -- probe --json
cargo run -- inspect-model --model /models/model.gguf
cargo run -- recommend \
  --model /models/model.gguf \
  --use-case interactive \
  --goal latency \
  --prompt-tokens 512 \
  --output-tokens 256
cargo run -- backend-status
cargo run -- plan-server \
  --model /models/model.gguf \
  --use-case interactive \
  --goal balanced \
  --prompt-tokens 512 \
  --output-tokens 256 \
  --concurrency 1
cargo run -- snapshots --limit 5
```

When free GPU-memory telemetry is unavailable, pass an explicit available
capacity to
generate exact layer candidates:

```bash
cargo run -- recommend \
  --model /models/model.gguf \
  --goal throughput \
  --prompt-tokens 2048 \
  --output-tokens 512 \
  --gpu-memory-mib 4096
```

The planner does not map objectives to fixed thread, layer, context, or batch
values. It derives a search space from:

- Current CPU topology, available memory, accelerators, power, and thermals
- GGUF architecture, layer count, context limit, attention shape, and file size
- Prompt length, expected output, concurrency, use case, and optimization goal

It rejects configurations that exceed calculated memory budgets, then ranks the
remaining candidates. Until `llama-bench` calibration is implemented, the
selected profile is explicitly labeled an analytical baseline rather than a
measured optimum.

By default, data is stored at
`$XDG_DATA_HOME/ai-local-runtime/runtime.db`, or under the equivalent user data
directory. Override it for experiments:

```bash
cargo run -- --database .runtime/test.db probe
```

The backend binaries are discovered from `PATH`. They can also be supplied
through `LLAMA_SERVER_PATH` and `LLAMA_BENCH_PATH`.

## Scope

Profile changes are initially applied between requests by restarting or
switching inference processes. Mid-token layer migration is deliberately out of
scope until benchmarks show that deeper `libllama` integration is worthwhile.

Only safety and search bounds are fixed. Energy efficiency and real throughput
cannot be inferred reliably from hardware names, so benchmark calibration will
replace analytical ranking as measured results become available.
