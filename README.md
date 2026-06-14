# AI Local Runtime

An experimental hardware-aware control plane for local AI inference engines.

The first backend is `llama.cpp`. This project does not replace its tensor
kernels or model support. It probes the host, records telemetry, recommends an
execution profile, and will supervise inference processes through a stable API.

## Current milestone

- Linux CPU, memory, accelerator, thermal, and battery probing
- SQLite snapshot history
- Explainable `performance`, `balanced`, `battery`, and `background` policies
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
cargo run -- recommend --mode balanced
cargo run -- backend-status
cargo run -- plan-server --model /models/model.gguf
cargo run -- snapshots --limit 5
```

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
