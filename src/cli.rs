use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::{Context, Result, bail};
use clap::{Args, Parser, Subcommand};
use serde::Serialize;

use crate::backend::llamacpp::{
    BenchmarkRun, InferenceOptions, InferenceResult, LlamaCppStatus, LlamaCppVariant,
    command_args as llama_command_args, run_benchmark, run_inference, server_command,
};
use crate::gguf::{ModelMetadata, inspect as inspect_gguf};
use crate::model::HardwareSnapshot;
use crate::policy::{
    ExecutionProfile, OptimizationGoal, PlanningDecision, PlanningOverrides, UseCase, WorkloadSpec,
    plan,
};
use crate::probe::system_probe;
use crate::store::Store;

#[derive(Debug, Parser)]
#[command(name = "airuntime", version, about)]
pub struct Cli {
    #[arg(long, global = true, value_name = "PATH")]
    database: Option<PathBuf>,

    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Capture and persist the current hardware state.
    Probe {
        #[arg(long)]
        json: bool,

        #[arg(long)]
        no_store: bool,
    },

    /// Inspect the scheduling metadata in a GGUF model.
    InspectModel {
        #[arg(long, value_name = "GGUF_PATH")]
        model: PathBuf,

        #[arg(long)]
        json: bool,
    },

    /// Plan candidates for a specific model, workload, and objective.
    Recommend {
        #[command(flatten)]
        planning: PlanningArgs,

        #[arg(long)]
        show_candidates: bool,

        #[arg(long)]
        json: bool,
    },

    /// Show whether llama.cpp executables can be used.
    BackendStatus {
        #[arg(long)]
        json: bool,
    },

    /// Show the model-aware llama-server configuration without launching it.
    PlanServer {
        #[command(flatten)]
        planning: PlanningArgs,

        #[arg(long, default_value = "127.0.0.1")]
        host: String,

        #[arg(long, default_value_t = 8080)]
        port: u16,

        #[arg(long)]
        json: bool,
    },

    /// Launch llama-server, run one prompt, record telemetry, and stop it.
    Infer {
        #[command(flatten)]
        planning: PlanningArgs,

        #[arg(long)]
        prompt: String,

        #[arg(long, default_value = "127.0.0.1")]
        host: String,

        #[arg(long, default_value_t = 8080)]
        port: u16,

        /// Disable model reasoning through the chat-template parameters.
        #[arg(long)]
        disable_thinking: bool,

        #[arg(long, default_value_t = 120)]
        startup_timeout_seconds: u64,

        #[arg(long, default_value_t = 300)]
        request_timeout_seconds: u64,

        #[arg(long)]
        json: bool,
    },

    /// Benchmark top scheduler candidates on CUDA and persist the measurements.
    Calibrate {
        #[command(flatten)]
        planning: PlanningArgs,

        #[arg(long, default_value_t = 3)]
        candidates: usize,

        #[arg(long, default_value_t = 3)]
        repetitions: u32,

        #[arg(long)]
        json: bool,
    },

    /// Show recently persisted hardware snapshots.
    Snapshots {
        #[arg(long, default_value_t = 5)]
        limit: usize,

        #[arg(long)]
        json: bool,
    },
}

#[derive(Debug, Clone, Args)]
struct PlanningArgs {
    #[arg(long, value_name = "GGUF_PATH")]
    model: PathBuf,

    #[arg(long, value_enum, default_value_t = UseCase::Interactive)]
    use_case: UseCase,

    #[arg(long, value_enum, default_value_t = OptimizationGoal::Balanced)]
    goal: OptimizationGoal,

    #[arg(long)]
    prompt_tokens: u32,

    #[arg(long)]
    output_tokens: u32,

    #[arg(long, default_value_t = 1)]
    concurrency: u32,

    /// Override unavailable free GPU-memory telemetry.
    #[arg(long, value_name = "MIB")]
    gpu_memory_mib: Option<u64>,
}

pub fn run(cli: Cli) -> Result<()> {
    let database = cli.database.unwrap_or_else(default_database_path);

    match cli.command {
        Command::Probe { json, no_store } => {
            let snapshot = capture()?;
            if !no_store {
                let id = Store::open(&database)?.insert_snapshot(&snapshot)?;
                if !json {
                    println!("Stored snapshot #{id} in {}", database.display());
                }
            }
            print_snapshot(&snapshot, json)?;
        }
        Command::InspectModel { model, json } => {
            let model = inspect_gguf(&model)?;
            if json {
                println!("{}", serde_json::to_string_pretty(&model)?);
            } else {
                print_model(&model);
            }
        }
        Command::Recommend {
            planning,
            show_candidates,
            json,
        } => {
            let (snapshot, decision) = create_calibrated_decision(&planning, &database)?;
            let _ = persist_decision(&database, &snapshot, &decision)?;
            if json {
                println!("{}", serde_json::to_string_pretty(&decision)?);
            } else {
                print_decision(&decision, show_candidates);
            }
        }
        Command::BackendStatus { json } => {
            let status = LlamaCppStatus::discover();
            if json {
                println!("{}", serde_json::to_string_pretty(&status)?);
            } else {
                println!(
                    "llama-server: {}",
                    display_optional_path(status.server_path.as_ref())
                );
                println!("Server backend: {}", display_variant(status.server_variant));
                println!(
                    "llama-bench: {}",
                    display_optional_path(status.bench_path.as_ref())
                );
                println!("Bench backend: {}", display_variant(status.bench_variant));
                println!("Ready for serving: {}", status.ready_for_serving);
                println!("Ready for benchmarking: {}", status.ready_for_benchmarking);
            }
        }
        Command::PlanServer {
            planning,
            host,
            port,
            json,
        } => {
            let (snapshot, decision) = create_calibrated_decision(&planning, &database)?;
            let _ = persist_decision(&database, &snapshot, &decision)?;

            let status = LlamaCppStatus::discover();
            let executable_found = status.server_path.is_some();
            let executable = status
                .server_path
                .unwrap_or_else(|| PathBuf::from("llama-server"));
            let command = server_command(
                &executable,
                &decision.model.path,
                &host,
                port,
                &decision.selected,
            );
            let plan = ServerPlan {
                executable: command.get_program().to_string_lossy().into_owned(),
                executable_found,
                args: llama_command_args(&command),
                selected_profile: decision.selected,
                candidate_count: decision.candidates.len(),
                calibration_required: decision.calibration_required,
                planning_notes: decision.notes,
            };

            if json {
                println!("{}", serde_json::to_string_pretty(&plan)?);
            } else {
                println!("Executable: {}", plan.executable);
                println!("Found: {}", plan.executable_found);
                println!("Arguments:");
                for argument in plan.args {
                    println!("- {argument}");
                }
                println!("Candidate configurations: {}", plan.candidate_count);
                println!("Calibration required: {}", plan.calibration_required);
                println!("Planning notes:");
                for note in plan.planning_notes {
                    println!("- {note}");
                }
            }
        }
        Command::Infer {
            planning,
            prompt,
            host,
            port,
            disable_thinking,
            startup_timeout_seconds,
            request_timeout_seconds,
            json,
        } => {
            let (snapshot, decision) = create_calibrated_decision(&planning, &database)?;
            let status = LlamaCppStatus::discover();
            let server_path = status
                .server_path
                .context("llama-server was not found; set LLAMA_SERVER_PATH")?;
            if status.server_variant != Some(LlamaCppVariant::Cuda) {
                bail!(
                    "the selected llama-server is not a CUDA build: {}; set LLAMA_SERVER_PATH to a CUDA build",
                    server_path.display()
                );
            }

            let (snapshot_id, decision_id) = persist_decision(&database, &snapshot, &decision)?;
            let result = run_inference(&InferenceOptions {
                server_path: &server_path,
                model_path: &decision.model.path,
                host: &host,
                port,
                profile: &decision.selected,
                prompt: &prompt,
                max_tokens: planning.output_tokens,
                disable_thinking,
                startup_timeout: Duration::from_secs(startup_timeout_seconds),
                request_timeout: Duration::from_secs(request_timeout_seconds),
            })?;
            Store::open(&database)?.insert_inference_run(
                snapshot_id,
                decision_id,
                &decision.model.path,
                &result,
            )?;

            if json {
                println!("{}", serde_json::to_string_pretty(&result)?);
            } else {
                print_inference_result(&result);
            }
        }
        Command::Calibrate {
            planning,
            candidates,
            repetitions,
            json,
        } => {
            if candidates == 0 {
                bail!("candidate count must be greater than zero");
            }
            let (snapshot, decision) = create_decision(&planning)?;
            let status = LlamaCppStatus::discover();
            let bench_path = status
                .bench_path
                .context("llama-bench was not found; set LLAMA_BENCH_PATH")?;
            if status.bench_variant != Some(LlamaCppVariant::Cuda) {
                bail!(
                    "the selected llama-bench is not a CUDA build: {}; set LLAMA_BENCH_PATH to a CUDA build",
                    bench_path.display()
                );
            }

            let (snapshot_id, decision_id) = persist_decision(&database, &snapshot, &decision)?;
            let store = Store::open(&database)?;
            let mut runs = Vec::new();
            for profile in decision.candidates.iter().take(candidates.min(10)) {
                let run = run_benchmark(
                    &bench_path,
                    &decision.model.path,
                    profile,
                    planning.prompt_tokens,
                    planning.output_tokens,
                    repetitions,
                )?;
                runs.push(run);
            }
            for run in &runs {
                store.insert_benchmark_run(snapshot_id, decision_id, &decision.model.path, run)?;
            }
            runs.sort_by(|left, right| {
                right
                    .effective_tokens_per_second
                    .total_cmp(&left.effective_tokens_per_second)
            });
            let selected = runs
                .first()
                .context("the scheduler produced no candidates to benchmark")?
                .profile
                .clone();
            let calibration = CalibrationResult {
                selection_basis:
                    "measured effective throughput for the requested prompt/output token mix".into(),
                selected,
                runs,
            };

            if json {
                println!("{}", serde_json::to_string_pretty(&calibration)?);
            } else {
                print_calibration(&calibration);
            }
        }
        Command::Snapshots { limit, json } => {
            let snapshots = Store::open(&database)?.recent_snapshots(limit)?;
            if json {
                println!("{}", serde_json::to_string_pretty(&snapshots)?);
            } else if snapshots.is_empty() {
                println!("No snapshots stored in {}", database.display());
            } else {
                for snapshot in snapshots {
                    println!(
                        "{}  {}  {:.1} GiB available  {} accelerator(s)",
                        snapshot.captured_at,
                        snapshot.hostname,
                        bytes_to_gib(snapshot.memory.available_bytes),
                        snapshot.accelerators.len()
                    );
                }
            }
        }
    }

    Ok(())
}

#[derive(Debug, Serialize)]
struct ServerPlan {
    executable: String,
    executable_found: bool,
    args: Vec<String>,
    selected_profile: ExecutionProfile,
    candidate_count: usize,
    calibration_required: bool,
    planning_notes: Vec<String>,
}

#[derive(Debug, Serialize)]
struct CalibrationResult {
    selection_basis: String,
    selected: ExecutionProfile,
    runs: Vec<BenchmarkRun>,
}

fn create_decision(planning: &PlanningArgs) -> Result<(HardwareSnapshot, PlanningDecision)> {
    let snapshot = capture()?;
    let model = inspect_gguf(&planning.model)?;
    let workload = WorkloadSpec {
        use_case: planning.use_case,
        optimization_goal: planning.goal,
        prompt_tokens: planning.prompt_tokens,
        output_tokens: planning.output_tokens,
        concurrency: planning.concurrency,
    };
    let gpu_memory_bytes = planning
        .gpu_memory_mib
        .map(|mib| {
            mib.checked_mul(1024 * 1024)
                .context("GPU memory override is too large")
        })
        .transpose()?;
    let overrides = PlanningOverrides { gpu_memory_bytes };
    let decision = plan(&snapshot, model, workload, overrides)?;
    Ok((snapshot, decision))
}

fn create_calibrated_decision(
    planning: &PlanningArgs,
    database: &Path,
) -> Result<(HardwareSnapshot, PlanningDecision)> {
    let (snapshot, mut decision) = create_decision(planning)?;
    if let Some(calibration) =
        Store::open(database)?.latest_compatible_calibration(&snapshot, &decision)?
    {
        decision.apply_measured_profile(
            &calibration.profile,
            calibration.effective_tokens_per_second,
        );
    }
    Ok((snapshot, decision))
}

fn persist_decision(
    database: &Path,
    snapshot: &HardwareSnapshot,
    decision: &PlanningDecision,
) -> Result<(i64, i64)> {
    let store = Store::open(database)?;
    let snapshot_id = store.insert_snapshot(snapshot)?;
    let decision_id = store.insert_planning_decision(snapshot_id, decision)?;
    Ok((snapshot_id, decision_id))
}

fn capture() -> Result<HardwareSnapshot> {
    system_probe()?.capture()
}

fn print_model(model: &ModelMetadata) {
    println!(
        "Model: {}",
        model.name.as_deref().unwrap_or("<unnamed model>")
    );
    println!("Path: {}", model.path.display());
    println!("Architecture: {}", model.architecture);
    println!("Layers: {}", model.block_count);
    println!("File size: {:.2} GiB", bytes_to_gib(model.file_size_bytes));
    println!(
        "Maximum context: {}",
        display_optional_number(model.context_length)
    );
    println!(
        "Embedding length: {}",
        display_optional_number(model.embedding_length)
    );
    println!(
        "Attention heads: {}",
        display_optional_number(model.attention_head_count)
    );
    println!(
        "KV heads: {}",
        display_optional_number(model.attention_head_count_kv)
    );
}

fn print_decision(decision: &PlanningDecision, show_candidates: bool) {
    println!(
        "Model: {} ({} layers, {:.2} GiB)",
        decision
            .model
            .name
            .as_deref()
            .unwrap_or(&decision.model.architecture),
        decision.model.block_count,
        bytes_to_gib(decision.model.file_size_bytes)
    );
    println!(
        "Workload: {:?}, {:?}, {}+{} tokens, concurrency {}",
        decision.workload.use_case,
        decision.workload.optimization_goal,
        decision.workload.prompt_tokens,
        decision.workload.output_tokens,
        decision.workload.concurrency
    );
    println!("Candidates: {}", decision.candidates.len());
    println!("Selection basis: {}", decision.selection_basis);
    println!("Calibration required: {}", decision.calibration_required);
    let label = if decision.calibration_required {
        "Selected baseline"
    } else {
        "Selected measured profile"
    };
    print_profile(label, &decision.selected);
    for note in &decision.notes {
        println!("- {note}");
    }

    if show_candidates {
        println!("Top candidates:");
        for (index, candidate) in decision.candidates.iter().take(10).enumerate() {
            println!(
                "{}. score {:.3}, {:?}, {} threads, batch {}, {:?}",
                index + 1,
                candidate.planning_score,
                candidate.gpu_placement,
                candidate.cpu_threads,
                candidate.batch_size,
                candidate.kv_cache_type
            );
        }
    }
}

fn print_profile(label: &str, profile: &ExecutionProfile) {
    println!("{label}:");
    println!("- CPU threads: {}", profile.cpu_threads);
    println!("- GPU placement: {:?}", profile.gpu_placement);
    println!("- Total context: {}", profile.context_size);
    println!("- Parallel slots: {}", profile.parallel_slots);
    println!(
        "- Batch / physical batch: {} / {}",
        profile.batch_size, profile.ubatch_size
    );
    println!("- KV cache: {:?}", profile.kv_cache_type);
    if let Some(total) = profile.memory.estimated_total_bytes {
        println!("- Estimated memory: {:.2} GiB", bytes_to_gib(total));
    } else {
        println!("- Estimated memory: incomplete");
    }
}

fn print_inference_result(result: &InferenceResult) {
    println!("Backend: {}", result.backend);
    println!("Model: {}", result.model);
    println!("Response: {}", result.response);
    if let Some(reasoning) = &result.reasoning
        && !reasoning.is_empty()
    {
        println!("Reasoning: {reasoning}");
    }
    println!(
        "Tokens: {} prompt, {} generated, {} total",
        result.usage.prompt_tokens, result.usage.completion_tokens, result.usage.total_tokens
    );
    if let Some(timings) = &result.timings {
        println!(
            "Throughput: {:.2} prompt tokens/s, {:.2} generated tokens/s",
            timings.prompt_per_second, timings.predicted_per_second
        );
    }
    println!("Request wall time: {} ms", result.wall_time_ms);
    println!("GPU telemetry samples: {}", result.gpu.sample_count);
    if let Some(memory) = result.gpu.peak_memory_used_mib {
        println!("Peak GPU memory: {memory:.0} MiB");
    }
    if let Some(utilization) = result.gpu.peak_utilization_percent {
        println!("Peak GPU utilization: {utilization:.0}%");
    }
    if let Some(temperature) = result.gpu.peak_temperature_celsius {
        println!("Peak GPU temperature: {temperature:.0} C");
    }
    if let Some(power) = result.gpu.average_power_watts {
        println!("Average GPU power: {power:.1} W");
    }
}

fn print_calibration(calibration: &CalibrationResult) {
    println!("Selection basis: {}", calibration.selection_basis);
    println!("Measured candidates: {}", calibration.runs.len());
    for (index, run) in calibration.runs.iter().enumerate() {
        let prompt_rate = run
            .measurements
            .iter()
            .find(|measurement| measurement.n_prompt > 0)
            .map(|measurement| measurement.avg_ts)
            .unwrap_or_default();
        let generation_rate = run
            .measurements
            .iter()
            .find(|measurement| measurement.n_gen > 0)
            .map(|measurement| measurement.avg_ts)
            .unwrap_or_default();
        println!(
            "{}. effective {:.2} tokens/s, prompt {:.2}, generation {:.2}, {:?}, {} threads, batch {}",
            index + 1,
            run.effective_tokens_per_second,
            prompt_rate,
            generation_rate,
            run.profile.gpu_placement,
            run.profile.cpu_threads,
            run.profile.batch_size
        );
    }
    print_profile("Measured selection", &calibration.selected);
}

fn print_snapshot(snapshot: &HardwareSnapshot, json: bool) -> Result<()> {
    if json {
        println!("{}", serde_json::to_string_pretty(snapshot)?);
        return Ok(());
    }

    println!("Host: {} ({})", snapshot.hostname, snapshot.os);
    println!(
        "CPU: {} ({} logical, {} physical)",
        snapshot.cpu.model,
        snapshot.cpu.logical_cores,
        snapshot
            .cpu
            .physical_cores
            .map_or_else(|| "unknown".into(), |cores| cores.to_string())
    );
    println!(
        "Memory: {:.1} GiB available / {:.1} GiB total",
        bytes_to_gib(snapshot.memory.available_bytes),
        bytes_to_gib(snapshot.memory.total_bytes)
    );
    if snapshot.accelerators.is_empty() {
        println!("Accelerators: none discovered");
    } else {
        println!("Accelerators:");
        for accelerator in &snapshot.accelerators {
            println!(
                "- {:?}: {} (available memory: {}, telemetry: {})",
                accelerator.kind,
                accelerator.name,
                accelerator.available_memory_bytes.map_or_else(
                    || "unknown".into(),
                    |bytes| format!("{:.1} GiB", bytes_to_gib(bytes))
                ),
                accelerator.telemetry_available
            );
        }
    }
    if let Some(temperature) = snapshot.highest_temperature_celsius() {
        println!("Highest temperature: {temperature:.1} C");
    }
    match snapshot.power.on_ac {
        Some(true) => println!("Power: AC"),
        Some(false) => println!(
            "Power: battery{}",
            snapshot
                .power
                .battery_percent
                .map_or_else(String::new, |value| format!(" ({value:.0}%)"))
        ),
        None => println!("Power: unknown"),
    }

    Ok(())
}

fn default_database_path() -> PathBuf {
    dirs::data_local_dir()
        .context("unable to determine the local data directory")
        .unwrap_or_else(|_| PathBuf::from(".runtime"))
        .join("ai-local-runtime")
        .join("runtime.db")
}

fn display_optional_path(path: Option<&PathBuf>) -> String {
    path.map_or_else(|| "not found".into(), |path| path.display().to_string())
}

fn display_variant(variant: Option<LlamaCppVariant>) -> &'static str {
    match variant {
        Some(LlamaCppVariant::Cuda) => "CUDA",
        Some(LlamaCppVariant::Vulkan) => "Vulkan",
        Some(LlamaCppVariant::Other) => "other",
        None => "not found",
    }
}

fn display_optional_number(value: Option<u32>) -> String {
    value.map_or_else(|| "unknown".into(), |value| value.to_string())
}

fn bytes_to_gib(bytes: u64) -> f64 {
    bytes as f64 / 1024_f64.powi(3)
}
