use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use clap::{Args, Parser, Subcommand};
use serde::Serialize;

use crate::backend::llamacpp::{
    LlamaCppStatus, command_args as llama_command_args, server_command,
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
            let (snapshot, decision) = create_decision(&planning)?;
            persist_decision(&database, &snapshot, &decision)?;
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
                println!(
                    "llama-bench: {}",
                    display_optional_path(status.bench_path.as_ref())
                );
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
            let (snapshot, decision) = create_decision(&planning)?;
            persist_decision(&database, &snapshot, &decision)?;

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

fn persist_decision(
    database: &Path,
    snapshot: &HardwareSnapshot,
    decision: &PlanningDecision,
) -> Result<()> {
    let store = Store::open(database)?;
    let snapshot_id = store.insert_snapshot(snapshot)?;
    store.insert_planning_decision(snapshot_id, decision)?;
    Ok(())
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
    print_profile("Selected baseline", &decision.selected);
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

fn display_optional_number(value: Option<u32>) -> String {
    value.map_or_else(|| "unknown".into(), |value| value.to_string())
}

fn bytes_to_gib(bytes: u64) -> f64 {
    bytes as f64 / 1024_f64.powi(3)
}
