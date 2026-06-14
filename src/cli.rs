use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use serde::Serialize;

use crate::backend::llamacpp::{
    LlamaCppStatus, command_args as llama_command_args, server_command,
};
use crate::model::HardwareSnapshot;
use crate::policy::{ExecutionProfile, RuntimeMode, recommend};
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

    /// Recommend an execution profile for current conditions.
    Recommend {
        #[arg(long, value_enum, default_value_t = RuntimeMode::Balanced)]
        mode: RuntimeMode,

        #[arg(long)]
        json: bool,
    },

    /// Show whether llama.cpp executables can be used.
    BackendStatus {
        #[arg(long)]
        json: bool,
    },

    /// Show the llama-server process configuration without launching it.
    PlanServer {
        #[arg(long, value_name = "GGUF_PATH")]
        model: PathBuf,

        #[arg(long, value_enum, default_value_t = RuntimeMode::Balanced)]
        mode: RuntimeMode,

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
        Command::Recommend { mode, json } => {
            let snapshot = capture()?;
            let profile = recommend(&snapshot, mode);
            let store = Store::open(&database)?;
            let snapshot_id = store.insert_snapshot(&snapshot)?;
            store.insert_recommendation(snapshot_id, &profile)?;
            if json {
                println!("{}", serde_json::to_string_pretty(&profile)?);
            } else {
                println!("Mode: {:?}", profile.mode);
                println!("CPU threads: {}", profile.cpu_threads);
                println!("GPU offload: {:?}", profile.gpu_offload);
                println!("Context: {} tokens", profile.context_size);
                println!("Batch size: {}", profile.batch_size);
                println!("KV cache: {}", profile.kv_cache_type);
                for reason in profile.reasons {
                    println!("- {reason}");
                }
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
            model,
            mode,
            host,
            port,
            json,
        } => {
            let snapshot = capture()?;
            let profile = recommend(&snapshot, mode);
            let store = Store::open(&database)?;
            let snapshot_id = store.insert_snapshot(&snapshot)?;
            store.insert_recommendation(snapshot_id, &profile)?;

            let status = LlamaCppStatus::discover();
            let executable_found = status.server_path.is_some();
            let executable = status
                .server_path
                .unwrap_or_else(|| PathBuf::from("llama-server"));
            let command = server_command(&executable, &model, &host, port, &profile);
            let plan = ServerPlan {
                executable: command.get_program().to_string_lossy().into_owned(),
                executable_found,
                args: llama_command_args(&command),
                profile,
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
                println!("Policy reasons:");
                for reason in plan.profile.reasons {
                    println!("- {reason}");
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
    profile: ExecutionProfile,
}

fn capture() -> Result<HardwareSnapshot> {
    system_probe()?.capture()
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
                "- {:?}: {} (telemetry: {})",
                accelerator.kind, accelerator.name, accelerator.telemetry_available
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

fn bytes_to_gib(bytes: u64) -> f64 {
    bytes as f64 / 1024_f64.powi(3)
}
