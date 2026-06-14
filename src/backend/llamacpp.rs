use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

use serde::Serialize;

use crate::policy::{ExecutionProfile, GpuOffload};

#[derive(Debug, Clone, Serialize)]
pub struct LlamaCppStatus {
    pub server_path: Option<PathBuf>,
    pub bench_path: Option<PathBuf>,
    pub ready_for_serving: bool,
    pub ready_for_benchmarking: bool,
}

impl LlamaCppStatus {
    pub fn discover() -> Self {
        let server_path = discover_binary("LLAMA_SERVER_PATH", "llama-server");
        let bench_path = discover_binary("LLAMA_BENCH_PATH", "llama-bench");

        Self {
            ready_for_serving: server_path.is_some(),
            ready_for_benchmarking: bench_path.is_some(),
            server_path,
            bench_path,
        }
    }
}

pub fn server_command(
    server_path: &Path,
    model_path: &Path,
    host: &str,
    port: u16,
    profile: &ExecutionProfile,
) -> Command {
    let mut command = Command::new(server_path);
    command
        .arg("--model")
        .arg(model_path)
        .arg("--host")
        .arg(host)
        .arg("--port")
        .arg(port.to_string())
        .arg("--threads")
        .arg(profile.cpu_threads.to_string())
        .arg("--ctx-size")
        .arg(profile.context_size.to_string())
        .arg("--batch-size")
        .arg(profile.batch_size.to_string())
        .arg("--cache-type-k")
        .arg(&profile.kv_cache_type)
        .arg("--cache-type-v")
        .arg(&profile.kv_cache_type);

    match profile.gpu_offload {
        GpuOffload::Disabled => {
            command.arg("--n-gpu-layers").arg("0");
        }
        GpuOffload::Conservative => {
            command.arg("--n-gpu-layers").arg("16");
        }
        GpuOffload::MaximumFitting => {
            // A high layer count asks llama.cpp to offload every layer that fits.
            command.arg("--n-gpu-layers").arg("999");
        }
    }

    command
}

fn discover_binary(env_name: &str, executable: &str) -> Option<PathBuf> {
    if let Some(path) = env::var_os(env_name).map(PathBuf::from)
        && is_executable_file(&path)
    {
        return Some(path);
    }

    env::var_os("PATH")
        .as_deref()
        .map(env::split_paths)
        .into_iter()
        .flatten()
        .map(|directory| directory.join(executable))
        .find(|path| is_executable_file(path))
}

fn is_executable_file(path: &Path) -> bool {
    path.is_file()
}

pub fn command_args(command: &Command) -> Vec<String> {
    command
        .get_args()
        .map(|value| value.to_string_lossy().into_owned())
        .collect()
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use crate::policy::{ExecutionProfile, GpuOffload, RuntimeMode};

    use super::{command_args, server_command};

    #[test]
    fn builds_a_server_command_from_a_profile() {
        let profile = ExecutionProfile {
            mode: RuntimeMode::Balanced,
            cpu_threads: 8,
            gpu_offload: GpuOffload::MaximumFitting,
            context_size: 4096,
            batch_size: 256,
            kv_cache_type: "f16".into(),
            reasons: vec![],
        };

        let command = server_command(
            Path::new("/opt/llama-server"),
            Path::new("/models/test.gguf"),
            "127.0.0.1",
            8080,
            &profile,
        );
        let args = command_args(&command);

        assert_eq!(command.get_program(), "/opt/llama-server");
        assert!(
            args.windows(2)
                .any(|pair| pair == ["--model", "/models/test.gguf"])
        );
        assert!(args.windows(2).any(|pair| pair == ["--threads", "8"]));
        assert!(
            args.windows(2)
                .any(|pair| pair == ["--n-gpu-layers", "999"])
        );
    }
}
