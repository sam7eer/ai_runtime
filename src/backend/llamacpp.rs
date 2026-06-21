use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::thread;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::backend::nvidia::{GpuTelemetrySummary, NvidiaSampler};
use crate::policy::{ExecutionProfile, GpuPlacement};

const HEALTH_POLL_INTERVAL: Duration = Duration::from_millis(250);
const TELEMETRY_INTERVAL: Duration = Duration::from_millis(200);

#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum LlamaCppVariant {
    Cuda,
    Vulkan,
    Other,
}

#[derive(Debug, Clone, Serialize)]
pub struct LlamaCppStatus {
    pub server_path: Option<PathBuf>,
    pub server_variant: Option<LlamaCppVariant>,
    pub bench_path: Option<PathBuf>,
    pub bench_variant: Option<LlamaCppVariant>,
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
            server_variant: server_path.as_deref().map(infer_variant),
            bench_variant: bench_path.as_deref().map(infer_variant),
            server_path,
            bench_path,
        }
    }
}

#[derive(Debug, Clone)]
pub struct InferenceOptions<'a> {
    pub server_path: &'a Path,
    pub model_path: &'a Path,
    pub host: &'a str,
    pub port: u16,
    pub profile: &'a ExecutionProfile,
    pub prompt: &'a str,
    pub max_tokens: u32,
    pub disable_thinking: bool,
    pub startup_timeout: Duration,
    pub request_timeout: Duration,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct InferenceResult {
    pub backend: String,
    pub model: String,
    pub response: String,
    pub reasoning: Option<String>,
    pub finish_reason: Option<String>,
    pub usage: TokenUsage,
    pub timings: Option<InferenceTimings>,
    pub wall_time_ms: u64,
    pub gpu: GpuTelemetrySummary,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct TokenUsage {
    #[serde(default)]
    pub prompt_tokens: u64,
    #[serde(default)]
    pub completion_tokens: u64,
    #[serde(default)]
    pub total_tokens: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct InferenceTimings {
    pub prompt_n: u64,
    pub prompt_ms: f64,
    pub prompt_per_second: f64,
    pub predicted_n: u64,
    pub predicted_ms: f64,
    pub predicted_per_second: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct BenchmarkRun {
    pub backend: String,
    pub profile: ExecutionProfile,
    pub measurements: Vec<BenchmarkMeasurement>,
    pub effective_tokens_per_second: f64,
    pub wall_time_ms: u64,
    pub gpu: GpuTelemetrySummary,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct BenchmarkMeasurement {
    pub n_prompt: u32,
    pub n_gen: u32,
    pub avg_ns: u64,
    pub avg_ts: f64,
    pub stddev_ts: f64,
}

#[derive(Debug, Deserialize)]
struct ChatCompletion {
    model: String,
    #[serde(default)]
    choices: Vec<ChatChoice>,
    #[serde(default)]
    usage: TokenUsage,
    timings: Option<InferenceTimings>,
}

#[derive(Debug, Deserialize)]
struct ChatChoice {
    finish_reason: Option<String>,
    message: ChatMessage,
}

#[derive(Debug, Deserialize)]
struct ChatMessage {
    #[serde(default)]
    content: String,
    reasoning_content: Option<String>,
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
        .arg("--ubatch-size")
        .arg(profile.ubatch_size.to_string())
        .arg("--parallel")
        .arg(profile.parallel_slots.to_string())
        .arg("--cache-type-k")
        .arg(profile.kv_cache_type.llama_cpp_value())
        .arg("--cache-type-v")
        .arg(profile.kv_cache_type.llama_cpp_value())
        .arg("--n-gpu-layers")
        .arg(profile.gpu_placement.llama_cpp_value());

    command
}

pub fn run_inference(options: &InferenceOptions<'_>) -> Result<InferenceResult> {
    let mut command = server_command(
        options.server_path,
        options.model_path,
        options.host,
        options.port,
        options.profile,
    );
    command
        .arg("--no-ui")
        .arg("--metrics")
        .arg("--log-disable")
        .stdout(Stdio::null())
        .stderr(Stdio::null());

    let sampler = NvidiaSampler::start(TELEMETRY_INTERVAL);
    let mut server = ManagedChild::spawn(&mut command).with_context(|| {
        format!(
            "failed to start llama-server at {}",
            options.server_path.display()
        )
    })?;
    let base_url = format!("http://{}:{}", options.host, options.port);
    wait_until_ready(&base_url, &mut server, options.startup_timeout)?;

    let agent: ureq::Agent = ureq::Agent::config_builder()
        .timeout_global(Some(options.request_timeout))
        .build()
        .into();
    let mut body = json!({
        "model": "local",
        "messages": [{"role": "user", "content": options.prompt}],
        "max_tokens": options.max_tokens,
        "stream": false
    });
    if options.disable_thinking {
        body["chat_template_kwargs"] = json!({"enable_thinking": false});
    }

    let request_started = Instant::now();
    let completion = agent
        .post(format!("{base_url}/v1/chat/completions"))
        .send_json(&body)
        .context("llama-server rejected the inference request")?
        .body_mut()
        .read_json::<ChatCompletion>()
        .context("llama-server returned an invalid chat completion")?;
    let wall_time_ms = request_started
        .elapsed()
        .as_millis()
        .try_into()
        .unwrap_or(u64::MAX);
    let choice = completion
        .choices
        .into_iter()
        .next()
        .context("llama-server returned no completion choices")?;
    server.stop();
    let gpu = sampler.finish();

    Ok(InferenceResult {
        backend: format!("{:?}", infer_variant(options.server_path)).to_lowercase(),
        model: completion.model,
        response: choice.message.content,
        reasoning: choice.message.reasoning_content,
        finish_reason: choice.finish_reason,
        usage: completion.usage,
        timings: completion.timings,
        wall_time_ms,
        gpu,
    })
}

pub fn run_benchmark(
    bench_path: &Path,
    model_path: &Path,
    profile: &ExecutionProfile,
    prompt_tokens: u32,
    output_tokens: u32,
    repetitions: u32,
) -> Result<BenchmarkRun> {
    if repetitions == 0 {
        bail!("benchmark repetitions must be greater than zero");
    }

    let gpu_layers = match profile.gpu_placement {
        GpuPlacement::CpuOnly => "0".into(),
        GpuPlacement::Exact { layers } => layers.to_string(),
        GpuPlacement::Auto => "-1".into(),
    };
    let mut command = Command::new(bench_path);
    command
        .arg("--model")
        .arg(model_path)
        .arg("--n-prompt")
        .arg(prompt_tokens.to_string())
        .arg("--n-gen")
        .arg(output_tokens.to_string())
        .arg("--batch-size")
        .arg(profile.batch_size.to_string())
        .arg("--ubatch-size")
        .arg(profile.ubatch_size.to_string())
        .arg("--threads")
        .arg(profile.cpu_threads.to_string())
        .arg("--cache-type-k")
        .arg(profile.kv_cache_type.llama_cpp_value())
        .arg("--cache-type-v")
        .arg(profile.kv_cache_type.llama_cpp_value())
        .arg("--n-gpu-layers")
        .arg(gpu_layers)
        .arg("--repetitions")
        .arg(repetitions.to_string())
        .arg("--output")
        .arg("json");

    let sampler = NvidiaSampler::start(TELEMETRY_INTERVAL);
    let started = Instant::now();
    let output = command
        .output()
        .with_context(|| format!("failed to start llama-bench at {}", bench_path.display()))?;
    let wall_time_ms = started.elapsed().as_millis().try_into().unwrap_or(u64::MAX);
    let gpu = sampler.finish();
    if !output.status.success() {
        bail!(
            "llama-bench failed with status {}: {}",
            output.status,
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }

    let measurements: Vec<BenchmarkMeasurement> = serde_json::from_slice(&output.stdout)
        .context("llama-bench returned invalid JSON measurements")?;
    let prompt_rate = measurements
        .iter()
        .find(|measurement| measurement.n_prompt > 0 && measurement.n_gen == 0)
        .map(|measurement| measurement.avg_ts)
        .context("llama-bench did not return a prompt-processing measurement")?;
    let generation_rate = measurements
        .iter()
        .find(|measurement| measurement.n_gen > 0 && measurement.n_prompt == 0)
        .map(|measurement| measurement.avg_ts)
        .context("llama-bench did not return a token-generation measurement")?;
    let effective_tokens_per_second =
        effective_throughput(prompt_tokens, output_tokens, prompt_rate, generation_rate)?;

    Ok(BenchmarkRun {
        backend: format!("{:?}", infer_variant(bench_path)).to_lowercase(),
        profile: profile.clone(),
        measurements,
        effective_tokens_per_second,
        wall_time_ms,
        gpu,
    })
}

fn effective_throughput(
    prompt_tokens: u32,
    output_tokens: u32,
    prompt_rate: f64,
    generation_rate: f64,
) -> Result<f64> {
    if prompt_rate <= 0.0 || generation_rate <= 0.0 {
        bail!("llama-bench returned a non-positive throughput");
    }
    let elapsed_seconds =
        f64::from(prompt_tokens) / prompt_rate + f64::from(output_tokens) / generation_rate;
    Ok(f64::from(prompt_tokens.saturating_add(output_tokens)) / elapsed_seconds)
}

fn wait_until_ready(base_url: &str, server: &mut ManagedChild, timeout: Duration) -> Result<()> {
    let agent: ureq::Agent = ureq::Agent::config_builder()
        .timeout_global(Some(Duration::from_secs(1)))
        .build()
        .into();
    let started = Instant::now();

    loop {
        if let Some(status) = server.try_wait()? {
            bail!("llama-server exited before becoming ready with status {status}");
        }
        if let Ok(mut response) = agent.get(format!("{base_url}/health")).call()
            && response
                .body_mut()
                .read_json::<serde_json::Value>()
                .ok()
                .and_then(|value| value["status"].as_str().map(str::to_owned))
                .as_deref()
                == Some("ok")
        {
            return Ok(());
        }

        if started.elapsed() >= timeout {
            bail!(
                "llama-server did not become ready at {base_url} within {:.0} seconds",
                timeout.as_secs_f64()
            );
        }
        thread::sleep(HEALTH_POLL_INTERVAL);
    }
}

struct ManagedChild {
    child: Option<Child>,
}

impl ManagedChild {
    fn spawn(command: &mut Command) -> std::io::Result<Self> {
        command.spawn().map(|child| Self { child: Some(child) })
    }

    fn try_wait(&mut self) -> std::io::Result<Option<std::process::ExitStatus>> {
        self.child.as_mut().expect("child is present").try_wait()
    }

    fn stop(&mut self) {
        let Some(mut child) = self.child.take() else {
            return;
        };

        #[cfg(unix)]
        unsafe {
            libc::kill(child.id() as libc::pid_t, libc::SIGINT);
        }
        #[cfg(not(unix))]
        let _ = child.kill();

        for _ in 0..50 {
            if child.try_wait().ok().flatten().is_some() {
                return;
            }
            thread::sleep(Duration::from_millis(100));
        }
        let _ = child.kill();
        let _ = child.wait();
    }
}

impl Drop for ManagedChild {
    fn drop(&mut self) {
        self.stop();
    }
}

fn discover_binary(env_name: &str, executable: &str) -> Option<PathBuf> {
    if let Some(path) = env::var_os(env_name).map(PathBuf::from)
        && is_executable_file(&path)
    {
        return Some(path);
    }

    discover_local_cuda_binary(executable).or_else(|| {
        env::var_os("PATH")
            .as_deref()
            .map(env::split_paths)
            .into_iter()
            .flatten()
            .map(|directory| directory.join(executable))
            .find(|path| is_executable_file(path))
    })
}

fn discover_local_cuda_binary(executable: &str) -> Option<PathBuf> {
    let root = dirs::data_local_dir()?.join("llama.cpp");
    fs::read_dir(root)
        .ok()?
        .flatten()
        .filter(|entry| {
            entry
                .file_name()
                .to_string_lossy()
                .to_ascii_lowercase()
                .ends_with("-cuda")
        })
        .map(|entry| entry.path().join(executable))
        .filter(|path| is_executable_file(path))
        .max_by_key(|path| local_build_number(path))
}

fn local_build_number(path: &Path) -> u64 {
    path.parent()
        .and_then(Path::file_name)
        .and_then(|name| name.to_str())
        .and_then(|name| name.strip_prefix('b'))
        .and_then(|name| name.strip_suffix("-cuda"))
        .and_then(|number| number.parse().ok())
        .unwrap_or_default()
}

fn infer_variant(path: &Path) -> LlamaCppVariant {
    let normalized = path.to_string_lossy().to_ascii_lowercase();
    if normalized.contains("cuda") {
        LlamaCppVariant::Cuda
    } else if normalized.contains("vulkan") {
        LlamaCppVariant::Vulkan
    } else {
        LlamaCppVariant::Other
    }
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

    use crate::policy::{ExecutionProfile, GpuPlacement, KvCacheType, MemoryEstimate};

    use super::{
        BenchmarkMeasurement, LlamaCppVariant, command_args, effective_throughput, infer_variant,
        server_command,
    };

    #[test]
    fn builds_a_server_command_from_a_profile() {
        let profile = test_profile();

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
        assert!(args.windows(2).any(|pair| pair == ["--n-gpu-layers", "24"]));
        assert!(args.windows(2).any(|pair| pair == ["--parallel", "2"]));
    }

    #[test]
    fn preserves_backend_auto_gpu_placement() {
        let mut profile = test_profile();
        profile.gpu_placement = GpuPlacement::Auto;

        let command = server_command(
            Path::new("/opt/llama-server"),
            Path::new("/models/test.gguf"),
            "127.0.0.1",
            8080,
            &profile,
        );
        let args = command_args(&command);

        assert!(
            args.windows(2)
                .any(|pair| pair == ["--n-gpu-layers", "auto"])
        );
    }

    #[test]
    fn identifies_packaged_gpu_variants() {
        assert_eq!(
            infer_variant(Path::new("/opt/llama.cpp/b100-cuda/llama-server")),
            LlamaCppVariant::Cuda
        );
        assert_eq!(
            infer_variant(Path::new("/opt/llama.cpp/b100-vulkan/llama-server")),
            LlamaCppVariant::Vulkan
        );
    }

    #[test]
    fn benchmark_measurements_deserialize_and_combine_rates() {
        let measurements: Vec<BenchmarkMeasurement> = serde_json::from_str(
            r#"[
                {"n_prompt": 512, "n_gen": 0, "avg_ns": 100, "avg_ts": 1000.0, "stddev_ts": 1.0},
                {"n_prompt": 0, "n_gen": 128, "avg_ns": 200, "avg_ts": 40.0, "stddev_ts": 0.5}
            ]"#,
        )
        .unwrap();
        let combined = effective_throughput(512, 128, 1000.0, 40.0).unwrap();

        assert_eq!(measurements.len(), 2);
        assert!((combined - 172.413_793).abs() < 0.001);
    }

    fn test_profile() -> ExecutionProfile {
        ExecutionProfile {
            cpu_threads: 8,
            gpu_placement: GpuPlacement::Exact { layers: 24 },
            context_size: 4096,
            parallel_slots: 2,
            batch_size: 256,
            ubatch_size: 128,
            kv_cache_type: KvCacheType::F16,
            memory: MemoryEstimate {
                model_bytes: 1,
                kv_cache_bytes: Some(1),
                compute_buffer_bytes: 1,
                estimated_total_bytes: Some(3),
                estimated_gpu_bytes: Some(3),
                system_budget_bytes: 10,
                gpu_budget_bytes: Some(10),
            },
            planning_score: 1.0,
            reasons: vec![],
        }
    }
}
