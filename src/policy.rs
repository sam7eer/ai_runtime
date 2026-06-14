use std::collections::BTreeSet;

use anyhow::{Result, bail};
use clap::ValueEnum;
use serde::{Deserialize, Serialize};

use crate::gguf::ModelMetadata;
use crate::model::{AcceleratorKind, HardwareSnapshot};

const SYSTEM_MEMORY_SAFETY_PERCENT: u64 = 90;
const GPU_MEMORY_SAFETY_PERCENT: u64 = 90;
const MIN_BATCH_SIZE: u32 = 32;
const MAX_BATCH_SIZE: u32 = 2048;
const MAX_UBATCH_SIZE: u32 = 512;
const MIN_COMPUTE_BUFFER_BYTES: u64 = 256 * 1024 * 1024;
const THERMAL_BASELINE_C: f64 = 60.0;
const THERMAL_CEILING_C: f64 = 95.0;

#[derive(Debug, Clone, Copy, ValueEnum, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum UseCase {
    Interactive,
    Batch,
    Background,
}

#[derive(Debug, Clone, Copy, ValueEnum, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum OptimizationGoal {
    Latency,
    Throughput,
    Efficiency,
    Balanced,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct WorkloadSpec {
    pub use_case: UseCase,
    pub optimization_goal: OptimizationGoal,
    pub prompt_tokens: u32,
    pub output_tokens: u32,
    pub concurrency: u32,
}

impl WorkloadSpec {
    pub fn context_per_request(&self) -> Result<u32> {
        self.prompt_tokens
            .checked_add(self.output_tokens)
            .ok_or_else(|| anyhow::anyhow!("workload context length overflowed"))
    }

    pub fn total_context_tokens(&self) -> Result<u64> {
        Ok(u64::from(self.context_per_request()?) * u64::from(self.concurrency))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum GpuPlacement {
    CpuOnly,
    Exact { layers: u32 },
    Auto,
}

impl GpuPlacement {
    pub fn llama_cpp_value(&self) -> String {
        match self {
            Self::CpuOnly => "0".into(),
            Self::Exact { layers } => layers.to_string(),
            Self::Auto => "auto".into(),
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum KvCacheType {
    F16,
    Q8_0,
    Q4_0,
}

impl KvCacheType {
    pub fn llama_cpp_value(self) -> &'static str {
        match self {
            Self::F16 => "f16",
            Self::Q8_0 => "q8_0",
            Self::Q4_0 => "q4_0",
        }
    }

    fn bytes_ratio(self) -> (u64, u64) {
        match self {
            Self::F16 => (2, 1),
            Self::Q8_0 => (34, 32),
            Self::Q4_0 => (18, 32),
        }
    }

    fn precision_rank(self) -> f64 {
        match self {
            Self::F16 => 1.0,
            Self::Q8_0 => 0.7,
            Self::Q4_0 => 0.4,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct MemoryEstimate {
    pub model_bytes: u64,
    pub kv_cache_bytes: Option<u64>,
    pub compute_buffer_bytes: u64,
    pub estimated_total_bytes: Option<u64>,
    pub estimated_gpu_bytes: Option<u64>,
    pub system_budget_bytes: u64,
    pub gpu_budget_bytes: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ExecutionProfile {
    pub cpu_threads: usize,
    pub gpu_placement: GpuPlacement,
    pub context_size: u32,
    pub parallel_slots: u32,
    pub batch_size: u32,
    pub ubatch_size: u32,
    pub kv_cache_type: KvCacheType,
    pub memory: MemoryEstimate,
    pub planning_score: f64,
    pub reasons: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PlanningDecision {
    pub model: ModelMetadata,
    pub workload: WorkloadSpec,
    pub selected: ExecutionProfile,
    pub candidates: Vec<ExecutionProfile>,
    pub selection_basis: String,
    pub calibration_required: bool,
    pub notes: Vec<String>,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct PlanningOverrides {
    pub gpu_memory_bytes: Option<u64>,
}

pub fn plan(
    snapshot: &HardwareSnapshot,
    model: ModelMetadata,
    workload: WorkloadSpec,
    overrides: PlanningOverrides,
) -> Result<PlanningDecision> {
    validate_workload(&model, &workload)?;

    let system_budget = snapshot
        .memory
        .available_bytes
        .saturating_mul(SYSTEM_MEMORY_SAFETY_PERCENT)
        / 100;
    let gpu_budget = discover_gpu_budget(snapshot, overrides);
    let thread_candidates = thread_candidates(snapshot);
    let batch_candidates = batch_candidates(&workload);
    let kv_types = [KvCacheType::F16, KvCacheType::Q8_0, KvCacheType::Q4_0];

    let mut candidates = Vec::new();
    for kv_cache_type in kv_types {
        for batch_size in &batch_candidates {
            let ubatch_size = (*batch_size).min(MAX_UBATCH_SIZE);
            let base_memory = estimate_memory(
                &model,
                &workload,
                kv_cache_type,
                ubatch_size,
                system_budget,
                gpu_budget,
                &GpuPlacement::CpuOnly,
            )?;
            if !fits_system_memory(&base_memory) {
                continue;
            }

            let placements = placement_candidates(&model, snapshot, &base_memory, gpu_budget);
            for cpu_threads in &thread_candidates {
                for gpu_placement in &placements {
                    let memory = estimate_memory(
                        &model,
                        &workload,
                        kv_cache_type,
                        ubatch_size,
                        system_budget,
                        gpu_budget,
                        gpu_placement,
                    )?;
                    if !fits_system_memory(&memory) || !fits_gpu_memory(&memory) {
                        continue;
                    }

                    let mut candidate = ExecutionProfile {
                        cpu_threads: *cpu_threads,
                        gpu_placement: gpu_placement.clone(),
                        context_size: workload
                            .context_per_request()?
                            .saturating_mul(workload.concurrency),
                        parallel_slots: workload.concurrency,
                        batch_size: *batch_size,
                        ubatch_size,
                        kv_cache_type,
                        memory,
                        planning_score: 0.0,
                        reasons: Vec::new(),
                    };
                    candidate.planning_score =
                        analytical_score(&candidate, &model, snapshot, &workload);
                    candidate.reasons =
                        candidate_reasons(&candidate, &model, snapshot, &workload, gpu_budget);
                    candidates.push(candidate);
                }
            }
        }
    }

    if candidates.is_empty() {
        bail!(
            "no safe execution candidate fits the current memory budget for this model and workload"
        );
    }

    candidates.sort_by(|left, right| {
        right
            .planning_score
            .total_cmp(&left.planning_score)
            .then_with(|| left.cpu_threads.cmp(&right.cpu_threads))
    });
    candidates.dedup_by(|left, right| same_configuration(left, right));

    let selected = candidates[0].clone();
    let mut notes = vec![
        "selection is an analytical baseline; benchmark measurements are required to claim optimal performance or energy efficiency".into(),
        "fixed values are safety/search bounds, not mode-to-configuration rules".into(),
    ];
    if gpu_budget.is_none()
        && snapshot
            .accelerators
            .iter()
            .any(|accelerator| accelerator.kind != AcceleratorKind::Npu)
    {
        notes.push(
            "GPU memory is unavailable, so GPU placement is delegated to llama.cpp auto mode; use --gpu-memory-mib to enable exact layer candidates".into(),
        );
    }
    if snapshot.power.on_ac == Some(false) {
        notes.push(
            "the host is on battery; balanced and efficiency rankings include a resource-pressure penalty without replacing the requested goal".into(),
        );
    }
    if let Some(temperature) = snapshot.highest_temperature_celsius() {
        notes.push(format!(
            "the current peak temperature of {temperature:.1} C contributes a continuous resource-pressure penalty"
        ));
    }
    notes.push(format!(
        "current system memory pressure is {:.0}%",
        snapshot.memory.pressure_ratio() * 100.0
    ));
    if model.kv_dimensions_per_layer().is_none() {
        notes.push(
            "the model lacks enough attention metadata for a complete KV-cache estimate".into(),
        );
    }

    Ok(PlanningDecision {
        model,
        workload,
        selected,
        candidates,
        selection_basis: "model-aware analytical candidate ranking".into(),
        calibration_required: true,
        notes,
    })
}

fn validate_workload(model: &ModelMetadata, workload: &WorkloadSpec) -> Result<()> {
    if workload.prompt_tokens == 0 {
        bail!("prompt tokens must be greater than zero");
    }
    if workload.output_tokens == 0 {
        bail!("output tokens must be greater than zero");
    }
    if workload.concurrency == 0 {
        bail!("concurrency must be greater than zero");
    }
    if model.block_count == 0 {
        bail!("model block count must be greater than zero");
    }
    if model.embedding_length.is_none() {
        bail!("model embedding length is required for safe memory planning");
    }

    let requested = workload.context_per_request()?;
    if let Some(model_limit) = model.context_length
        && requested > model_limit
    {
        bail!("requested context of {requested} tokens exceeds the model limit of {model_limit}");
    }
    let total_context = workload.total_context_tokens()?;
    if total_context > u64::from(u32::MAX) {
        bail!("total context across concurrent slots exceeds the backend limit");
    }
    Ok(())
}

fn discover_gpu_budget(snapshot: &HardwareSnapshot, overrides: PlanningOverrides) -> Option<u64> {
    overrides
        .gpu_memory_bytes
        .or_else(|| {
            snapshot
                .accelerators
                .iter()
                .filter(|accelerator| accelerator.kind == AcceleratorKind::DiscreteGpu)
                .filter_map(|accelerator| accelerator.available_memory_bytes)
                .max()
        })
        .map(|bytes| bytes.saturating_mul(GPU_MEMORY_SAFETY_PERCENT) / 100)
}

fn thread_candidates(snapshot: &HardwareSnapshot) -> Vec<usize> {
    let logical = snapshot.cpu.logical_cores.max(1);
    let physical = snapshot.cpu.physical_cores.unwrap_or(logical).max(1);
    let mut values = BTreeSet::new();
    values.insert(1);
    values.insert((physical / 2).max(1));
    values.insert(physical);
    values.insert(logical);
    values.into_iter().collect()
}

fn batch_candidates(workload: &WorkloadSpec) -> Vec<u32> {
    let useful_tokens = workload
        .prompt_tokens
        .saturating_mul(workload.concurrency)
        .clamp(MIN_BATCH_SIZE, MAX_BATCH_SIZE);
    let target = useful_tokens.next_power_of_two().min(MAX_BATCH_SIZE);
    let mut values = BTreeSet::new();
    values.insert(MIN_BATCH_SIZE.min(target));
    values.insert((target / 2).max(MIN_BATCH_SIZE));
    values.insert(target);
    values.into_iter().collect()
}

fn placement_candidates(
    model: &ModelMetadata,
    snapshot: &HardwareSnapshot,
    base_memory: &MemoryEstimate,
    gpu_budget: Option<u64>,
) -> Vec<GpuPlacement> {
    let has_gpu = snapshot
        .accelerators
        .iter()
        .any(|accelerator| accelerator.kind != AcceleratorKind::Npu);
    if !has_gpu {
        return vec![GpuPlacement::CpuOnly];
    }

    let Some(gpu_budget) = gpu_budget else {
        return vec![GpuPlacement::CpuOnly, GpuPlacement::Auto];
    };

    let fixed_gpu_bytes = base_memory
        .kv_cache_bytes
        .unwrap_or_default()
        .saturating_add(base_memory.compute_buffer_bytes);
    if fixed_gpu_bytes >= gpu_budget {
        return vec![GpuPlacement::CpuOnly];
    }

    let bytes_per_layer = model.file_size_bytes.div_ceil(u64::from(model.block_count));
    let max_layers =
        ((gpu_budget - fixed_gpu_bytes) / bytes_per_layer).min(u64::from(model.block_count)) as u32;
    if max_layers == 0 {
        return vec![GpuPlacement::CpuOnly];
    }

    let mut layers = BTreeSet::new();
    layers.insert((max_layers / 4).max(1));
    layers.insert((max_layers / 2).max(1));
    layers.insert((max_layers * 3 / 4).max(1));
    layers.insert(max_layers);

    let mut placements = vec![GpuPlacement::CpuOnly];
    placements.extend(
        layers
            .into_iter()
            .map(|layers| GpuPlacement::Exact { layers }),
    );
    placements
}

fn estimate_memory(
    model: &ModelMetadata,
    workload: &WorkloadSpec,
    kv_cache_type: KvCacheType,
    ubatch_size: u32,
    system_budget: u64,
    gpu_budget: Option<u64>,
    placement: &GpuPlacement,
) -> Result<MemoryEstimate> {
    let total_context_tokens = workload.total_context_tokens()?;
    let kv_cache_bytes = model.kv_dimensions_per_layer().and_then(|(key, value)| {
        let elements = u64::from(model.block_count)
            .checked_mul(key.checked_add(value)?)?
            .checked_mul(total_context_tokens)?;
        let (numerator, denominator) = kv_cache_type.bytes_ratio();
        elements.checked_mul(numerator)?.checked_div(denominator)
    });
    let compute_buffer_bytes = model
        .embedding_length
        .map(|embedding| {
            u64::from(embedding)
                .saturating_mul(u64::from(ubatch_size))
                .saturating_mul(4)
                .saturating_mul(4)
        })
        .unwrap_or_default()
        .max(MIN_COMPUTE_BUFFER_BYTES);
    let estimated_total_bytes = kv_cache_bytes.map(|kv| {
        model
            .file_size_bytes
            .saturating_add(kv)
            .saturating_add(compute_buffer_bytes)
    });
    let estimated_gpu_bytes = match placement {
        GpuPlacement::CpuOnly => Some(0),
        GpuPlacement::Exact { layers } => {
            let layer_weights = model
                .file_size_bytes
                .div_ceil(u64::from(model.block_count))
                .saturating_mul(u64::from(*layers));
            Some(
                layer_weights
                    .saturating_add(kv_cache_bytes.unwrap_or_default())
                    .saturating_add(compute_buffer_bytes),
            )
        }
        GpuPlacement::Auto => None,
    };

    Ok(MemoryEstimate {
        model_bytes: model.file_size_bytes,
        kv_cache_bytes,
        compute_buffer_bytes,
        estimated_total_bytes,
        estimated_gpu_bytes,
        system_budget_bytes: system_budget,
        gpu_budget_bytes: gpu_budget,
    })
}

fn fits_system_memory(memory: &MemoryEstimate) -> bool {
    memory
        .estimated_total_bytes
        .is_none_or(|required| required <= memory.system_budget_bytes)
}

fn fits_gpu_memory(memory: &MemoryEstimate) -> bool {
    match (memory.estimated_gpu_bytes, memory.gpu_budget_bytes) {
        (Some(required), Some(budget)) => required <= budget,
        _ => true,
    }
}

fn analytical_score(
    candidate: &ExecutionProfile,
    model: &ModelMetadata,
    snapshot: &HardwareSnapshot,
    workload: &WorkloadSpec,
) -> f64 {
    let gpu_fraction = match candidate.gpu_placement {
        GpuPlacement::CpuOnly => 0.0,
        GpuPlacement::Exact { layers } => f64::from(layers) / f64::from(model.block_count.max(1)),
        GpuPlacement::Auto => 1.0,
    };
    let thread_fraction = candidate.cpu_threads as f64 / snapshot.cpu.logical_cores.max(1) as f64;
    let batch_fraction = f64::from(candidate.batch_size) / f64::from(MAX_BATCH_SIZE);
    let precision = candidate.kv_cache_type.precision_rank();
    let memory_efficiency = candidate
        .memory
        .estimated_total_bytes
        .map(|required| {
            1.0 - (required as f64 / candidate.memory.system_budget_bytes.max(1) as f64).min(1.0)
        })
        .unwrap_or(0.0);

    let objective_score = match workload.optimization_goal {
        OptimizationGoal::Latency => {
            0.55 * gpu_fraction + 0.20 * thread_fraction + 0.15 * batch_fraction + 0.10 * precision
        }
        OptimizationGoal::Throughput => {
            0.45 * gpu_fraction + 0.20 * thread_fraction + 0.25 * batch_fraction + 0.10 * precision
        }
        OptimizationGoal::Efficiency => {
            0.25 * gpu_fraction
                + 0.15 * thread_fraction
                + 0.10 * batch_fraction
                + 0.40 * memory_efficiency
                + 0.10 * precision
        }
        OptimizationGoal::Balanced => {
            0.40 * gpu_fraction
                + 0.20 * thread_fraction
                + 0.15 * batch_fraction
                + 0.15 * memory_efficiency
                + 0.10 * precision
        }
    };

    let use_case_adjustment = match workload.use_case {
        UseCase::Interactive => 0.10 * gpu_fraction,
        UseCase::Batch => 0.10 * batch_fraction,
        UseCase::Background => 0.10 * memory_efficiency - 0.05 * thread_fraction,
    };
    let resource_intensity = (gpu_fraction + thread_fraction + batch_fraction) / 3.0;
    let thermal_pressure = snapshot
        .highest_temperature_celsius()
        .map(|temperature| {
            ((temperature - THERMAL_BASELINE_C) / (THERMAL_CEILING_C - THERMAL_BASELINE_C))
                .clamp(0.0, 1.0)
        })
        .unwrap_or(0.0);
    let thermal_penalty = 0.20 * thermal_pressure * resource_intensity;
    let battery_penalty = if snapshot.power.on_ac == Some(false)
        && matches!(
            workload.optimization_goal,
            OptimizationGoal::Balanced | OptimizationGoal::Efficiency
        ) {
        0.10 * resource_intensity
    } else {
        0.0
    };

    objective_score + use_case_adjustment - thermal_penalty - battery_penalty
}

fn candidate_reasons(
    candidate: &ExecutionProfile,
    model: &ModelMetadata,
    snapshot: &HardwareSnapshot,
    workload: &WorkloadSpec,
    gpu_budget: Option<u64>,
) -> Vec<String> {
    let mut reasons = vec![
        format!(
            "preserves {} tokens per request across {} slot(s)",
            workload.context_per_request().unwrap_or_default(),
            workload.concurrency
        ),
        format!(
            "uses {} of {} detected logical CPU threads",
            candidate.cpu_threads,
            snapshot.cpu.logical_cores.max(1)
        ),
    ];
    match candidate.gpu_placement {
        GpuPlacement::CpuOnly => reasons.push("keeps all model layers on CPU".into()),
        GpuPlacement::Exact { layers } => reasons.push(format!(
            "offloads {layers} of {} model layers within the estimated GPU budget",
            model.block_count
        )),
        GpuPlacement::Auto => reasons.push(
            "delegates layer count to llama.cpp because live GPU memory capacity is unknown".into(),
        ),
    }
    if gpu_budget.is_some() {
        reasons.push("GPU candidate was derived from a 90% memory safety budget".into());
    }
    reasons
}

fn same_configuration(left: &ExecutionProfile, right: &ExecutionProfile) -> bool {
    left.cpu_threads == right.cpu_threads
        && left.gpu_placement == right.gpu_placement
        && left.context_size == right.context_size
        && left.parallel_slots == right.parallel_slots
        && left.batch_size == right.batch_size
        && left.ubatch_size == right.ubatch_size
        && left.kv_cache_type == right.kv_cache_type
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use crate::gguf::ModelMetadata;
    use crate::model::{
        AcceleratorInfo, AcceleratorKind, CpuInfo, HardwareSnapshot, MemoryInfo, PowerInfo,
        ThermalReading,
    };

    use super::{GpuPlacement, OptimizationGoal, PlanningOverrides, UseCase, WorkloadSpec, plan};

    fn snapshot() -> HardwareSnapshot {
        HardwareSnapshot {
            captured_at: "2026-06-14T00:00:00Z".into(),
            hostname: "test".into(),
            os: "Linux".into(),
            kernel: "test".into(),
            cpu: CpuInfo {
                model: "Test CPU".into(),
                architecture: "x86_64".into(),
                logical_cores: 16,
                physical_cores: Some(8),
            },
            memory: MemoryInfo {
                total_bytes: 16 * 1024_u64.pow(3),
                available_bytes: 12 * 1024_u64.pow(3),
                swap_total_bytes: 0,
                swap_free_bytes: 0,
            },
            accelerators: vec![AcceleratorInfo {
                kind: AcceleratorKind::DiscreteGpu,
                name: "Test GPU".into(),
                vendor: "Test".into(),
                device_path: None,
                dedicated_memory_bytes: Some(4 * 1024_u64.pow(3)),
                available_memory_bytes: Some(4 * 1024_u64.pow(3)),
                telemetry_available: true,
            }],
            thermals: vec![ThermalReading {
                source: "cpu".into(),
                temperature_celsius: 55.0,
            }],
            power: PowerInfo {
                on_ac: Some(true),
                battery_percent: Some(100.0),
                battery_status: Some("Full".into()),
            },
        }
    }

    fn model() -> ModelMetadata {
        ModelMetadata {
            path: PathBuf::from("/models/test.gguf"),
            file_size_bytes: 4 * 1024_u64.pow(3),
            gguf_version: 3,
            name: Some("Test".into()),
            architecture: "llama".into(),
            block_count: 32,
            context_length: Some(8192),
            embedding_length: Some(4096),
            attention_head_count: Some(32),
            attention_head_count_kv: Some(8),
            attention_key_length: None,
            attention_value_length: None,
        }
    }

    fn workload() -> WorkloadSpec {
        WorkloadSpec {
            use_case: UseCase::Interactive,
            optimization_goal: OptimizationGoal::Balanced,
            prompt_tokens: 512,
            output_tokens: 256,
            concurrency: 1,
        }
    }

    #[test]
    fn derives_exact_gpu_candidates_from_model_and_vram() {
        let decision = plan(
            &snapshot(),
            model(),
            workload(),
            PlanningOverrides::default(),
        )
        .unwrap();

        assert!(decision.candidates.iter().any(|candidate| matches!(
            candidate.gpu_placement,
            GpuPlacement::Exact { layers } if layers > 0 && layers <= 32
        )));
        assert!(decision.calibration_required);
    }

    #[test]
    fn unknown_vram_uses_backend_auto_instead_of_a_magic_layer_count() {
        let mut host = snapshot();
        host.accelerators[0].available_memory_bytes = None;

        let decision = plan(&host, model(), workload(), PlanningOverrides::default()).unwrap();

        assert!(
            decision
                .candidates
                .iter()
                .any(|candidate| candidate.gpu_placement == GpuPlacement::Auto)
        );
    }

    #[test]
    fn battery_state_does_not_replace_the_requested_goal() {
        let mut host = snapshot();
        host.power.on_ac = Some(false);
        let mut requested = workload();
        requested.optimization_goal = OptimizationGoal::Latency;

        let decision = plan(&host, model(), requested, PlanningOverrides::default()).unwrap();

        assert_eq!(
            decision.workload.optimization_goal,
            OptimizationGoal::Latency
        );
        assert!(
            decision
                .notes
                .iter()
                .any(|note| note.contains("without replacing"))
        );
    }

    #[test]
    fn rejects_workloads_beyond_the_model_context() {
        let mut requested = workload();
        requested.prompt_tokens = 8000;
        requested.output_tokens = 1000;

        let error = plan(
            &snapshot(),
            model(),
            requested,
            PlanningOverrides::default(),
        )
        .unwrap_err();

        assert!(error.to_string().contains("exceeds the model limit"));
    }

    #[test]
    fn thermal_pressure_changes_ranking_scores_without_a_fixed_profile_mapping() {
        let cool = plan(
            &snapshot(),
            model(),
            workload(),
            PlanningOverrides::default(),
        )
        .unwrap();
        let mut hot_host = snapshot();
        hot_host.thermals[0].temperature_celsius = 95.0;
        let hot = plan(&hot_host, model(), workload(), PlanningOverrides::default()).unwrap();

        assert!(hot.selected.planning_score < cool.selected.planning_score);
        assert!(
            hot.notes
                .iter()
                .any(|note| note.contains("continuous resource-pressure"))
        );
    }
}
