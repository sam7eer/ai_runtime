use clap::ValueEnum;
use serde::{Deserialize, Serialize};

use crate::model::HardwareSnapshot;

#[derive(Debug, Clone, Copy, ValueEnum, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum RuntimeMode {
    Performance,
    Balanced,
    Battery,
    Background,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum GpuOffload {
    Disabled,
    Conservative,
    MaximumFitting,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ExecutionProfile {
    pub mode: RuntimeMode,
    pub cpu_threads: usize,
    pub gpu_offload: GpuOffload,
    pub context_size: u32,
    pub batch_size: u32,
    pub kv_cache_type: String,
    pub reasons: Vec<String>,
}

pub fn recommend(snapshot: &HardwareSnapshot, requested_mode: RuntimeMode) -> ExecutionProfile {
    let mode = effective_mode(snapshot, requested_mode);
    let logical_cores = snapshot.cpu.logical_cores.max(1);
    let has_gpu = !snapshot.accelerators.is_empty();
    let memory_gib = snapshot.memory.available_bytes as f64 / 1024_f64.powi(3);
    let pressure = snapshot.memory.pressure_ratio();
    let hottest = snapshot.highest_temperature_celsius();
    let thermally_hot = hottest.is_some_and(|temperature| temperature >= 85.0);
    let memory_tight = pressure >= 0.80 || memory_gib < 3.0;

    let (mut cpu_threads, mut gpu_offload, mut context_size, mut batch_size) = match mode {
        RuntimeMode::Performance => (
            logical_cores,
            if has_gpu {
                GpuOffload::MaximumFitting
            } else {
                GpuOffload::Disabled
            },
            8192,
            512,
        ),
        RuntimeMode::Balanced => (
            (logical_cores * 3 / 4).max(1),
            if has_gpu {
                GpuOffload::MaximumFitting
            } else {
                GpuOffload::Disabled
            },
            4096,
            256,
        ),
        RuntimeMode::Battery => (
            (logical_cores / 2).max(1),
            if has_gpu {
                GpuOffload::Conservative
            } else {
                GpuOffload::Disabled
            },
            4096,
            128,
        ),
        RuntimeMode::Background => ((logical_cores / 4).max(1), GpuOffload::Disabled, 2048, 64),
    };

    let mut reasons = vec![format!("selected {mode:?} policy")];

    if requested_mode != mode {
        reasons.push("battery state overrode the requested mode".into());
    }
    if !has_gpu {
        reasons.push("no usable GPU was discovered".into());
    }
    if thermally_hot {
        cpu_threads = cpu_threads.min((logical_cores / 2).max(1));
        gpu_offload = if has_gpu {
            GpuOffload::Conservative
        } else {
            GpuOffload::Disabled
        };
        batch_size = batch_size.min(128);
        reasons.push(format!(
            "thermal guard applied at {:.1} C",
            hottest.unwrap_or_default()
        ));
    }
    if memory_tight {
        context_size = context_size.min(2048);
        batch_size = batch_size.min(128);
        reasons.push(format!(
            "memory guard applied with {:.1} GiB available and {:.0}% pressure",
            memory_gib,
            pressure * 100.0
        ));
    }

    ExecutionProfile {
        mode,
        cpu_threads,
        gpu_offload,
        context_size,
        batch_size,
        kv_cache_type: if memory_tight { "q8_0" } else { "f16" }.into(),
        reasons,
    }
}

fn effective_mode(snapshot: &HardwareSnapshot, requested: RuntimeMode) -> RuntimeMode {
    match (snapshot.power.on_ac, requested) {
        (Some(false), RuntimeMode::Performance | RuntimeMode::Balanced) => RuntimeMode::Battery,
        _ => requested,
    }
}

#[cfg(test)]
mod tests {
    use crate::model::{
        AcceleratorInfo, AcceleratorKind, CpuInfo, HardwareSnapshot, MemoryInfo, PowerInfo,
        ThermalReading,
    };

    use super::{GpuOffload, RuntimeMode, recommend};

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
                available_bytes: 10 * 1024_u64.pow(3),
                swap_total_bytes: 0,
                swap_free_bytes: 0,
            },
            accelerators: vec![AcceleratorInfo {
                kind: AcceleratorKind::DiscreteGpu,
                name: "Test GPU".into(),
                vendor: "Test".into(),
                device_path: None,
                dedicated_memory_bytes: Some(4 * 1024_u64.pow(3)),
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

    #[test]
    fn balanced_profile_uses_available_gpu() {
        let profile = recommend(&snapshot(), RuntimeMode::Balanced);

        assert_eq!(profile.cpu_threads, 12);
        assert_eq!(profile.gpu_offload, GpuOffload::MaximumFitting);
        assert_eq!(profile.context_size, 4096);
    }

    #[test]
    fn battery_state_overrides_performance() {
        let mut host = snapshot();
        host.power.on_ac = Some(false);

        let profile = recommend(&host, RuntimeMode::Performance);

        assert_eq!(profile.mode, RuntimeMode::Battery);
        assert_eq!(profile.gpu_offload, GpuOffload::Conservative);
    }

    #[test]
    fn memory_pressure_reduces_context_and_quantizes_kv_cache() {
        let mut host = snapshot();
        host.memory.available_bytes = 2 * 1024_u64.pow(3);

        let profile = recommend(&host, RuntimeMode::Balanced);

        assert_eq!(profile.context_size, 2048);
        assert_eq!(profile.kv_cache_type, "q8_0");
    }
}
