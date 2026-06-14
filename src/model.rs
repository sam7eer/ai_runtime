use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct HardwareSnapshot {
    pub captured_at: String,
    pub hostname: String,
    pub os: String,
    pub kernel: String,
    pub cpu: CpuInfo,
    pub memory: MemoryInfo,
    pub accelerators: Vec<AcceleratorInfo>,
    pub thermals: Vec<ThermalReading>,
    pub power: PowerInfo,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CpuInfo {
    pub model: String,
    pub architecture: String,
    pub logical_cores: usize,
    pub physical_cores: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MemoryInfo {
    pub total_bytes: u64,
    pub available_bytes: u64,
    pub swap_total_bytes: u64,
    pub swap_free_bytes: u64,
}

impl MemoryInfo {
    pub fn pressure_ratio(&self) -> f64 {
        if self.total_bytes == 0 {
            return 0.0;
        }

        1.0 - self.available_bytes as f64 / self.total_bytes as f64
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AcceleratorInfo {
    pub kind: AcceleratorKind,
    pub name: String,
    pub vendor: String,
    pub device_path: Option<String>,
    pub dedicated_memory_bytes: Option<u64>,
    pub telemetry_available: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum AcceleratorKind {
    IntegratedGpu,
    DiscreteGpu,
    Npu,
    Unknown,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ThermalReading {
    pub source: String,
    pub temperature_celsius: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PowerInfo {
    pub on_ac: Option<bool>,
    pub battery_percent: Option<f64>,
    pub battery_status: Option<String>,
}

impl HardwareSnapshot {
    pub fn highest_temperature_celsius(&self) -> Option<f64> {
        self.thermals
            .iter()
            .map(|reading| reading.temperature_celsius)
            .filter(|value| value.is_finite())
            .max_by(f64::total_cmp)
    }
}
