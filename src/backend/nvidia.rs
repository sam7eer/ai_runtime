use std::process::Command;
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct GpuTelemetrySample {
    pub elapsed_ms: u64,
    pub device_index: u32,
    pub device_name: String,
    pub memory_used_mib: f64,
    pub memory_free_mib: f64,
    pub utilization_percent: f64,
    pub temperature_celsius: f64,
    pub power_watts: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct GpuTelemetrySummary {
    pub device_name: Option<String>,
    pub sample_count: usize,
    pub peak_memory_used_mib: Option<f64>,
    pub minimum_memory_free_mib: Option<f64>,
    pub peak_utilization_percent: Option<f64>,
    pub peak_temperature_celsius: Option<f64>,
    pub average_power_watts: Option<f64>,
}

pub struct NvidiaSampler {
    stop: Arc<AtomicBool>,
    handle: Option<JoinHandle<Vec<GpuTelemetrySample>>>,
}

impl NvidiaSampler {
    pub fn start(interval: Duration) -> Self {
        let stop = Arc::new(AtomicBool::new(false));
        let thread_stop = Arc::clone(&stop);
        let handle = thread::spawn(move || {
            let started = Instant::now();
            let mut samples = Vec::new();
            while !thread_stop.load(Ordering::Relaxed) {
                if let Some(sample) = capture_sample(started.elapsed()) {
                    samples.push(sample);
                }
                thread::sleep(interval);
            }
            if let Some(sample) = capture_sample(started.elapsed()) {
                samples.push(sample);
            }
            samples
        });

        Self {
            stop,
            handle: Some(handle),
        }
    }

    pub fn finish(mut self) -> GpuTelemetrySummary {
        self.stop.store(true, Ordering::Relaxed);
        let samples = self
            .handle
            .take()
            .and_then(|handle| handle.join().ok())
            .unwrap_or_default();
        summarize(&samples)
    }
}

impl Drop for NvidiaSampler {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

fn capture_sample(elapsed: Duration) -> Option<GpuTelemetrySample> {
    let output = Command::new("nvidia-smi")
        .args([
            "--query-gpu=index,name,memory.used,memory.free,utilization.gpu,temperature.gpu,power.draw",
            "--format=csv,noheader,nounits",
        ])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }

    let stdout = String::from_utf8(output.stdout).ok()?;
    parse_sample(stdout.lines().next()?, elapsed)
}

fn parse_sample(line: &str, elapsed: Duration) -> Option<GpuTelemetrySample> {
    let fields: Vec<_> = line.split(',').map(str::trim).collect();
    if fields.len() != 7 {
        return None;
    }

    Some(GpuTelemetrySample {
        elapsed_ms: elapsed.as_millis().try_into().unwrap_or(u64::MAX),
        device_index: fields[0].parse().ok()?,
        device_name: fields[1].to_owned(),
        memory_used_mib: parse_number(fields[2])?,
        memory_free_mib: parse_number(fields[3])?,
        utilization_percent: parse_number(fields[4])?,
        temperature_celsius: parse_number(fields[5])?,
        power_watts: parse_number(fields[6])?,
    })
}

fn parse_number(value: &str) -> Option<f64> {
    value.parse().ok().filter(|value: &f64| value.is_finite())
}

fn summarize(samples: &[GpuTelemetrySample]) -> GpuTelemetrySummary {
    GpuTelemetrySummary {
        device_name: samples.first().map(|sample| sample.device_name.clone()),
        sample_count: samples.len(),
        peak_memory_used_mib: maximum(samples.iter().map(|sample| sample.memory_used_mib)),
        minimum_memory_free_mib: minimum(samples.iter().map(|sample| sample.memory_free_mib)),
        peak_utilization_percent: maximum(samples.iter().map(|sample| sample.utilization_percent)),
        peak_temperature_celsius: maximum(samples.iter().map(|sample| sample.temperature_celsius)),
        average_power_watts: average(samples.iter().map(|sample| sample.power_watts)),
    }
}

fn maximum(values: impl Iterator<Item = f64>) -> Option<f64> {
    values.max_by(f64::total_cmp)
}

fn minimum(values: impl Iterator<Item = f64>) -> Option<f64> {
    values.min_by(f64::total_cmp)
}

fn average(values: impl Iterator<Item = f64>) -> Option<f64> {
    let (sum, count) = values.fold((0.0, 0_u64), |(sum, count), value| (sum + value, count + 1));
    (count > 0).then_some(sum / count as f64)
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::{parse_sample, summarize};

    #[test]
    fn parses_and_summarizes_nvidia_csv() {
        let first = parse_sample(
            "0, NVIDIA GeForce RTX 2050, 1200, 2800, 75, 61, 32.5",
            Duration::from_millis(100),
        )
        .unwrap();
        let second = parse_sample(
            "0, NVIDIA GeForce RTX 2050, 1500, 2500, 90, 66, 37.5",
            Duration::from_millis(200),
        )
        .unwrap();

        let summary = summarize(&[first, second]);

        assert_eq!(summary.sample_count, 2);
        assert_eq!(summary.peak_memory_used_mib, Some(1500.0));
        assert_eq!(summary.minimum_memory_free_mib, Some(2500.0));
        assert_eq!(summary.peak_utilization_percent, Some(90.0));
        assert_eq!(summary.average_power_watts, Some(35.0));
    }
}
