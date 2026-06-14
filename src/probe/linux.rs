use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{Context, Result};
use chrono::Utc;

use crate::model::{
    AcceleratorInfo, AcceleratorKind, CpuInfo, HardwareSnapshot, MemoryInfo, PowerInfo,
    ThermalReading,
};
use crate::probe::HardwareProbe;

pub struct LinuxProbe;

impl HardwareProbe for LinuxProbe {
    fn capture(&self) -> Result<HardwareSnapshot> {
        Ok(HardwareSnapshot {
            captured_at: Utc::now().to_rfc3339(),
            hostname: read_trimmed("/proc/sys/kernel/hostname").unwrap_or_else(|| "unknown".into()),
            os: read_os_name(),
            kernel: read_trimmed("/proc/sys/kernel/osrelease").unwrap_or_else(|| "unknown".into()),
            cpu: probe_cpu()?,
            memory: probe_memory()?,
            accelerators: probe_accelerators(),
            thermals: probe_thermals(),
            power: probe_power(),
        })
    }
}

fn probe_cpu() -> Result<CpuInfo> {
    let cpuinfo = fs::read_to_string("/proc/cpuinfo").context("failed to read /proc/cpuinfo")?;
    let model = cpuinfo
        .lines()
        .find_map(|line| {
            let (key, value) = line.split_once(':')?;
            matches!(key.trim(), "model name" | "Hardware").then(|| value.trim().to_owned())
        })
        .unwrap_or_else(|| "unknown".into());

    Ok(CpuInfo {
        model,
        architecture: std::env::consts::ARCH.into(),
        logical_cores: std::thread::available_parallelism()
            .map(usize::from)
            .unwrap_or(1),
        physical_cores: count_physical_cores(),
    })
}

fn count_physical_cores() -> Option<usize> {
    let mut cores = HashSet::new();
    let entries = fs::read_dir("/sys/devices/system/cpu").ok()?;

    for entry in entries.flatten() {
        let name = entry.file_name();
        let name = name.to_string_lossy();
        let Some(cpu_number) = name.strip_prefix("cpu") else {
            continue;
        };
        if cpu_number.parse::<usize>().is_err() {
            continue;
        }

        let topology = entry.path().join("topology");
        let package = read_trimmed(topology.join("physical_package_id"))?;
        let core = read_trimmed(topology.join("core_id"))?;
        cores.insert((package, core));
    }

    (!cores.is_empty()).then_some(cores.len())
}

fn probe_memory() -> Result<MemoryInfo> {
    let meminfo = fs::read_to_string("/proc/meminfo").context("failed to read /proc/meminfo")?;
    let values: HashMap<&str, u64> = meminfo
        .lines()
        .filter_map(|line| {
            let (key, value) = line.split_once(':')?;
            let kib = value.split_whitespace().next()?.parse::<u64>().ok()?;
            Some((key, kib * 1024))
        })
        .collect();

    Ok(MemoryInfo {
        total_bytes: values.get("MemTotal").copied().unwrap_or(0),
        available_bytes: values.get("MemAvailable").copied().unwrap_or(0),
        swap_total_bytes: values.get("SwapTotal").copied().unwrap_or(0),
        swap_free_bytes: values.get("SwapFree").copied().unwrap_or(0),
    })
}

fn probe_accelerators() -> Vec<AcceleratorInfo> {
    let mut accelerators = Vec::new();
    let Ok(entries) = fs::read_dir("/sys/class/drm") else {
        return accelerators;
    };

    for entry in entries.flatten() {
        let card_name = entry.file_name().to_string_lossy().into_owned();
        if !is_primary_card_name(&card_name) {
            continue;
        }

        let device_path = entry.path().join("device");
        let vendor_id = read_trimmed(device_path.join("vendor")).unwrap_or_default();
        let device_id = read_trimmed(device_path.join("device")).unwrap_or_default();
        let vendor = vendor_name(&vendor_id).to_owned();
        let driver = fs::read_link(device_path.join("driver"))
            .ok()
            .and_then(|path| {
                path.file_name()
                    .map(|name| name.to_string_lossy().into_owned())
            });
        let boot_vga = read_trimmed(device_path.join("boot_vga")).as_deref() == Some("1");
        let kind = if vendor_id == "0x10de" || !boot_vga {
            AcceleratorKind::DiscreteGpu
        } else {
            AcceleratorKind::IntegratedGpu
        };
        let name = match driver {
            Some(driver) => format!("{vendor} GPU {device_id} ({driver})"),
            None => format!("{vendor} GPU {device_id}"),
        };
        let dedicated_memory_bytes = read_vram_total(&device_path);
        let available_memory_bytes =
            read_available_vram(&device_path, &vendor_id, dedicated_memory_bytes);

        accelerators.push(AcceleratorInfo {
            kind,
            name,
            vendor,
            device_path: Some(format!("/sys/class/drm/{card_name}")),
            dedicated_memory_bytes,
            available_memory_bytes,
            telemetry_available: available_memory_bytes.is_some()
                || telemetry_available(&device_path, &vendor_id),
        });
    }

    accelerators
}

fn is_primary_card_name(name: &str) -> bool {
    name.strip_prefix("card")
        .is_some_and(|suffix| !suffix.is_empty() && suffix.chars().all(|c| c.is_ascii_digit()))
}

fn vendor_name(vendor_id: &str) -> &'static str {
    match vendor_id {
        "0x10de" => "NVIDIA",
        "0x1002" => "AMD",
        "0x8086" => "Intel",
        "0x106b" => "Apple",
        "0x17cb" => "Qualcomm",
        _ => "Unknown",
    }
}

fn read_vram_total(device_path: &Path) -> Option<u64> {
    ["mem_info_vram_total", "mem_info_vis_vram_total"]
        .iter()
        .find_map(|file| read_trimmed(device_path.join(file))?.parse().ok())
}

fn read_available_vram(
    device_path: &Path,
    vendor_id: &str,
    total_bytes: Option<u64>,
) -> Option<u64> {
    match vendor_id {
        "0x1002" => {
            let total = total_bytes?;
            let used = read_trimmed(device_path.join("mem_info_vram_used"))?
                .parse::<u64>()
                .ok()?;
            Some(total.saturating_sub(used))
        }
        "0x10de" => nvidia_free_memory_bytes(),
        _ => None,
    }
}

fn nvidia_free_memory_bytes() -> Option<u64> {
    let output = Command::new("nvidia-smi")
        .args(["--query-gpu=memory.free", "--format=csv,noheader,nounits"])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }

    let output = String::from_utf8(output.stdout).ok()?;
    let free_mib = output.lines().next()?.trim().parse::<u64>().ok()?;
    free_mib.checked_mul(1024 * 1024)
}

fn telemetry_available(device_path: &Path, vendor_id: &str) -> bool {
    match vendor_id {
        "0x10de" => Command::new("nvidia-smi")
            .args(["--query-gpu=name", "--format=csv,noheader"])
            .output()
            .is_ok_and(|output| output.status.success()),
        "0x1002" => {
            device_path.join("gpu_busy_percent").is_file() || device_path.join("hwmon").is_dir()
        }
        _ => false,
    }
}

fn probe_thermals() -> Vec<ThermalReading> {
    let mut readings = Vec::new();
    let Ok(entries) = fs::read_dir("/sys/class/thermal") else {
        return readings;
    };

    for entry in entries.flatten() {
        if !entry
            .file_name()
            .to_string_lossy()
            .starts_with("thermal_zone")
        {
            continue;
        }

        let path = entry.path();
        let Some(raw_temp) = read_trimmed(path.join("temp")).and_then(|value| value.parse().ok())
        else {
            continue;
        };
        let source = read_trimmed(path.join("type"))
            .unwrap_or_else(|| entry.file_name().to_string_lossy().into_owned());
        let temperature_celsius = if raw_temp > 1000.0 {
            raw_temp / 1000.0
        } else {
            raw_temp
        };

        readings.push(ThermalReading {
            source,
            temperature_celsius,
        });
    }

    readings
}

fn probe_power() -> PowerInfo {
    let mut power = PowerInfo {
        on_ac: None,
        battery_percent: None,
        battery_status: None,
    };
    let Ok(entries) = fs::read_dir("/sys/class/power_supply") else {
        return power;
    };

    for entry in entries.flatten() {
        let path = entry.path();
        match read_trimmed(path.join("type")).as_deref() {
            Some("Mains") | Some("USB") | Some("USB_C") => {
                if let Some(online) = read_trimmed(path.join("online")) {
                    power.on_ac = Some(online == "1");
                }
            }
            Some("Battery") => {
                power.battery_percent =
                    read_trimmed(path.join("capacity")).and_then(|value| value.parse().ok());
                power.battery_status = read_trimmed(path.join("status"));
            }
            _ => {}
        }
    }

    power
}

fn read_os_name() -> String {
    let Ok(content) = fs::read_to_string("/etc/os-release") else {
        return "Linux".into();
    };

    content
        .lines()
        .find_map(|line| {
            let value = line.strip_prefix("PRETTY_NAME=")?;
            Some(value.trim_matches('"').into())
        })
        .unwrap_or_else(|| "Linux".into())
}

fn read_trimmed(path: impl Into<PathBuf>) -> Option<String> {
    fs::read_to_string(path.into())
        .ok()
        .map(|value| value.trim().to_owned())
        .filter(|value| !value.is_empty())
}

#[cfg(test)]
mod tests {
    use super::is_primary_card_name;

    #[test]
    fn identifies_only_primary_drm_cards() {
        assert!(is_primary_card_name("card0"));
        assert!(is_primary_card_name("card12"));
        assert!(!is_primary_card_name("card0-DP-1"));
        assert!(!is_primary_card_name("renderD128"));
        assert!(!is_primary_card_name("card"));
    }
}
