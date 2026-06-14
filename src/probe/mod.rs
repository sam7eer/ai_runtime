#[cfg(target_os = "linux")]
mod linux;

use anyhow::Result;

use crate::model::HardwareSnapshot;

pub trait HardwareProbe {
    fn capture(&self) -> Result<HardwareSnapshot>;
}

#[cfg(target_os = "linux")]
pub fn system_probe() -> Result<Box<dyn HardwareProbe>> {
    Ok(Box::new(linux::LinuxProbe))
}

#[cfg(not(target_os = "linux"))]
pub fn system_probe() -> Result<Box<dyn HardwareProbe>> {
    anyhow::bail!("hardware probing is currently implemented only for Linux")
}
