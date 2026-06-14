use std::fs;
use std::path::Path;

use anyhow::{Context, Result};
use rusqlite::{Connection, params};

use crate::model::HardwareSnapshot;
use crate::policy::ExecutionProfile;

pub struct Store {
    connection: Connection,
}

impl Store {
    pub fn open(path: &Path) -> Result<Self> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).with_context(|| {
                format!("failed to create database directory {}", parent.display())
            })?;
        }

        let connection = Connection::open(path)
            .with_context(|| format!("failed to open database {}", path.display()))?;
        let store = Self { connection };
        store.migrate()?;
        Ok(store)
    }

    fn migrate(&self) -> Result<()> {
        self.connection.execute_batch(
            "
            PRAGMA journal_mode = WAL;
            PRAGMA foreign_keys = ON;

            CREATE TABLE IF NOT EXISTS hardware_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                captured_at TEXT NOT NULL,
                payload_json TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_hardware_snapshots_captured_at
                ON hardware_snapshots(captured_at DESC);

            CREATE TABLE IF NOT EXISTS profile_recommendations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                snapshot_id INTEGER NOT NULL,
                mode TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(snapshot_id) REFERENCES hardware_snapshots(id)
            );

            CREATE INDEX IF NOT EXISTS idx_profile_recommendations_snapshot
                ON profile_recommendations(snapshot_id);
            ",
        )?;
        Ok(())
    }

    pub fn insert_snapshot(&self, snapshot: &HardwareSnapshot) -> Result<i64> {
        let payload = serde_json::to_string(snapshot)?;
        self.connection.execute(
            "INSERT INTO hardware_snapshots (captured_at, payload_json) VALUES (?1, ?2)",
            params![snapshot.captured_at, payload],
        )?;
        Ok(self.connection.last_insert_rowid())
    }

    pub fn insert_recommendation(
        &self,
        snapshot_id: i64,
        profile: &ExecutionProfile,
    ) -> Result<i64> {
        let payload = serde_json::to_string(profile)?;
        self.connection.execute(
            "INSERT INTO profile_recommendations (snapshot_id, mode, payload_json)
             VALUES (?1, ?2, ?3)",
            params![snapshot_id, format!("{:?}", profile.mode), payload],
        )?;
        Ok(self.connection.last_insert_rowid())
    }

    pub fn recent_snapshots(&self, limit: usize) -> Result<Vec<HardwareSnapshot>> {
        let mut statement = self.connection.prepare(
            "SELECT payload_json
             FROM hardware_snapshots
             ORDER BY captured_at DESC, id DESC
             LIMIT ?1",
        )?;
        let rows = statement.query_map([limit as i64], |row| row.get::<_, String>(0))?;

        rows.map(|row| {
            let json = row?;
            serde_json::from_str(&json).map_err(Into::into)
        })
        .collect()
    }
}

#[cfg(test)]
mod tests {
    use tempfile::tempdir;

    use crate::model::{CpuInfo, HardwareSnapshot, MemoryInfo, PowerInfo};
    use crate::policy::{GpuOffload, RuntimeMode, recommend};

    use super::Store;

    fn snapshot() -> HardwareSnapshot {
        HardwareSnapshot {
            captured_at: "2026-06-14T00:00:00Z".into(),
            hostname: "test".into(),
            os: "Linux".into(),
            kernel: "test".into(),
            cpu: CpuInfo {
                model: "Test CPU".into(),
                architecture: "x86_64".into(),
                logical_cores: 8,
                physical_cores: Some(4),
            },
            memory: MemoryInfo {
                total_bytes: 16,
                available_bytes: 8,
                swap_total_bytes: 0,
                swap_free_bytes: 0,
            },
            accelerators: vec![],
            thermals: vec![],
            power: PowerInfo {
                on_ac: None,
                battery_percent: None,
                battery_status: None,
            },
        }
    }

    #[test]
    fn persists_and_loads_snapshots() {
        let directory = tempdir().unwrap();
        let store = Store::open(&directory.path().join("runtime.db")).unwrap();
        let expected = snapshot();

        let snapshot_id = store.insert_snapshot(&expected).unwrap();
        let profile = recommend(&expected, RuntimeMode::Balanced);
        store.insert_recommendation(snapshot_id, &profile).unwrap();
        let snapshots = store.recent_snapshots(1).unwrap();

        assert_eq!(snapshots, vec![expected]);
        assert_eq!(profile.gpu_offload, GpuOffload::Disabled);
    }
}
