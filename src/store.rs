use std::fs;
use std::path::Path;

use anyhow::{Context, Result};
use rusqlite::{Connection, params};

use crate::model::HardwareSnapshot;
use crate::policy::PlanningDecision;

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

            CREATE TABLE IF NOT EXISTS planning_decisions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                snapshot_id INTEGER NOT NULL,
                model_path TEXT NOT NULL,
                optimization_goal TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(snapshot_id) REFERENCES hardware_snapshots(id)
            );

            CREATE INDEX IF NOT EXISTS idx_planning_decisions_snapshot
                ON planning_decisions(snapshot_id);
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

    pub fn insert_planning_decision(
        &self,
        snapshot_id: i64,
        decision: &PlanningDecision,
    ) -> Result<i64> {
        let payload = serde_json::to_string(decision)?;
        self.connection.execute(
            "INSERT INTO planning_decisions
                (snapshot_id, model_path, optimization_goal, payload_json)
             VALUES (?1, ?2, ?3, ?4)",
            params![
                snapshot_id,
                decision.model.path.to_string_lossy(),
                format!("{:?}", decision.workload.optimization_goal),
                payload
            ],
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

    use std::path::PathBuf;

    use crate::gguf::ModelMetadata;
    use crate::model::{CpuInfo, HardwareSnapshot, MemoryInfo, PowerInfo};
    use crate::policy::{
        GpuPlacement, OptimizationGoal, PlanningOverrides, UseCase, WorkloadSpec, plan,
    };

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
                total_bytes: 16 * 1024_u64.pow(3),
                available_bytes: 8 * 1024_u64.pow(3),
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
        let decision = plan(
            &expected,
            ModelMetadata {
                path: PathBuf::from("/models/test.gguf"),
                file_size_bytes: 1,
                gguf_version: 3,
                name: Some("Test".into()),
                architecture: "llama".into(),
                block_count: 1,
                context_length: Some(4096),
                embedding_length: Some(128),
                attention_head_count: Some(1),
                attention_head_count_kv: Some(1),
                attention_key_length: None,
                attention_value_length: None,
            },
            WorkloadSpec {
                use_case: UseCase::Interactive,
                optimization_goal: OptimizationGoal::Balanced,
                prompt_tokens: 64,
                output_tokens: 64,
                concurrency: 1,
            },
            PlanningOverrides::default(),
        )
        .unwrap();
        store
            .insert_planning_decision(snapshot_id, &decision)
            .unwrap();
        let snapshots = store.recent_snapshots(1).unwrap();

        assert_eq!(snapshots, vec![expected]);
        assert_eq!(decision.selected.gpu_placement, GpuPlacement::CpuOnly);
    }
}
