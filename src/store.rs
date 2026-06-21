use std::fs;
use std::path::Path;

use anyhow::{Context, Result};
use rusqlite::{Connection, params};

use crate::backend::llamacpp::{BenchmarkRun, InferenceResult};
use crate::model::HardwareSnapshot;
use crate::policy::{ExecutionProfile, PlanningDecision, runtime_configuration_matches};

#[derive(Debug, Clone)]
pub struct CalibrationMatch {
    pub profile: ExecutionProfile,
    pub effective_tokens_per_second: f64,
}

#[derive(Debug)]
struct CalibrationRecord {
    planning_decision_id: i64,
    benchmark_json: String,
    planning_json: String,
    snapshot_json: String,
}

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

            CREATE TABLE IF NOT EXISTS inference_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                snapshot_id INTEGER NOT NULL,
                planning_decision_id INTEGER NOT NULL,
                model_path TEXT NOT NULL,
                backend TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(snapshot_id) REFERENCES hardware_snapshots(id),
                FOREIGN KEY(planning_decision_id) REFERENCES planning_decisions(id)
            );

            CREATE INDEX IF NOT EXISTS idx_inference_runs_planning_decision
                ON inference_runs(planning_decision_id);

            CREATE TABLE IF NOT EXISTS benchmark_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                snapshot_id INTEGER NOT NULL,
                planning_decision_id INTEGER NOT NULL,
                model_path TEXT NOT NULL,
                backend TEXT NOT NULL,
                effective_tokens_per_second REAL NOT NULL,
                payload_json TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(snapshot_id) REFERENCES hardware_snapshots(id),
                FOREIGN KEY(planning_decision_id) REFERENCES planning_decisions(id)
            );

            CREATE INDEX IF NOT EXISTS idx_benchmark_runs_planning_decision
                ON benchmark_runs(planning_decision_id);
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

    pub fn insert_inference_run(
        &self,
        snapshot_id: i64,
        planning_decision_id: i64,
        model_path: &Path,
        result: &InferenceResult,
    ) -> Result<i64> {
        let payload = serde_json::to_string(result)?;
        self.connection.execute(
            "INSERT INTO inference_runs
                (snapshot_id, planning_decision_id, model_path, backend, payload_json)
             VALUES (?1, ?2, ?3, ?4, ?5)",
            params![
                snapshot_id,
                planning_decision_id,
                model_path.to_string_lossy(),
                result.backend,
                payload
            ],
        )?;
        Ok(self.connection.last_insert_rowid())
    }

    pub fn insert_benchmark_run(
        &self,
        snapshot_id: i64,
        planning_decision_id: i64,
        model_path: &Path,
        run: &BenchmarkRun,
    ) -> Result<i64> {
        let payload = serde_json::to_string(run)?;
        self.connection.execute(
            "INSERT INTO benchmark_runs
                (snapshot_id, planning_decision_id, model_path, backend,
                 effective_tokens_per_second, payload_json)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            params![
                snapshot_id,
                planning_decision_id,
                model_path.to_string_lossy(),
                run.backend,
                run.effective_tokens_per_second,
                payload
            ],
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

    pub fn latest_compatible_calibration(
        &self,
        current_snapshot: &HardwareSnapshot,
        current_decision: &PlanningDecision,
    ) -> Result<Option<CalibrationMatch>> {
        let mut statement = self.connection.prepare(
            "SELECT br.planning_decision_id, br.payload_json,
                    pd.payload_json, hs.payload_json
             FROM benchmark_runs br
             JOIN planning_decisions pd ON pd.id = br.planning_decision_id
             JOIN hardware_snapshots hs ON hs.id = br.snapshot_id
             WHERE br.backend = 'cuda'
             ORDER BY br.id DESC
             LIMIT 500",
        )?;
        let records = statement
            .query_map([], |row| {
                Ok(CalibrationRecord {
                    planning_decision_id: row.get(0)?,
                    benchmark_json: row.get(1)?,
                    planning_json: row.get(2)?,
                    snapshot_json: row.get(3)?,
                })
            })?
            .collect::<rusqlite::Result<Vec<_>>>()?;

        let mut offset = 0;
        while offset < records.len() {
            let planning_decision_id = records[offset].planning_decision_id;
            let end = records[offset..]
                .iter()
                .position(|record| record.planning_decision_id != planning_decision_id)
                .map(|position| offset + position)
                .unwrap_or(records.len());
            let group = &records[offset..end];
            offset = end;

            let Some(stored_decision) = group.first().and_then(|record| {
                serde_json::from_str::<PlanningDecision>(&record.planning_json).ok()
            }) else {
                continue;
            };
            let Some(stored_snapshot) = group.first().and_then(|record| {
                serde_json::from_str::<HardwareSnapshot>(&record.snapshot_json).ok()
            }) else {
                continue;
            };
            if !calibration_context_matches(
                current_snapshot,
                current_decision,
                &stored_snapshot,
                &stored_decision,
            ) {
                continue;
            }

            let best = group
                .iter()
                .filter_map(|record| {
                    serde_json::from_str::<BenchmarkRun>(&record.benchmark_json).ok()
                })
                .filter(|run| {
                    run.effective_tokens_per_second.is_finite()
                        && run.effective_tokens_per_second > 0.0
                })
                .filter_map(|run| {
                    current_decision
                        .candidates
                        .iter()
                        .find(|candidate| runtime_configuration_matches(candidate, &run.profile))
                        .cloned()
                        .map(|profile| CalibrationMatch {
                            profile,
                            effective_tokens_per_second: run.effective_tokens_per_second,
                        })
                })
                .max_by(|left, right| {
                    left.effective_tokens_per_second
                        .total_cmp(&right.effective_tokens_per_second)
                });
            if best.is_some() {
                return Ok(best);
            }
        }

        Ok(None)
    }
}

fn calibration_context_matches(
    current_snapshot: &HardwareSnapshot,
    current_decision: &PlanningDecision,
    stored_snapshot: &HardwareSnapshot,
    stored_decision: &PlanningDecision,
) -> bool {
    current_decision.workload == stored_decision.workload
        && model_identity_matches(current_decision, stored_decision)
        && hardware_identity(current_snapshot) == hardware_identity(stored_snapshot)
}

fn model_identity_matches(current: &PlanningDecision, stored: &PlanningDecision) -> bool {
    current.model.file_size_bytes == stored.model.file_size_bytes
        && current.model.gguf_version == stored.model.gguf_version
        && current.model.architecture == stored.model.architecture
        && current.model.block_count == stored.model.block_count
        && current.model.embedding_length == stored.model.embedding_length
        && current.model.attention_head_count == stored.model.attention_head_count
        && current.model.attention_head_count_kv == stored.model.attention_head_count_kv
}

fn hardware_identity(
    snapshot: &HardwareSnapshot,
) -> (String, String, usize, Option<usize>, Vec<String>) {
    let mut accelerators: Vec<_> = snapshot
        .accelerators
        .iter()
        .map(|accelerator| {
            format!(
                "{:?}|{}|{}|{:?}",
                accelerator.kind,
                accelerator.vendor,
                accelerator.name,
                accelerator.dedicated_memory_bytes
            )
        })
        .collect();
    accelerators.sort();
    (
        snapshot.cpu.model.clone(),
        snapshot.cpu.architecture.clone(),
        snapshot.cpu.logical_cores,
        snapshot.cpu.physical_cores,
        accelerators,
    )
}

#[cfg(test)]
mod tests {
    use tempfile::tempdir;

    use std::path::PathBuf;

    use crate::backend::llamacpp::{
        BenchmarkMeasurement, BenchmarkRun, InferenceResult, TokenUsage,
    };
    use crate::backend::nvidia::GpuTelemetrySummary;
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
        let decision_id = store
            .insert_planning_decision(snapshot_id, &decision)
            .unwrap();
        let run_id = store
            .insert_inference_run(
                snapshot_id,
                decision_id,
                &decision.model.path,
                &InferenceResult {
                    backend: "cuda".into(),
                    model: "test.gguf".into(),
                    response: "ok".into(),
                    reasoning: None,
                    finish_reason: Some("stop".into()),
                    usage: TokenUsage {
                        prompt_tokens: 1,
                        completion_tokens: 1,
                        total_tokens: 2,
                    },
                    timings: None,
                    wall_time_ms: 10,
                    gpu: GpuTelemetrySummary {
                        device_name: Some("Test GPU".into()),
                        sample_count: 1,
                        peak_memory_used_mib: Some(100.0),
                        minimum_memory_free_mib: Some(900.0),
                        peak_utilization_percent: Some(50.0),
                        peak_temperature_celsius: Some(60.0),
                        average_power_watts: Some(20.0),
                    },
                },
            )
            .unwrap();
        let benchmark_id = store
            .insert_benchmark_run(
                snapshot_id,
                decision_id,
                &decision.model.path,
                &BenchmarkRun {
                    backend: "cuda".into(),
                    profile: decision.selected.clone(),
                    measurements: vec![BenchmarkMeasurement {
                        n_prompt: 64,
                        n_gen: 0,
                        avg_ns: 1,
                        avg_ts: 100.0,
                        stddev_ts: 0.0,
                    }],
                    effective_tokens_per_second: 100.0,
                    wall_time_ms: 10,
                    gpu: GpuTelemetrySummary {
                        device_name: Some("Test GPU".into()),
                        sample_count: 1,
                        peak_memory_used_mib: Some(100.0),
                        minimum_memory_free_mib: Some(900.0),
                        peak_utilization_percent: Some(50.0),
                        peak_temperature_celsius: Some(60.0),
                        average_power_watts: Some(20.0),
                    },
                },
            )
            .unwrap();
        let snapshots = store.recent_snapshots(1).unwrap();
        let calibration = store
            .latest_compatible_calibration(&expected, &decision)
            .unwrap()
            .unwrap();
        let mut different_hardware = expected.clone();
        different_hardware.cpu.model = "Different CPU".into();

        assert!(run_id > 0);
        assert!(benchmark_id > 0);
        assert_eq!(snapshots, vec![expected]);
        assert_eq!(calibration.effective_tokens_per_second, 100.0);
        assert!(
            store
                .latest_compatible_calibration(&different_hardware, &decision)
                .unwrap()
                .is_none()
        );
        assert_eq!(decision.selected.gpu_placement, GpuPlacement::CpuOnly);
    }
}
