use std::{sync::Arc, time::Duration};

use litellm_traces::{Trace, query::named::ReadAccessParams};
use moka::future::Cache;
use sha2::{Digest, Sha256};

mod error;

pub use error::Error;

#[derive(Clone, Eq, Hash, PartialEq)]
pub struct SnapshotKey(String);

impl SnapshotKey {
    pub fn new(
        source: &str,
        access: &ReadAccessParams,
        trace_id: &str,
        trace_ref: &str,
        snapshot_ms: u64,
    ) -> Result<Self, Error> {
        let encoded = serde_json::to_vec(&(source, access, trace_id, trace_ref, snapshot_ms))?;
        Ok(Self(format!("{:x}", Sha256::digest(encoded))))
    }
}

pub struct Snapshot {
    trace: Trace,
    version: String,
    weight: u32,
}

impl Snapshot {
    pub fn trace(&self) -> &Trace {
        &self.trace
    }

    pub fn version(&self) -> &str {
        &self.version
    }
}

pub struct SnapshotCache {
    entries: Cache<SnapshotKey, Arc<Snapshot>>,
    max_graph_bytes: usize,
}

impl SnapshotCache {
    pub fn new(max_graph_bytes: usize, ttl: Duration) -> Self {
        Self {
            entries: Cache::builder()
                .max_capacity((max_graph_bytes as u64).saturating_mul(2))
                .weigher(|_: &SnapshotKey, snapshot: &Arc<Snapshot>| snapshot.weight)
                .time_to_live(ttl)
                .build(),
            max_graph_bytes,
        }
    }

    pub async fn get(&self, key: &SnapshotKey) -> Option<Arc<Snapshot>> {
        self.entries.get(key).await
    }

    pub async fn insert(&self, key: SnapshotKey, trace: Trace) -> Result<Arc<Snapshot>, Error> {
        let encoded = serde_json::to_vec(&trace)?;
        if encoded.len() > self.max_graph_bytes {
            return Err(Error::ReadTooLarge);
        }

        let span_ids: Vec<&str> = trace
            .spans
            .iter()
            .map(|span| span.span_id.as_str())
            .collect();

        let version = format!("{:x}", Sha256::digest(serde_json::to_vec(&span_ids)?));

        let snapshot = Arc::new(Snapshot {
            trace,
            version,
            weight: u32::try_from(encoded.len().saturating_mul(2)).unwrap_or(u32::MAX),
        });

        self.entries.insert(key, Arc::clone(&snapshot)).await;
        Ok(snapshot)
    }
}

#[cfg(test)]
mod tests {
    use litellm_traces::{
        SpanStatus,
        query::named::{SpendByResponseIdsRow, TraceSpansRow},
        resolve_trace,
    };

    use super::*;

    fn trace(span_id: &str) -> Trace {
        let rows = [TraceSpansRow {
            trace_id: String::new(),
            span_id: span_id.into(),
            parent_span_id: String::new(),
            name: "run".into(),
            kind: litellm_traces::ObservationType::Agent,
            wrapper_candidate: false,
            agent: "agent".into(),
            framework: String::new(),
            status: SpanStatus::Ok,
            status_message: String::new(),
            error_truncated: false,
            start_ns: 1_790_742_989_000_000_000,
            duration_ns: 10_000_000,
            service: "agent-demo".into(),
            input_preview: format!("input of {span_id}"),
            model: String::new(),
            input_tokens: 0,
            output_tokens: 0,
            litellm_request_id: String::new(),
            call_keys: Vec::new(),
            call_evidence: None,
            tool_call_id: String::new(),
            team_id: String::new(),
            api_key_hash: String::new(),
            user_id: String::new(),
        }];
        resolve_trace("trace", "ref", &rows, &[] as &[SpendByResponseIdsRow])
            .expect("fixture should resolve")
    }

    fn key(suffix: &str) -> SnapshotKey {
        SnapshotKey::new(
            "source",
            &ReadAccessParams {
                all_teams: false,
                user_id: String::new(),
                team_ids: vec!["team".into()],
            },
            suffix,
            "ref",
            100,
        )
        .unwrap()
    }

    #[tokio::test]
    async fn weighted_capacity_bounds_retained_snapshots() {
        let limit = ["first", "second", "third"]
            .iter()
            .map(|span_id| serde_json::to_vec(&trace(span_id)).unwrap().len())
            .max()
            .unwrap();
        let cache = SnapshotCache::new(limit, Duration::from_secs(120));

        for (key, span_id) in [
            (key("a"), "first"),
            (key("b"), "second"),
            (key("c"), "third"),
        ] {
            cache.insert(key, trace(span_id)).await.unwrap();
        }

        cache.entries.run_pending_tasks().await;
        assert!(cache.entries.weighted_size() <= (limit as u64) * 2);
    }
}
