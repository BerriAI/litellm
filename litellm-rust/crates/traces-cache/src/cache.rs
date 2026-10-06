use std::{future::Future, sync::Arc, time::Duration};

use litellm_traces::{
    Trace, TraceSummary,
    query::named::{ReadAccessParams, TraceSpansRow},
};
use moka::{Expiry, future::Cache};
use serde::Serialize;
use sha2::{Digest, Sha256};

use crate::Error;

pub const LIVE_TTL: Duration = Duration::from_secs(5);
pub const SETTLED_TTL: Duration = Duration::from_secs(10 * 60);
const SETTLED_AFTER_MS: u64 = 5 * 60 * 1000;
const MAX_INDEX_ENTRIES: u64 = 100_000;

#[derive(Clone, Eq, Hash, PartialEq)]
pub struct SnapshotKey(String);

impl SnapshotKey {
    fn digest(fields: &impl Serialize) -> Result<Self, Error> {
        Ok(Self(format!(
            "{:x}",
            Sha256::digest(serde_json::to_vec(fields)?)
        )))
    }

    pub fn new(
        source: &str,
        access: &ReadAccessParams,
        trace_id: &str,
        trace_ref: &str,
        snapshot_ms: u64,
    ) -> Result<Self, Error> {
        Self::digest(&(source, access, trace_id, trace_ref, snapshot_ms))
    }

    pub fn latest(
        source: &str,
        access: &ReadAccessParams,
        trace_id: &str,
        trace_ref: &str,
    ) -> Result<Self, Error> {
        Self::digest(&(source, access, trace_id, trace_ref))
    }

    pub(crate) fn run(
        source: &str,
        access: &ReadAccessParams,
        run: (&str, &str, &str, &str),
    ) -> Result<Self, Error> {
        Self::digest(&("run", source, access, run))
    }

    pub(crate) fn scope(source: &str, access: &ReadAccessParams) -> Result<Self, Error> {
        Self::digest(&("scope", source, access))
    }
}

/// How long a read result stays reusable: traces still receiving spans, or read with spend
/// unavailable, are re-read after `LIVE_TTL`; traces quiet for `SETTLED_AFTER_MS` are kept for
/// `SETTLED_TTL`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Freshness {
    Live,
    Settled,
}

impl Freshness {
    pub fn of(rows: &[TraceSpansRow], spend_known: bool, snapshot_ms: u64) -> Self {
        let last_end_ms = rows
            .iter()
            .map(|row| row.start_ns.saturating_add_unsigned(row.duration_ns) / 1_000_000)
            .max()
            .unwrap_or(i64::MAX);
        let quiet_ms = i64::try_from(snapshot_ms)
            .unwrap_or(i64::MAX)
            .saturating_sub(last_end_ms);
        if spend_known && quiet_ms >= SETTLED_AFTER_MS as i64 {
            Self::Settled
        } else {
            Self::Live
        }
    }

    fn ttl(self) -> Duration {
        match self {
            Self::Live => LIVE_TTL,
            Self::Settled => SETTLED_TTL,
        }
    }
}

trait Fresh {
    fn freshness(&self) -> Freshness;
}

struct ByFreshness;

impl<K, V: Fresh> Expiry<K, V> for ByFreshness {
    fn expire_after_create(&self, _: &K, value: &V, _: std::time::Instant) -> Option<Duration> {
        Some(value.freshness().ttl())
    }
}

pub struct Snapshot {
    trace: Trace,
    version: String,
    snapshot_ms: u64,
    freshness: Freshness,
    weight: u32,
}

impl Snapshot {
    pub fn trace(&self) -> &Trace {
        &self.trace
    }

    pub fn version(&self) -> &str {
        &self.version
    }

    pub fn snapshot_ms(&self) -> u64 {
        self.snapshot_ms
    }

    pub fn freshness(&self) -> Freshness {
        self.freshness
    }
}

#[derive(Clone, Copy)]
struct Latest {
    snapshot_ms: u64,
    freshness: Freshness,
}

impl Fresh for Latest {
    fn freshness(&self) -> Freshness {
        self.freshness
    }
}

/// Resolved trace snapshots pinned by `snapshot_ms` for paging, plus which snapshot each trace
/// currently serves so repeated opens reuse one read until its freshness expires.
pub struct SnapshotCache {
    pinned: Cache<SnapshotKey, Arc<Snapshot>>,
    latest: Cache<SnapshotKey, Latest>,
    max_graph_bytes: usize,
}

impl SnapshotCache {
    pub fn new(max_graph_bytes: usize, idle: Duration) -> Self {
        Self {
            pinned: Cache::builder()
                .max_capacity((max_graph_bytes as u64).saturating_mul(2))
                .weigher(|_: &SnapshotKey, snapshot: &Arc<Snapshot>| snapshot.weight)
                .time_to_idle(idle)
                .build(),
            latest: Cache::builder()
                .max_capacity(MAX_INDEX_ENTRIES)
                .expire_after(ByFreshness)
                .build(),
            max_graph_bytes,
        }
    }

    pub async fn get(&self, key: &SnapshotKey) -> Option<Arc<Snapshot>> {
        self.pinned.get(key).await
    }

    /// Returns the snapshot pinned at `key`, running `load` once for all concurrent callers on a
    /// miss. A failed load is not cached.
    pub async fn pinned_or_load<E, F>(
        &self,
        key: SnapshotKey,
        snapshot_ms: u64,
        load: F,
    ) -> Result<Arc<Snapshot>, Arc<E>>
    where
        E: From<Error> + Send + Sync + 'static,
        F: Future<Output = Result<(Trace, Freshness), E>>,
    {
        self.pinned
            .try_get_with(key, async {
                let (trace, freshness) = load.await?;
                Ok(Arc::new(self.snapshot(trace, snapshot_ms, freshness)?))
            })
            .await
    }

    /// Returns the snapshot `latest` currently serves. On a miss, `load_at(now_ms)` runs once for
    /// all concurrent callers and its snapshot is served until its freshness expires.
    pub async fn latest_or_load<E, F, Fut>(
        &self,
        latest: SnapshotKey,
        now_ms: u64,
        load_at: F,
    ) -> Result<Arc<Snapshot>, Arc<E>>
    where
        E: Send + Sync + 'static,
        F: Fn(u64) -> Fut,
        Fut: Future<Output = Result<Arc<Snapshot>, Arc<E>>>,
    {
        let entry = self
            .latest
            .try_get_with(latest, async {
                let snapshot = load_at(now_ms).await?;
                Ok::<_, Arc<E>>(Latest {
                    snapshot_ms: snapshot.snapshot_ms,
                    freshness: snapshot.freshness,
                })
            })
            .await
            .map_err(|error| Arc::clone(&*error))?;
        load_at(entry.snapshot_ms).await
    }

    fn snapshot(
        &self,
        trace: Trace,
        snapshot_ms: u64,
        freshness: Freshness,
    ) -> Result<Snapshot, Error> {
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
        Ok(Snapshot {
            trace,
            version,
            snapshot_ms,
            freshness,
            weight: u32::try_from(encoded.len().saturating_mul(2)).unwrap_or(u32::MAX),
        })
    }

    #[cfg(test)]
    async fn weighted_size(&self) -> u64 {
        self.pinned.run_pending_tasks().await;
        self.pinned.weighted_size()
    }
}

#[derive(Clone)]
pub(crate) enum ListedRun {
    Resolved(Box<TraceSummary>, Freshness),
    Limited,
}

impl Fresh for ListedRun {
    fn freshness(&self) -> Freshness {
        match self {
            Self::Resolved(_, freshness) => *freshness,
            Self::Limited => Freshness::Settled,
        }
    }
}

pub(crate) struct ListCache {
    pub(crate) runs: Cache<SnapshotKey, ListedRun>,
    pub(crate) limits: Cache<SnapshotKey, u32>,
}

impl ListCache {
    pub(crate) fn new() -> Self {
        Self {
            runs: Cache::builder()
                .max_capacity(MAX_INDEX_ENTRIES)
                .expire_after(ByFreshness)
                .build(),
            limits: Cache::builder()
                .max_capacity(MAX_INDEX_ENTRIES)
                .time_to_live(SETTLED_TTL)
                .build(),
        }
    }
}

#[cfg(test)]
mod tests {
    use litellm_traces::{
        SpanStatus,
        query::named::{SpendByResponseIdsRow, TraceSpansRow},
        resolve_trace,
    };
    use rstest::rstest;

    use super::*;

    fn row(span_id: &str) -> TraceSpansRow {
        TraceSpansRow {
            trace_id: String::new(),
            original_trace_id: String::new(),
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
        }
    }

    fn trace(span_id: &str) -> Trace {
        resolve_trace(
            "trace",
            "ref",
            &[row(span_id)],
            &[] as &[SpendByResponseIdsRow],
        )
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
            cache
                .pinned_or_load(key, 100, async {
                    Ok::<_, Error>((trace(span_id), Freshness::Settled))
                })
                .await
                .unwrap();
        }

        assert!(cache.weighted_size().await <= (limit as u64) * 2);
    }

    const LAST_END_MS: u64 = 1_790_742_989_010;

    #[rstest]
    #[case::just_ended(LAST_END_MS, true, Freshness::Live)]
    #[case::quiet_just_under(LAST_END_MS + SETTLED_AFTER_MS - 1, true, Freshness::Live)]
    #[case::quiet_long_enough(LAST_END_MS + SETTLED_AFTER_MS, true, Freshness::Settled)]
    #[case::spend_unknown(LAST_END_MS + SETTLED_AFTER_MS, false, Freshness::Live)]
    fn freshness_settles_once_spans_stop_and_spend_is_known(
        #[case] snapshot_ms: u64,
        #[case] spend_known: bool,
        #[case] expected: Freshness,
    ) {
        assert_eq!(
            Freshness::of(&[row("root")], spend_known, snapshot_ms),
            expected
        );
    }
}
