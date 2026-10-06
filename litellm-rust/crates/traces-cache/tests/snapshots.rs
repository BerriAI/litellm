use std::time::Duration;

use litellm_traces::{
    SpanStatus, Trace,
    query::named::{ReadAccessParams, SpendByResponseIdsRow, TraceSpansRow},
    resolve_trace,
};
use std::sync::Arc;

use litellm_traces_cache::{Error, Freshness, Snapshot, SnapshotCache, SnapshotKey};
use rstest::{fixture, rstest};

const T0: i64 = 1_790_742_989_000_000_000;
const MS: i64 = 1_000_000;
const TTL: Duration = Duration::from_secs(120);

fn row(span_id: &str, parent: &str, name: &str, kind: &str, agent: &str) -> TraceSpansRow {
    TraceSpansRow {
        trace_id: String::new(),
        original_trace_id: String::new(),
        span_id: span_id.into(),
        parent_span_id: parent.into(),
        name: name.into(),
        kind: kind.parse().unwrap(),
        wrapper_candidate: false,
        agent: agent.into(),
        framework: String::new(),
        status: SpanStatus::Ok,
        status_message: String::new(),
        error_truncated: false,
        start_ns: T0,
        duration_ns: 10 * MS as u64,
        service: "agent-demo".into(),
        input_preview: format!("input of {name}"),
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

fn access() -> ReadAccessParams {
    ReadAccessParams {
        all_teams: false,
        user_id: String::new(),
        team_ids: vec!["team".into()],
    }
}

fn key(
    source: &str,
    access: &ReadAccessParams,
    trace_id: &str,
    trace_ref: &str,
    ms: u64,
) -> SnapshotKey {
    SnapshotKey::new(source, access, trace_id, trace_ref, ms).unwrap()
}

async fn insert(
    cache: &SnapshotCache,
    key: SnapshotKey,
    trace: Trace,
) -> Result<Arc<Snapshot>, Arc<Error>> {
    cache
        .pinned_or_load(key, 100, async {
            Ok::<_, Error>((trace, Freshness::Settled))
        })
        .await
}

#[fixture]
fn trace() -> Trace {
    resolve_trace(
        "trace",
        "ref",
        &[row("root", "", "run", "agent", "agent")],
        &[] as &[SpendByResponseIdsRow],
    )
    .expect("fixture should resolve")
}

#[rstest]
#[case::different_team(false, "", "other-team")]
#[case::different_user(false, "other-user", "team")]
#[case::different_scope(true, "", "team")]
#[tokio::test]
async fn cached_trace_is_isolated_by_access_scope(
    trace: Trace,
    #[case] all_teams: bool,
    #[case] user_id: &str,
    #[case] team_id: &str,
) {
    let cache = SnapshotCache::new(1024 * 1024, TTL);
    let stored = key("source", &access(), "trace", "ref", 100);

    insert(&cache, stored.clone(), trace.clone()).await.unwrap();

    let other_access = ReadAccessParams {
        all_teams,
        user_id: user_id.into(),
        team_ids: vec![team_id.into()],
    };
    let other = key("source", &other_access, "trace", "ref", 100);

    assert!(cache.get(&other).await.is_none());
    let cached = cache.get(&stored).await.unwrap();
    assert_eq!(cached.trace(), &trace);
}

#[rstest]
#[case::different_source("other-source", "trace", "ref", 100)]
#[case::different_trace_id("source", "other-trace", "ref", 100)]
#[case::different_trace_ref("source", "trace", "other-ref", 100)]
#[case::different_snapshot_ms("source", "trace", "ref", 200)]
#[tokio::test]
async fn cached_trace_is_isolated_by_key_fields(
    trace: Trace,
    #[case] source: &str,
    #[case] trace_id: &str,
    #[case] trace_ref: &str,
    #[case] snapshot_ms: u64,
) {
    let cache = SnapshotCache::new(1024 * 1024, TTL);
    let stored = key("source", &access(), "trace", "ref", 100);

    insert(&cache, stored.clone(), trace.clone()).await.unwrap();

    let other = key(source, &access(), trace_id, trace_ref, snapshot_ms);
    assert!(cache.get(&other).await.is_none());
    assert!(cache.get(&stored).await.is_some());
}

#[rstest]
#[tokio::test]
async fn snapshot_at_the_size_limit_is_accepted(trace: Trace) {
    let size = serde_json::to_vec(&trace).unwrap().len();
    let cache = SnapshotCache::new(size, TTL);
    let stored = key("source", &access(), "trace", "ref", 100);

    insert(&cache, stored.clone(), trace).await.unwrap();
    assert!(cache.get(&stored).await.is_some());
}

#[rstest]
#[tokio::test]
async fn snapshot_one_byte_over_the_size_limit_is_rejected(trace: Trace) {
    let size = serde_json::to_vec(&trace).unwrap().len();
    let cache = SnapshotCache::new(size - 1, TTL);
    let stored = key("source", &access(), "trace", "ref", 100);

    assert!(matches!(
        insert(&cache, stored.clone(), trace).await,
        Err(error) if matches!(*error, Error::ReadTooLarge)
    ));
    assert!(cache.get(&stored).await.is_none());
}

#[rstest]
#[case::same_ids(&["root", "child"], &["root", "child"], true)]
#[case::different_ids(&["root", "child"], &["root", "other"], false)]
#[tokio::test]
async fn snapshot_version_tracks_the_ordered_span_ids(
    #[case] first_ids: &[&str],
    #[case] second_ids: &[&str],
    #[case] equal: bool,
) {
    let build = |ids: &[&str]| -> Trace {
        let rows: Vec<TraceSpansRow> = ids
            .iter()
            .map(|span_id| row(span_id, "", "run", "agent", "agent"))
            .collect();
        resolve_trace("trace", "ref", &rows, &[] as &[SpendByResponseIdsRow])
            .expect("fixture should resolve")
    };
    let cache = SnapshotCache::new(1024 * 1024, TTL);

    let first = insert(
        &cache,
        key("source", &access(), "a", "ref", 100),
        build(first_ids),
    )
    .await
    .unwrap();
    let second = insert(
        &cache,
        key("source", &access(), "b", "ref", 100),
        build(second_ids),
    )
    .await
    .unwrap();

    assert_eq!(first.version() == second.version(), equal);
}

#[rstest]
#[tokio::test]
async fn snapshots_expire_when_idle(trace: Trace) {
    let cache = SnapshotCache::new(1024 * 1024, Duration::from_millis(50));
    let stored = key("source", &access(), "trace", "ref", 100);

    insert(&cache, stored.clone(), trace).await.unwrap();
    tokio::time::sleep(Duration::from_millis(200)).await;

    assert!(cache.get(&stored).await.is_none());
}
