use std::{
    collections::{HashMap, HashSet},
    sync::{
        Mutex,
        atomic::{AtomicUsize, Ordering},
    },
    time::Duration,
};

use litellm_traces::{
    CallEvidenceKind, CallKey, ObservationType, SpanStatus,
    query::named::{
        ListTracesParams, ListTracesRow, ReadAccessParams, SpanDetailParams, SpanDetailRow,
        SpanErrorParams, SpanErrorRow, SpendByResponseIdsParams, SpendByResponseIdsRow,
        TraceIdentityParams, TracePageSpansParams, TraceSpansParams, TraceSpansRow,
    },
};
use litellm_traces_cache::{LIVE_TTL, ReadError, StoreError, TraceReader, TraceStore};
use rstest::rstest;

const START_NS: i64 = 1_790_742_989_000_000_000;

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
enum Operation {
    TraceRefs,
    ListRuns,
    TraceSpans,
    RunSpans,
    Spend,
    SpanDetail,
    SpanError,
}

#[derive(Clone, Copy)]
enum Failure {
    TooLarge,
    Failed,
}

#[derive(Debug, thiserror::Error)]
#[error("fake trace store failed")]
struct FakeError;

#[derive(Default)]
struct State {
    failures: HashMap<Operation, Failure>,
    trace_refs: Vec<String>,
    list_runs: Vec<ListTracesRow>,
    trace_spans: HashMap<String, Vec<TraceSpansRow>>,
    run_spans: Vec<TraceSpansRow>,
    spend: Vec<SpendByResponseIdsRow>,
    span_detail: Option<SpanDetailRow>,
    span_error: Option<SpanErrorRow>,
    list_runs_too_large_above: Option<u32>,
    trace_too_large_refs: HashSet<String>,
    spend_fails_above_response_ids: Option<usize>,
}

#[derive(Default)]
struct Calls {
    trace_refs: AtomicUsize,
    list_runs: AtomicUsize,
    trace_spans: AtomicUsize,
    run_spans: AtomicUsize,
    spend: AtomicUsize,
    span_detail: AtomicUsize,
    span_error: AtomicUsize,
}

#[derive(Default)]
struct FakeStore {
    state: Mutex<State>,
    calls: Calls,
}

impl FakeStore {
    fn with_spans(trace_ref: &str, spans: Vec<TraceSpansRow>) -> Self {
        Self {
            state: Mutex::new(State {
                trace_spans: HashMap::from([(trace_ref.to_owned(), spans)]),
                ..State::default()
            }),
            calls: Calls::default(),
        }
    }

    fn set_failure(&self, operation: Operation, failure: Failure) {
        self.state
            .lock()
            .unwrap()
            .failures
            .insert(operation, failure);
    }

    fn set_trace_refs(&self, trace_refs: Vec<String>) {
        self.state.lock().unwrap().trace_refs = trace_refs;
    }

    fn set_list_runs(&self, rows: Vec<ListTracesRow>) {
        self.state.lock().unwrap().list_runs = rows;
    }

    fn set_list_runs_too_large_above(&self, limit: u32) {
        self.state.lock().unwrap().list_runs_too_large_above = Some(limit);
    }

    fn set_run_spans(&self, rows: Vec<TraceSpansRow>) {
        self.state.lock().unwrap().run_spans = rows;
    }

    fn set_trace_spans_too_large(&self, trace_ref: &str) {
        self.state
            .lock()
            .unwrap()
            .trace_too_large_refs
            .insert(trace_ref.to_owned());
    }

    /// Fails `spend` only when the lookup covers more than `limit` response ids, so a batch
    /// covering several runs fails while each run's own narrower lookup still succeeds.
    fn set_spend_fails_above_response_ids(&self, limit: usize) {
        self.state.lock().unwrap().spend_fails_above_response_ids = Some(limit);
    }

    fn calls(&self, operation: Operation) -> usize {
        match operation {
            Operation::TraceRefs => self.calls.trace_refs.load(Ordering::SeqCst),
            Operation::ListRuns => self.calls.list_runs.load(Ordering::SeqCst),
            Operation::TraceSpans => self.calls.trace_spans.load(Ordering::SeqCst),
            Operation::RunSpans => self.calls.run_spans.load(Ordering::SeqCst),
            Operation::Spend => self.calls.spend.load(Ordering::SeqCst),
            Operation::SpanDetail => self.calls.span_detail.load(Ordering::SeqCst),
            Operation::SpanError => self.calls.span_error.load(Ordering::SeqCst),
        }
    }

    fn failure(state: &State, operation: Operation) -> Result<(), StoreError<FakeError>> {
        match state.failures.get(&operation) {
            Some(Failure::TooLarge) => Err(StoreError::TooLarge),
            Some(Failure::Failed) => Err(StoreError::Failed(FakeError)),
            None => Ok(()),
        }
    }
}

impl TraceStore for FakeStore {
    type Error = FakeError;

    fn source(&self) -> &str {
        "fake"
    }

    async fn trace_refs(
        &self,
        _: &TraceIdentityParams,
    ) -> Result<Vec<String>, StoreError<Self::Error>> {
        self.calls.trace_refs.fetch_add(1, Ordering::SeqCst);
        let state = self.state.lock().unwrap();
        Self::failure(&state, Operation::TraceRefs)?;
        Ok(state.trace_refs.clone())
    }

    async fn list_runs(
        &self,
        params: &ListTracesParams,
    ) -> Result<Vec<ListTracesRow>, StoreError<Self::Error>> {
        self.calls.list_runs.fetch_add(1, Ordering::SeqCst);
        let state = self.state.lock().unwrap();
        Self::failure(&state, Operation::ListRuns)?;
        if state
            .list_runs_too_large_above
            .is_some_and(|limit| params.limit > limit)
        {
            return Err(StoreError::TooLarge);
        }
        Ok(state
            .list_runs
            .iter()
            .take(params.limit as usize)
            .cloned()
            .collect())
    }

    async fn trace_spans(
        &self,
        params: &TraceSpansParams,
        _: u64,
    ) -> Result<Vec<TraceSpansRow>, StoreError<Self::Error>> {
        self.calls.trace_spans.fetch_add(1, Ordering::SeqCst);
        tokio::task::yield_now().await;
        let state = self.state.lock().unwrap();
        Self::failure(&state, Operation::TraceSpans)?;
        if state.trace_too_large_refs.contains(&params.trace_ref) {
            return Err(StoreError::TooLarge);
        }
        Ok(state
            .trace_spans
            .get(&params.trace_ref)
            .cloned()
            .unwrap_or_default())
    }

    async fn run_spans(
        &self,
        _: &TracePageSpansParams,
        _: u64,
    ) -> Result<Vec<TraceSpansRow>, StoreError<Self::Error>> {
        self.calls.run_spans.fetch_add(1, Ordering::SeqCst);
        tokio::task::yield_now().await;
        let state = self.state.lock().unwrap();
        Self::failure(&state, Operation::RunSpans)?;
        Ok(state.run_spans.clone())
    }

    async fn spend(
        &self,
        params: &SpendByResponseIdsParams,
    ) -> Result<Vec<SpendByResponseIdsRow>, StoreError<Self::Error>> {
        self.calls.spend.fetch_add(1, Ordering::SeqCst);
        let state = self.state.lock().unwrap();
        Self::failure(&state, Operation::Spend)?;
        if state
            .spend_fails_above_response_ids
            .is_some_and(|limit| params.response_ids.len() > limit)
        {
            return Err(StoreError::Failed(FakeError));
        }
        Ok(state.spend.clone())
    }

    async fn span_detail(
        &self,
        _: &SpanDetailParams,
    ) -> Result<Option<SpanDetailRow>, StoreError<Self::Error>> {
        self.calls.span_detail.fetch_add(1, Ordering::SeqCst);
        let state = self.state.lock().unwrap();
        Self::failure(&state, Operation::SpanDetail)?;
        Ok(state.span_detail.clone())
    }

    async fn span_error(
        &self,
        _: &SpanErrorParams,
    ) -> Result<Option<SpanErrorRow>, StoreError<Self::Error>> {
        self.calls.span_error.fetch_add(1, Ordering::SeqCst);
        let state = self.state.lock().unwrap();
        Self::failure(&state, Operation::SpanError)?;
        Ok(state.span_error.clone())
    }
}

fn access() -> ReadAccessParams {
    ReadAccessParams {
        all_teams: true,
        user_id: String::new(),
        team_ids: Vec::new(),
    }
}

fn span(index: usize) -> TraceSpansRow {
    TraceSpansRow {
        trace_id: "trace".into(),
        original_trace_id: String::new(),
        span_id: format!("span-{index}"),
        parent_span_id: if index == 0 {
            String::new()
        } else {
            "span-0".into()
        },
        name: "agent".into(),
        kind: ObservationType::Agent,
        wrapper_candidate: false,
        agent: "agent".into(),
        framework: String::new(),
        status: SpanStatus::Ok,
        status_message: String::new(),
        error_truncated: false,
        start_ns: START_NS + index as i64 * 1_000_000,
        duration_ns: 10_000_000,
        service: "test".into(),
        input_preview: format!("span input {index}"),
        model: String::new(),
        input_tokens: 0,
        output_tokens: 0,
        litellm_request_id: String::new(),
        call_keys: Vec::new(),
        call_evidence: None,
        tool_call_id: String::new(),
        source_type: String::new(),
        source_url: String::new(),
        source_title: String::new(),
        source_user: String::new(),
        team_id: "team".into(),
        api_key_hash: "key".into(),
        user_id: "user".into(),
    }
}

fn run(trace_id: &str, trace_ref: &str) -> ListTracesRow {
    ListTracesRow {
        trace_id: trace_id.into(),
        trace_ref: trace_ref.into(),
        team_id: "team".into(),
        api_key_hash: "key".into(),
        user_id: "user".into(),
        name: "listed".into(),
        service: "test".into(),
        input_preview: String::new(),
        status: SpanStatus::Ok,
        start_ms: 1_790_742_989_000,
        duration_ms: 10,
        span_count: 1,
        agent_count: 1,
        agent_invocations: 1,
        agent_names: vec!["agent".into()],
        frameworks: Vec::new(),
        llm_calls: 0,
        tool_calls: 0,
        input_tokens: 0,
        output_tokens: 0,
        models: Vec::new(),
        error_count: 0,
        request_ids: Vec::new(),
    }
}

#[rstest]
#[tokio::test]
async fn pages_reuse_one_trace_snapshot_and_concatenate_in_order() {
    let store = FakeStore::with_spans("ref", (0..5).map(span).collect());
    let reader = TraceReader::new(usize::MAX);
    let access = access();
    let first = reader
        .get_trace_page(&store, &access, "trace", "ref", None, 2)
        .await
        .unwrap()
        .unwrap();
    let second = reader
        .get_trace_page(
            &store,
            &access,
            "trace",
            "ref",
            first.next_cursor.as_deref(),
            2,
        )
        .await
        .unwrap()
        .unwrap();
    let third = reader
        .get_trace_page(
            &store,
            &access,
            "trace",
            "ref",
            second.next_cursor.as_deref(),
            2,
        )
        .await
        .unwrap()
        .unwrap();
    let ids: Vec<_> = first
        .spans
        .iter()
        .chain(&second.spans)
        .chain(&third.spans)
        .map(|span| span.span_id.as_str())
        .collect();
    assert_eq!(ids, ["span-0", "span-1", "span-2", "span-3", "span-4"]);
    assert!(third.next_cursor.is_none());
    assert_eq!(store.calls(Operation::TraceSpans), 1);
}

#[rstest]
#[tokio::test]
async fn snapshot_versions_are_stable_across_readers_and_detect_changes() {
    let access = access();
    let original = FakeStore::with_spans("ref", vec![span(0), span(1)]);
    let reader_a = TraceReader::new(usize::MAX);
    let first = reader_a
        .get_trace_page(&original, &access, "trace", "ref", None, 1)
        .await
        .unwrap()
        .unwrap();
    let cursor = first.next_cursor.unwrap();

    let changed = FakeStore::with_spans("ref", vec![span(0), span(1), span(2)]);
    let reader_b = TraceReader::new(usize::MAX);
    let result = reader_b
        .get_trace_page(&changed, &access, "trace", "ref", Some(&cursor), 1)
        .await;
    assert!(matches!(result, Err(ReadError::TraceChanged)));

    let unchanged = FakeStore::with_spans("ref", vec![span(0), span(1)]);
    let reader_c = TraceReader::new(usize::MAX);
    let next = reader_c
        .get_trace_page(&unchanged, &access, "trace", "ref", Some(&cursor), 1)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(next.spans[0].span_id, "span-1");
}

#[rstest]
#[tokio::test]
async fn response_size_splits_pages_and_rejects_a_single_oversized_span() {
    let spans: Vec<_> = (0..4)
        .map(|index| {
            let mut row = span(index);
            row.input_preview = "x".repeat(256);
            row
        })
        .collect();
    let access = access();
    let full_budget_reader = TraceReader::new(usize::MAX);
    let one_span = full_budget_reader
        .get_trace_page(
            &FakeStore::with_spans("ref", spans.clone()),
            &access,
            "trace",
            "ref",
            None,
            1,
        )
        .await
        .unwrap()
        .unwrap();
    let response_bytes = serde_json::to_vec(&one_span).unwrap().len() + 128;
    let reader = TraceReader::new(response_bytes);
    let store = FakeStore::with_spans("ref", spans.clone());
    let page = reader
        .get_trace_page(&store, &access, "trace", "ref", None, 4)
        .await
        .unwrap()
        .unwrap();
    assert!(!page.spans.is_empty());
    assert!(page.spans.len() < 4);
    assert!(page.next_cursor.is_some());
    let continued = reader
        .get_trace_page(
            &store,
            &access,
            "trace",
            "ref",
            page.next_cursor.as_deref(),
            4,
        )
        .await
        .unwrap()
        .unwrap();
    assert!(!continued.spans.is_empty());
    assert!(matches!(
        TraceReader::new(1)
            .get_trace_page(&store, &access, "trace", "ref", None, 1)
            .await,
        Err(ReadError::TooLarge)
    ));
}

#[rstest]
#[tokio::test]
async fn list_run_budget_halves_the_limit_and_cursor_requires_a_full_page() {
    let store = FakeStore::default();
    store.set_list_runs(
        (0..3)
            .map(|index| run(&format!("trace-{index}"), &format!("ref-{index}")))
            .collect(),
    );
    store.set_list_runs_too_large_above(2);
    let reader = TraceReader::new(usize::MAX);
    let access = access();
    let page = reader
        .list_traces(&store, &access, 0, i64::MAX, None, 8)
        .await
        .unwrap();
    assert_eq!(page.data.len(), 2);
    assert!(page.next_cursor.is_some());
    assert_eq!(store.calls(Operation::ListRuns), 3);

    let shorter = FakeStore::default();
    shorter.set_list_runs(vec![run("only", "ref-only")]);
    shorter.set_list_runs_too_large_above(2);
    let page = reader
        .list_traces(&shorter, &access, 0, i64::MAX, None, 8)
        .await
        .unwrap();
    assert_eq!(page.data.len(), 1);
    assert!(page.next_cursor.is_none());
    assert_eq!(shorter.calls(Operation::ListRuns), 1);
}

#[rstest]
#[tokio::test]
async fn oversized_run_batch_falls_back_to_each_run_and_keeps_listed_summaries() {
    let store = FakeStore::with_spans("ref-good", vec![span(0)]);
    store.set_list_runs(vec![
        run("trace-large", "ref-large"),
        run("trace-good", "ref-good"),
    ]);
    store.set_trace_spans_too_large("ref-large");
    store.set_run_spans(Vec::new());
    store.set_failure(Operation::RunSpans, Failure::TooLarge);
    let reader = TraceReader::new(usize::MAX);
    let page = reader
        .list_traces(&store, &access(), 0, i64::MAX, None, 2)
        .await
        .unwrap();
    assert_eq!(page.data.len(), 2);
    assert!(page.data[0].resolution_limited);
    assert_eq!(page.data[0].trace_ref, "ref-large");
    assert!(!page.data[1].resolution_limited);
    assert_eq!(page.data[1].trace_ref, "ref-good");
    assert_eq!(store.calls(Operation::RunSpans), 1);
    assert_eq!(store.calls(Operation::TraceSpans), 2);

    let again = reader
        .list_traces(&store, &access(), 0, i64::MAX, None, 2)
        .await
        .unwrap();
    assert_eq!(again.data, page.data);
    assert_eq!(store.calls(Operation::RunSpans), 1);
    assert_eq!(store.calls(Operation::TraceSpans), 2);
}

fn spend_row(response_id: &str, cost: f64) -> SpendByResponseIdsRow {
    SpendByResponseIdsRow {
        request_id: response_id.into(),
        litellm_call_id: String::new(),
        response_id: response_id.into(),
        upstream_response_id: String::new(),
        provider_request_id: String::new(),
        trace_id: String::new(),
        span_id: String::new(),
        team_id: "team".into(),
        api_key: "key".into(),
        user: "user".into(),
        spend: Some(cost),
        start_ms: START_NS / 1_000_000,
    }
}

#[rstest]
#[tokio::test]
async fn failed_batch_spend_lookup_falls_back_to_each_run_instead_of_losing_every_cost() {
    let mut first = span(0);
    first.trace_id = "trace-a".into();
    first.kind = ObservationType::Llm;
    first.litellm_request_id = "response-a".into();
    first.call_keys = vec![CallKey::ProviderResponse("response-a".into())];
    first.call_evidence = Some(CallEvidenceKind::Complete);
    let mut second = span(0);
    second.trace_id = "trace-b".into();
    second.kind = ObservationType::Llm;
    second.litellm_request_id = "response-b".into();
    second.call_keys = vec![CallKey::ProviderResponse("response-b".into())];
    second.call_evidence = Some(CallEvidenceKind::Complete);

    let store = FakeStore::default();
    store.set_list_runs(vec![run("trace-a", "ref-a"), run("trace-b", "ref-b")]);
    store.set_run_spans(vec![first.clone(), second.clone()]);
    {
        let mut state = store.state.lock().unwrap();
        state.trace_spans.insert("ref-a".to_owned(), vec![first]);
        state.trace_spans.insert("ref-b".to_owned(), vec![second]);
        state.spend = vec![spend_row("response-a", 1.5), spend_row("response-b", 2.5)];
    }
    // The batch covers both runs' response ids (2); each run resolved on its own only ever
    // asks for its own (1), so this fails only the combined read, not the per-run fallback.
    store.set_spend_fails_above_response_ids(1);

    let page = TraceReader::new(usize::MAX)
        .list_traces(&store, &access(), 0, i64::MAX, None, 8)
        .await
        .unwrap();

    assert_eq!(page.data.len(), 2);
    let by_ref: HashMap<&str, f64> = page
        .data
        .iter()
        .map(|run| {
            (
                run.trace_ref.as_str(),
                run.spend
                    .expect("run's own spend read should have succeeded"),
            )
        })
        .collect();
    assert_eq!(by_ref["ref-a"], 1.5);
    assert_eq!(by_ref["ref-b"], 2.5);
    assert_eq!(store.calls(Operation::RunSpans), 1);
    assert_eq!(store.calls(Operation::TraceSpans), 2);
}

#[rstest]
#[tokio::test]
async fn failed_spend_lookup_preserves_the_trace_with_unknown_spend() {
    let mut row = span(0);
    row.litellm_request_id = "response".into();
    row.call_keys = vec![CallKey::ProviderResponse("response".into())];
    row.call_evidence = Some(CallEvidenceKind::Complete);
    let store = FakeStore::with_spans("ref", vec![row]);
    store.set_failure(Operation::Spend, Failure::Failed);
    let trace = TraceReader::new(usize::MAX)
        .get_trace(&store, &access(), "trace", "ref")
        .await
        .unwrap()
        .unwrap();
    assert_eq!(trace.summary.spend, None);
    assert_eq!(trace.spans[0].spend, None);
    assert_eq!(store.calls(Operation::Spend), 1);
}

#[rstest]
#[tokio::test]
async fn ambiguous_trace_references_fail_and_a_single_reference_is_resolved() {
    let reader = TraceReader::new(usize::MAX);
    let access = access();
    let ambiguous = FakeStore::default();
    ambiguous.set_trace_refs(vec!["ref-a".into(), "ref-b".into()]);
    assert!(matches!(
        reader.get_trace(&ambiguous, &access, "trace", "").await,
        Err(ReadError::AmbiguousTrace)
    ));

    let unique = FakeStore::with_spans("ref-only", vec![span(0)]);
    unique.set_trace_refs(vec!["ref-only".into()]);
    let trace = reader
        .get_trace(&unique, &access, "trace", "")
        .await
        .unwrap()
        .unwrap();
    assert_eq!(trace.summary.trace_ref, "ref-only");
}

#[rstest]
#[case::zero(0)]
#[case::above_max(501)]
#[tokio::test]
async fn invalid_page_sizes_are_rejected(#[case] page_size: u32) {
    let store = FakeStore::default();
    let reader = TraceReader::new(usize::MAX);
    let access = access();
    assert!(matches!(
        reader
            .get_trace_page(&store, &access, "trace", "ref", None, page_size)
            .await,
        Err(ReadError::InvalidParameters)
    ));
}

#[rstest]
#[tokio::test]
async fn zero_list_limit_is_rejected() {
    let store = FakeStore::default();
    let reader = TraceReader::new(usize::MAX);
    let access = access();
    assert!(matches!(
        reader
            .list_traces(&store, &access, 0, i64::MAX, None, 0)
            .await,
        Err(ReadError::InvalidParameters)
    ));
}

fn now_ns() -> i64 {
    time::OffsetDateTime::now_utc().unix_timestamp_nanos() as i64
}

#[rstest]
#[tokio::test]
async fn concurrent_and_repeated_opens_share_one_storage_read() {
    let store = FakeStore::with_spans("ref", (0..3).map(span).collect());
    let reader = TraceReader::new(usize::MAX);
    let access = access();
    let (first, second) = tokio::join!(
        reader.get_trace(&store, &access, "trace", "ref"),
        reader.get_trace_page(&store, &access, "trace", "ref", None, 2),
    );
    let first = first.unwrap().unwrap();
    let second = second.unwrap().unwrap();
    let reopened = reader
        .get_trace_page(&store, &access, "trace", "ref", None, 2)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(first.spans.len(), 3);
    assert_eq!(second.spans, first.spans[..2]);
    assert_eq!(reopened.next_cursor, second.next_cursor);
    assert_eq!(store.calls(Operation::TraceSpans), 1);
}

#[rstest]
#[tokio::test]
async fn failed_reads_are_not_cached() {
    let store = FakeStore::with_spans("ref", vec![span(0)]);
    store.set_failure(Operation::TraceSpans, Failure::Failed);
    let reader = TraceReader::new(usize::MAX);
    let access = access();
    assert!(matches!(
        reader.get_trace(&store, &access, "trace", "ref").await,
        Err(ReadError::Store(_))
    ));
    store.state.lock().unwrap().failures.clear();
    let trace = reader
        .get_trace(&store, &access, "trace", "ref")
        .await
        .unwrap()
        .unwrap();
    assert_eq!(trace.spans.len(), 1);
    assert_eq!(store.calls(Operation::TraceSpans), 2);
}

#[rstest]
#[case::claude_code("claude-code")]
#[case::claude_agent_sdk("claude-agent-sdk")]
#[tokio::test]
async fn resumed_native_sessions_refresh_after_live_ttl(#[case] framework: &str) {
    let original = TraceSpansRow {
        framework: framework.into(),
        ..span(0)
    };
    let store = FakeStore::with_spans("ref", vec![original.clone()]);
    let reader = TraceReader::new(usize::MAX);
    let access = access();
    let first = reader
        .get_trace(&store, &access, "trace", "ref")
        .await
        .unwrap()
        .unwrap();
    assert_eq!(first.spans.len(), 1);

    store.state.lock().unwrap().trace_spans.insert(
        "ref".into(),
        vec![
            original,
            TraceSpansRow {
                start_ns: now_ns(),
                ..span(1)
            },
        ],
    );
    let cached = reader
        .get_trace(&store, &access, "trace", "ref")
        .await
        .unwrap()
        .unwrap();
    assert_eq!(cached.spans.len(), 1);
    assert_eq!(store.calls(Operation::TraceSpans), 1);

    tokio::time::sleep(LIVE_TTL + Duration::from_millis(200)).await;
    let resumed = reader
        .get_trace(&store, &access, "trace", "ref")
        .await
        .unwrap()
        .unwrap();
    assert_eq!(resumed.spans.len(), 2);
    assert_eq!(store.calls(Operation::TraceSpans), 2);
    assert_eq!(first.spans.len(), 1);
}

#[rstest]
#[tokio::test]
async fn listed_runs_are_read_once_until_a_live_run_expires() {
    let live = TraceSpansRow {
        trace_id: "trace-live".into(),
        start_ns: now_ns(),
        ..span(0)
    };
    let settled = TraceSpansRow {
        trace_id: "trace-settled".into(),
        ..span(0)
    };
    let store = FakeStore::default();
    store.set_list_runs(vec![
        run("trace-live", "ref-live"),
        run("trace-settled", "ref-settled"),
    ]);
    store.set_run_spans(vec![live, settled]);
    let reader = TraceReader::new(usize::MAX);
    let access = access();
    let list = || reader.list_traces(&store, &access, 0, i64::MAX, None, 2);

    let first = list().await.unwrap();
    assert!(first.data.iter().all(|summary| summary.name == "agent"));
    list().await.unwrap();
    assert_eq!(store.calls(Operation::RunSpans), 1);

    store.set_run_spans(Vec::new());
    tokio::time::sleep(LIVE_TTL + Duration::from_millis(200)).await;
    let after = list().await.unwrap();
    assert_eq!(store.calls(Operation::RunSpans), 2);
    assert_eq!(after.data[0].name, "listed");
    assert_eq!(after.data[1], first.data[1]);
}

#[rstest]
#[tokio::test]
async fn concurrent_pages_of_an_evicted_snapshot_share_one_storage_read() {
    let access = access();
    let first = TraceReader::new(usize::MAX)
        .get_trace_page(
            &FakeStore::with_spans("ref", (0..3).map(span).collect()),
            &access,
            "trace",
            "ref",
            None,
            1,
        )
        .await
        .unwrap()
        .unwrap();
    let cursor = first.next_cursor.as_deref();
    let store = FakeStore::with_spans("ref", (0..3).map(span).collect());
    let reader = TraceReader::new(usize::MAX);
    let (left, right) = tokio::join!(
        reader.get_trace_page(&store, &access, "trace", "ref", cursor, 1),
        reader.get_trace_page(&store, &access, "trace", "ref", cursor, 1),
    );
    assert_eq!(left.unwrap().unwrap().spans[0].span_id, "span-1");
    assert_eq!(right.unwrap().unwrap().spans[0].span_id, "span-1");
    assert_eq!(store.calls(Operation::TraceSpans), 1);
}
