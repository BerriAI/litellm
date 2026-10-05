use std::{
    collections::{HashMap, HashSet},
    sync::{
        Mutex,
        atomic::{AtomicUsize, Ordering},
    },
    time::Duration,
};

use litellm_traces::{
    CallEvidenceKind, CallKey, ObservationType, QueryScope, SpanStatus, TraceSummary,
    search::{AgentRuns, HistogramBucket, RunField, RunFilter, RunSearch},
    store::{
        CallQuery, CallRow, CountBy, CountValue, RunCount, RunCountQuery, RunOrder, RunQuery,
        RunRow, RunSelection, RunSortKey, SpanPart, SpanQuery, SpanRow, SpanSelection, SpanText,
        SpanTextQuery, TextRange,
    },
};
use litellm_traces_cache::{
    LIVE_TTL, PageRequest, ReadError, StoreError, StoreResult, TraceReader, TraceStore,
};
use rstest::{fixture, rstest};

const START_NS: i64 = 1_790_742_989_000_000_000;

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
enum Operation {
    TraceRefs,
    ListRuns,
    TraceSpans,
    RunSpans,
    Spend,
    SpanText,
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
    list_runs: Vec<RunRow>,
    trace_spans: HashMap<String, Vec<SpanRow>>,
    run_spans: Vec<SpanRow>,
    spend: Vec<CallRow>,
    texts: HashMap<(String, SpanPart), String>,
    list_runs_too_large_above: Option<u32>,
    trace_too_large_refs: HashSet<String>,
    spend_fails_above_response_ids: Option<usize>,
    count_rows: Vec<RunCount>,
    count_reads: Vec<RunCountQuery>,
}

#[derive(Default)]
struct FakeStore {
    source: Option<&'static str>,
    state: Mutex<State>,
    calls: Mutex<HashMap<Operation, AtomicUsize>>,
}

impl FakeStore {
    fn with_spans(trace_ref: &str, spans: Vec<SpanRow>) -> Self {
        Self {
            source: None,
            state: Mutex::new(State {
                trace_spans: HashMap::from([(trace_ref.to_owned(), spans)]),
                ..State::default()
            }),
            calls: Mutex::default(),
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

    fn set_list_runs(&self, rows: Vec<RunRow>) {
        self.state.lock().unwrap().list_runs = rows;
    }

    fn set_list_runs_too_large_above(&self, limit: u32) {
        self.state.lock().unwrap().list_runs_too_large_above = Some(limit);
    }

    fn set_run_spans(&self, rows: Vec<SpanRow>) {
        self.state.lock().unwrap().run_spans = rows;
    }

    fn set_trace_spans_too_large(&self, trace_ref: &str) {
        self.state
            .lock()
            .unwrap()
            .trace_too_large_refs
            .insert(trace_ref.to_owned());
    }

    fn set_text(&self, span_id: &str, part: SpanPart, text: &str) {
        self.state
            .lock()
            .unwrap()
            .texts
            .insert((span_id.to_owned(), part), text.to_owned());
    }

    /// Fails `calls` only when the lookup covers more than `limit` response ids, so a batch
    /// covering several runs fails while each run's own narrower lookup still succeeds.
    fn set_spend_fails_above_response_ids(&self, limit: usize) {
        self.state.lock().unwrap().spend_fails_above_response_ids = Some(limit);
    }

    fn calls(&self, operation: Operation) -> usize {
        self.calls
            .lock()
            .unwrap()
            .get(&operation)
            .map_or(0, |count| count.load(Ordering::SeqCst))
    }

    fn record(&self, operation: Operation) {
        self.calls
            .lock()
            .unwrap()
            .entry(operation)
            .or_default()
            .fetch_add(1, Ordering::SeqCst);
    }

    fn failure(state: &State, operation: Operation) -> StoreResult<(), FakeError> {
        match state.failures.get(&operation) {
            Some(Failure::TooLarge) => Err(StoreError::TooLarge),
            Some(Failure::Failed) => Err(StoreError::Failed(FakeError)),
            None => Ok(()),
        }
    }
}

fn keyset<R, C: Ord>(rows: &[R], cursor: impl Fn(&R) -> C, after: Option<C>, limit: u32) -> Vec<R>
where
    R: Clone,
{
    let mut rows: Vec<R> = rows
        .iter()
        .filter(|row| after.as_ref().is_none_or(|after| cursor(row) > *after))
        .cloned()
        .collect();
    rows.sort_by_key(|row| cursor(row));
    rows.truncate(limit as usize);
    rows
}

impl TraceStore for FakeStore {
    type Error = FakeError;

    fn source(&self) -> &str {
        self.source.unwrap_or("fake")
    }

    async fn runs(&self, _: &QueryScope, query: &RunQuery) -> StoreResult<Vec<RunRow>, FakeError> {
        let state = self.state.lock().unwrap();
        if let RunSelection::TraceId(trace_id) = &query.selection {
            self.record(Operation::TraceRefs);
            Self::failure(&state, Operation::TraceRefs)?;
            return Ok(state
                .trace_refs
                .iter()
                .take(query.limit as usize)
                .map(|trace_ref| run(trace_id, trace_ref))
                .collect());
        }
        self.record(Operation::ListRuns);
        Self::failure(&state, Operation::ListRuns)?;
        if state
            .list_runs_too_large_above
            .is_some_and(|limit| query.limit > limit)
        {
            return Err(StoreError::TooLarge);
        }
        let mut rows: Vec<_> = state
            .list_runs
            .iter()
            .filter(|row| {
                query.after.as_ref().is_none_or(|after| {
                    let value = (query.order.value(row), &row.trace_ref);
                    let cursor = (after.value, &after.trace_ref);
                    if query.order.descending {
                        value < cursor
                    } else {
                        value > cursor
                    }
                })
            })
            .cloned()
            .collect();
        rows.sort_by(|left, right| query.order.compare(left, right));
        rows.truncate(query.limit as usize);
        Ok(rows)
    }

    async fn run_counts(
        &self,
        _: &QueryScope,
        query: &RunCountQuery,
    ) -> StoreResult<Vec<RunCount>, FakeError> {
        let mut state = self.state.lock().unwrap();
        state.count_reads.push(query.clone());
        Ok(state.count_rows.clone())
    }

    async fn spans(
        &self,
        _: &QueryScope,
        query: &SpanQuery,
    ) -> StoreResult<Vec<SpanRow>, FakeError> {
        tokio::task::yield_now().await;
        let state = self.state.lock().unwrap();
        let rows = match &query.selection {
            SpanSelection::Trace { trace_ref, .. } => {
                self.record(Operation::TraceSpans);
                Self::failure(&state, Operation::TraceSpans)?;
                if state.trace_too_large_refs.contains(trace_ref) {
                    return Err(StoreError::TooLarge);
                }
                state
                    .trace_spans
                    .get(trace_ref)
                    .cloned()
                    .unwrap_or_default()
            }
            SpanSelection::Runs { window, .. } => {
                self.record(Operation::RunSpans);
                Self::failure(&state, Operation::RunSpans)?;
                state
                    .run_spans
                    .iter()
                    .filter(|row| {
                        i128::from(row.start_ns) >= i128::from(window.start) * 1_000_000
                            && i128::from(row.start_ns) < i128::from(window.end) * 1_000_000
                    })
                    .cloned()
                    .collect()
            }
        };
        Ok(keyset(
            &rows,
            SpanRow::cursor,
            query.after.clone(),
            query.limit,
        ))
    }

    async fn span_text(
        &self,
        _: &QueryScope,
        query: &SpanTextQuery,
    ) -> StoreResult<Vec<SpanText>, FakeError> {
        self.record(Operation::SpanText);
        let state = self.state.lock().unwrap();
        Self::failure(&state, Operation::SpanText)?;
        Ok(query
            .span_ids
            .iter()
            .filter_map(|span_id| {
                let text = state.texts.get(&(span_id.clone(), query.part))?;
                let total = text.chars().count() as u64;
                let (skip, take) = match query.range {
                    TextRange::From { offset, max_chars } => {
                        (offset, max_chars.unwrap_or(u64::MAX))
                    }
                    TextRange::Last { chars } => (total.saturating_sub(chars), chars),
                };
                Some(SpanText {
                    span_id: span_id.clone(),
                    text: text
                        .chars()
                        .skip(skip as usize)
                        .take(take as usize)
                        .collect(),
                    total_chars: total,
                    version: format!("{:0>64}", text.len()),
                    contains: query
                        .contains
                        .as_ref()
                        .is_some_and(|needle| text.contains(needle.as_str())),
                })
            })
            .collect())
    }

    async fn calls(
        &self,
        _: &QueryScope,
        query: &CallQuery,
    ) -> StoreResult<Vec<CallRow>, FakeError> {
        self.record(Operation::Spend);
        let state = self.state.lock().unwrap();
        Self::failure(&state, Operation::Spend)?;
        if state
            .spend_fails_above_response_ids
            .is_some_and(|limit| query.response_ids.len() > limit)
        {
            return Err(StoreError::Failed(FakeError));
        }
        Ok(keyset(
            &state.spend,
            CallRow::cursor,
            query.after.clone(),
            query.limit,
        ))
    }
}

fn access() -> QueryScope {
    QueryScope::All
}

fn everything() -> RunFilter {
    RunFilter {
        start_ms: 0,
        end_ms: i64::MAX,
        search: RunSearch::default(),
        ..Default::default()
    }
}

fn window(start_ms: i64, end_ms: i64, q: &str) -> RunFilter {
    RunFilter {
        start_ms,
        end_ms,
        search: RunSearch::parse(q).unwrap(),
        ..Default::default()
    }
}

fn newest(limit: u32) -> PageRequest {
    PageRequest {
        cursor: None,
        limit,
    }
}

fn span(index: usize) -> SpanRow {
    SpanRow {
        trace_id: "trace".into(),
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
        team_id: "team".into(),
        api_key_hash: "key".into(),
        user_id: "user".into(),
    }
}

fn run(trace_id: &str, trace_ref: &str) -> RunRow {
    RunRow {
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
        duration_ns: 10_000_000,
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
#[case::smaller(1)]
#[case::larger(3)]
#[tokio::test]
async fn span_cursors_reject_changed_page_sizes(#[case] page_size: u32) {
    let store = FakeStore::with_spans("ref", (0..5).map(span).collect());
    let reader = TraceReader::new(usize::MAX);
    let access = access();
    let first = reader
        .get_trace_page(&store, &access, "trace", "ref", None, 2)
        .await
        .unwrap()
        .unwrap();
    let result = reader
        .get_trace_page(
            &store,
            &access,
            "trace",
            "ref",
            first.next_cursor.as_deref(),
            page_size,
        )
        .await;
    assert!(matches!(result, Err(ReadError::InvalidCursor("span"))));
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
async fn listed_run_resolution_keeps_spans_crossing_a_fractional_millisecond() {
    let store = FakeStore::default();
    let root = SpanRow {
        start_ns: START_NS + 900_000,
        duration_ns: 900_000,
        ..span(0)
    };
    let child = SpanRow {
        start_ns: START_NS + 1_200_000,
        duration_ns: 100_000,
        status: SpanStatus::Error,
        agent: "child".into(),
        ..span(1)
    };
    let listed = RunRow {
        duration_ns: root.duration_ns,
        span_count: 2,
        error_count: 1,
        agent_count: 2,
        agent_invocations: 2,
        agent_names: vec!["agent".into(), "child".into()],
        ..run("trace", "ref")
    };
    store.set_list_runs(vec![listed.clone()]);
    store.set_run_spans(vec![root, child]);
    let page = TraceReader::new(usize::MAX)
        .list_traces(
            &store,
            &access(),
            &everything(),
            RunOrder::NEWEST,
            &newest(1),
        )
        .await
        .unwrap();
    assert_eq!(page.data.len(), 1);
    assert_eq!(page.data[0].span_count, listed.span_count);
    assert_eq!(page.data[0].error_count, listed.error_count);
    assert_eq!(page.data[0].agent_count, listed.agent_count);
    assert_eq!(page.data[0].agent_names, listed.agent_names);
    assert_eq!(
        page.data[0].duration_ms,
        listed.duration_ns as f64 / 1_000_000.0
    );
    assert!(!page.data[0].resolution_limited);
}

#[rstest]
#[tokio::test]
async fn list_run_budget_halves_the_limit_and_cursor_requires_a_run_past_the_page() {
    let runs = |count: usize| {
        (0..count)
            .map(|index| run(&format!("trace-{index}"), &format!("ref-{index}")))
            .collect()
    };
    let store = FakeStore::default();
    store.set_list_runs(runs(3));
    store.set_list_runs_too_large_above(3);
    let reader = TraceReader::new(usize::MAX);
    let access = access();
    let page = reader
        .list_traces(&store, &access, &everything(), RunOrder::NEWEST, &newest(8))
        .await
        .unwrap();
    assert_eq!(page.data.len(), 2);
    assert!(page.next_cursor.is_some());
    assert_eq!(store.calls(Operation::ListRuns), 3);

    for (remaining, listed) in [(1, 1), (2, 2)] {
        let rest = FakeStore::default();
        rest.set_list_runs(runs(remaining));
        rest.set_list_runs_too_large_above(3);
        let page = reader
            .list_traces(&rest, &access, &everything(), RunOrder::NEWEST, &newest(8))
            .await
            .unwrap();
        assert_eq!(page.data.len(), listed);
        assert!(page.next_cursor.is_none());
        assert_eq!(rest.calls(Operation::ListRuns), 1);
    }
}

#[fixture]
fn paging_scope() -> QueryScope {
    QueryScope::Owned {
        user_id: "user".into(),
        team_ids: vec!["team".into()],
    }
}

#[rstest]
#[case::start(window(1, i64::MAX, ""), paging_scope(), "fake", RunOrder::NEWEST)]
#[case::end(window(0, i64::MAX - 1, ""), paging_scope(), "fake", RunOrder::NEWEST)]
#[case::text(window(0, i64::MAX, "find"), paging_scope(), "fake", RunOrder::NEWEST)]
#[case::field(
    window(0, i64::MAX, "agent:worker"),
    paging_scope(),
    "fake",
    RunOrder::NEWEST
)]
#[case::attribute(
    window(0, i64::MAX, "attr.stage:production"),
    paging_scope(),
    "fake",
    RunOrder::NEWEST
)]
#[case::trace_refs(RunFilter { trace_refs: vec!["ref".into()], ..everything() }, paging_scope(), "fake", RunOrder::NEWEST)]
#[case::user(everything(), QueryScope::Owned { user_id: "another-user".into(), team_ids: vec!["team".into()] }, "fake", RunOrder::NEWEST)]
#[case::teams(everything(), QueryScope::Owned { user_id: "user".into(), team_ids: vec!["another-team".into()] }, "fake", RunOrder::NEWEST)]
#[case::source(everything(), paging_scope(), "other", RunOrder::NEWEST)]
#[case::sort(everything(), paging_scope(), "fake", RunOrder { descending: false, ..RunOrder::NEWEST })]
#[tokio::test]
async fn run_cursors_reject_a_changed_query_before_reading_storage(
    #[case] filter: RunFilter,
    #[case] scope: QueryScope,
    #[case] source: &'static str,
    #[case] order: RunOrder,
) {
    let original = FakeStore::default();
    original.set_list_runs(vec![run("trace-a", "ref-a"), run("trace-b", "ref-b")]);
    let reader = TraceReader::new(usize::MAX);
    let first = reader
        .list_traces(
            &original,
            &paging_scope(),
            &everything(),
            RunOrder::NEWEST,
            &newest(1),
        )
        .await
        .unwrap();
    let changed = FakeStore {
        source: Some(source),
        ..Default::default()
    };
    let result = reader
        .list_traces(
            &changed,
            &scope,
            &filter,
            order,
            &PageRequest {
                cursor: first.next_cursor,
                limit: 1,
            },
        )
        .await;
    assert!(matches!(result, Err(ReadError::InvalidCursor("trace"))));
    assert_eq!(changed.calls(Operation::ListRuns), 0);
}

#[rstest]
#[tokio::test]
async fn run_cursors_continue_without_gaps_when_page_size_changes() {
    let store = FakeStore::default();
    store.set_list_runs(vec![
        run("trace-a", "ref-a"),
        run("trace-b", "ref-b"),
        run("trace-c", "ref-c"),
    ]);
    let reader = TraceReader::new(usize::MAX);
    let first = reader
        .list_traces(
            &store,
            &access(),
            &everything(),
            RunOrder::NEWEST,
            &newest(1),
        )
        .await
        .unwrap();
    let cursor = first.next_cursor.unwrap();
    let second = reader
        .list_traces(
            &store,
            &access(),
            &everything(),
            RunOrder::NEWEST,
            &PageRequest {
                cursor: Some(cursor),
                limit: 2,
            },
        )
        .await
        .unwrap();
    let refs: Vec<_> = first
        .data
        .iter()
        .chain(&second.data)
        .map(|run| run.trace_ref.as_str())
        .collect();
    assert_eq!(refs, ["ref-c", "ref-b", "ref-a"]);
    assert!(second.next_cursor.is_none());
    assert_eq!(store.calls(Operation::ListRuns), 2);
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
        .list_traces(
            &store,
            &access(),
            &everything(),
            RunOrder::NEWEST,
            &newest(2),
        )
        .await
        .unwrap();
    assert_eq!(page.data.len(), 2);
    assert!(page.data[0].resolution_limited);
    assert_eq!(page.data[0].trace_ref, "ref-large");
    assert!(!page.data[1].resolution_limited);
    assert_eq!(page.data[1].trace_ref, "ref-good");
    let reads = (
        store.calls(Operation::RunSpans),
        store.calls(Operation::TraceSpans),
    );

    let again = reader
        .list_traces(
            &store,
            &access(),
            &everything(),
            RunOrder::NEWEST,
            &newest(2),
        )
        .await
        .unwrap();
    assert_eq!(again.data, page.data);
    assert_eq!(
        (
            store.calls(Operation::RunSpans),
            store.calls(Operation::TraceSpans)
        ),
        reads
    );
}

fn spend_row(response_id: &str, cost: f64) -> CallRow {
    CallRow {
        request_id: response_id.into(),
        litellm_call_id: String::new(),
        response_id: response_id.into(),
        upstream_response_id: String::new(),
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
        .list_traces(
            &store,
            &access(),
            &everything(),
            RunOrder::NEWEST,
            &newest(8),
        )
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
#[case::zero_limit(window(0, 10, ""), 0)]
#[case::empty_window(window(10, 10, ""), 8)]
#[case::reversed_window(window(10, 0, ""), 8)]
#[tokio::test]
async fn invalid_list_reads_are_rejected_before_storage(
    #[case] filter: RunFilter,
    #[case] limit: u32,
) {
    let store = FakeStore::default();
    assert!(matches!(
        TraceReader::new(usize::MAX)
            .list_traces(&store, &access(), &filter, RunOrder::NEWEST, &newest(limit))
            .await,
        Err(ReadError::InvalidParameters)
    ));
    assert_eq!(store.calls(Operation::ListRuns), 0);
}

#[rstest]
#[case::empty_window(window(10, 10, ""), 4)]
#[case::reversed_window(window(10, 0, ""), 4)]
#[case::zero_buckets(window(0, 10, ""), 0)]
#[case::too_many_buckets(window(0, 10, ""), 241)]
#[tokio::test]
async fn invalid_histogram_reads_are_rejected_before_storage(
    #[case] filter: RunFilter,
    #[case] buckets: u32,
) {
    let store = FakeStore::default();
    assert!(matches!(
        TraceReader::new(usize::MAX)
            .histogram(&store, &access(), &filter, buckets)
            .await,
        Err(ReadError::InvalidParameters)
    ));
    assert!(store.state.lock().unwrap().count_reads.is_empty());
}

#[rstest]
#[tokio::test]
async fn histogram_reads_the_search_and_fills_every_bucket() {
    let store = FakeStore::default();
    store.state.lock().unwrap().count_rows = vec![
        RunCount {
            bucket: 1,
            failed: false,
            value: "writer".into(),
            runs: 2,
        },
        RunCount {
            bucket: 1,
            failed: true,
            value: "reviewer".into(),
            runs: 1,
        },
    ];
    let filter = window(0, 30, "model:gpt*");
    let histogram = TraceReader::new(usize::MAX)
        .histogram(&store, &access(), &filter, 3)
        .await
        .unwrap();

    let empty = |start_ms, end_ms| HistogramBucket {
        start_ms,
        end_ms,
        total: 0,
        failed: 0,
        agents: Vec::new(),
    };
    assert_eq!(
        histogram.buckets,
        vec![
            empty(0, 10),
            HistogramBucket {
                start_ms: 10,
                end_ms: 20,
                total: 3,
                failed: 1,
                agents: vec![AgentRuns {
                    agent: "writer".into(),
                    runs: 2,
                }],
            },
            empty(20, 30),
        ]
    );
    let state = store.state.lock().unwrap();
    let [query] = state.count_reads.as_slice() else {
        panic!("expected one histogram read");
    };
    assert_eq!(
        query.by,
        CountBy {
            buckets: Some(3),
            failed: true,
            value: Some(CountValue::PrimaryAgent),
        }
    );
    assert_eq!((query.filter.clone(), query.limit), (filter, None));
}

#[rstest]
#[case::empty_window(window(10, 10, ""), 20)]
#[case::zero_limit(window(0, 10, ""), 0)]
#[case::too_many_values(window(0, 10, ""), 101)]
#[tokio::test]
async fn invalid_value_reads_are_rejected_before_storage(
    #[case] filter: RunFilter,
    #[case] limit: u32,
) {
    let store = FakeStore::default();
    assert!(matches!(
        TraceReader::new(usize::MAX)
            .values(&store, &access(), &filter, RunField::Agent, "", limit)
            .await,
        Err(ReadError::InvalidParameters)
    ));
    assert!(store.state.lock().unwrap().count_reads.is_empty());
}

#[rstest]
#[tokio::test]
async fn values_narrow_by_the_search_and_the_needle() {
    let store = FakeStore::default();
    store.state.lock().unwrap().count_rows = vec![
        RunCount {
            bucket: 0,
            failed: false,
            value: "researcher".into(),
            runs: 3,
        },
        RunCount {
            bucket: 0,
            failed: false,
            value: "re_writer".into(),
            runs: 1,
        },
    ];
    let filter = window(0, 10, "-has_error:true");
    let values = TraceReader::new(usize::MAX)
        .values(&store, &access(), &filter, RunField::Agent, "re_", 20)
        .await
        .unwrap();

    assert_eq!(values.values, vec!["researcher", "re_writer"]);
    let state = store.state.lock().unwrap();
    let [query] = state.count_reads.as_slice() else {
        panic!("expected one values read");
    };
    assert_eq!(
        query.by,
        CountBy {
            value: Some(CountValue::Field(RunField::Agent)),
            ..CountBy::default()
        }
    );
    assert_eq!(
        (query.contains.as_str(), query.limit, &query.filter),
        ("re_", Some(20), &filter)
    );
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
#[case::start(RunSortKey::StartMs)]
#[case::duration(RunSortKey::DurationMs)]
#[case::spans(RunSortKey::SpanCount)]
#[case::errors(RunSortKey::ErrorCount)]
#[case::reference(RunSortKey::TraceRef)]
#[tokio::test]
async fn cached_run_keeps_all_canonical_metrics_current_for_every_sort(#[case] key: RunSortKey) {
    let store = FakeStore::default();
    store.set_list_runs(vec![run("trace", "ref")]);
    store.set_run_spans(vec![SpanRow {
        kind: ObservationType::Llm,
        litellm_request_id: "response".into(),
        call_keys: vec![CallKey::ProviderResponse("response".into())],
        call_evidence: Some(CallEvidenceKind::Complete),
        ..span(0)
    }]);
    store.state.lock().unwrap().spend = vec![spend_row("response", 1.5)];
    let reader = TraceReader::new(usize::MAX);
    let first = reader
        .list_traces(
            &store,
            &access(),
            &everything(),
            RunOrder::NEWEST,
            &newest(1),
        )
        .await
        .unwrap();
    let changed = RunRow {
        start_ms: START_NS / 1_000_000 + 20,
        duration_ns: 23_000_000,
        span_count: 7,
        error_count: 3,
        agent_names: vec!["new-agent".into()],
        frameworks: vec!["new-framework".into()],
        ..run("trace", "ref")
    };
    store.set_list_runs(vec![changed.clone()]);
    let second = reader
        .list_traces(
            &store,
            &access(),
            &everything(),
            RunOrder {
                key,
                descending: false,
            },
            &newest(1),
        )
        .await
        .unwrap();
    let expected = TraceSummary {
        start_time: litellm_traces::iso_time(changed.start_ms),
        duration_ms: changed.duration_ns as f64 / 1_000_000.0,
        span_count: changed.span_count,
        error_count: changed.error_count,
        has_error: changed.error_count > 0,
        status: changed.status,
        ..first.data[0].clone()
    };
    assert_eq!(expected.spend, Some(1.5));
    assert_eq!(second.data, vec![expected]);
    assert_eq!(store.calls(Operation::RunSpans), 1);
    assert_eq!(store.calls(Operation::Spend), 1);
}

#[rstest]
#[tokio::test]
async fn listed_runs_are_read_once_until_a_live_run_expires() {
    let live = SpanRow {
        trace_id: "trace-live".into(),
        start_ns: now_ns(),
        ..span(0)
    };
    let settled = SpanRow {
        trace_id: "trace-settled".into(),
        ..span(0)
    };
    let store = FakeStore::default();
    store.set_list_runs(vec![
        RunRow {
            start_ms: live.start_ns / 1_000_000,
            ..run("trace-live", "ref-live")
        },
        run("trace-settled", "ref-settled"),
    ]);
    store.set_run_spans(vec![live, settled]);
    let reader = TraceReader::new(usize::MAX);
    let access = access();
    let filter = everything();
    let page = newest(2);
    let list = || reader.list_traces(&store, &access, &filter, RunOrder::NEWEST, &page);

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

#[rstest]
#[tokio::test]
async fn trace_reads_follow_the_keyset_past_one_storage_page() {
    let spans: Vec<_> = (0..600).map(span).collect();
    let store = FakeStore::with_spans("ref", spans);
    let trace = TraceReader::new(usize::MAX)
        .get_trace(&store, &access(), "trace", "ref")
        .await
        .unwrap()
        .unwrap();
    assert_eq!(trace.spans.len(), 600);
    assert_eq!(store.calls(Operation::TraceSpans), 3);
}

fn llm(index: usize, offset_ms: i64) -> SpanRow {
    SpanRow {
        kind: ObservationType::Llm,
        start_ns: START_NS + offset_ms * 1_000_000,
        ..span(index)
    }
}

#[rstest]
#[case::agent_with_output("answer", "answer")]
#[case::agent_without_output("", "latest")]
#[tokio::test]
async fn span_detail_answers_a_silent_agent_with_its_latest_llm_output(
    #[case] recorded: &str,
    #[case] expected: &str,
) {
    let store = FakeStore::with_spans("ref", vec![span(0), llm(1, 1), llm(2, 5), llm(3, 9)]);
    store.set_text("span-0", SpanPart::Input, "question");
    store.set_text("span-0", SpanPart::Output, recorded);
    store.set_text("span-0", SpanPart::Attributes, r#"{"k":"v"}"#);
    store.set_text("span-1", SpanPart::Output, "earliest");
    store.set_text("span-2", SpanPart::Output, "latest");
    store.set_text("span-3", SpanPart::Output, "");
    let detail = TraceReader::new(usize::MAX)
        .get_span(&store, &access(), "trace", "span-0", "ref")
        .await
        .unwrap()
        .unwrap();
    assert_eq!(
        (detail.input.as_str(), detail.output.as_str()),
        ("question", expected)
    );
    assert_eq!(detail.attributes["k"], "v");
}

#[rstest]
#[tokio::test]
async fn span_detail_keeps_an_empty_output_for_spans_that_are_not_agents() {
    let store = FakeStore::with_spans("ref", vec![span(0), llm(1, 1), llm(2, 2)]);
    store.set_text("span-1", SpanPart::Input, "question");
    store.set_text("span-2", SpanPart::Output, "child output");
    let detail = TraceReader::new(usize::MAX)
        .get_span(&store, &access(), "trace", "span-1", "ref")
        .await
        .unwrap()
        .unwrap();
    assert_eq!(detail.output, "");
}

#[rstest]
#[tokio::test]
async fn span_detail_of_an_invisible_span_is_absent() {
    let store = FakeStore::with_spans("ref", vec![span(0)]);
    assert!(
        TraceReader::new(usize::MAX)
            .get_span(&store, &access(), "trace", "span-0", "ref")
            .await
            .unwrap()
            .is_none()
    );
}

#[rstest]
#[tokio::test]
async fn span_errors_page_by_characters_until_the_message_changes() {
    let store = FakeStore::default();
    let message = "é".repeat(20_000);
    store.set_text("span", SpanPart::Error, &message);
    let reader = TraceReader::new(usize::MAX);
    let first = reader
        .get_span_error(&store, &access(), "trace", "span", "ref", None)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(first.message.chars().count(), 16_384);
    assert_eq!(first.total_chars, 20_000);
    let cursor = first.next_cursor.unwrap();
    let second = reader
        .get_span_error(&store, &access(), "trace", "span", "ref", Some(&cursor))
        .await
        .unwrap()
        .unwrap();
    assert_eq!(second.message.chars().count(), 20_000 - 16_384);
    assert!(second.next_cursor.is_none());

    store.set_text("span", SpanPart::Error, "changed");
    assert!(
        reader
            .get_span_error(&store, &access(), "trace", "span", "ref", Some(&cursor))
            .await
            .unwrap()
            .is_none()
    );
}
