use std::{
    sync::{Arc, Mutex},
    time::{SystemTime, UNIX_EPOCH},
};

use axum::{
    Extension, Router,
    body::{Body, to_bytes},
    http::{Request, StatusCode, header},
    response::Response,
};
use litellm_gateway_traces::{Traces, router};
use litellm_traces::{
    QueryScope,
    search::{RunField, RunFilter, RunSearch},
    store::{
        CallQuery, CallRow, CountValue, RunCount, RunCountQuery, RunQuery, RunRow, RunSelection,
        SpanQuery, SpanRow, SpanText, SpanTextQuery,
    },
};
use litellm_traces_cache::{StoreError, StoreResult, TraceReader, TraceStore};
use rstest::rstest;
use serde_json::{Value, json};
use tower::ServiceExt;

#[derive(Debug, thiserror::Error)]
#[error("fake trace store failed")]
struct FakeError;

#[derive(Clone, Copy)]
enum Outcome {
    Empty,
    TooLarge,
    Failed,
}

struct FakeStore {
    outcome: Outcome,
    lists: Mutex<Vec<(QueryScope, RunQuery)>>,
    counts: Mutex<Vec<RunCountQuery>>,
}

impl FakeStore {
    fn listed_filter(&self) -> RunFilter {
        let lists = self.lists.lock().unwrap();
        let [(_, query)] = lists.as_slice() else {
            panic!("expected one list read, got {}", lists.len());
        };
        let RunSelection::Matching(filter) = &query.selection else {
            panic!("expected a run search");
        };
        filter.clone()
    }

    fn counted(&self) -> RunCountQuery {
        self.counts.lock().unwrap()[0].clone()
    }
}

impl TraceStore for FakeStore {
    type Error = FakeError;

    fn source(&self) -> &str {
        "fake"
    }

    async fn runs(
        &self,
        access: &QueryScope,
        query: &RunQuery,
    ) -> StoreResult<Vec<RunRow>, FakeError> {
        self.lists
            .lock()
            .unwrap()
            .push((access.clone(), query.clone()));
        match self.outcome {
            Outcome::Empty => Ok(Vec::new()),
            Outcome::TooLarge => Err(StoreError::TooLarge),
            Outcome::Failed => Err(StoreError::Failed(FakeError)),
        }
    }

    async fn run_counts(
        &self,
        _: &QueryScope,
        query: &RunCountQuery,
    ) -> StoreResult<Vec<RunCount>, FakeError> {
        self.counts.lock().unwrap().push(query.clone());
        Ok(vec![RunCount {
            bucket: 0,
            failed: false,
            value: "researcher".into(),
            runs: 2,
        }])
    }

    async fn spans(&self, _: &QueryScope, _: &SpanQuery) -> StoreResult<Vec<SpanRow>, FakeError> {
        Ok(Vec::new())
    }

    async fn span_text(
        &self,
        _: &QueryScope,
        _: &SpanTextQuery,
    ) -> StoreResult<Vec<SpanText>, FakeError> {
        Ok(Vec::new())
    }

    async fn calls(&self, _: &QueryScope, _: &CallQuery) -> StoreResult<Vec<CallRow>, FakeError> {
        Ok(Vec::new())
    }
}

fn access() -> QueryScope {
    QueryScope::Owned {
        user_id: "user".into(),
        team_ids: vec!["team".into()],
    }
}

fn app(outcome: Outcome) -> (Router, Arc<Traces<FakeStore>>) {
    let traces = Arc::new(Traces {
        reader: TraceReader::new(usize::MAX),
        store: FakeStore {
            outcome,
            lists: Mutex::new(Vec::new()),
            counts: Mutex::new(Vec::new()),
        },
    });
    let app = router(Arc::clone(&traces)).layer(Extension(access()));
    (app, traces)
}

async fn get(app: Router, uri: &str) -> Response {
    app.oneshot(Request::get(uri).body(Body::empty()).unwrap())
        .await
        .unwrap()
}

async fn json_body(response: Response) -> Value {
    serde_json::from_slice(&to_bytes(response.into_body(), 65536).await.unwrap()).unwrap()
}

#[tokio::test]
async fn list_reads_the_window_and_parsed_search_in_the_callers_scope() {
    let (app, traces) = app(Outcome::Empty);
    let response = get(
        app,
        "/v1/traces?start_ms=10&end_ms=20&q=plan%20-agent:res*%20model:%22gpt%20x%22",
    )
    .await;

    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        json_body(response).await,
        json!({"data": [], "next_cursor": null})
    );
    let filter = traces.store.listed_filter();
    assert_eq!((filter.start_ms, filter.end_ms), (10, 20));
    let lists = traces.store.lists.lock().unwrap();
    assert_eq!((&lists[0].0, lists[0].1.limit), (&access(), 50));
    assert_eq!(
        filter.search,
        RunSearch::parse(r#"plan -agent:res* model:"gpt x""#)
    );
}

#[tokio::test]
async fn list_defaults_to_the_last_day() {
    let (app, traces) = app(Outcome::Empty);
    let before = now_ms();
    assert_eq!(get(app, "/v1/traces").await.status(), StatusCode::OK);
    let after = now_ms();

    let filter = traces.store.listed_filter();
    assert!((before..=after).contains(&filter.end_ms));
    assert_eq!(filter.end_ms - filter.start_ms, 24 * 60 * 60 * 1000);
    assert_eq!(filter.search, RunSearch::default());
}

#[tokio::test]
async fn histogram_buckets_the_matching_runs_of_the_window() {
    let (app, traces) = app(Outcome::Empty);
    let response = get(
        app,
        "/v1/traces/histogram?start_ms=0&end_ms=40&q=status:ok&buckets=4",
    )
    .await;

    assert_eq!(response.status(), StatusCode::OK);
    let body = json_body(response).await;
    assert_eq!(body["buckets"].as_array().map(Vec::len), Some(4));
    assert_eq!(
        body["buckets"][0],
        json!({"start_ms": 0, "end_ms": 10, "total": 2, "failed": 0, "agents": [{"agent": "researcher", "runs": 2}]})
    );
    let query = traces.store.counted();
    assert_eq!(query.by.buckets, Some(4));
    assert_eq!(query.filter.search, RunSearch::parse("status:ok"));
}

#[tokio::test]
async fn values_suggest_a_field_narrowed_by_the_search() {
    let (app, traces) = app(Outcome::Empty);
    let response = get(
        app,
        "/v1/traces/values/agent?start_ms=0&end_ms=40&q=model:gpt*&contains=res&limit=5",
    )
    .await;

    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(json_body(response).await, json!({"values": ["researcher"]}));
    let query = traces.store.counted();
    assert_eq!(query.by.value, Some(CountValue::Field(RunField::Agent)));
    assert_eq!((query.contains.as_str(), query.limit), ("res", Some(5)));
    assert_eq!(query.filter.search, RunSearch::parse("model:gpt*"));
}

#[rstest]
#[case::bad_cursor(
    Outcome::Empty,
    "/v1/traces?cursor=not-a-cursor",
    StatusCode::BAD_REQUEST,
    "invalid_request"
)]
#[case::bad_window(Outcome::Empty, "/v1/traces?start_ms=x", StatusCode::BAD_REQUEST, "")]
#[case::reversed_window(
    Outcome::Empty,
    "/v1/traces?start_ms=20&end_ms=10",
    StatusCode::BAD_REQUEST,
    "invalid_request"
)]
#[case::too_many_buckets(
    Outcome::Empty,
    "/v1/traces/histogram?buckets=241",
    StatusCode::BAD_REQUEST,
    "invalid_request"
)]
#[case::too_many_values(
    Outcome::Empty,
    "/v1/traces/values/agent?limit=101",
    StatusCode::BAD_REQUEST,
    "invalid_request"
)]
#[case::unknown_field(Outcome::Empty, "/v1/traces/values/color", StatusCode::BAD_REQUEST, "")]
#[case::too_large(
    Outcome::TooLarge,
    "/v1/traces",
    StatusCode::PAYLOAD_TOO_LARGE,
    "too_large"
)]
#[case::store_down(
    Outcome::Failed,
    "/v1/traces",
    StatusCode::SERVICE_UNAVAILABLE,
    "unavailable"
)]
#[tokio::test]
async fn failures_keep_the_python_status_and_code(
    #[case] outcome: Outcome,
    #[case] uri: &str,
    #[case] status: StatusCode,
    #[case] code: &str,
) {
    let (app, _) = app(outcome);
    let response = get(app, uri).await;

    assert_eq!(response.status(), status);
    assert_eq!(
        response.headers().get(header::RETRY_AFTER).is_some(),
        status == StatusCode::SERVICE_UNAVAILABLE
    );
    if !code.is_empty() {
        assert_eq!(json_body(response).await["detail"]["code"], code);
    }
}

fn now_ms() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_millis() as i64
}
