//! Prometheus metrics served at `GET /metrics`.
//!
//! Label sets are resolved without allocating on the request path: the route comes from axum's
//! [`MatchedPath`] (an `Arc<str>`), the model group and deployment are indexes into tables built
//! from the config at startup, and each series lives behind a read-mostly lock keyed by those
//! small values. A series is created once, on its first request.

use std::{
    collections::HashMap,
    fmt::Write as _,
    sync::{
        Arc, RwLock,
        atomic::{AtomicBool, AtomicI64, AtomicU64, Ordering},
    },
    time::{Duration, Instant},
};

use axum::{
    body::Body,
    extract::{MatchedPath, Request, State},
    http::{StatusCode, header},
    middleware::Next,
    response::{IntoResponse, Response},
};
use http_body_util::BodyExt;
use litellm_config::Config;
use litellm_gateway_inference::observe;

/// Upper bounds, in seconds, shared by the latency and time to first byte histograms. They reach
/// 30 minutes because streams that long are normal here.
const BUCKETS: [f64; 19] = [
    0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0, 600.0,
    1200.0, 1800.0, 3600.0,
];

const UNMATCHED_ROUTE: &str = "unmatched";

#[derive(Default)]
struct Histogram {
    /// Per-bucket counts, not cumulative; the last slot is `+Inf`.
    buckets: [AtomicU64; BUCKETS.len() + 1],
    sum_micros: AtomicU64,
    count: AtomicU64,
}

impl Histogram {
    fn observe(&self, elapsed: Duration) {
        let seconds = elapsed.as_secs_f64();
        let index = BUCKETS
            .iter()
            .position(|bound| seconds <= *bound)
            .unwrap_or(BUCKETS.len());
        self.buckets[index].fetch_add(1, Ordering::Relaxed);
        self.sum_micros.fetch_add(
            u64::try_from(elapsed.as_micros()).unwrap_or(u64::MAX),
            Ordering::Relaxed,
        );
        self.count.fetch_add(1, Ordering::Relaxed);
    }

    fn render(&self, out: &mut String, name: &str, labels: &str) {
        let mut cumulative = 0;
        for (index, bucket) in self.buckets.iter().enumerate() {
            cumulative += bucket.load(Ordering::Relaxed);
            match BUCKETS.get(index) {
                Some(bound) => {
                    let _ = writeln!(out, "{name}_bucket{{{labels},le=\"{bound}\"}} {cumulative}");
                }
                None => {
                    let _ = writeln!(out, "{name}_bucket{{{labels},le=\"+Inf\"}} {cumulative}");
                }
            }
        }
        let sum = self.sum_micros.load(Ordering::Relaxed) as f64 / 1e6;
        let _ = writeln!(out, "{name}_sum{{{labels}}} {sum}");
        let _ = writeln!(
            out,
            "{name}_count{{{labels}}} {}",
            self.count.load(Ordering::Relaxed)
        );
    }
}

#[derive(Default)]
struct Series {
    latency: Histogram,
    time_to_first_byte: Histogram,
}

/// `(model index, status code)` under a route.
type SeriesKey = (u32, u16);

type RouteSeries = HashMap<Arc<str>, HashMap<SeriesKey, Arc<Series>>>;

#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
struct FailureKey {
    deployment: u32,
    status: u16,
}

struct ModelEntry {
    index: u32,
    /// `(api_base as configured, deployment index)` for each deployment of the group.
    deployments: Vec<(Option<String>, u32)>,
}

struct DeploymentLabel {
    model: u32,
    api_base: String,
}

/// Process-wide request metrics. Cheap to clone.
#[derive(Clone)]
pub struct Metrics(Arc<Inner>);

struct Inner {
    /// Index 0 is the empty label, for requests that never resolved a model.
    model_labels: Vec<String>,
    models: HashMap<String, ModelEntry>,
    deployments: Vec<DeploymentLabel>,
    /// Keyed by matched route, then [`SeriesKey`], so a lookup borrows the route as `&str`.
    series: RwLock<RouteSeries>,
    failures: RwLock<HashMap<FailureKey, AtomicU64>>,
    in_flight: AtomicI64,
    draining: Arc<AtomicBool>,
}

impl Metrics {
    pub fn new(config: &Config, draining: Arc<AtomicBool>) -> Self {
        let mut model_labels = vec![String::new()];
        let mut models: HashMap<String, ModelEntry> = HashMap::new();
        let mut deployments = Vec::new();
        for model in &config.model_list {
            let next_index = u32::try_from(model_labels.len()).unwrap_or(u32::MAX);
            let entry = models
                .entry(model.model_name.clone())
                .or_insert_with(|| ModelEntry {
                    index: next_index,
                    deployments: Vec::new(),
                });
            if entry.index == next_index {
                model_labels.push(model.model_name.clone());
            }
            let api_base = model.litellm_params.api_base.clone();
            if entry
                .deployments
                .iter()
                .any(|(existing, _)| *existing == api_base)
            {
                continue;
            }
            let deployment = u32::try_from(deployments.len()).unwrap_or(u32::MAX);
            deployments.push(DeploymentLabel {
                model: entry.index,
                api_base: api_base.as_deref().map(api_base_label).unwrap_or_default(),
            });
            entry.deployments.push((api_base, deployment));
        }
        Self(Arc::new(Inner {
            model_labels,
            models,
            deployments,
            series: RwLock::default(),
            failures: RwLock::default(),
            in_flight: AtomicI64::new(0),
            draining,
        }))
    }

    fn series(&self, route: &str, key: SeriesKey) -> Arc<Series> {
        if let Some(series) = self
            .0
            .series
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .get(route)
            .and_then(|by_key| by_key.get(&key))
        {
            return series.clone();
        }
        self.0
            .series
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .entry(Arc::from(route))
            .or_default()
            .entry(key)
            .or_default()
            .clone()
    }

    fn record_failure(&self, resolved: &observe::Resolved, status: u16) {
        let Some(model) = self.0.models.get(&resolved.model_group) else {
            return;
        };
        let Some(deployment) = model
            .deployments
            .iter()
            .find(|(api_base, _)| *api_base == resolved.api_base)
            .or_else(|| model.deployments.first())
            .map(|(_, deployment)| *deployment)
        else {
            return;
        };
        let key = FailureKey { deployment, status };
        if let Some(counter) = self
            .0
            .failures
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .get(&key)
        {
            counter.fetch_add(1, Ordering::Relaxed);
            return;
        }
        self.0
            .failures
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .entry(key)
            .or_default()
            .fetch_add(1, Ordering::Relaxed);
    }

    pub fn render(&self) -> String {
        let inner = &self.0;
        let mut out = String::with_capacity(4096);
        let mut series: Vec<(Arc<str>, SeriesKey, Arc<Series>)> = inner
            .series
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .iter()
            .flat_map(|(route, by_key)| {
                by_key
                    .iter()
                    .map(|(key, series)| (route.clone(), *key, series.clone()))
            })
            .collect();
        series.sort_by(|left, right| (&left.0, left.1).cmp(&(&right.0, right.1)));
        let labels: Vec<String> = series
            .iter()
            .map(|(route, (model, status), _)| {
                format!(
                    "route=\"{}\",requested_model=\"{}\",status_code=\"{}\"",
                    escape(route),
                    escape(&inner.model_labels[*model as usize]),
                    status
                )
            })
            .collect();

        out.push_str("# HELP litellm_proxy_total_requests_metric Total requests the proxy answered, by route, requested model and status code\n");
        out.push_str("# TYPE litellm_proxy_total_requests_metric counter\n");
        for ((_, _, series), labels) in series.iter().zip(&labels) {
            let _ = writeln!(
                out,
                "litellm_proxy_total_requests_metric_total{{{labels}}} {}",
                series.latency.count.load(Ordering::Relaxed)
            );
        }
        out.push_str("# HELP litellm_request_total_latency_metric Seconds from request start to the end of the response body\n");
        out.push_str("# TYPE litellm_request_total_latency_metric histogram\n");
        for ((_, _, series), labels) in series.iter().zip(&labels) {
            series
                .latency
                .render(&mut out, "litellm_request_total_latency_metric", labels);
        }
        out.push_str("# HELP litellm_request_time_to_first_byte_metric Seconds from request start to the first response body byte\n");
        out.push_str("# TYPE litellm_request_time_to_first_byte_metric histogram\n");
        for ((_, _, series), labels) in series.iter().zip(&labels) {
            series.time_to_first_byte.render(
                &mut out,
                "litellm_request_time_to_first_byte_metric",
                labels,
            );
        }

        out.push_str("# HELP litellm_deployment_failure_responses Upstream calls that failed with 429 or 5xx, by model group and deployment api_base\n");
        out.push_str("# TYPE litellm_deployment_failure_responses counter\n");
        let mut failures: Vec<(FailureKey, u64)> = inner
            .failures
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .iter()
            .map(|(key, count)| (*key, count.load(Ordering::Relaxed)))
            .collect();
        failures.sort_by_key(|(key, _)| *key);
        for (key, count) in failures {
            let deployment = &inner.deployments[key.deployment as usize];
            let _ = writeln!(
                out,
                "litellm_deployment_failure_responses_total{{requested_model=\"{}\",api_base=\"{}\",exception_status=\"{}\"}} {count}",
                escape(&inner.model_labels[deployment.model as usize]),
                escape(&deployment.api_base),
                key.status
            );
        }

        out.push_str("# HELP litellm_in_flight_requests Requests whose response has not finished, streams included\n");
        out.push_str("# TYPE litellm_in_flight_requests gauge\n");
        let _ = writeln!(
            out,
            "litellm_in_flight_requests {}",
            inner.in_flight.load(Ordering::Relaxed)
        );
        out.push_str("# HELP litellm_gateway_draining 1 once shutdown has started and readiness reports unhealthy\n");
        out.push_str("# TYPE litellm_gateway_draining gauge\n");
        let _ = writeln!(
            out,
            "litellm_gateway_draining {}",
            u8::from(inner.draining.load(Ordering::Relaxed))
        );
        out
    }
}

/// `scheme://host[:port]/path` of a configured `api_base`, without credentials or query string.
fn api_base_label(api_base: &str) -> String {
    let without_query = api_base.split(['?', '#']).next().unwrap_or_default();
    match without_query.split_once("://") {
        Some((scheme, rest)) => {
            let (authority, path) = rest.split_at(rest.find('/').unwrap_or(rest.len()));
            let host = authority.rsplit('@').next().unwrap_or(authority);
            format!("{scheme}://{host}{}", path.trim_end_matches('/'))
        }
        None => without_query.trim_end_matches('/').to_owned(),
    }
}

fn escape(value: &str) -> std::borrow::Cow<'_, str> {
    if value.contains(['\\', '"', '\n']) {
        value
            .replace('\\', "\\\\")
            .replace('"', "\\\"")
            .replace('\n', "\\n")
            .into()
    } else {
        value.into()
    }
}

fn is_upstream_failure(status: StatusCode) -> bool {
    status == StatusCode::TOO_MANY_REQUESTS
        || (status.is_server_error() && status != StatusCode::NOT_IMPLEMENTED)
}

/// Decrements the in-flight gauge and records latency once the response body is done or dropped.
struct Pending {
    metrics: Metrics,
    started: Instant,
    series: Option<Arc<Series>>,
    first_byte: bool,
}

impl Pending {
    fn on_frame(&mut self) {
        if self.first_byte {
            return;
        }
        self.first_byte = true;
        if let Some(series) = &self.series {
            series.time_to_first_byte.observe(self.started.elapsed());
        }
    }
}

impl Drop for Pending {
    fn drop(&mut self) {
        if let Some(series) = &self.series {
            series.latency.observe(self.started.elapsed());
        }
        self.metrics.0.in_flight.fetch_sub(1, Ordering::Relaxed);
    }
}

/// Middleware recording request count, latency, time to first byte and in-flight requests.
pub async fn track(State(metrics): State<Metrics>, request: Request, next: Next) -> Response {
    let started = Instant::now();
    metrics.0.in_flight.fetch_add(1, Ordering::Relaxed);
    // Created before the handler runs so a cancelled request still leaves the gauge.
    let mut pending = Pending {
        metrics: metrics.clone(),
        started,
        series: None,
        first_byte: false,
    };
    let route = request.extensions().get::<MatchedPath>().cloned();
    let (response, resolved) = observe::scope(next.run(request)).await;
    let status = response.status();
    let model = resolved
        .as_ref()
        .and_then(|resolved| metrics.0.models.get(&resolved.model_group))
        .map_or(0, |entry| entry.index);
    if let Some(resolved) = &resolved
        && is_upstream_failure(status)
    {
        metrics.record_failure(resolved, status.as_u16());
    }
    pending.series = Some(metrics.series(
        route.as_ref().map_or(UNMATCHED_ROUTE, MatchedPath::as_str),
        (model, status.as_u16()),
    ));
    response.map(|body| {
        Body::new(body.map_frame(move |frame| {
            if frame.is_data() {
                pending.on_frame();
            }
            frame
        }))
    })
}

pub async fn handler(State(metrics): State<Metrics>) -> impl IntoResponse {
    (
        [(
            header::CONTENT_TYPE,
            "text/plain; version=0.0.4; charset=utf-8",
        )],
        metrics.render(),
    )
}

#[cfg(test)]
mod tests {
    use axum::{
        Router,
        body::{Bytes, to_bytes},
        middleware,
        routing::{get, post},
    };
    use futures_util::stream;
    use rstest::{fixture, rstest};
    use tower::ServiceExt;

    use super::*;

    const CONFIG: &str = r#"
model_list:
  - model_name: chat-group
    litellm_params:
      model: openai_like/gpt
      api_base: http://user:pass@mock:9000/d/chat-0/?token=x
  - model_name: chat-group
    litellm_params:
      model: openai_like/gpt
      api_base: http://mock:9000/d/chat-1
  - model_name: claude-group
    litellm_params:
      model: anthropic/claude
"#;

    #[fixture]
    fn metrics() -> Metrics {
        let config = Config::from_yaml(CONFIG).unwrap();
        Metrics::new(&config, Arc::default())
    }

    fn resolved(model_group: &str, api_base: Option<&str>) -> observe::Resolved {
        observe::Resolved {
            model_group: model_group.into(),
            api_base: api_base.map(Into::into),
        }
    }

    fn app(metrics: Metrics) -> Router {
        Router::new()
            .route("/ok", post(|| async { "done" }))
            .route(
                "/stream",
                post(|| async {
                    Body::from_stream(stream::iter([
                        Ok::<_, std::convert::Infallible>(Bytes::from_static(b"a")),
                        Ok(Bytes::from_static(b"b")),
                    ]))
                }),
            )
            .route("/fail", post(|| async { StatusCode::SERVICE_UNAVAILABLE }))
            .layer(middleware::from_fn_with_state(metrics.clone(), track))
            .route("/metrics", get(handler).with_state(metrics))
    }

    async fn call(app: &Router, method: &str, path: &str) -> (StatusCode, String) {
        let response = app
            .clone()
            .oneshot(
                Request::builder()
                    .method(method)
                    .uri(path)
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        let status = response.status();
        let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        (status, String::from_utf8(body.to_vec()).unwrap())
    }

    #[rstest]
    #[case::histogram_floor(Duration::from_millis(1), 0)]
    #[case::on_a_bound(Duration::from_millis(100), 4)]
    #[case::between_bounds(Duration::from_secs(45), 12)]
    #[case::above_every_bound(Duration::from_secs(7200), BUCKETS.len())]
    fn histogram_places_observations_in_the_first_bucket_that_holds_them(
        #[case] elapsed: Duration,
        #[case] bucket: usize,
    ) {
        let histogram = Histogram::default();

        histogram.observe(elapsed);

        assert_eq!(histogram.buckets[bucket].load(Ordering::Relaxed), 1);
        assert_eq!(histogram.count.load(Ordering::Relaxed), 1);
    }

    #[rstest]
    #[case::credentials_and_query(
        "http://user:pass@mock:9000/d/x/?token=y",
        "http://mock:9000/d/x"
    )]
    #[case::host_only("https://api.example.com", "https://api.example.com")]
    #[case::no_scheme("mock:9000/v1/", "mock:9000/v1")]
    fn api_base_labels_drop_credentials_and_query(#[case] api_base: &str, #[case] label: &str) {
        assert_eq!(api_base_label(api_base), label);
    }

    #[rstest]
    #[case::rate_limited(StatusCode::TOO_MANY_REQUESTS, true)]
    #[case::bad_gateway(StatusCode::BAD_GATEWAY, true)]
    #[case::not_implemented(StatusCode::NOT_IMPLEMENTED, false)]
    #[case::bad_request(StatusCode::BAD_REQUEST, false)]
    fn upstream_failures_are_429_and_5xx_except_501(
        #[case] status: StatusCode,
        #[case] failure: bool,
    ) {
        assert_eq!(is_upstream_failure(status), failure);
    }

    #[rstest]
    #[tokio::test]
    async fn requests_are_counted_by_route_and_status(metrics: Metrics) {
        let app = app(metrics);
        call(&app, "POST", "/ok").await;
        call(&app, "POST", "/ok").await;
        call(&app, "POST", "/stream").await;

        let (status, body) = call(&app, "GET", "/metrics").await;

        assert_eq!(status, StatusCode::OK);
        assert!(body.contains(
            "litellm_proxy_total_requests_metric_total{route=\"/ok\",requested_model=\"\",status_code=\"200\"} 2\n"
        ));
        assert!(body.contains(
            "litellm_request_total_latency_metric_count{route=\"/stream\",requested_model=\"\",status_code=\"200\"} 1\n"
        ));
        assert!(body.contains(
            "litellm_request_time_to_first_byte_metric_bucket{route=\"/stream\",requested_model=\"\",status_code=\"200\",le=\"+Inf\"} 1\n"
        ));
        assert!(body.contains("litellm_in_flight_requests 0\n"));
        assert!(body.contains("litellm_gateway_draining 0\n"));
        assert!(!body.contains("route=\"/metrics\""));
    }

    #[rstest]
    #[tokio::test]
    async fn in_flight_counts_a_response_until_its_body_is_dropped(metrics: Metrics) {
        let app = app(metrics.clone());
        let response = app
            .clone()
            .oneshot(Request::post("/stream").body(Body::empty()).unwrap())
            .await
            .unwrap();

        assert_eq!(metrics.0.in_flight.load(Ordering::Relaxed), 1);
        drop(response);
        assert_eq!(metrics.0.in_flight.load(Ordering::Relaxed), 0);
    }

    #[rstest]
    #[case::configured_deployment(Some("http://mock:9000/d/chat-1"), "http://mock:9000/d/chat-1")]
    #[case::sanitized_label(
        Some("http://user:pass@mock:9000/d/chat-0/?token=x"),
        "http://mock:9000/d/chat-0"
    )]
    #[case::unknown_api_base_falls_back_to_first(
        Some("http://elsewhere"),
        "http://mock:9000/d/chat-0"
    )]
    fn failures_are_labelled_by_model_group_and_deployment(
        metrics: Metrics,
        #[case] api_base: Option<&str>,
        #[case] label: &str,
    ) {
        metrics.record_failure(&resolved("chat-group", api_base), 503);

        assert!(metrics.render().contains(&format!(
            "litellm_deployment_failure_responses_total{{requested_model=\"chat-group\",api_base=\"{label}\",exception_status=\"503\"}} 1\n"
        )));
    }

    #[rstest]
    fn failures_for_unknown_model_groups_are_ignored(metrics: Metrics) {
        metrics.record_failure(&resolved("missing", None), 503);

        assert!(
            !metrics
                .render()
                .contains("litellm_deployment_failure_responses_total{")
        );
    }

    #[rstest]
    #[tokio::test]
    async fn unresolved_failures_do_not_count_as_upstream_failures(metrics: Metrics) {
        let app = app(metrics);
        call(&app, "POST", "/fail").await;

        let (_, body) = call(&app, "GET", "/metrics").await;

        assert!(body.contains("status_code=\"503\"} 1\n"));
        assert!(!body.contains("litellm_deployment_failure_responses_total{"));
    }

    #[rstest]
    fn label_values_are_escaped() {
        assert_eq!(escape("a\"b\\c\nd"), "a\\\"b\\\\c\\nd");
    }
}
