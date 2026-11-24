//! The proxy's `prometheus` callback as a callback of the stage: each attempt that ends is
//! recorded into the metric families `PrometheusLogger` declares, by the logic in
//! `litellm-callbacks-prometheus`, and `/metrics` writes them as `prometheus_client` does.
//!
//! The logger runs with its default settings, since `admit` leaves a config that sets any
//! Prometheus setting to Python. Accepted differences: Python's registry also holds the client's
//! `process_*`, `python_*` and `platform` collectors, which a Rust process has no counterpart of;
//! the metrics the proxy records outside the success and failure logs (the proxy failure hook,
//! deployment state, budgets, the dynamic limiter's shares) are not recorded yet; and a scrape is
//! always answered in the text format, as a scraper that asks for OpenMetrics still reads it.

use std::{
    sync::Arc,
    time::{SystemTime, UNIX_EPOCH},
};

use std::io::Write;

use axum::{
    body::Body,
    extract::{Request, State},
    http::Uri,
    http::{HeaderMap, HeaderValue, StatusCode, header},
    middleware::Next,
    response::{IntoResponse, Response},
    routing::get,
};
use flate2::{Compression, write::GzEncoder};
use futures_util::future::BoxFuture;
use litellm_callbacks::{
    Activation, Callback, CallbackError, FieldSet, Hook, HookInput, HookOutput, Manifest, Scalar,
    Selection, Subscription,
};
use litellm_callbacks_prometheus::{
    Attempt, AttemptFacts, Event, HostFacts, Level, LogLine, LogSink, Moment, Prometheus, READS,
    SeriesLimits, Store, SystemClock,
    exposition::{CONTENT_TYPE, Scrape, created_series, render},
    families::{FamiliesError, default_store},
};
use litellm_config::{Config, LitellmRole};
use litellm_env::Lookup;
use litellm_gateway_auth::Auth;
use litellm_router_core::Router;
use serde_json::{Map, Value};

use super::Registered;
use crate::host::NativeRows;

/// The name the config's `callbacks` lists the logger by.
pub const NAME: &str = "prometheus";

/// What a call costs until the gateway prices calls, as the logging payload holds a cost it
/// could not work out.
const UNPRICED: f64 = 0.0;

/// Whether the `_created` series are written, which `prometheus_client` reads at import.
const DISABLE_CREATED: &str = "PROMETHEUS_DISABLE_CREATED_SERIES";

/// Python writes the logger's errors through `verbose_logger`, at the level the logger chose.
struct Logged;

impl LogSink for Logged {
    fn log(&self, line: &LogLine) {
        let message = formatted(line);
        match line.level {
            Level::Debug => tracing::debug!(callback = NAME, "{message}"),
            Level::Error => tracing::error!(callback = NAME, "{message}"),
        }
    }
}

/// `format % args` for the `%s` placeholders the logger writes.
fn formatted(line: &LogLine) -> String {
    let mut args = line.args.iter();
    let mut parts = line.format.split("%s");
    let mut text = parts.next().unwrap_or_default().to_owned();
    for part in parts {
        text.push_str(args.next().map_or("%s", String::as_str));
        text.push_str(part);
    }
    text
}

/// The model a deployment is logged under, by its id: the one its provider is sent, as the
/// logging payload's `model` names it.
type ServedModel = Box<dyn Fn(&str) -> Option<String> + Send + Sync>;

/// The metrics of every call the gateway serves, and how a scrape reads them.
pub struct Metrics {
    logger: Prometheus,
    manifest: Manifest,
    created: bool,
    served_model: ServedModel,
}

/// `get_llm_provider`'s model: the deployment's with the provider it names stripped.
fn served_model(router: &Router, id: &str) -> Option<String> {
    let deployment = router.registry().deployment(id)?;
    let model = &deployment.params.model;
    let provider = deployment
        .resolved_provider
        .as_deref()
        .or(deployment.params.custom_llm_provider.as_deref());
    Some(
        provider
            .and_then(|provider| model.strip_prefix(provider)?.strip_prefix('/'))
            .unwrap_or(model)
            .to_owned(),
    )
}

impl Metrics {
    /// `None` when the config's `callbacks` does not list the logger.
    pub fn read(
        config: &Config,
        environment: &dyn Lookup,
        router: Router,
    ) -> Result<Option<Arc<Self>>, FamiliesError> {
        let named = config
            .litellm_settings
            .callbacks
            .iter()
            .flatten()
            .any(|callback| callback.as_str() == Some(NAME));
        if !named {
            return Ok(None);
        }
        let store = default_store(SeriesLimits::default(), Box::new(SystemClock::default()))?;
        Ok(Some(Arc::new(Self::over(
            store,
            created_series(environment.get(DISABLE_CREATED).as_deref()),
            Box::new(move |id| served_model(&router, id)),
        ))))
    }

    fn over(store: Store, created: bool, served_model: ServedModel) -> Self {
        Self {
            logger: Prometheus::new(Arc::new(store), Arc::new(Logged), false),
            manifest: Manifest {
                selection: Selection {
                    activation: Activation::Required,
                    ..Selection::named(NAME)
                },
                subscriptions: vec![Subscription {
                    reads: FieldSet::of(&READS),
                    ..Subscription::on(Hook::AttemptEnded)
                }],
                has_request_state: false,
            },
            created,
            served_model,
        }
    }

    pub(crate) fn registered(self: &Arc<Self>) -> Registered {
        Registered::from(self.clone() as Arc<dyn litellm_callbacks::ErasedCallback>)
    }

    fn attempt_ended(&self, input: &HookInput) -> HookOutput {
        let Some((outcome, mut facts)) = AttemptFacts::read(input, host_facts(input))? else {
            return Ok(None);
        };
        facts
            .cost
            .response_cost
            .get_or_insert(Scalar::Float(UNPRICED));
        if let Some(model) = input
            .call
            .as_ref()
            .and_then(|call| call.target.deployment_id.as_deref())
            .and_then(&self.served_model)
        {
            facts.model = Value::from(model);
        }
        let Some(event) = Event::of(outcome) else {
            return Ok(None);
        };
        self.logger
            .run(&Attempt { event, facts })
            .map_err(CallbackError::from)?;
        Ok(None)
    }

    /// What a scrape asking for `names` is answered with.
    pub fn scrape(&self, names: Vec<String>) -> String {
        render(
            self.logger.store(),
            &Scrape {
                names,
                created: self.created,
            },
        )
    }
}

/// The deployment labels the logging payload carries, by the deployment the router chose, and
/// the logger's clock, which stands in for an end time the call did not log.
fn host_facts(input: &HookInput) -> HostFacts {
    let target = input.call.as_ref().map(|call| &call.target);
    let text = |value: Option<&String>| value.map_or(Value::Null, |text| Value::from(text.clone()));
    let deployment_labels = Map::from_iter([
        (
            "model_group".to_owned(),
            text(target.and_then(|target| target.model_group.as_ref())),
        ),
        (
            "model_id".to_owned(),
            text(target.and_then(|target| target.deployment_id.as_ref())),
        ),
        (
            "api_base".to_owned(),
            Value::from(
                target
                    .and_then(|target| target.api_base.clone())
                    .unwrap_or_default(),
            ),
        ),
        (
            "custom_llm_provider".to_owned(),
            text(target.and_then(|target| target.provider.as_ref())),
        ),
    ]);
    HostFacts {
        settings: litellm_callbacks_prometheus::labels::Settings::default(),
        deployment_labels,
        now: Some(now()),
    }
}

fn now() -> Moment {
    let elapsed = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    Moment {
        us: i64::try_from(elapsed.as_micros()).unwrap_or(i64::MAX),
        aware: false,
    }
}

impl Callback for Metrics {
    type State = ();

    fn manifest(&self) -> &Manifest {
        &self.manifest
    }

    fn handle<'a>(
        &'a self,
        input: &'a HookInput,
        _state: Option<&'a Self::State>,
    ) -> BoxFuture<'a, HookOutput> {
        Box::pin(async move {
            match input.hook {
                Hook::AttemptEnded => self.attempt_ended(input),
                _ => Ok(None),
            }
        })
    }
}

/// `/metrics` as the proxy mounts it, behind `PrometheusAuthMiddleware`, which authenticates every
/// scrape while `require_auth_for_metrics_endpoint` keeps its default. The app is mounted at
/// `/metrics/`, so Starlette redirects `/metrics` there.
pub(crate) fn router(metrics: Arc<Metrics>, auth: Auth<NativeRows>) -> axum::Router {
    axum::Router::new()
        .route("/metrics", get(to_the_mount))
        .route("/metrics/", get(scrape))
        .with_state(metrics)
        .route_layer(axum::middleware::from_fn_with_state(auth, authorized))
}

/// The proxy answers a scrape its auth refuses 401 with why, as a JSON string. A key that is no
/// proxy admin is refused, as the proxy's route checks refuse it a route that is no LLM route.
async fn authorized(
    State(auth): State<Auth<NativeRows>>,
    request: Request,
    next: Next,
) -> Response {
    let (parts, body) = request.into_parts();
    let why = match auth.authenticate_parts(&parts).await {
        Ok(caller) if caller.role() == LitellmRole::ProxyAdmin => {
            return next.run(Request::from_parts(parts, body)).await;
        }
        Ok(_) => "only a proxy admin can read the metrics".to_owned(),
        Err(error) => error.to_string(),
    };
    let message = format!(
        "Unauthorized access to metrics endpoint: {why} To allow unauthenticated access, set `litellm_settings.require_auth_for_metrics_endpoint: false` in your proxy_config.yaml."
    );
    (
        StatusCode::UNAUTHORIZED,
        [(header::CONTENT_TYPE, "application/json")],
        Value::String(message).to_string(),
    )
        .into_response()
}

/// Starlette's `redirect_slashes`: a 307 to the same URL with the slash added, on the host the
/// request named.
async fn to_the_mount(uri: Uri, headers: HeaderMap) -> Response {
    let query = uri
        .query()
        .map(|query| format!("?{query}"))
        .unwrap_or_default();
    let location = headers
        .get(header::HOST)
        .and_then(|host| host.to_str().ok())
        .map_or_else(
            || format!("/metrics/{query}"),
            |host| format!("http://{host}/metrics/{query}"),
        );
    let mut response = StatusCode::TEMPORARY_REDIRECT.into_response();
    if let Ok(location) = HeaderValue::try_from(location) {
        response.headers_mut().insert(header::LOCATION, location);
    }
    response
}

/// `prometheus_client.exposition.gzip_accepted`.
fn gzip_accepted(headers: &HeaderMap) -> bool {
    headers
        .get_all(header::ACCEPT_ENCODING)
        .iter()
        .filter_map(|value| value.to_str().ok())
        .flat_map(|value| value.split(','))
        .any(|accepted| {
            accepted
                .split(';')
                .next()
                .is_some_and(|encoding| encoding.trim().eq_ignore_ascii_case("gzip"))
        })
}

fn gzipped(text: &[u8]) -> std::io::Result<Vec<u8>> {
    let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
    encoder.write_all(text)?;
    encoder.finish()
}

/// `make_metrics_asgi_app`: `name[]` restricts the scrape, it is written off the request path,
/// and a client that accepts gzip gets it gzipped.
async fn scrape(State(metrics): State<Arc<Metrics>>, uri: Uri, headers: HeaderMap) -> Response {
    let names = url::form_urlencoded::parse(uri.query().unwrap_or_default().as_bytes())
        .filter(|(key, _)| key == "name[]")
        .map(|(_, value)| value.into_owned())
        .collect();
    let gzip = gzip_accepted(&headers);
    let written = tokio::task::spawn_blocking(move || {
        let text = metrics.scrape(names).into_bytes();
        if gzip { gzipped(&text) } else { Ok(text) }
    })
    .await;
    let Ok(Ok(body)) = written else {
        return StatusCode::INTERNAL_SERVER_ERROR.into_response();
    };
    let mut response = Response::new(Body::from(body));
    let response_headers = response.headers_mut();
    response_headers.insert(header::CONTENT_TYPE, HeaderValue::from_static(CONTENT_TYPE));
    if gzip {
        response_headers.insert(header::CONTENT_ENCODING, HeaderValue::from_static("gzip"));
    }
    response
}

#[cfg(test)]
mod tests {
    use litellm_callbacks::{CallFacts, CallTarget, Field, FieldValues};
    use litellm_callbacks_prometheus::ManualClock;
    use rstest::rstest;
    use serde_json::json;

    use super::*;

    fn metrics() -> Metrics {
        let store = default_store(SeriesLimits::default(), Box::new(ManualClock::default()))
            .unwrap_or_else(|error| panic!("{error}"));
        Metrics::over(
            store,
            false,
            Box::new(|_| Some("claude-haiku-4-5".to_owned())),
        )
    }

    fn ended(values: Vec<(Field, Value)>) -> HookInput {
        let identity = json!({
            "metadata_kind": "mapping",
            "key_hash": "litellm_proxy_master_key",
            "user_id": "default_user_id",
        });
        let values = [
            vec![
                (Field::LoggedKeyIdentity, identity),
                (Field::RequestTags, json!([])),
            ],
            values,
        ]
        .concat();
        let mut fields = FieldValues::new(FieldSet::of(&READS));
        for (field, value) in values {
            fields.insert(field, value);
        }
        HookInput {
            hook: Hook::AttemptEnded,
            fields,
            call: Some(CallFacts {
                target: CallTarget {
                    model: "claude".to_owned(),
                    model_group: Some("claude".to_owned()),
                    provider: Some("anthropic".to_owned()),
                    api_base: Some("https://api.anthropic.com".to_owned()),
                    deployment_id: Some("deployment-1".to_owned()),
                    attempt: None,
                },
                ..CallFacts::default()
            }),
        }
    }

    /// `ms` past a fixed second, naive as the logging payload's times are.
    fn moment(ms: u32) -> Value {
        json!({"year": 2026, "month": 11, "day": 23, "hour": 5, "minute": 0, "second": 0, "microsecond": ms * 1000})
    }

    fn sample<'a>(text: &'a str, name: &str) -> Vec<&'a str> {
        text.lines()
            .filter(|line| line.starts_with(&format!("{name}{{")))
            .collect()
    }

    #[rstest]
    fn the_callback_is_built_only_when_the_config_names_it() {
        let lookup = |_: &str| None::<String>;
        let router = || {
            Router::new(
                litellm_router_core::RouterConfig::default(),
                litellm_router_core::RouterDynamicConfig::default(),
            )
            .unwrap()
        };
        let unnamed = Config::from_yaml("litellm_settings:\n  callbacks: []\n").unwrap();
        let named = Config::from_yaml("litellm_settings:\n  callbacks: [prometheus]\n").unwrap();
        assert!(
            Metrics::read(&unnamed, &lookup, router())
                .unwrap()
                .is_none()
        );
        assert!(Metrics::read(&named, &lookup, router()).unwrap().is_some());
    }

    #[rstest]
    fn a_served_call_counts_its_request_and_tokens_under_the_deployment_it_reached() {
        let metrics = metrics();
        let input = ended(vec![
            (Field::Model, json!("claude")),
            (Field::Outcome, json!("success")),
            (
                Field::Usage,
                json!({"logged": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30}}),
            ),
            (Field::LoggedModelGroup, json!({"value": "claude"})),
            (
                Field::RateLimitState,
                json!({"key_model_group": "claude", "hidden_params_present": true, "additional_headers_present": true}),
            ),
            (
                Field::Deployment,
                json!({"model_info": {"id": "deployment-1"}, "provider": "anthropic"}),
            ),
            (
                Field::AttemptTiming,
                json!({
                    "start": moment(0),
                    "api_call_start": moment(100),
                    "end": moment(600),
                    "logged_start": moment(0),
                    "logged_end": moment(600),
                }),
            ),
        ]);
        let ended = metrics.attempt_ended(&input);
        assert!(matches!(ended, Ok(None)), "{ended:?}");
        let text = metrics.scrape(Vec::new());
        assert_eq!(
            sample(&text, "litellm_deployment_success_responses_total").len(),
            1,
            "{text}"
        );
        assert!(
            sample(&text, "litellm_deployment_success_responses_total")[0]
                .contains("api_base=\"https://api.anthropic.com\""),
            "{text}"
        );
        assert_eq!(
            sample(&text, "litellm_proxy_total_requests_metric_total").len(),
            1
        );
    }

    #[rstest]
    fn a_failed_call_counts_a_failure() {
        let metrics = metrics();
        let input = ended(vec![
            (Field::Model, json!("claude")),
            (Field::Outcome, json!("failure")),
        ]);
        let _ = metrics.attempt_ended(&input);
        let text = metrics.scrape(Vec::new());
        assert_eq!(
            sample(&text, "litellm_llm_api_failed_requests_metric_total").len(),
            1,
            "{text}"
        );
        assert!(sample(&text, "litellm_requests_metric_total").is_empty());
    }

    #[rstest]
    #[case::one_placeholder("a %s b", vec!["x"], "a x b")]
    #[case::missing_argument("%s and %s", vec!["x"], "x and %s")]
    fn a_log_line_is_formatted_as_python_formats_it(
        #[case] format: &'static str,
        #[case] args: Vec<&str>,
        #[case] expected: &str,
    ) {
        let line = LogLine::new(
            Level::Error,
            format,
            args.into_iter().map(str::to_owned).collect(),
        );
        assert_eq!(formatted(&line), expected);
    }
}
