//! The headers every response on the inference routes carries beyond the route's own. The call's
//! record holds what the call did, and [`render_response_headers`] is the one place that turns it
//! into headers, in one pass: it knows every header's name and the order they go out in, and reads
//! the record only through what `CallState` and `AttemptRecord` expose. It writes the custom
//! headers of Python's `get_custom_headers` (the cache key, what the router did for the call, and
//! the call's details: the deployment's model and api base, its durations, the time spent in
//! callbacks and the provider's own headers, or on an error what the proxy's handler adds to it),
//! then the headers callbacks and Rust code added, then the defaults. It runs inside
//! `record::scoped`, so it reads the record of the call it responds to, and it closes the record's
//! header bag as it writes, so a header added after the head, a stream's end included, is refused.
//!
//! Python only sends a custom header whose value is set, so a value that is empty is left out. It
//! builds the call's details from the response's `_hidden_params`, which a `/v1/messages`
//! response only carries for a stream, so the api base, the callback duration and the provider's
//! headers of a complete Messages response are not sent. An OCR response carries them, but its
//! provider headers are never copied in, so only its error response sends them.

use std::time::Instant;

use axum::{
    extract::Request,
    http::{HeaderMap, HeaderName, HeaderValue},
    middleware::Next,
    response::Response,
};
use litellm_python_compat::{datetime::milliseconds, number::round, repr::float_repr};
use litellm_router_core::RouterResponse;
use serde_json::{Map, Value};

use crate::{
    CallType,
    record::{self, AttemptRecord, CallState, ErrorResponse},
};

/// `LITELLM_UI_ALLOW_HEADERS` in `litellm/constants.py`.
#[derive(Clone, Copy, Debug, strum::VariantArray, strum::IntoStaticStr)]
pub enum ExposedHeader {
    #[strum(serialize = "x-litellm-semantic-filter")]
    SemanticFilter,
    #[strum(serialize = "x-litellm-semantic-filter-tools")]
    SemanticFilterTools,
    #[strum(serialize = "x-litellm-adaptive-router-model")]
    AdaptiveRouterModel,
    #[strum(serialize = "x-litellm-applied-guardrails")]
    AppliedGuardrails,
    #[strum(serialize = "x-litellm-guardrail-scan-id")]
    GuardrailScanId,
    #[strum(serialize = "x-litellm-guardrail-scan-metadata")]
    GuardrailScanMetadata,
    #[strum(serialize = "x-litellm-cache-key")]
    CacheKey,
}

impl From<ExposedHeader> for HeaderName {
    fn from(header: ExposedHeader) -> Self {
        HeaderName::from_static(header.into())
    }
}

/// What the router did for a request. The route writes it to the call's record, and
/// [`render_response_headers`] turns it into the proxy's `x-litellm-*` headers, the same for a
/// complete response and for a stream, whose head is written after its attempt was served.
#[derive(Clone, Debug)]
pub struct Routing {
    /// The deployment the last attempt ran on.
    deployment_id: String,
    /// `None` for a call that ended on a failed attempt, which the proxy names only the
    /// deployment of.
    served: Option<Served>,
}

#[derive(Clone, Debug)]
struct Served {
    /// The name the call ran under, an alias as the client wrote it (`x-litellm-model-group`).
    model_group: String,
    /// The group that name resolves to, which the router counts usage under.
    counted_group: String,
    attempted_retries: u32,
    attempted_fallbacks: u32,
    max_retries: Option<u32>,
}

impl Routing {
    pub(crate) fn served<R>(routed: &RouterResponse<R>) -> Self {
        Self {
            deployment_id: routed.deployment_id.clone(),
            served: Some(Served {
                model_group: routed.name.clone(),
                counted_group: routed.model_group.clone(),
                attempted_retries: routed.attempted_retries,
                attempted_fallbacks: routed.attempted_fallbacks,
                max_retries: routed.max_retries,
            }),
        }
    }

    /// The model group that served the call, an alias resolved, `None` for a call that ended on
    /// a failed attempt.
    pub(crate) fn counted_group(&self) -> Option<&str> {
        Some(&self.served.as_ref()?.counted_group)
    }

    /// For a call that ended on a failed attempt on `deployment_id`.
    pub(crate) fn failed_on(deployment_id: &str) -> Self {
        Self {
            deployment_id: deployment_id.to_owned(),
            served: None,
        }
    }
}

/// The layer just inside `record::scoped` that writes the head of every response from the
/// record of its call.
pub(crate) async fn render_response_headers(request: Request, next: Next) -> Response {
    let mut response = next.run(request).await;
    record::write(|entries| head(entries, response.headers_mut(), Instant::now()));
    response
}

/// Writes the head `entries` give a response over `headers`, the route's own, as `now`. Each
/// layer replaces a header of the same name an earlier one set:
///
/// 1. the served attempt's cache key, on a served response
/// 2. what the router did for the call, or for the attempt an error ended on
/// 3. the call's details on a served response, or what the proxy's handler adds to an error
/// 4. the headers callbacks and Rust code added, on a served response, as Python applies
///    `post_call_response_headers_hook` over `get_custom_headers`
///
/// Last come the defaults, each only where no header of its name was set, as Python's
/// `apply_remaining_usage_headers` does.
fn head(entries: &mut CallState, headers: &mut HeaderMap, now: Instant) {
    let error = entries.error_response().cloned();
    let served = error.is_none();
    let mut layers = cache_key(entries, served);
    layers.extend(routing_headers(entries.routing()));
    layers.extend(match &error {
        None => served_details(entries, now),
        Some(error) => error_details(entries.call_type, error),
    });
    let (added, defaults) = entries.send_head();
    if served {
        layers.extend(as_text(added));
    }
    for (name, value) in layers {
        if let Some((name, value)) = header(&name, &value) {
            headers.insert(name, value);
        }
    }
    for (name, value) in as_text(defaults) {
        if let Some((name, value)) = header(&name, &value) {
            headers.entry(name).or_insert(value);
        }
    }
}

/// One header as the response carries it, `None` when its name or value is no header.
fn header(name: &str, value: &str) -> Option<(HeaderName, HeaderValue)> {
    Some((
        HeaderName::try_from(name).ok()?,
        HeaderValue::try_from(value).ok()?,
    ))
}

/// A header value the record holds as text: a text as written and any other value as its JSON.
fn text(value: &Value) -> String {
    match value {
        Value::String(text) => text.clone(),
        other => other.to_string(),
    }
}

fn as_text(headers: &Map<String, Value>) -> Vec<(String, String)> {
    headers
        .iter()
        .map(|(name, value)| (name.clone(), text(value)))
        .collect()
}

/// `x-litellm-cache-key`, the key of the cached response a served call came from.
fn cache_key(entries: &CallState, served: bool) -> Vec<(String, String)> {
    entries
        .attempt
        .as_ref()
        .and_then(AttemptRecord::cache_key)
        .filter(|_| served)
        .map(|key| {
            (
                <&str>::from(ExposedHeader::CacheKey).to_owned(),
                key.to_owned(),
            )
        })
        .into_iter()
        .collect()
}

/// The routing headers in the order the proxy sends them, each only when the call has its value.
fn routing_headers(routing: Option<&Routing>) -> Vec<(String, String)> {
    let Some(routing) = routing else {
        return Vec::new();
    };
    let served = routing.served.as_ref();
    [
        ("x-litellm-model-id", Some(routing.deployment_id.clone())),
        (
            "x-litellm-model-group",
            served.map(|served| served.model_group.clone()),
        ),
        (
            "x-litellm-attempted-retries",
            served.map(|served| served.attempted_retries.to_string()),
        ),
        (
            "x-litellm-attempted-fallbacks",
            served.map(|served| served.attempted_fallbacks.to_string()),
        ),
        (
            "x-litellm-max-retries",
            served
                .and_then(|served| served.max_retries)
                .map(|max| max.to_string()),
        ),
    ]
    .into_iter()
    .filter_map(|(name, value)| Some((name.to_owned(), value?)))
    .collect()
}

/// `OPENAI_RESPONSE_HEADERS` in `litellm/types/utils.py`.
const OPENAI_RESPONSE_HEADERS: [&str; 6] = [
    "x-ratelimit-remaining-requests",
    "x-ratelimit-remaining-tokens",
    "x-ratelimit-limit-requests",
    "x-ratelimit-limit-tokens",
    "x-ratelimit-reset-requests",
    "x-ratelimit-reset-tokens",
];

/// The OpenAI headers `get_response_headers` keeps unprefixed, in its order.
const OPENAI_ERROR_HEADERS: [&str; 4] = [
    "x-ratelimit-limit-requests",
    "x-ratelimit-remaining-requests",
    "x-ratelimit-limit-tokens",
    "x-ratelimit-remaining-tokens",
];

/// `_REQUEST_ID_HEADERS` in `get_headers.py`, in the order `get_provider_request_id` reads them.
const REQUEST_ID_HEADERS: [&str; 4] = [
    "request-id",
    "x-request-id",
    "llm_provider-request-id",
    "llm_provider-x-request-id",
];

const PREFIX: &str = "llm_provider-";

/// Headers by name in the order each name was first written, a later write replacing the value
/// in place, as a Python `dict` keeps them.
#[derive(Default)]
struct Headers(Vec<(String, String)>);

impl Headers {
    fn put(&mut self, name: impl Into<String>, value: impl Into<String>) {
        let (name, value) = (name.into(), value.into());
        match self.0.iter_mut().find(|(held, _)| *held == name) {
            Some((_, held)) => *held = value,
            None => self.0.push((name, value)),
        }
    }

    fn get(&self, name: &str) -> Option<&str> {
        self.0
            .iter()
            .find(|(held, _)| held == name)
            .map(|(_, value)| value.as_str())
    }
}

/// The provider's headers as httpx's `Headers.items()` yields them: lowercase names, the values
/// of a repeated name joined with `, ` at its first place.
fn provider(headers: &[(String, String)]) -> Headers {
    let mut joined = Headers::default();
    for (name, value) in headers {
        let name = name.to_ascii_lowercase();
        let value = match joined.get(&name) {
            Some(held) => format!("{held}, {value}"),
            None => value.clone(),
        };
        joined.put(name, value);
    }
    joined
}

/// `get_provider_request_id`: the first request id header that has a value.
fn request_id(headers: &Headers) -> Option<String> {
    REQUEST_ID_HEADERS
        .iter()
        .find_map(|name| headers.get(name).filter(|value| !value.is_empty()))
        .map(str::to_owned)
}

/// `process_response_headers` over a provider's raw headers: the OpenAI rate limit headers as
/// they are, then every header prefixed `llm_provider-` unless it already is, then the provider's
/// request id as `request-id`.
fn served_provider_headers(headers: &[(String, String)]) -> Vec<(String, String)> {
    let raw = provider(headers);
    let mut out = Headers::default();
    for (name, value) in &raw.0 {
        if OPENAI_RESPONSE_HEADERS.contains(&name.as_str()) {
            out.put(name, value);
        }
    }
    for (name, value) in &raw.0 {
        if name.starts_with(PREFIX) {
            out.put(name, value);
        }
    }
    for (name, value) in &raw.0 {
        if !name.starts_with(PREFIX) {
            out.put(format!("{PREFIX}{name}"), value);
        }
    }
    if let Some(id) = request_id(&raw) {
        out.put("request-id", id);
    }
    out.0
}

/// `get_response_headers` over the headers of a provider's error: every header prefixed
/// `llm_provider-` unless its name holds `llm_provider`, then the OpenAI rate limit headers as
/// they are, then the provider's request id as `request-id`.
fn error_provider_headers(headers: &[(String, String)]) -> Vec<(String, String)> {
    let raw = provider(headers);
    let mut out = Headers::default();
    for (name, value) in &raw.0 {
        if name.contains("llm_provider") {
            out.put(name, value);
        } else {
            out.put(format!("{PREFIX}{name}"), value);
        }
    }
    for name in OPENAI_ERROR_HEADERS {
        if let Some(value) = raw.get(name) {
            out.put(name, value.to_owned());
        }
    }
    if let Some(id) = request_id(&raw) {
        out.put("request-id", id);
    }
    out.0
}

/// The api base without its query, which could carry a secret, as Python's `split("?")[0]`.
fn api_base(base: &str) -> String {
    base.split('?').next().unwrap_or_default().to_owned()
}

/// The call's details on a served response, `now` being when its head is written.
///
/// The response duration runs from the record's start to the provider's response head for a
/// stream, as Python stamps it when the provider stream is handed back, and to when the result
/// was ready for a complete response. The overhead is the time from the record's start to the
/// first attempt's start, and from the served attempt's start to that same end, less the served
/// provider's wait from its request sent to its response head, which is what
/// `litellm_overhead_time_ms` takes off, so failed attempts and the backoff before a retry are not
/// counted. Without the served attempt's start it is the response duration less that wait. It is
/// not sent for a response no provider was asked for, such as a cache hit.
fn served_details(entries: &CallState, now: Instant) -> Vec<(String, String)> {
    let stream = entries.stream_response;
    let hidden = stream || entries.call_type == Some(CallType::Aocr);
    let attempt = entries.attempt.as_ref();
    let deployment = entries.served_deployment.as_ref();
    let end = attempt
        .and_then(|record| {
            if stream {
                record.head.or(record.ready)
            } else {
                record.ready.or(record.head)
            }
        })
        .unwrap_or(now);
    let total = milliseconds(end.saturating_duration_since(entries.started()));
    let mut out = Headers::default();
    if let Some(deployment) = deployment {
        out.put("x-litellm-model-name", deployment.model.clone());
    }
    if hidden && let Some(base) = deployment.and_then(|deployment| deployment.api_base.as_deref()) {
        out.put("x-litellm-model-api-base", api_base(base));
    }
    out.put("x-litellm-response-duration-ms", float_repr(total));
    if let Some(wait) = attempt.and_then(|record| record.wait) {
        let outside = match attempt.and_then(|record| record.started) {
            Some(served) => {
                let first = entries
                    .failed_attempts
                    .iter()
                    .filter_map(|record| record.started)
                    .fold(served, Instant::min);
                milliseconds(first.saturating_duration_since(entries.started()))
                    + milliseconds(end.saturating_duration_since(served))
            }
            None => total,
        };
        out.put(
            "x-litellm-overhead-duration-ms",
            float_repr(round(outside - milliseconds(wait), 4)),
        );
    }
    if hidden {
        out.put(
            "x-litellm-callback-duration-ms",
            float_repr(round(milliseconds(entries.callback_time), 4)),
        );
    }
    let mut out = out.0;
    if stream && let Some(record) = attempt {
        out.extend(served_provider_headers(&record.provider_headers));
    }
    present(out)
}

/// The headers the proxy's handler adds to a response that ended on `error`: a zero response
/// cost, the timeout of the provider attempt that raised it, and on OCR, whose errors go through
/// `handle_llm_api_exception`, the provider's own headers. `/v1/messages` builds its error
/// headers without them.
fn error_details(call_type: Option<CallType>, error: &ErrorResponse) -> Vec<(String, String)> {
    if !error.in_handler {
        return Vec::new();
    }
    let mut out = vec![("x-litellm-response-cost".to_owned(), "0".to_owned())];
    if let Some(timeout) = error.timeout {
        out.push((
            "x-litellm-timeout".to_owned(),
            float_repr(timeout.as_secs_f64()),
        ));
    }
    if call_type == Some(CallType::Aocr) {
        out.extend(error_provider_headers(&error.provider_headers));
    }
    present(out)
}

/// The headers Python's `exclude_values` lets through.
fn present(headers: Vec<(String, String)>) -> Vec<(String, String)> {
    headers
        .into_iter()
        .filter(|(_, value)| !matches!(value.as_str(), "" | "None"))
        .collect()
}

#[cfg(test)]
impl Routing {
    /// A call `deployment_id` served under `model_group` after `retries` retries.
    pub(crate) fn served_by(deployment_id: &str, model_group: &str, retries: u32) -> Self {
        Self {
            deployment_id: deployment_id.to_owned(),
            served: Some(Served {
                model_group: model_group.to_owned(),
                counted_group: model_group.to_owned(),
                attempted_retries: retries,
                attempted_fallbacks: 0,
                max_retries: (retries > 0).then_some(retries),
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use rstest::rstest;
    use serde_json::json;

    use super::*;
    use crate::record::{HeadSent, ServedDeployment};

    fn pairs(expected: &[(&str, &str)]) -> Vec<(String, String)> {
        expected
            .iter()
            .map(|(name, value)| ((*name).to_owned(), (*value).to_owned()))
            .collect()
    }

    /// A call `d1` served from the cached response `k1` after one retry, with `added` added to its
    /// response, and whether it then ended on an error on `d1`.
    fn entries(added: &[(&str, Value)], ended: bool) -> CallState {
        let mut entries = CallState::new("trace-1".to_owned());
        entries.routed(
            Routing::served_by("d1", "public/model", 1),
            None,
            AttemptRecord::cached("k1"),
        );
        for (name, value) in added {
            entries
                .add_header((*name).to_owned(), value.clone())
                .unwrap();
        }
        if ended {
            entries.respond_with_error(Some(Routing::failed_on("d1")), ErrorResponse::default());
        }
        entries
    }

    /// The head `entries` give a response whose route wrote only its content type, written as the
    /// call began.
    fn written(entries: &mut CallState) -> Vec<(String, String)> {
        let mut headers = HeaderMap::from_iter([(
            HeaderName::from_static("content-type"),
            HeaderValue::from_static("application/json"),
        )]);
        let started = entries.started();
        head(entries, &mut headers, started);
        headers
            .iter()
            .map(|(name, value)| (name.to_string(), value.to_str().unwrap().to_owned()))
            .collect()
    }

    #[rstest]
    #[case::a_served_response(
        &[("x-callback", json!("yes")), ("x-count", json!(3))],
        false,
        &[
            ("content-type", "application/json"),
            ("x-litellm-cache-key", "k1"),
            ("x-litellm-model-id", "d1"),
            ("x-litellm-model-group", "public/model"),
            ("x-litellm-attempted-retries", "1"),
            ("x-litellm-attempted-fallbacks", "0"),
            ("x-litellm-max-retries", "1"),
            ("x-litellm-response-duration-ms", "0.0"),
            ("x-callback", "yes"),
            ("x-count", "3"),
        ]
    )]
    #[case::a_response_to_a_call_that_ended_on_an_error(
        &[("x-callback", json!("yes"))],
        true,
        &[("content-type", "application/json"), ("x-litellm-model-id", "d1")]
    )]
    fn the_head_is_the_routes_own_then_the_record_then_what_was_added(
        #[case] added: &[(&str, Value)],
        #[case] ended: bool,
        #[case] expected: &[(&str, &str)],
    ) {
        assert_eq!(written(&mut entries(added, ended)), pairs(expected));
    }

    #[rstest]
    #[case::the_routes_own_header(
        "content-type",
        &[
            ("content-type", "text/plain"),
            ("x-litellm-cache-key", "k1"),
            ("x-litellm-model-id", "d1"),
            ("x-litellm-model-group", "public/model"),
            ("x-litellm-attempted-retries", "1"),
            ("x-litellm-attempted-fallbacks", "0"),
            ("x-litellm-max-retries", "1"),
            ("x-litellm-response-duration-ms", "0.0"),
        ]
    )]
    #[case::the_cache_key(
        "x-litellm-cache-key",
        &[
            ("content-type", "application/json"),
            ("x-litellm-cache-key", "text/plain"),
            ("x-litellm-model-id", "d1"),
            ("x-litellm-model-group", "public/model"),
            ("x-litellm-attempted-retries", "1"),
            ("x-litellm-attempted-fallbacks", "0"),
            ("x-litellm-max-retries", "1"),
            ("x-litellm-response-duration-ms", "0.0"),
        ]
    )]
    #[case::a_routing_header(
        "x-litellm-model-id",
        &[
            ("content-type", "application/json"),
            ("x-litellm-cache-key", "k1"),
            ("x-litellm-model-id", "text/plain"),
            ("x-litellm-model-group", "public/model"),
            ("x-litellm-attempted-retries", "1"),
            ("x-litellm-attempted-fallbacks", "0"),
            ("x-litellm-max-retries", "1"),
            ("x-litellm-response-duration-ms", "0.0"),
        ]
    )]
    fn a_callbacks_header_replaces_any_of_the_same_name_in_place(
        #[case] name: &str,
        #[case] expected: &[(&str, &str)],
    ) {
        let mut entries = entries(&[(name, json!("text/plain"))], false);
        assert_eq!(written(&mut entries), pairs(expected));
    }

    #[rstest]
    fn a_default_goes_out_only_where_no_header_of_its_name_was_set() {
        let mut entries = entries(&[("x-ratelimit-remaining-requests", json!("5"))], false);
        for (name, value) in [
            ("x-ratelimit-remaining-requests", "9"),
            ("x-ratelimit-limit-requests", "10"),
            ("x-litellm-model-id", "d9"),
        ] {
            entries
                .add_default_header(name.to_owned(), json!(value))
                .unwrap();
        }
        assert_eq!(
            written(&mut entries),
            pairs(&[
                ("content-type", "application/json"),
                ("x-litellm-cache-key", "k1"),
                ("x-litellm-model-id", "d1"),
                ("x-litellm-model-group", "public/model"),
                ("x-litellm-attempted-retries", "1"),
                ("x-litellm-attempted-fallbacks", "0"),
                ("x-litellm-max-retries", "1"),
                ("x-litellm-response-duration-ms", "0.0"),
                ("x-ratelimit-remaining-requests", "5"),
                ("x-ratelimit-limit-requests", "10"),
            ])
        );
    }

    #[rstest]
    fn a_header_added_after_the_head_is_refused_and_never_sent() {
        let mut entries = entries(&[], false);
        written(&mut entries);
        assert_eq!(
            (
                entries.add_header("x-late".to_owned(), json!("yes")),
                entries.response_headers().contains_key("x-late"),
            ),
            (Err(HeadSent), false)
        );
    }

    #[rstest]
    #[case::text("x-a", json!("one"), Some("one"))]
    #[case::a_number_as_python_formats_it("x-a", json!(7), Some("7"))]
    #[case::a_name_no_header_can_have("x a", json!("one"), None)]
    fn a_header_added_goes_out_as_text(
        #[case] name: &str,
        #[case] value: Value,
        #[case] sent: Option<&str>,
    ) {
        let header = header(name, &text(&value));
        assert_eq!(
            header.as_ref().and_then(|(_, value)| value.to_str().ok()),
            sent
        );
    }

    #[rstest]
    #[case::an_openai_rate_limit_header_is_kept_and_prefixed(
        &[("x-ratelimit-remaining-tokens", "9"), ("Content-Type", "application/json")],
        &[
            ("x-ratelimit-remaining-tokens", "9"),
            ("llm_provider-x-ratelimit-remaining-tokens", "9"),
            ("llm_provider-content-type", "application/json"),
        ]
    )]
    #[case::an_already_prefixed_header_is_kept(
        &[("llm_provider-a", "1"), ("b", "2")],
        &[("llm_provider-a", "1"), ("llm_provider-b", "2")]
    )]
    #[case::the_request_id_is_sent_bare(
        &[("x-request-id", "x1"), ("request-id", "r1")],
        &[
            ("llm_provider-x-request-id", "x1"),
            ("llm_provider-request-id", "r1"),
            ("request-id", "r1"),
        ]
    )]
    #[case::a_repeated_header_is_joined(
        &[("via", "a"), ("via", "b")],
        &[("llm_provider-via", "a, b")]
    )]
    fn a_served_streams_provider_headers_go_out_as_process_response_headers_makes_them(
        #[case] headers: &[(&str, &str)],
        #[case] expected: &[(&str, &str)],
    ) {
        assert_eq!(served_provider_headers(&pairs(headers)), pairs(expected));
    }

    #[rstest]
    #[case::the_openai_headers_follow_the_prefixed_ones(
        &[("x-ratelimit-limit-tokens", "9"), ("retry-after", "3")],
        &[
            ("llm_provider-x-ratelimit-limit-tokens", "9"),
            ("llm_provider-retry-after", "3"),
            ("x-ratelimit-limit-tokens", "9"),
        ]
    )]
    #[case::a_name_holding_the_prefix_anywhere_is_kept(
        &[("x-llm_provider-a", "1"), ("x-request-id", "x1")],
        &[
            ("x-llm_provider-a", "1"),
            ("llm_provider-x-request-id", "x1"),
            ("request-id", "x1"),
        ]
    )]
    fn an_errors_provider_headers_go_out_as_get_response_headers_makes_them(
        #[case] headers: &[(&str, &str)],
        #[case] expected: &[(&str, &str)],
    ) {
        assert_eq!(error_provider_headers(&pairs(headers)), pairs(expected));
    }

    #[rstest]
    #[case::no_query("https://a.test/v1", "https://a.test/v1")]
    #[case::a_query_with_a_secret("https://a.test/v1?key=s", "https://a.test/v1")]
    fn an_api_base_goes_out_without_its_query(#[case] base: &str, #[case] expected: &str) {
        assert_eq!(api_base(base), expected);
    }

    /// A call of `call_type` that `d1` served, its response a stream when `stream` says so.
    fn served_call(call_type: CallType, stream: bool) -> CallState {
        let mut entries = CallState::new("trace-1".to_owned());
        entries.call_type = Some(call_type);
        entries.routed(
            Routing::served_by("d1", "public/model", 0),
            Some(ServedDeployment {
                model: "anthropic/claude-x".to_owned(),
                api_base: Some("https://a.test/v1?key=s".to_owned()),
                timeout: Some(Duration::from_secs(5)),
            }),
            AttemptRecord::default(),
        );
        if stream {
            entries.stream_response = true;
        }
        entries
    }

    #[rstest]
    #[case::a_complete_messages_response(
        CallType::AnthropicMessages,
        false,
        &["x-litellm-model-name", "x-litellm-response-duration-ms"]
    )]
    #[case::a_messages_stream(
        CallType::AnthropicMessages,
        true,
        &[
            "x-litellm-model-name",
            "x-litellm-model-api-base",
            "x-litellm-response-duration-ms",
            "x-litellm-callback-duration-ms",
        ]
    )]
    #[case::an_ocr_response(
        CallType::Aocr,
        false,
        &[
            "x-litellm-model-name",
            "x-litellm-model-api-base",
            "x-litellm-response-duration-ms",
            "x-litellm-callback-duration-ms",
        ]
    )]
    fn a_served_response_sends_what_pythons_hidden_params_carry_for_it(
        #[case] call_type: CallType,
        #[case] stream: bool,
        #[case] expected: &[&str],
    ) {
        let entries = served_call(call_type, stream);
        let sent = served_details(&entries, Instant::now());
        assert_eq!(
            sent.iter()
                .map(|(name, _)| name.as_str())
                .collect::<Vec<_>>(),
            expected
        );
    }

    /// The response and overhead durations of a Messages call, stream or complete, whose served
    /// attempt began `served` ms after the call began, after a failed one that began at `failed`
    /// ms and took 3 s, whose response head came at `head` ms, whose result was ready at `ready`
    /// ms and whose provider took `wait` µs, its head written at 500 ms. The overhead is rounded
    /// to 4 places, so `120 - 100.456`, `19.543999999999997` as a float, goes out as `19.544`.
    #[rstest]
    #[case::a_stream_ends_at_the_providers_head(
        true, None, Some(0), Some(120), Some(300), Some(100_456), &[("x-litellm-response-duration-ms", "120.0"), ("x-litellm-overhead-duration-ms", "19.544")]
    )]
    #[case::a_complete_answer_ends_when_it_was_ready(
        false, None, Some(0), Some(120), Some(300), Some(100_456), &[("x-litellm-response-duration-ms", "300.0"), ("x-litellm-overhead-duration-ms", "199.544")]
    )]
    #[case::a_stream_with_no_head_ends_when_it_was_ready(
        true, None, Some(0), None, Some(300), Some(100_456), &[("x-litellm-response-duration-ms", "300.0"), ("x-litellm-overhead-duration-ms", "199.544")]
    )]
    #[case::a_complete_answer_never_ready_ends_at_the_head(
        false, None, Some(0), Some(120), None, Some(100_456), &[("x-litellm-response-duration-ms", "120.0"), ("x-litellm-overhead-duration-ms", "19.544")]
    )]
    #[case::no_provider_wait_sends_no_overhead(
        false, None, Some(0), Some(120), Some(300), None, &[("x-litellm-response-duration-ms", "300.0")]
    )]
    #[case::an_attempt_that_never_responded_ends_at_the_head_written(
        true, None, Some(0), None, None, None, &[("x-litellm-response-duration-ms", "500.0")]
    )]
    #[case::a_failed_attempt_and_the_backoff_after_it_are_not_overhead(
        false, Some(10), Some(3020), Some(3070), Some(3100), Some(50_000), &[("x-litellm-response-duration-ms", "3100.0"), ("x-litellm-overhead-duration-ms", "40.0")]
    )]
    #[case::without_the_served_attempts_start_the_overhead_is_the_duration_less_the_wait(
        false, Some(10), None, Some(3070), Some(3100), Some(50_000), &[("x-litellm-response-duration-ms", "3100.0"), ("x-litellm-overhead-duration-ms", "3050.0")]
    )]
    fn the_durations_run_from_the_calls_start_to_its_answer(
        #[case] stream: bool,
        #[case] failed: Option<u64>,
        #[case] served: Option<u64>,
        #[case] head: Option<u64>,
        #[case] ready: Option<u64>,
        #[case] wait: Option<u64>,
        #[case] expected: &[(&str, &str)],
    ) {
        let mut entries = served_call(CallType::AnthropicMessages, stream);
        let started = entries.started();
        let at = |ms: u64| started + Duration::from_millis(ms);
        entries
            .failed_attempts
            .extend(failed.map(|ms| AttemptRecord {
                started: Some(at(ms)),
                head: Some(at(ms + 3000)),
                wait: Some(Duration::from_millis(3000)),
                ..AttemptRecord::default()
            }));
        let attempt = entries.attempt.as_mut().unwrap();
        attempt.started = served.map(at);
        attempt.head = head.map(at);
        attempt.ready = ready.map(at);
        attempt.wait = wait.map(Duration::from_micros);
        let sent = served_details(&entries, at(500));
        assert_eq!(
            sent.into_iter()
                .filter(|(name, _)| name.ends_with("-duration-ms") && !name.contains("callback"))
                .collect::<Vec<_>>(),
            pairs(expected)
        );
    }

    #[rstest]
    fn the_model_name_and_api_base_are_the_served_deployments() {
        let entries = served_call(CallType::Aocr, false);
        let sent = served_details(&entries, Instant::now());
        assert_eq!(
            &sent[..2],
            &pairs(&[
                ("x-litellm-model-name", "anthropic/claude-x"),
                ("x-litellm-model-api-base", "https://a.test/v1"),
            ])[..]
        );
    }

    #[rstest]
    #[case::refused_before_the_handler(
        CallType::Aocr,
        ErrorResponse { in_handler: false, ..ErrorResponse::default() },
        &[]
    )]
    #[case::a_refusal_in_the_handler(
        CallType::AnthropicMessages,
        ErrorResponse { in_handler: true, ..ErrorResponse::default() },
        &[("x-litellm-response-cost", "0")]
    )]
    #[case::a_messages_provider_error(
        CallType::AnthropicMessages,
        ErrorResponse {
            in_handler: true,
            timeout: Some(Duration::from_secs(5)),
            provider_headers: vec![("x-request-id".to_owned(), "x1".to_owned())],
        },
        &[("x-litellm-response-cost", "0"), ("x-litellm-timeout", "5.0")]
    )]
    #[case::an_ocr_provider_error(
        CallType::Aocr,
        ErrorResponse {
            in_handler: true,
            timeout: Some(Duration::from_millis(2500)),
            provider_headers: vec![("x-request-id".to_owned(), "x1".to_owned())],
        },
        &[
            ("x-litellm-response-cost", "0"),
            ("x-litellm-timeout", "2.5"),
            ("llm_provider-x-request-id", "x1"),
            ("request-id", "x1"),
        ]
    )]
    fn an_error_response_sends_what_the_proxys_handler_adds(
        #[case] call_type: CallType,
        #[case] error: ErrorResponse,
        #[case] expected: &[(&str, &str)],
    ) {
        assert_eq!(error_details(Some(call_type), &error), pairs(expected));
    }
}
