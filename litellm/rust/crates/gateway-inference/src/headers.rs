//! The headers a response on the inference routes carries beyond the route's own. The record of
//! the call holds what the call did, and [`render_response_headers`] is the one place that turns it
//! into headers: it knows every header's name and the order they go out in, and reads the record
//! only through its getters. It runs inside `record::scoped`, so it reads the record of the call it
//! responds to, and it closes the record's header bag as it writes, so a header added after the
//! head, a stream's end included, is refused.

use axum::{
    extract::Request,
    http::{HeaderMap, HeaderName, HeaderValue},
    middleware::Next,
    response::Response,
};
use litellm_router_core::RouterResponse;
use serde_json::Value;

use crate::record::{self, AttemptRecord, CallState};

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
    record::write(|entries| head(entries, response.headers_mut()));
    response
}

/// One header of the record as the response carries it, `None` when its name or value is no
/// header. A text is sent as written and any other value as its JSON.
fn header(name: &str, value: &Value) -> Option<(HeaderName, HeaderValue)> {
    let value = match value {
        Value::String(text) => HeaderValue::try_from(text.as_str()),
        other => HeaderValue::try_from(other.to_string()),
    };
    Some((HeaderName::try_from(name).ok()?, value.ok()?))
}

/// The routing headers in the order the proxy sends them, each with its value when the call has
/// one.
fn routing_headers(routing: &Routing) -> [(&'static str, Option<String>); 5] {
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
}

/// Writes the head `entries` give a response over `headers`, the route's own. A served response
/// carries the served attempt's cache key, then the headers callbacks and Rust code added, each
/// replacing a header of the route's by its name, then what the router did for the call, which
/// replaces a callback's header of the same name, and last the defaults, each only where no
/// header of its name was set, as Python's `apply_remaining_usage_headers` does. A response to a
/// call that ended on an error carries only what the router did for the attempt it ended on.
fn head(entries: &mut CallState, headers: &mut HeaderMap) {
    let error = entries.is_error_response();
    let cache_key = entries
        .attempt()
        .and_then(AttemptRecord::cache_key)
        .filter(|_| !error)
        .and_then(|key| HeaderValue::from_str(key).ok());
    let routing = entries.routing().map(routing_headers);
    if let Some(key) = cache_key {
        headers.insert(HeaderName::from(ExposedHeader::CacheKey), key);
    }
    let (added, defaults) = entries.send_head();
    if !error {
        for (name, value) in added.iter().filter_map(|(name, value)| header(name, value)) {
            headers.insert(name, value);
        }
    }
    for (name, value) in routing.into_iter().flatten() {
        if let Some(value) = value.and_then(|value| HeaderValue::from_str(&value).ok()) {
            headers.insert(HeaderName::from_static(name), value);
        }
    }
    for (name, value) in defaults
        .iter()
        .filter_map(|(name, value)| header(name, value))
    {
        headers.entry(name).or_insert(value);
    }
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
    use rstest::rstest;
    use serde_json::json;

    use super::*;
    use crate::record::HeadSent;

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
            entries.respond_with_error(Some(Routing::failed_on("d1")));
        }
        entries
    }

    /// The head `entries` give a response whose route wrote only its content type.
    fn written(entries: &mut CallState) -> Vec<(String, String)> {
        let mut headers = HeaderMap::from_iter([(
            HeaderName::from_static("content-type"),
            HeaderValue::from_static("application/json"),
        )]);
        head(entries, &mut headers);
        headers
            .iter()
            .map(|(name, value)| (name.to_string(), value.to_str().unwrap().to_owned()))
            .collect()
    }

    fn pairs(expected: &[(&str, &str)]) -> Vec<(String, String)> {
        expected
            .iter()
            .map(|(name, value)| ((*name).to_owned(), (*value).to_owned()))
            .collect()
    }

    #[rstest]
    #[case::a_served_response(
        &[("x-callback", json!("yes")), ("x-count", json!(3))],
        false,
        &[
            ("content-type", "application/json"),
            ("x-litellm-cache-key", "k1"),
            ("x-callback", "yes"),
            ("x-count", "3"),
            ("x-litellm-model-id", "d1"),
            ("x-litellm-model-group", "public/model"),
            ("x-litellm-attempted-retries", "1"),
            ("x-litellm-attempted-fallbacks", "0"),
            ("x-litellm-max-retries", "1"),
        ]
    )]
    #[case::a_response_to_a_call_that_ended_on_an_error(
        &[("x-callback", json!("yes"))],
        true,
        &[("content-type", "application/json"), ("x-litellm-model-id", "d1")]
    )]
    fn the_head_is_the_routes_own_then_the_cache_key_then_what_was_added_then_the_routing(
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
        ]
    )]
    #[case::never_a_routing_header(
        "x-litellm-model-id",
        &[
            ("content-type", "application/json"),
            ("x-litellm-cache-key", "k1"),
            ("x-litellm-model-id", "d1"),
            ("x-litellm-model-group", "public/model"),
            ("x-litellm-attempted-retries", "1"),
            ("x-litellm-attempted-fallbacks", "0"),
            ("x-litellm-max-retries", "1"),
        ]
    )]
    fn a_callbacks_header_replaces_in_place(#[case] name: &str, #[case] expected: &[(&str, &str)]) {
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
                ("x-ratelimit-remaining-requests", "5"),
                ("x-litellm-model-id", "d1"),
                ("x-litellm-model-group", "public/model"),
                ("x-litellm-attempted-retries", "1"),
                ("x-litellm-attempted-fallbacks", "0"),
                ("x-litellm-max-retries", "1"),
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
        let header = header(name, &value);
        assert_eq!(
            header.as_ref().and_then(|(_, value)| value.to_str().ok()),
            sent
        );
    }
}
