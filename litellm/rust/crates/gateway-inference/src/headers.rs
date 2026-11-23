use std::{
    sync::Arc,
    time::{Duration, Instant},
};

use axum::{
    extract::{Request, State},
    http::{HeaderName, HeaderValue},
    middleware::Next,
    response::Response,
};
use litellm_host::interceptors::Cost;
use litellm_python_compat::repr::float_repr;
use litellm_router_core::RouterResponse;

use crate::{Gateway, attempted::Attempted, routing::Host};

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

/// What the router did for a request. A route attaches it to its response as an extension, and
/// [`told`] is the one place that turns it into the proxy's `x-litellm-*` headers, the same for
/// a complete answer and for a stream, whose head is written after its attempt was served.
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

/// A call that ended on an error, which a route attaches to the answer. Python answers one with
/// the cost of a call that ran nothing and, once an attempt ran, the timeout that attempt had.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct Failed {
    pub(crate) timeout: Option<Duration>,
}

/// The provider's headers Python reads a request id from, first found first.
const REQUEST_IDS: [&str; 2] = ["request-id", "x-request-id"];
const PROVIDER_PREFIX: &str = "llm_provider-";
const RESPONSE_COST: &str = "x-litellm-response-cost";

fn milliseconds(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1000.0
}

/// `round(value, 4)`, as Python rounds the overhead it names.
fn rounded(value: f64) -> f64 {
    (value * 10_000.0).round() / 10_000.0
}

/// `CostBreakdownHeaderValues`: the parts of a call's cost Python names beside its total, each
/// as its own `x-litellm-response-cost-*` header. A part that is `None` names no header. A priced
/// call names its cache read, cache creation and reasoning parts only when they are above zero.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct CostParts {
    /// The cost before discount and margin.
    pub(crate) original: Option<f64>,
    pub(crate) discount_amount: Option<f64>,
    pub(crate) margin_amount: Option<f64>,
    pub(crate) margin_percent: Option<f64>,
    /// The prompt's cost without its cache reads and writes, which have their own parts.
    pub(crate) input: Option<f64>,
    pub(crate) output: Option<f64>,
    pub(crate) cache_read: Option<f64>,
    pub(crate) cache_creation: Option<f64>,
    pub(crate) reasoning: Option<f64>,
    pub(crate) tool_usage: Option<f64>,
}

impl CostParts {
    /// `_ZERO_COST_BREAKDOWN`: the parts of a call that cost nothing, which a cache hit names.
    const ZERO: Self = Self {
        original: Some(0.0),
        discount_amount: Some(0.0),
        margin_amount: Some(0.0),
        margin_percent: Some(0.0),
        input: Some(0.0),
        output: Some(0.0),
        cache_read: None,
        cache_creation: None,
        reasoning: None,
        tool_usage: Some(0.0),
    };

    /// Each part under its header, in Python's order. `_header_value` names a finite part only.
    fn headers(self) -> impl Iterator<Item = (String, String)> {
        [
            ("original", self.original),
            ("discount-amount", self.discount_amount),
            ("margin-amount", self.margin_amount),
            ("margin-percent", self.margin_percent),
            ("input", self.input),
            ("output", self.output),
            ("cache-read", self.cache_read),
            ("cache-creation", self.cache_creation),
            ("reasoning", self.reasoning),
            ("tool-usage", self.tool_usage),
        ]
        .into_iter()
        .filter_map(|(part, cost)| {
            let cost = cost.filter(|cost| cost.is_finite())?;
            Some((format!("{RESPONSE_COST}-{part}"), float_repr(cost)))
        })
    }
}

/// What a served call cost, as Python's cost headers name it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Charged {
    pub(crate) total: f64,
    pub(crate) parts: CostParts,
}

/// The cost the route settled for an attempt. A provider's own figure comes without parts, and a
/// cost the route has yet to settle names nothing.
pub(crate) fn charged(attempted: &Attempted) -> Option<Charged> {
    match attempted.cost()? {
        Cost::Reported { amount } => Some(Charged {
            total: amount.as_f64()?,
            parts: CostParts::default(),
        }),
        Cost::CacheHit => Some(Charged {
            total: 0.0,
            parts: CostParts::ZERO,
        }),
        Cost::Deferred => None,
    }
}

/// What a served call cost, for the mounting server's `x-litellm-key-spend`. `None` for an answer
/// that names no cost.
pub fn response_cost(response: &Response) -> Option<f64> {
    Some(charged(response.extensions().get::<Attempted>()?)?.total)
}

/// `process_response_headers`: every header of the provider's answer under `llm_provider-`, then
/// the request id it names. Python tells these on a stream only.
fn provider_headers(head: Vec<(String, String)>) -> Vec<(String, String)> {
    let request_id = REQUEST_IDS.into_iter().find_map(|name| {
        let (_, id) = head
            .iter()
            .find(|(header, _)| header.eq_ignore_ascii_case(name))?;
        Some(("request-id".to_owned(), id.clone()))
    });
    head.into_iter()
        .map(|(name, value)| match name.starts_with(PROVIDER_PREFIX) {
            true => (name, value),
            false => (format!("{PROVIDER_PREFIX}{name}"), value),
        })
        .chain(request_id)
        .collect()
}

/// What the router counted for a served call, after the headers Python's `get_custom_headers`
/// builds itself.
fn served_headers(served: &Served) -> impl Iterator<Item = (String, String)> {
    let named = |name: &str, value: String| (name.to_owned(), value);
    [
        Some(named("x-litellm-model-group", served.model_group.clone())),
        Some(named(
            "x-litellm-attempted-retries",
            served.attempted_retries.to_string(),
        )),
        Some(named(
            "x-litellm-attempted-fallbacks",
            served.attempted_fallbacks.to_string(),
        )),
        served
            .max_retries
            .map(|max| named("x-litellm-max-retries", max.to_string())),
    ]
    .into_iter()
    .flatten()
}

/// Every header a response's extensions name, in the order Python writes them: what the router
/// did ([`Routing`]), what the attempt that served it told ([`Attempted`]) or that the call
/// failed ([`Failed`]), and the version on every answer. `took` is the time from the request to
/// the answer, a stream's to its head.
fn named(response: &Response, version: Option<&str>, took: Duration) -> Vec<(String, String)> {
    let extensions = response.extensions();
    let routing = extensions.get::<Routing>();
    let attempted = extensions.get::<Attempted>();
    let failed = extensions.get::<Failed>();
    let stream_head = attempted.and_then(Attempted::stream_head);
    let streamed = stream_head.is_some();
    let header = |name: &str, value: Option<String>| Some((name.to_owned(), value?));
    let api_base = attempted
        .and_then(Attempted::api_base)
        .filter(|_| streamed)
        .and_then(|api_base| api_base.split('?').next());
    let charged = attempted.and_then(charged).filter(|_| failed.is_none());
    let cost = match failed {
        Some(_) => Some("0".to_owned()),
        None => charged.map(|charged| float_repr(charged.total)),
    };
    let cost_parts = charged
        .into_iter()
        .flat_map(|charged| charged.parts.headers());
    let overhead = attempted
        .and_then(Attempted::provider)
        .map(|provider| rounded(milliseconds(took.saturating_sub(provider))));
    let timeout = failed.and_then(|failed| failed.timeout);
    [
        header(
            "x-litellm-model-id",
            routing.map(|routing| routing.deployment_id.clone()),
        ),
        header(
            "x-litellm-model-name",
            attempted.map(|attempted| attempted.model().to_owned()),
        ),
        header(
            ExposedHeader::CacheKey.into(),
            attempted.and_then(Attempted::cache_key),
        ),
        header("x-litellm-model-api-base", api_base.map(str::to_owned)),
        header("x-litellm-version", version.map(str::to_owned)),
        header(RESPONSE_COST, cost),
    ]
    .into_iter()
    .flatten()
    .chain(cost_parts)
    .chain(
        [
            header(
                "x-litellm-response-duration-ms",
                attempted.map(|_| float_repr(milliseconds(took))),
            ),
            header("x-litellm-overhead-duration-ms", overhead.map(float_repr)),
            // No logging callback has run when a stream's head goes out.
            header(
                "x-litellm-callback-duration-ms",
                streamed.then(|| float_repr(0.0)),
            ),
            header(
                "x-litellm-timeout",
                timeout.map(|timeout| float_repr(timeout.as_secs_f64())),
            ),
        ]
        .into_iter()
        .flatten(),
    )
    .chain(provider_headers(stream_head.unwrap_or_default()))
    .chain(
        routing
            .and_then(|routing| routing.served.as_ref())
            .into_iter()
            .flat_map(served_headers),
    )
    .collect()
}

/// Writes the proxy's headers on every answer of a route, from what the route attached to it.
pub(crate) async fn told<H: Host>(
    State(gateway): State<Arc<Gateway<H>>>,
    request: Request,
    next: Next,
) -> Response {
    let received = Instant::now();
    let mut response = next.run(request).await;
    let named = named(&response, gateway.version.as_deref(), received.elapsed());
    for (name, value) in named {
        if let (Ok(name), Ok(value)) = (HeaderName::try_from(name), HeaderValue::try_from(value)) {
            response.headers_mut().insert(name, value);
        }
    }
    response
}

#[cfg(test)]
mod tests {
    use litellm_host::interceptors::{
        ExecutionFacts, Interceptors, ProviderIdentity, RawResponse, RequestContext, ResponseHead,
        ResultSource, WireRequest,
    };
    use rstest::rstest;

    use super::*;

    const TOOK: Duration = Duration::from_millis(1500);

    /// An attempt that heard a provider answer with `head`, as a stream or whole, and `cost`.
    async fn attempted(
        api_base: Option<&str>,
        streamed: bool,
        head: &[(&str, &str)],
        cost: Cost,
    ) -> Attempted {
        let attempted = Attempted::new(
            "anthropic/claude-test".to_owned(),
            api_base.map(str::to_owned),
        );
        let head = ResponseHead {
            status: 200,
            headers: head
                .iter()
                .map(|(name, value)| ((*name).to_owned(), (*value).to_owned()))
                .collect(),
            sent_body: Default::default(),
            streamed,
        };
        let wire = WireRequest {
            url: "https://provider.test/v1/messages".to_owned(),
            headers: Vec::new(),
            body: serde_json::Value::Null,
        };
        let context = RequestContext {
            model: "claude-test".to_owned(),
            custom_llm_provider: "anthropic".to_owned(),
            optional_params: serde_json::Value::Null,
            secret_fields: Vec::new(),
            api_key: None,
        };
        attempted
            .before_provider_request(wire, context)
            .await
            .unwrap();
        attempted
            .after_provider_response(RawResponse::new(String::new()).with_head(head))
            .await
            .unwrap();
        let facts = ExecutionFacts {
            provider: ProviderIdentity {
                model: "claude-test".to_owned(),
                provider: "anthropic".to_owned(),
            },
            source: ResultSource::Provider,
            cost,
        };
        attempted.result_ready(facts).await.unwrap();
        attempted
    }

    fn answer(attempted: Attempted) -> Response {
        let routing = Routing {
            deployment_id: "deployment-1".to_owned(),
            served: Some(Served {
                model_group: "claude".to_owned(),
                counted_group: "claude".to_owned(),
                attempted_retries: 0,
                attempted_fallbacks: 0,
                max_retries: None,
            }),
        };
        let mut response = attempted.told(Response::default());
        response.extensions_mut().insert(routing);
        response
    }

    fn value<'a>(named: &'a [(String, String)], name: &str) -> Option<&'a str> {
        named
            .iter()
            .find(|(header, _)| header == name)
            .map(|(_, value)| value.as_str())
    }

    #[tokio::test]
    async fn a_stream_names_the_providers_headers_after_its_own_in_pythons_order() {
        let head = [
            ("x-request-id", "req_1"),
            ("x-litellm-model-id", "spoofed"),
            ("llm_provider-kept", "as it is"),
        ];
        let attempted = attempted(
            Some("https://provider.test/v1?key=secret"),
            true,
            &head,
            Cost::Deferred,
        )
        .await;

        let named = named(&answer(attempted), Some("1.2.3"), TOOK);

        let overhead: f64 = value(&named, "x-litellm-overhead-duration-ms")
            .unwrap()
            .parse()
            .unwrap();
        assert!((0.0..=1500.0).contains(&overhead));
        let rest: Vec<_> = named
            .iter()
            .filter(|(name, _)| name != "x-litellm-overhead-duration-ms")
            .map(|(name, value)| (name.as_str(), value.as_str()))
            .collect();
        assert_eq!(
            rest,
            [
                ("x-litellm-model-id", "deployment-1"),
                ("x-litellm-model-name", "anthropic/claude-test"),
                ("x-litellm-model-api-base", "https://provider.test/v1"),
                ("x-litellm-version", "1.2.3"),
                ("x-litellm-response-duration-ms", "1500.0"),
                ("x-litellm-callback-duration-ms", "0.0"),
                ("llm_provider-x-request-id", "req_1"),
                ("llm_provider-x-litellm-model-id", "spoofed"),
                ("llm_provider-kept", "as it is"),
                ("request-id", "req_1"),
                ("x-litellm-model-group", "claude"),
                ("x-litellm-attempted-retries", "0"),
                ("x-litellm-attempted-fallbacks", "0"),
            ]
        );
    }

    #[tokio::test]
    async fn a_complete_answer_names_neither_the_providers_headers_nor_its_api_base() {
        let head = [("request-id", "req_1")];
        let attempted =
            attempted(Some("https://provider.test"), false, &head, Cost::Deferred).await;

        let named = named(&answer(attempted), None, TOOK);

        let names: Vec<_> = named
            .iter()
            .map(|(name, _)| name.as_str())
            .filter(|name| *name != "x-litellm-overhead-duration-ms")
            .collect();
        assert_eq!(
            names,
            [
                "x-litellm-model-id",
                "x-litellm-model-name",
                "x-litellm-response-duration-ms",
                "x-litellm-model-group",
                "x-litellm-attempted-retries",
                "x-litellm-attempted-fallbacks",
            ]
        );
    }

    #[rstest]
    #[case::the_provider_request_id_comes_before_the_generic_one(
        &[("x-request-id", "generic"), ("request-id", "provider")],
        Some("provider")
    )]
    #[case::a_provider_that_names_none(&[("content-type", "text/event-stream")], None)]
    #[tokio::test]
    async fn a_stream_names_the_request_id_the_provider_gave(
        #[case] head: &[(&str, &str)],
        #[case] expected: Option<&str>,
    ) {
        let attempted = attempted(None, true, head, Cost::Deferred).await;

        let named = named(&answer(attempted), None, TOOK);

        assert_eq!(value(&named, "request-id"), expected);
    }

    fn reported(amount: f64) -> Cost {
        Cost::Reported {
            amount: serde_json::Number::from_f64(amount).unwrap(),
        }
    }

    #[rstest]
    #[case::a_cost_the_provider_reported_reads_as_pythons_float(reported(8.2e-6), Some("8.2e-06"))]
    #[case::a_cache_hit_costs_nothing(Cost::CacheHit, Some("0.0"))]
    #[case::a_cost_not_settled_yet_is_not_named(Cost::Deferred, None)]
    #[tokio::test]
    async fn a_served_answer_names_what_the_call_cost(
        #[case] cost: Cost,
        #[case] expected: Option<&str>,
    ) {
        let response = answer(attempted(None, false, &[], cost).await);

        let named = named(&response, None, TOOK);

        assert_eq!(value(&named, "x-litellm-response-cost"), expected);
        assert_eq!(
            response_cost(&response).map(float_repr).as_deref(),
            expected
        );
    }

    #[rstest]
    #[case::a_cache_hit_names_the_parts_of_a_call_that_cost_nothing(
        Cost::CacheHit,
        &[
            ("x-litellm-response-cost", "0.0"),
            ("x-litellm-response-cost-original", "0.0"),
            ("x-litellm-response-cost-discount-amount", "0.0"),
            ("x-litellm-response-cost-margin-amount", "0.0"),
            ("x-litellm-response-cost-margin-percent", "0.0"),
            ("x-litellm-response-cost-input", "0.0"),
            ("x-litellm-response-cost-output", "0.0"),
            ("x-litellm-response-cost-tool-usage", "0.0"),
        ]
    )]
    #[case::a_providers_own_figure_comes_without_parts(
        reported(0.25),
        &[("x-litellm-response-cost", "0.25")]
    )]
    #[tokio::test]
    async fn a_served_answer_names_the_parts_of_its_cost_after_the_total(
        #[case] cost: Cost,
        #[case] expected: &[(&str, &str)],
    ) {
        let response = answer(attempted(None, false, &[], cost).await);

        let named = named(&response, None, TOOK);

        let costs: Vec<_> = named
            .iter()
            .filter(|(name, _)| name.starts_with(RESPONSE_COST))
            .map(|(name, value)| (name.as_str(), value.as_str()))
            .collect();
        assert_eq!(costs, expected);
    }

    #[test]
    fn a_part_is_named_only_when_it_is_a_finite_number() {
        let parts = CostParts {
            input: Some(1.2e-6),
            output: Some(f64::NAN),
            cache_read: Some(f64::INFINITY),
            reasoning: Some(7e-6),
            ..CostParts::default()
        };

        assert_eq!(
            parts.headers().collect::<Vec<_>>(),
            [
                (
                    "x-litellm-response-cost-input".to_owned(),
                    "1.2e-06".to_owned()
                ),
                (
                    "x-litellm-response-cost-reasoning".to_owned(),
                    "7e-06".to_owned()
                ),
            ]
        );
    }

    #[rstest]
    #[case::an_attempt_ran(Some(Duration::from_secs(600)), Some("600.0"))]
    #[case::a_fraction_of_a_second(Some(Duration::from_millis(2500)), Some("2.5"))]
    #[case::no_attempt_ran(None, None)]
    fn a_failed_call_costs_nothing_and_names_the_timeout_its_attempt_had(
        #[case] timeout: Option<Duration>,
        #[case] expected: Option<&str>,
    ) {
        let mut response = Response::default();
        response
            .extensions_mut()
            .insert(Routing::failed_on("deployment-1"));
        response.extensions_mut().insert(Failed { timeout });

        let named = named(&response, Some("1.2.3"), TOOK);

        let timeout = expected.map(|timeout| ("x-litellm-timeout", timeout));
        let expected: Vec<_> = [
            ("x-litellm-model-id", "deployment-1"),
            ("x-litellm-version", "1.2.3"),
            ("x-litellm-response-cost", "0"),
        ]
        .into_iter()
        .chain(timeout)
        .collect();
        let named: Vec<_> = named
            .iter()
            .map(|(name, value)| (name.as_str(), value.as_str()))
            .collect();
        assert_eq!(named, expected);
    }
}
