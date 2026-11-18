use axum::{
    http::{HeaderName, HeaderValue},
    response::Response,
};
use litellm_router_core::RouterResponse;
use tower_http::set_header::SetResponseHeaderLayer;

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

/// What the router did to serve a request. A route attaches it to its response as an extension,
/// and [`layer`] is the one place that turns it into the proxy's `x-litellm-*` headers, the same
/// for a complete answer and for a stream, whose head is written after its attempt was served.
#[derive(Clone, Debug)]
pub(crate) struct Routing {
    deployment_id: String,
    model_group: String,
    attempted_retries: u32,
    attempted_fallbacks: u32,
    max_retries: Option<u32>,
}

impl Routing {
    pub(crate) fn served<R>(routed: &RouterResponse<R>) -> Self {
        Self {
            deployment_id: routed.deployment_id.clone(),
            model_group: routed.model_group.clone(),
            attempted_retries: routed.attempted_retries,
            attempted_fallbacks: routed.attempted_fallbacks,
            max_retries: routed.max_retries,
        }
    }
}

/// Reads one header's value off a response.
type Read = fn(&Response) -> Option<HeaderValue>;
type SetHeader = SetResponseHeaderLayer<Read>;

/// `value` of what the router did for `response`, `None` for a response no route attached a
/// [`Routing`] to.
fn told(response: &Response, value: fn(&Routing) -> Option<String>) -> Option<HeaderValue> {
    let routing = response.extensions().get::<Routing>()?;
    HeaderValue::from_str(&value(routing)?).ok()
}

/// Writes the routing headers of every response that carries a [`Routing`]. The layers run on a
/// response from the last to the first, so the headers go out in the proxy's order.
pub(crate) fn layer() -> (SetHeader, SetHeader, SetHeader, SetHeader, SetHeader) {
    let set = |name: &'static str, read: Read| {
        SetResponseHeaderLayer::overriding(HeaderName::from_static(name), read)
    };
    (
        set("x-litellm-max-retries", |response| {
            told(response, |routing| {
                routing.max_retries.map(|max| max.to_string())
            })
        }),
        set("x-litellm-attempted-fallbacks", |response| {
            told(response, |routing| {
                Some(routing.attempted_fallbacks.to_string())
            })
        }),
        set("x-litellm-attempted-retries", |response| {
            told(response, |routing| {
                Some(routing.attempted_retries.to_string())
            })
        }),
        set("x-litellm-model-group", |response| {
            told(response, |routing| Some(routing.model_group.clone()))
        }),
        set("x-litellm-model-id", |response| {
            told(response, |routing| Some(routing.deployment_id.clone()))
        }),
    )
}
