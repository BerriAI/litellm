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

/// What the router did for a request. A route attaches it to its response as an extension, and
/// [`layer`] is the one place that turns it into the proxy's `x-litellm-*` headers, the same for
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
                routing
                    .served
                    .as_ref()?
                    .max_retries
                    .map(|max| max.to_string())
            })
        }),
        set("x-litellm-attempted-fallbacks", |response| {
            told(response, |routing| {
                Some(routing.served.as_ref()?.attempted_fallbacks.to_string())
            })
        }),
        set("x-litellm-attempted-retries", |response| {
            told(response, |routing| {
                Some(routing.served.as_ref()?.attempted_retries.to_string())
            })
        }),
        set("x-litellm-model-group", |response| {
            told(response, |routing| {
                Some(routing.served.as_ref()?.model_group.clone())
            })
        }),
        set("x-litellm-model-id", |response| {
            told(response, |routing| Some(routing.deployment_id.clone()))
        }),
    )
}
