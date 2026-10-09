use super::{CallEvidence, CallKey, ObservationType, RoleEvidence, SpanContext, SpanFacts};
use super::{Integration, Rule};

const SCOPES: [&str; 7] = [
    "opentelemetry.instrumentation.httpx",
    "opentelemetry.instrumentation.requests",
    "opentelemetry.instrumentation.aiohttp_client",
    "opentelemetry.instrumentation.urllib3",
    "opentelemetry.instrumentation.urllib",
    "@opentelemetry/instrumentation-http",
    "@opentelemetry/instrumentation-undici",
];

pub(super) fn matches(context: &SpanContext<'_>) -> bool {
    SCOPES.contains(&context.scope) || matches_gateway_attempt(context)
}

fn matches_gateway_attempt(context: &SpanContext<'_>) -> bool {
    context.scope == "litellm.gateway.client"
        && context.name == "gateway.request"
        && context
            .attributes
            .get("litellm.gateway.attempt")
            .is_some_and(|value| value == "true")
        && context
            .attributes
            .get("http.request.method")
            .is_some_and(|value| value == "POST")
}

pub(super) fn adjust(context: &SpanContext<'_>, facts: SpanFacts) -> SpanFacts {
    SpanFacts {
        role: Some(RoleEvidence::Declared(ObservationType::Framework)),
        calls: CallEvidence::complete(if matches_gateway_attempt(context) {
            CallKey::GatewayAttempt
        } else {
            CallKey::Transport
        }),
        ..facts
    }
}

pub(super) struct HttpClient;

impl Rule for HttpClient {
    fn matches(&self, context: &SpanContext<'_>) -> bool {
        matches(context)
    }
    fn integration(&self, _: &SpanContext<'_>) -> Option<Integration> {
        None
    }
    fn adjust(
        &self,
        context: &SpanContext<'_>,
        extraction: super::Extraction,
    ) -> super::Extraction {
        extraction.map_facts(|facts| adjust(context, facts))
    }
}
