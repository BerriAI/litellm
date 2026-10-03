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
    SCOPES.contains(&context.scope)
}

pub(super) fn adjust(facts: SpanFacts) -> SpanFacts {
    SpanFacts {
        role: Some(RoleEvidence::Declared(ObservationType::Framework)),
        calls: CallEvidence::complete(CallKey::Transport),
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
    fn adjust(&self, _: &SpanContext<'_>, extraction: super::Extraction) -> super::Extraction {
        extraction.map_facts(adjust)
    }
}
