use super::{CallEvidence, CallKey, ObservationType, RoleEvidence, SpanContext, SpanFacts};

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
