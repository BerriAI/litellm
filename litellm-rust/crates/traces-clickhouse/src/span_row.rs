//! Decoded spans as `otel_traces` rows: payloads capped, the sending tenant stamped over whatever
//! the export claimed, and resource maps shared across the rows that came from one resource.

use std::collections::{BTreeMap, HashMap};

use litellm_traces::{
    CallEvidence, CallKey, DecodedEvent, DecodedSpan, Shared, SharedIdentity, Tenant,
    truncate_messages, truncate_value,
};
use serde::Serialize;
use serde_json::{Map, Value};

use crate::InsertRow;

/// Converts each distinct shared source once; keeping the source pins its identity.
struct SharedValues<T>(HashMap<SharedIdentity, (Shared<T>, Shared<Value>)>);

impl<T: Clone> SharedValues<T> {
    fn new() -> Self {
        Self(HashMap::new())
    }

    fn get(&mut self, source: &Shared<T>, convert: impl FnOnce(&T) -> Value) -> Shared<Value> {
        self.0
            .entry(source.identity())
            .or_insert_with(|| (source.clone(), Shared::new(convert(source))))
            .1
            .clone()
    }
}

fn stamped(attributes: &BTreeMap<String, String>, tenant: &Tenant) -> Value {
    let mut stamped: Map<String, Value> = attributes
        .iter()
        .map(|(key, value)| (key.clone(), Value::from(value.as_str())))
        .collect();
    for (key, value) in [
        ("litellm.team_id", &tenant.team_id),
        ("litellm.api_key_hash", &tenant.api_key_hash),
        ("litellm.org_id", &tenant.org_id),
        ("litellm.user_id", &tenant.user_id),
    ] {
        stamped.insert(key.to_owned(), Value::from(value.as_str()));
    }
    Value::Object(stamped)
}

fn exception_message(events: &[DecodedEvent]) -> String {
    events
        .iter()
        .find(|event| event.name == "exception")
        .and_then(|event| {
            event
                .attributes
                .get("exception.message")
                .filter(|message| !message.is_empty())
                .or_else(|| event.attributes.get("exception.type"))
        })
        .cloned()
        .unwrap_or_default()
}

fn json<T: Serialize>(value: T) -> Value {
    serde_json::to_value(value).unwrap_or(Value::Null)
}

fn present_fields<T: Serialize>(value: &T) -> String {
    match json(value) {
        Value::Object(fields) => Value::Object(
            fields
                .into_iter()
                .filter(|(_, value)| !value.is_null())
                .collect(),
        )
        .to_string(),
        other => other.to_string(),
    }
}

pub fn span_rows(
    spans: Vec<DecodedSpan>,
    tenant: &Tenant,
    max_value_bytes: usize,
) -> Vec<InsertRow> {
    let mut resources = SharedValues::new();
    let mut scopes = SharedValues::new();
    spans
        .into_iter()
        .map(|span| {
            let normalized = span.normalized;
            let service = span
                .resource_attributes
                .get("service.name")
                .cloned()
                .unwrap_or_default();
            let status_message = if span.status_message.is_empty() {
                exception_message(&span.events)
            } else {
                span.status_message
            };
            let attributes: Map<String, Value> = span
                .attributes
                .into_iter()
                .filter(|(key, _)| !span.consumed_attributes.contains(&key.as_str()))
                .map(|(key, value)| (key, Value::String(truncate_value(value, max_value_bytes))))
                .collect();
            let shared = [
                (
                    "ResourceAttributes",
                    resources.get(&span.resource_attributes, |attributes| {
                        stamped(attributes, tenant)
                    }),
                ),
                (
                    "ScopeName",
                    scopes.get(&span.scope_name, |name| Value::from(name.as_str())),
                ),
                (
                    "ScopeVersion",
                    scopes.get(&span.scope_version, |version| Value::from(version.as_str())),
                ),
            ];
            let owned = [
                ("Timestamp", json(span.start_ns)),
                ("TraceId", Value::String(span.trace_id)),
                ("SpanId", Value::String(span.span_id)),
                ("ParentSpanId", Value::String(span.parent_span_id)),
                ("TraceState", Value::String(span.trace_state)),
                ("SpanName", Value::String(span.name)),
                ("SpanKind", Value::String(span.kind)),
                ("ServiceName", Value::String(service)),
                ("SpanAttributes", Value::Object(attributes)),
                ("Duration", json(span.end_ns - span.start_ns)),
                ("StatusCode", Value::String(span.status_code)),
                ("StatusMessage", Value::String(status_message)),
                ("TeamId", Value::from(tenant.team_id.as_str())),
                ("ApiKeyHash", Value::from(tenant.api_key_hash.as_str())),
                ("UserId", Value::from(tenant.user_id.as_str())),
                ("ObservationType", json(normalized.observation_type)),
                (
                    "WrapperCandidate",
                    Value::Bool(normalized.wrapper_candidate),
                ),
                (
                    "AgentName",
                    Value::String(normalized.agent_name.unwrap_or_default()),
                ),
                (
                    "Framework",
                    Value::String(
                        normalized
                            .framework
                            .map(|integration| integration.to_string())
                            .unwrap_or_default(),
                    ),
                ),
                (
                    "AgentMetadata",
                    Value::String(present_fields(&normalized.agent_metadata)),
                ),
                (
                    "LiteLLMRequestId",
                    Value::String(request_id(&normalized.calls).to_owned()),
                ),
                (
                    "CallKeys",
                    json(
                        normalized
                            .calls
                            .key_set()
                            .into_iter()
                            .flatten()
                            .collect::<Vec<_>>(),
                    ),
                ),
                ("CallEvidence", json(normalized.calls.kind())),
                ("Model", Value::String(normalized.model.unwrap_or_default())),
                ("InputTokens", Value::from(normalized.input_tokens)),
                ("OutputTokens", Value::from(normalized.output_tokens)),
                (
                    "Input",
                    Value::String(truncate_messages(normalized.input, max_value_bytes)),
                ),
                ("InputPreview", Value::String(normalized.input_preview)),
                (
                    "Output",
                    Value::String(truncate_value(normalized.output, max_value_bytes)),
                ),
                (
                    "ToolCallId",
                    Value::String(normalized.tool_call_id.unwrap_or_default()),
                ),
            ];
            shared
                .into_iter()
                .chain(
                    owned
                        .into_iter()
                        .map(|(column, value)| (column, Shared::new(value))),
                )
                .map(|(column, value)| (column.to_owned(), value))
                .collect()
        })
        .collect()
}

fn request_id(evidence: &CallEvidence) -> &str {
    evidence
        .key_set()
        .into_iter()
        .flatten()
        .find_map(|key| match key {
            CallKey::ProviderResponse(id) => Some(id.as_str()),
            CallKey::LiteLlmRequest(_) | CallKey::Transport | CallKey::GatewayAttempt => None,
        })
        .unwrap_or_default()
}
