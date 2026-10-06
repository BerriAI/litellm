use opentelemetry_proto::tonic::{
    collector::{logs::v1::ExportLogsServiceRequest, trace::v1::ExportTraceServiceRequest},
    common::v1::{KeyValue, any_value::Value},
    logs::v1::LogRecord,
    trace::v1::{ResourceSpans, ScopeSpans, Span, Status},
};
use sha2::{Digest, Sha256};

use super::{DecodeLimits, DecodedSpan, span};
use crate::{
    Error,
    normalize::{CLAUDE_CODE_EVENTS_SCOPE, visible_claude_response},
};

fn value<'a>(attributes: &'a [KeyValue], key: &str) -> Option<&'a Value> {
    attributes
        .iter()
        .rev()
        .find(|entry| entry.key == key)
        .and_then(|entry| entry.value.as_ref())
        .and_then(|value| value.value.as_ref())
}

fn text<'a>(attributes: &'a [KeyValue], key: &str) -> &'a str {
    match value(attributes, key) {
        Some(Value::StringValue(text)) => text,
        _ => "",
    }
}

fn absent_id(id: &[u8], length: usize) -> bool {
    id.is_empty() || (id.len() == length && id.iter().all(|byte| *byte == 0))
}

struct LogContext {
    missing_trace: bool,
    missing_parent: bool,
}

fn message(record: LogRecord) -> (Span, LogContext) {
    let timestamp = if record.time_unix_nano == 0 {
        record.observed_time_unix_nano
    } else {
        record.time_unix_nano
    };
    let missing_trace = absent_id(&record.trace_id, 16);
    let missing_parent = absent_id(&record.span_id, 8);
    let mut hash = Sha256::new();
    hash.update(b"litellm.claude.message.v1\0");
    if !missing_trace {
        hash.update(&record.trace_id);
    }
    if !missing_parent {
        hash.update(&record.span_id);
    }
    hash.update(text(&record.attributes, "event.name"));
    let uuid = text(&record.attributes, "message.uuid");
    if uuid.is_empty() {
        hash.update(timestamp.to_be_bytes());
        match value(&record.attributes, "event.sequence") {
            Some(Value::IntValue(sequence)) => hash.update(sequence.to_string()),
            _ => hash.update(text(&record.attributes, "event.sequence")),
        }
        hash.update(text(&record.attributes, "response"));
    } else {
        hash.update(uuid);
    }
    if missing_trace {
        hash.update(b"\0unassigned\0");
        hash.update(text(&record.attributes, "session.id"));
    }
    let identity = hash.finalize();
    let trace_id = if missing_trace {
        let session = text(&record.attributes, "session.id");
        if session.is_empty() {
            identity[..16].to_vec()
        } else {
            span::session_trace_id(session)
        }
    } else {
        record.trace_id
    };
    let failed = match value(&record.attributes, "success") {
        Some(Value::BoolValue(success)) => !success,
        Some(Value::StringValue(success)) => success == "false",
        _ => false,
    };
    (
        Span {
            trace_id,
            span_id: identity[..8].to_vec(),
            parent_span_id: if missing_trace || missing_parent {
                Vec::new()
            } else {
                record.span_id
            },
            name: format!("claude_code.{}", text(&record.attributes, "event.name")),
            kind: 1,
            start_time_unix_nano: timestamp,
            end_time_unix_nano: timestamp,
            status: (text(&record.attributes, "event.name") == "tool_result" && failed).then(
                || Status {
                    code: 2,
                    message: text(&record.attributes, "error").to_owned(),
                },
            ),
            attributes: record.attributes,
            ..Span::default()
        },
        LogContext {
            missing_trace,
            missing_parent,
        },
    )
}

pub(super) fn flatten(
    request: ExportLogsServiceRequest,
    limits: DecodeLimits,
) -> Result<Vec<DecodedSpan>, Error> {
    let mut count = 0usize;
    let mut contexts = Vec::new();
    let mut resources = Vec::new();
    for resource in request.resource_logs {
        let mut scopes = Vec::new();
        for scope in resource.scope_logs {
            count = count
                .checked_add(scope.log_records.len())
                .ok_or(Error::TooLarge)?;
            if count > limits.spans
                || scope
                    .log_records
                    .iter()
                    .any(|record| record.attributes.len() > limits.attributes)
            {
                return Err(Error::TooLarge);
            }
            let supported = scope
                .scope
                .as_ref()
                .is_some_and(|scope| scope.name == CLAUDE_CODE_EVENTS_SCOPE);
            let spans = scope
                .log_records
                .into_iter()
                .filter(|record| {
                    let event = text(&record.attributes, "event.name");
                    supported
                        && (matches!(event, "tool_result" | "compaction")
                            || (matches!(event, "assistant_response" | "api_request_body")
                                && visible_claude_response(
                                    "assistant_response",
                                    text(&record.attributes, "query_source"),
                                )))
                })
                .map(|record| {
                    let (span, context) = message(record);
                    contexts.push(context);
                    span
                })
                .collect();
            scopes.push(ScopeSpans {
                scope: scope.scope,
                spans,
                schema_url: scope.schema_url,
            });
        }
        resources.push(ResourceSpans {
            resource: resource.resource,
            scope_spans: scopes,
            schema_url: resource.schema_url,
        });
    }
    let (mut spans, mut budget) = span::flatten_with_budget(
        ExportTraceServiceRequest {
            resource_spans: resources,
        },
        limits,
    )?;
    budget.consume(contexts.len() * size_of::<LogContext>())?;
    for (span, context) in spans.iter_mut().zip(contexts) {
        if context.missing_trace {
            span.attributes.remove("lens.original_trace_id");
        }
        if context.missing_trace || context.missing_parent {
            let key = "lens.capture.warning";
            let warning = "This native log has incomplete trace context. Its content is retained, but its execution parent is unconfirmed.";
            budget.consume(key.len() + warning.len() + 96)?;
            span.attributes.insert(key.to_owned(), warning.to_owned());
        }
    }
    Ok(spans)
}
