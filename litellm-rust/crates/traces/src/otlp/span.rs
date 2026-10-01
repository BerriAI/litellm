use std::collections::BTreeMap;

use opentelemetry_proto::tonic::{
    collector::trace::v1::ExportTraceServiceRequest,
    trace::v1::{ResourceSpans, ScopeSpans, Span, span::SpanKind, status::StatusCode},
};

use super::{
    DecodedEvent, DecodedSpan,
    attributes::{attribute_size, attributes},
    limits::{Budget, MAX_ATTRIBUTES, MAX_EVENTS, MAX_SPANS},
};
use crate::DecodeError;

pub(super) fn flatten(
    request: ExportTraceServiceRequest,
    max_decoded_bytes: usize,
) -> Result<Vec<DecodedSpan>, DecodeError> {
    let mut budget = Budget::new(max_decoded_bytes);
    let mut spans = Vec::new();
    for resource in request.resource_spans {
        append_resource(resource, &mut budget, &mut spans)?;
    }
    Ok(spans)
}

fn append_resource(
    resource: ResourceSpans,
    budget: &mut Budget,
    spans: &mut Vec<DecodedSpan>,
) -> Result<(), DecodeError> {
    let attributes = attributes(
        resource
            .resource
            .map(|resource| resource.attributes)
            .unwrap_or_default(),
        budget,
    )?;
    for scope in resource.scope_spans {
        append_scope(scope, &attributes, budget, spans)?;
    }
    Ok(())
}

fn append_scope(
    scope_spans: ScopeSpans,
    resource: &BTreeMap<String, String>,
    budget: &mut Budget,
    spans: &mut Vec<DecodedSpan>,
) -> Result<(), DecodeError> {
    let scope = scope_spans.scope.unwrap_or_default();
    if scope.attributes.len() > MAX_ATTRIBUTES {
        return Err(DecodeError::TooLarge);
    }
    let shared_size = attribute_size(resource) + scope.name.len() + scope.version.len();
    for span in scope_spans.spans {
        if spans.len() >= MAX_SPANS {
            return Err(DecodeError::TooLarge);
        }
        validate_span(&span)?;
        budget.consume(
            shared_size
                + span.name.len()
                + span.trace_state.len()
                + span
                    .status
                    .as_ref()
                    .map_or(0, |status| status.message.len())
                + size_of::<DecodedSpan>()
                + 128,
        )?;
        spans.push(decoded_span(
            span,
            resource,
            &scope.name,
            &scope.version,
            budget,
        )?);
    }
    Ok(())
}

fn valid_id(value: &[u8], length: usize) -> bool {
    value.len() == length && value.iter().any(|byte| *byte != 0)
}

fn validate_span(span: &Span) -> Result<(), DecodeError> {
    if !valid_id(&span.trace_id, 16)
        || !valid_id(&span.span_id, 8)
        || (!span.parent_span_id.is_empty() && !valid_id(&span.parent_span_id, 8))
        || span.start_time_unix_nano > i64::MAX as u64
        || span.end_time_unix_nano > i64::MAX as u64
        || span.end_time_unix_nano < span.start_time_unix_nano
        || span
            .links
            .iter()
            .any(|link| !valid_id(&link.trace_id, 16) || !valid_id(&link.span_id, 8))
    {
        return Err(DecodeError::InvalidPayload);
    }
    if span.events.len() > MAX_EVENTS
        || span.links.len() > MAX_EVENTS
        || span.attributes.len() > MAX_ATTRIBUTES
        || span
            .links
            .iter()
            .any(|link| link.attributes.len() > MAX_ATTRIBUTES)
        || span
            .events
            .iter()
            .any(|event| event.attributes.len() > MAX_ATTRIBUTES)
    {
        return Err(DecodeError::TooLarge);
    }
    Ok(())
}

fn hex_bytes(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn decoded_span(
    span: Span,
    resource_attributes: &BTreeMap<String, String>,
    scope_name: &str,
    scope_version: &str,
    budget: &mut Budget,
) -> Result<DecodedSpan, DecodeError> {
    let status = span.status.unwrap_or_default();
    Ok(DecodedSpan {
        trace_id: hex_bytes(&span.trace_id),
        span_id: hex_bytes(&span.span_id),
        parent_span_id: hex_bytes(&span.parent_span_id),
        trace_state: span.trace_state,
        name: span.name,
        kind: SpanKind::try_from(span.kind)
            .unwrap_or(SpanKind::Unspecified)
            .as_str_name()
            .to_owned(),
        resource_attributes: resource_attributes.clone(),
        scope_name: scope_name.to_owned(),
        scope_version: scope_version.to_owned(),
        attributes: attributes(span.attributes, budget)?,
        start_ns: span.start_time_unix_nano,
        end_ns: span.end_time_unix_nano,
        status_code: StatusCode::try_from(status.code)
            .unwrap_or(StatusCode::Unset)
            .as_str_name()
            .to_owned(),
        status_message: status.message,
        events: span
            .events
            .into_iter()
            .map(|event| {
                budget.consume(event.name.len() + 96)?;
                Ok(DecodedEvent {
                    name: event.name,
                    attributes: attributes(event.attributes, budget)?,
                })
            })
            .collect::<Result<Vec<_>, DecodeError>>()?,
    })
}
