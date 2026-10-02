use std::collections::BTreeMap;

use opentelemetry_proto::tonic::{
    collector::trace::v1::ExportTraceServiceRequest,
    trace::v1::{ResourceSpans, ScopeSpans, Span, span::SpanKind, status::StatusCode},
};

use super::{
    DecodedEvent, DecodedSpan,
    attributes::attributes,
    limits::{Budget, MAX_ATTRIBUTES, MAX_DECODED_SPAN_BYTES, MAX_EVENTS, MAX_SPANS},
};
use crate::{DecodeError, Shared, normalize::normalize};

pub(super) fn flatten(request: ExportTraceServiceRequest) -> Result<Vec<DecodedSpan>, DecodeError> {
    let mut budget = Budget::new(MAX_DECODED_SPAN_BYTES);
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
    let attributes = Shared::new(attributes(
        resource
            .resource
            .map(|resource| resource.attributes)
            .unwrap_or_default(),
        budget,
    )?);
    for scope in resource.scope_spans {
        append_scope(scope, &attributes, budget, spans)?;
    }
    Ok(())
}

fn append_scope(
    scope_spans: ScopeSpans,
    resource: &Shared<BTreeMap<String, String>>,
    budget: &mut Budget,
    spans: &mut Vec<DecodedSpan>,
) -> Result<(), DecodeError> {
    let scope = scope_spans.scope.unwrap_or_default();
    if scope.attributes.len() > MAX_ATTRIBUTES {
        return Err(DecodeError::TooLarge);
    }
    budget.consume(scope.name.len() + scope.version.len())?;
    let scope_name: Shared<String> = scope.name.into();
    let scope_version: Shared<String> = scope.version.into();
    for span in scope_spans.spans {
        if spans.len() >= MAX_SPANS {
            return Err(DecodeError::TooLarge);
        }
        validate_span(&span)?;
        budget.consume(
            span.name.len()
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
            &scope_name,
            &scope_version,
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
    resource_attributes: &Shared<BTreeMap<String, String>>,
    scope_name: &Shared<String>,
    scope_version: &Shared<String>,
    budget: &mut Budget,
) -> Result<DecodedSpan, DecodeError> {
    let status = span.status.unwrap_or_default();
    let parent_span_id = hex_bytes(&span.parent_span_id);
    let span_attributes = attributes(span.attributes, budget)?;
    let normalization = normalize(
        scope_name.as_ref(),
        &span.name,
        &parent_span_id,
        &span_attributes,
    )?;
    let resource_agent_name = resource_attributes
        .get("gen_ai.agent.name")
        .filter(|name| !name.is_empty());
    let agent_name = match (resource_agent_name, normalization.span.agent_name.as_str()) {
        (Some(name), "") => name.clone(),
        (Some(name), "hermes-agent") if scope_name.as_ref() == "hermes-otel-plugin" => name.clone(),
        (_, name) => name.to_owned(),
    };
    let normalized = crate::normalize::NormalizedSpan {
        agent_name,
        ..normalization.span
    };
    budget.consume(
        normalized.input.len()
            + normalized.output.len()
            + normalized.agent_name.len()
            + normalized.litellm_request_id.len()
            + normalized.model.len(),
    )?;
    Ok(DecodedSpan {
        trace_id: hex_bytes(&span.trace_id),
        span_id: hex_bytes(&span.span_id),
        parent_span_id,
        trace_state: span.trace_state,
        name: span.name,
        kind: SpanKind::try_from(span.kind)
            .unwrap_or(SpanKind::Unspecified)
            .as_str_name()
            .to_owned(),
        resource_attributes: budget.clone_shared(resource_attributes, |attributes| {
            attributes
                .iter()
                .map(|(key, value)| key.len() + value.len() + 96)
                .sum()
        })?,
        scope_name: budget.clone_shared(scope_name, String::len)?,
        scope_version: budget.clone_shared(scope_version, String::len)?,
        attributes: span_attributes,
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
        normalized,
        consumed_attributes: normalization.consumed_attributes,
    })
}
