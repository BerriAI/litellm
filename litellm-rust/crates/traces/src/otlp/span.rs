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
use crate::{
    Error, Shared,
    normalize::{SpanContext, normalize},
};

pub(super) fn flatten(request: ExportTraceServiceRequest) -> Result<Vec<DecodedSpan>, Error> {
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
) -> Result<(), Error> {
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
) -> Result<(), Error> {
    let scope = scope_spans.scope.unwrap_or_default();
    if scope.attributes.len() > MAX_ATTRIBUTES {
        return Err(Error::TooLarge);
    }
    budget.consume(scope.name.len() + scope.version.len())?;
    let scope_name: Shared<String> = scope.name.into();
    let scope_version: Shared<String> = scope.version.into();
    for span in scope_spans.spans {
        if spans.len() >= MAX_SPANS {
            return Err(Error::TooLarge);
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

fn validate_span(span: &Span) -> Result<(), Error> {
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
        return Err(Error::InvalidPayload);
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
        return Err(Error::TooLarge);
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
) -> Result<DecodedSpan, Error> {
    let status = span.status.unwrap_or_default();
    let parent_span_id = hex_bytes(&span.parent_span_id);
    let span_attributes = attributes(span.attributes, budget)?;
    let events = span
        .events
        .into_iter()
        .map(|event| {
            budget.consume(event.name.len() + 96)?;
            Ok(DecodedEvent {
                name: event.name,
                attributes: attributes(event.attributes, budget)?,
            })
        })
        .collect::<Result<Vec<_>, Error>>()?;
    let normalization = normalize(&SpanContext {
        scope: scope_name.as_ref(),
        name: &span.name,
        parent_span_id: &parent_span_id,
        attributes: &span_attributes,
        events: &events,
        resource_attributes: resource_attributes.as_ref(),
    })?;
    let normalized = normalization.span;
    budget.consume(
        normalized.input.len()
            + normalized.output.len()
            + normalized.agent_name.as_ref().map_or(0, String::len)
            + normalized
                .framework
                .as_ref()
                .map_or(0, |integration| match integration {
                    crate::Integration::Other(name) => name.len(),
                    _ => 0,
                })
            + normalized.agent_metadata.byte_len()
            + normalized
                .calls
                .key_set()
                .into_iter()
                .flatten()
                .map(|key| match key {
                    crate::CallKey::LiteLlmRequest(id) | crate::CallKey::ProviderResponse(id) => {
                        id.len() + size_of::<crate::CallKey>()
                    }
                    crate::CallKey::Transport | crate::CallKey::GatewayAttempt => {
                        size_of::<crate::CallKey>()
                    }
                })
                .sum::<usize>()
            + normalized.model.as_ref().map_or(0, String::len)
            + normalization.display_name.as_ref().map_or(0, String::len),
    )?;
    Ok(DecodedSpan {
        trace_id: hex_bytes(&span.trace_id),
        span_id: hex_bytes(&span.span_id),
        parent_span_id,
        trace_state: span.trace_state,
        name: normalization.display_name.unwrap_or(span.name),
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
        events,
        normalized,
        consumed_attributes: normalization.consumed_attributes,
    })
}
