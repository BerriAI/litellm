use std::{collections::BTreeMap, io::Read};

use base64::Engine;
use flate2::read::GzDecoder;
use opentelemetry_proto::tonic::{
    collector::trace::v1::ExportTraceServiceRequest,
    common::v1::{AnyValue, KeyValue, any_value::Value as AttributeValue},
    trace::v1::{Span, span::SpanKind, status::StatusCode},
};
use prost::Message;
use serde::Serialize;
use serde_json::Value;

use crate::DecodeError;

#[derive(Serialize)]
pub struct DecodedEvent {
    pub name: String,
    pub attributes: BTreeMap<String, String>,
}

#[derive(Serialize)]
pub struct DecodedSpan {
    pub trace_id: String,
    pub span_id: String,
    pub parent_span_id: String,
    pub trace_state: String,
    pub name: String,
    pub kind: String,
    pub resource_attributes: BTreeMap<String, String>,
    pub scope_name: String,
    pub scope_version: String,
    pub attributes: BTreeMap<String, String>,
    pub start_ns: u64,
    pub end_ns: u64,
    pub status_code: String,
    pub status_message: String,
    pub events: Vec<DecodedEvent>,
}

pub fn decode_otlp(
    body: &[u8],
    content_type: Option<&str>,
    content_encoding: Option<&str>,
    max_decompressed_bytes: usize,
) -> Result<Vec<DecodedSpan>, DecodeError> {
    let payload = if content_encoding == Some("gzip") || body.starts_with(&[0x1f, 0x8b]) {
        let limit = u64::try_from(max_decompressed_bytes).map_err(|_| DecodeError::TooLarge)?;
        let mut decoded = Vec::new();
        GzDecoder::new(body)
            .take(limit + 1)
            .read_to_end(&mut decoded)
            .map_err(|_| DecodeError::InvalidPayload)?;
        decoded
    } else {
        body.to_vec()
    };
    if payload.len() > max_decompressed_bytes {
        return Err(DecodeError::TooLarge);
    }
    let request = if content_type.is_some_and(|value| value.contains("json")) {
        let value: Value =
            serde_json::from_slice(&payload).map_err(|_| DecodeError::InvalidPayload)?;
        serde_json::from_value(normalize_json_ids(value)?)
            .map_err(|_| DecodeError::InvalidPayload)?
    } else {
        ExportTraceServiceRequest::decode(payload.as_slice())
            .map_err(|_| DecodeError::InvalidPayload)?
    };
    Ok(request
        .resource_spans
        .into_iter()
        .flat_map(|resource_spans| {
            let resource_attributes = attributes(
                resource_spans
                    .resource
                    .map(|resource| resource.attributes)
                    .unwrap_or_default(),
            );
            resource_spans
                .scope_spans
                .into_iter()
                .flat_map(move |scope_spans| {
                    let scope = scope_spans.scope.unwrap_or_default();
                    let resource_attributes = resource_attributes.clone();
                    scope_spans.spans.into_iter().map(move |span| {
                        decoded_span(span, &resource_attributes, &scope.name, &scope.version)
                    })
                })
        })
        .collect())
}

fn normalize_json_ids(value: Value) -> Result<Value, DecodeError> {
    match value {
        Value::Object(fields) => fields
            .into_iter()
            .map(|(name, value)| {
                let normalized = if matches!(name.as_str(), "traceId" | "spanId" | "parentSpanId") {
                    let encoded = value.as_str().ok_or(DecodeError::InvalidPayload)?;
                    let bytes = base64::engine::general_purpose::STANDARD
                        .decode(encoded)
                        .map_err(|_| DecodeError::InvalidPayload)?;
                    Value::String(hex_bytes(&bytes))
                } else if name == "kind" && value.is_string() {
                    let kind = SpanKind::from_str_name(value.as_str().unwrap_or_default())
                        .ok_or(DecodeError::InvalidPayload)?;
                    Value::from(kind as i32)
                } else if name == "code" && value.is_string() {
                    let code = StatusCode::from_str_name(value.as_str().unwrap_or_default())
                        .ok_or(DecodeError::InvalidPayload)?;
                    Value::from(code as i32)
                } else {
                    normalize_json_ids(value)?
                };
                Ok((name, normalized))
            })
            .collect::<Result<serde_json::Map<_, _>, _>>()
            .map(Value::Object),
        Value::Array(values) => values
            .into_iter()
            .map(normalize_json_ids)
            .collect::<Result<Vec<_>, _>>()
            .map(Value::Array),
        value => Ok(value),
    }
}

fn hex_bytes(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn decoded_span(
    span: Span,
    resource_attributes: &BTreeMap<String, String>,
    scope_name: &str,
    scope_version: &str,
) -> DecodedSpan {
    let status = span.status.unwrap_or_default();
    DecodedSpan {
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
        attributes: attributes(span.attributes),
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
            .map(|event| DecodedEvent {
                name: event.name,
                attributes: attributes(event.attributes),
            })
            .collect(),
    }
}

fn attributes(values: Vec<KeyValue>) -> BTreeMap<String, String> {
    values
        .into_iter()
        .map(|entry| {
            (
                entry.key,
                entry.value.as_ref().map(attribute_text).unwrap_or_default(),
            )
        })
        .collect()
}

fn attribute_text(value: &AnyValue) -> String {
    match value.value.as_ref() {
        Some(AttributeValue::StringValue(value)) => value.clone(),
        Some(AttributeValue::BoolValue(value)) => value.to_string(),
        Some(AttributeValue::IntValue(value)) => value.to_string(),
        Some(AttributeValue::DoubleValue(value)) => {
            serde_json::to_string(value).unwrap_or_default()
        }
        Some(AttributeValue::BytesValue(value)) => String::from_utf8_lossy(value).into_owned(),
        Some(AttributeValue::ArrayValue(value)) => format!(
            "[{}]",
            value
                .values
                .iter()
                .map(|value| serde_json::to_string(&attribute_text(value)).unwrap_or_default())
                .collect::<Vec<_>>()
                .join(", ")
        ),
        Some(AttributeValue::KvlistValue(value)) => format!(
            "{{{}}}",
            value
                .values
                .iter()
                .map(|entry| format!(
                    "{}: {}",
                    serde_json::to_string(&entry.key).unwrap_or_default(),
                    serde_json::to_string(
                        &entry.value.as_ref().map(attribute_text).unwrap_or_default()
                    )
                    .unwrap_or_default()
                ))
                .collect::<Vec<_>>()
                .join(", ")
        ),
        Some(AttributeValue::StringValueStrindex(value)) => value.to_string(),
        None => String::new(),
    }
}
