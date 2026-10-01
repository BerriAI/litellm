mod attributes;
mod limits;
mod span;
mod wire;

use serde::Serialize;
use std::collections::BTreeMap;

use crate::DecodeError;

pub const OTLP_DEFAULT_MAX_BODY_BYTES: usize = 16 * 1024 * 1024;

pub fn otlp_max_body_bytes() -> usize {
    std::env::var("OTLP_MAX_BODY_BYTES")
        .ok()
        .and_then(|value| value.trim().parse().ok())
        .unwrap_or(OTLP_DEFAULT_MAX_BODY_BYTES)
}

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
) -> Result<Vec<DecodedSpan>, DecodeError> {
    let max_body_bytes = otlp_max_body_bytes();
    let request = wire::decode(body, content_type, max_body_bytes)?;
    span::flatten(request, max_body_bytes)
}
