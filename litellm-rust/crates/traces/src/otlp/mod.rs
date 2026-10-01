mod attributes;
mod limits;
mod span;
mod wire;

use serde::Serialize;
use std::collections::BTreeMap;

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
    decode_budget_bytes: usize,
) -> Result<Vec<DecodedSpan>, DecodeError> {
    let request = wire::decode(body, content_type, decode_budget_bytes)?;
    span::flatten(request, decode_budget_bytes)
}
