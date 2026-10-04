mod attributes;
mod limits;
mod span;
mod wire;

pub use limits::DecodeLimits;

use serde::Serialize;
use std::collections::BTreeMap;

use crate::{Error, NormalizedSpan, Shared};

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
    pub resource_attributes: Shared<BTreeMap<String, String>>,
    pub scope_name: Shared<String>,
    pub scope_version: Shared<String>,
    pub attributes: BTreeMap<String, String>,
    pub start_ns: u64,
    pub end_ns: u64,
    pub status_code: String,
    pub status_message: String,
    pub events: Vec<DecodedEvent>,
    pub normalized: NormalizedSpan,
    pub consumed_attributes: Box<[&'static str]>,
}

pub fn decode_otlp(body: &[u8], content_type: Option<&str>) -> Result<Vec<DecodedSpan>, Error> {
    decode_otlp_with_limits(body, content_type, DecodeLimits::from_env()?)
}

pub fn decode_otlp_with_limits(
    body: &[u8],
    content_type: Option<&str>,
    limits: DecodeLimits,
) -> Result<Vec<DecodedSpan>, Error> {
    let request = wire::decode(body, content_type, &limits)?;
    span::flatten(request, limits)
}
