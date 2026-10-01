use opentelemetry_proto::tonic::collector::trace::v1::ExportTraceServiceRequest;
use prost::Message;

use super::limits::{json_preflight, protobuf_preflight};
use crate::DecodeError;

pub(super) fn decode(
    body: &[u8],
    content_type: Option<&str>,
    max_body_bytes: usize,
) -> Result<ExportTraceServiceRequest, DecodeError> {
    if body.len() > max_body_bytes {
        return Err(DecodeError::TooLarge);
    }
    let media_type = content_type
        .unwrap_or("application/x-protobuf")
        .split(';')
        .next()
        .unwrap_or_default()
        .trim();
    let request = if media_type.eq_ignore_ascii_case("application/json") {
        json_preflight(body)?;
        serde_json::from_slice(body).map_err(|_| DecodeError::InvalidPayload)?
    } else if media_type.eq_ignore_ascii_case("application/x-protobuf")
        || media_type.eq_ignore_ascii_case("application/protobuf")
    {
        protobuf_preflight(body)?;
        ExportTraceServiceRequest::decode(body).map_err(|_| DecodeError::InvalidPayload)?
    } else {
        return Err(DecodeError::InvalidPayload);
    };
    Ok(request)
}
