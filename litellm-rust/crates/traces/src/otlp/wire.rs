use opentelemetry_proto::tonic::collector::trace::v1::ExportTraceServiceRequest;
use prost::Message;

use super::limits::{json_preflight, protobuf_preflight};
use crate::Error;

#[derive(strum::EnumString)]
#[strum(ascii_case_insensitive)]
enum OtlpMediaType {
    #[strum(serialize = "application/json")]
    Json,
    #[strum(
        serialize = "application/x-protobuf",
        serialize = "application/protobuf"
    )]
    Protobuf,
}

pub(super) fn decode(
    body: &[u8],
    content_type: Option<&str>,
) -> Result<ExportTraceServiceRequest, Error> {
    let media_type = content_type
        .unwrap_or("application/x-protobuf")
        .split(';')
        .next()
        .unwrap_or_default()
        .trim()
        .parse::<OtlpMediaType>()
        .map_err(|_| Error::InvalidPayload)?;

    let request = match media_type {
        OtlpMediaType::Json => {
            json_preflight(body)?;
            serde_json::from_slice(body).map_err(|_| Error::InvalidPayload)?
        }
        OtlpMediaType::Protobuf => {
            protobuf_preflight(body)?;
            ExportTraceServiceRequest::decode(body).map_err(|_| Error::InvalidPayload)?
        }
    };
    Ok(request)
}
