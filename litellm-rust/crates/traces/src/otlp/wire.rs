use opentelemetry_proto::tonic::collector::logs::v1::ExportLogsServiceRequest;
use opentelemetry_proto::tonic::collector::trace::v1::ExportTraceServiceRequest;
use prost::Message;

use super::limits::{DecodeLimits, json_preflight, protobuf_logs_preflight, protobuf_preflight};
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
    limits: &DecodeLimits,
) -> Result<ExportTraceServiceRequest, Error> {
    decode_request(body, content_type, limits, protobuf_preflight)
}

pub(super) fn decode_logs(
    body: &[u8],
    content_type: Option<&str>,
    limits: &DecodeLimits,
) -> Result<ExportLogsServiceRequest, Error> {
    decode_request(body, content_type, limits, protobuf_logs_preflight)
}

fn decode_request<T: Message + Default + serde::de::DeserializeOwned>(
    body: &[u8],
    content_type: Option<&str>,
    limits: &DecodeLimits,
    preflight: fn(&[u8], &DecodeLimits) -> Result<(), Error>,
) -> Result<T, Error> {
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
            json_preflight(body, limits)?;
            serde_json::from_slice(body).map_err(|_| Error::InvalidPayload)?
        }
        OtlpMediaType::Protobuf => {
            preflight(body, limits)?;
            T::decode(body).map_err(|_| Error::InvalidPayload)?
        }
    };
    Ok(request)
}
