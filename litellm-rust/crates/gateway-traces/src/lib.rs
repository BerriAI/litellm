use std::{future::Future, io::Read, sync::Arc};

use axum::{
    Router,
    body::to_bytes,
    extract::{Request, State},
    http::{StatusCode, header},
    response::{IntoResponse, Response},
    routing::post,
};
use flate2::read::MultiGzDecoder;
use litellm_traces::{DecodeError, DecodedSpan, decode_otlp};

pub const MAX_BODY_BYTES: usize = 8 * 1024 * 1024;

pub trait SpanSink: Send + Sync + 'static {
    fn write(&self, spans: Vec<DecodedSpan>) -> impl Future<Output = Result<(), ()>> + Send;
}

pub fn router<S: SpanSink>(sink: S) -> Router {
    Router::new()
        .route("/v1/traces", post(ingest::<S>))
        .with_state(Arc::new(sink))
}

async fn ingest<S: SpanSink>(State(sink): State<Arc<S>>, request: Request) -> Response {
    let content_type = request
        .headers()
        .get(header::CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .map(str::to_owned);
    let encoding = request
        .headers()
        .get(header::CONTENT_ENCODING)
        .map(|value| value.to_str().map(str::to_owned));
    let response_type = if content_type.as_deref().is_some_and(|value| {
        value
            .split(';')
            .next()
            .is_some_and(|value| value.trim().eq_ignore_ascii_case("application/json"))
    }) {
        "application/json"
    } else {
        "application/x-protobuf"
    };
    let result = async {
        let encoding = encoding.transpose().map_err(|_| StatusCode::BAD_REQUEST)?;
        let body = to_bytes(request.into_body(), MAX_BODY_BYTES + 1)
            .await
            .map_err(|_| StatusCode::PAYLOAD_TOO_LARGE)?;
        let payload = decode_content_encoding(&body, encoding.as_deref())?;
        let spans =
            decode_otlp(&payload, content_type.as_deref(), MAX_BODY_BYTES).map_err(|error| {
                match error {
                    DecodeError::TooLarge => StatusCode::PAYLOAD_TOO_LARGE,
                    DecodeError::InvalidPayload => StatusCode::BAD_REQUEST,
                }
            })?;
        sink.write(spans)
            .await
            .map_err(|_| StatusCode::SERVICE_UNAVAILABLE)
    }
    .await;
    match result {
        Ok(()) if response_type == "application/json" => {
            ([(header::CONTENT_TYPE, response_type)], "{}").into_response()
        }
        Ok(()) => ([(header::CONTENT_TYPE, response_type)], "").into_response(),
        Err(status) => status.into_response(),
    }
}

fn decode_content_encoding(body: &[u8], encoding: Option<&str>) -> Result<Vec<u8>, StatusCode> {
    if body.len() > MAX_BODY_BYTES {
        return Err(StatusCode::PAYLOAD_TOO_LARGE);
    }
    match encoding {
        None => Ok(body.to_vec()),
        Some(value) if value.eq_ignore_ascii_case("identity") => Ok(body.to_vec()),
        Some(value) if value.eq_ignore_ascii_case("gzip") => {
            let mut payload = Vec::new();
            MultiGzDecoder::new(body)
                .take(MAX_BODY_BYTES as u64 + 1)
                .read_to_end(&mut payload)
                .map_err(|_| StatusCode::BAD_REQUEST)?;
            if payload.len() > MAX_BODY_BYTES {
                return Err(StatusCode::PAYLOAD_TOO_LARGE);
            }
            Ok(payload)
        }
        _ => Err(StatusCode::BAD_REQUEST),
    }
}

#[cfg(test)]
mod tests {
    use std::io::Write;

    use flate2::{Compression, write::GzEncoder};
    use rstest::rstest;

    use super::*;

    #[rstest]
    fn decompressed_size_limit_rejects_expansion() {
        let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
        encoder.write_all(&vec![b' '; MAX_BODY_BYTES + 1]).unwrap();
        let compressed = encoder.finish().unwrap();
        assert!(compressed.len() < MAX_BODY_BYTES);
        assert!(matches!(
            decode_content_encoding(&compressed, Some("gzip")),
            Err(StatusCode::PAYLOAD_TOO_LARGE)
        ));
    }
}
