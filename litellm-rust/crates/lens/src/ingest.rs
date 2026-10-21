use crate::{Error, State};
use axum::{
    body::{Body, to_bytes},
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
};
use flate2::read::MultiGzDecoder;
use litellm_traces::Tenant;
use litellm_traces_clickhouse::{InsertTable, insert_shared_rows, span_rows};
use prost::Message;
use std::{io::Read, sync::Arc, time::Duration};
use tokio::sync::OwnedSemaphorePermit;

pub const MAX_BODY_BYTES: usize = 16 * 1024 * 1024;
pub const UPLOAD_TIMEOUT: Duration = Duration::from_secs(30);

#[derive(Message)]
struct OtlpError {
    #[prost(int32, tag = "1")]
    code: i32,
    #[prost(string, tag = "2")]
    message: String,
}

fn decompress(payload: &[u8], encoding: Option<&str>) -> Result<Vec<u8>, Error> {
    match encoding {
        None | Some("identity" | "") => Ok(payload.to_vec()),
        Some("gzip") => {
            let mut decoded = Vec::new();
            MultiGzDecoder::new(payload)
                .take((MAX_BODY_BYTES + 1) as u64)
                .read_to_end(&mut decoded)
                .map_err(|_| Error::InvalidRequest)?;
            if decoded.len() > MAX_BODY_BYTES {
                return Err(Error::TooLarge);
            }
            Ok(decoded)
        }
        Some(_) => Err(Error::InvalidRequest),
    }
}

pub fn response(content_type: Option<&str>, outcome: Result<(), Error>) -> Response {
    let status = outcome
        .as_ref()
        .map(|_| StatusCode::OK)
        .unwrap_or_else(|error| error.status());
    let message = status.canonical_reason().unwrap_or("Trace request failed");
    let rpc_code = match status {
        StatusCode::OK => 0,
        StatusCode::BAD_REQUEST => 3,
        StatusCode::UNAUTHORIZED => 16,
        StatusCode::PAYLOAD_TOO_LARGE | StatusCode::TOO_MANY_REQUESTS => 8,
        StatusCode::CONFLICT => 10,
        StatusCode::SERVICE_UNAVAILABLE => 14,
        _ => 2,
    };
    let protobuf = content_type.is_some_and(|value| {
        value
            .split(';')
            .next()
            .is_some_and(|value| value.trim() == "application/x-protobuf")
    });
    let (body, media_type) = if protobuf {
        (
            if outcome.is_ok() {
                Vec::new()
            } else {
                OtlpError {
                    code: rpc_code,
                    message: message.into(),
                }
                .encode_to_vec()
            },
            "application/x-protobuf",
        )
    } else {
        (
            if outcome.is_ok() {
                b"{}".to_vec()
            } else {
                serde_json::json!({"code": rpc_code, "message": message})
                    .to_string()
                    .into_bytes()
            },
            "application/json",
        )
    };
    let mut response = (status, [(http::header::CONTENT_TYPE, media_type)], body).into_response();
    if matches!(
        status,
        StatusCode::SERVICE_UNAVAILABLE | StatusCode::TOO_MANY_REQUESTS
    ) {
        response
            .headers_mut()
            .insert("retry-after", http::HeaderValue::from_static("5"));
    }
    response
}

pub async fn receive(state: Arc<State>, headers: HeaderMap, body: Body, logs: bool) -> Response {
    let content_type = headers
        .get("content-type")
        .and_then(|value| value.to_str().ok())
        .map(str::to_owned);
    let outcome = receive_authorized(state, &headers, body, logs).await;
    response(content_type.as_deref(), outcome)
}

async fn receive_authorized(
    state: Arc<State>,
    headers: &HeaderMap,
    body: Body,
    logs: bool,
) -> Result<(), Error> {
    let tenant = state.credentials.tenant(headers)?;
    state.require_storage()?;
    let permit = state
        .ingest_slots
        .clone()
        .try_acquire_owned()
        .map_err(|_| Error::Unavailable)?;
    let payload = tokio::time::timeout(UPLOAD_TIMEOUT, to_bytes(body, MAX_BODY_BYTES))
        .await
        .map_err(|_| Error::Unavailable)?
        .map_err(|_| Error::TooLarge)?;
    let content_type = headers
        .get("content-type")
        .and_then(|value| value.to_str().ok())
        .map(str::to_owned);
    let encoding = headers
        .get("content-encoding")
        .and_then(|value| value.to_str().ok())
        .map(str::to_owned);
    tokio::spawn(store(
        state,
        payload,
        encoding,
        content_type,
        tenant,
        logs,
        permit,
    ))
    .await
    .map_err(|_| Error::Unavailable)?
}

async fn store(
    state: Arc<State>,
    payload: bytes::Bytes,
    encoding: Option<String>,
    content_type: Option<String>,
    tenant: Tenant,
    logs: bool,
    permit: OwnedSemaphorePermit,
) -> Result<(), Error> {
    let max_value_bytes = state.storage.config.max_attribute_value_bytes();
    let (rows, _permit) = tokio::task::spawn_blocking(move || {
        let payload = decompress(&payload, encoding.as_deref())?;
        let decode = if logs {
            litellm_traces::decode_otlp_logs
        } else {
            litellm_traces::decode_otlp
        };
        let spans = decode(&payload, content_type.as_deref())
            .map_err(litellm_traces_clickhouse::Error::from)?;
        Ok::<_, Error>((span_rows(spans, &tenant, max_value_bytes), permit))
    })
    .await
    .map_err(|_| Error::Unavailable)??;
    insert_shared_rows(
        &state.storage.client,
        state.storage.config.storage().writer(),
        state.storage.config.storage().database(),
        InsertTable::OtelTraces,
        rows,
    )
    .await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{OtlpError, response};
    use crate::Error;
    use axum::body::to_bytes;
    use prost::Message;
    use rstest::rstest;

    #[rstest]
    #[case::invalid_request(Error::InvalidRequest, 400, 3, false)]
    #[case::unauthenticated(Error::Unauthorized, 401, 16, false)]
    #[case::payload_too_large(Error::TooLarge, 413, 8, false)]
    #[case::credentials_pending(Error::CredentialsPending, 429, 8, true)]
    #[case::storage_unavailable(Error::Unavailable, 503, 14, true)]
    #[case::conflict(Error::TraceChanged, 409, 10, false)]
    #[tokio::test]
    async fn rejected_batches_have_matching_http_and_rpc_errors(
        #[case] error: Error,
        #[case] http_status: u16,
        #[case] rpc_code: i32,
        #[case] retryable: bool,
        #[values("application/json", "application/x-protobuf")] content_type: &str,
    ) {
        let reply = response(Some(content_type), Err(error));
        assert_eq!(reply.status().as_u16(), http_status);
        assert_eq!(reply.headers()["content-type"], content_type);
        assert_eq!(
            reply
                .headers()
                .get("retry-after")
                .map(|v| v.to_str().unwrap()),
            retryable.then_some("5")
        );
        let message = reply.status().canonical_reason().unwrap();
        let body = to_bytes(reply.into_body(), 1024).await.unwrap();
        if content_type == "application/x-protobuf" {
            let status = OtlpError::decode(body).unwrap();
            assert_eq!(status.code, rpc_code);
            assert_eq!(status.message, message);
        } else {
            let status: serde_json::Value = serde_json::from_slice(&body).unwrap();
            assert_eq!(
                status,
                serde_json::json!({"code": rpc_code, "message": message})
            );
        }
    }

    #[rstest]
    #[case::json("application/json", b"{}")]
    #[case::protobuf("application/x-protobuf", b"")]
    #[tokio::test]
    async fn accepted_batches_keep_the_empty_export_response(
        #[case] content_type: &str,
        #[case] expected: &[u8],
    ) {
        let reply = response(Some(content_type), Ok(()));
        assert_eq!(reply.status(), 200);
        assert_eq!(reply.headers()["content-type"], content_type);
        assert!(!reply.headers().contains_key("retry-after"));
        assert_eq!(to_bytes(reply.into_body(), 1024).await.unwrap(), expected);
    }
}
