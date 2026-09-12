use std::sync::Arc;
use std::time::Duration;

use reqwest::Url;
use tokio::time::Instant;

use crate::constants::{AZURE_DI_SUBSCRIPTION_HEADER, OCR_POLL_RETRY_SECS};
use crate::ocr::client::read_json_response;
use crate::ocr::codecs::document_intelligence::{
    AzureDocumentIntelligenceOperation, OperationStatus,
};
use crate::ocr::error::{OcrError, OcrPollingError, OcrResponseError};
use crate::ocr::hooks::OcrHooks;
use crate::ocr::types::OcrConnection;
use crate::ocr::wire::DecodedOcrResponse;

pub(super) async fn read_operation_response(
    http_client: &reqwest::Client,
    response: reqwest::Response,
    original_url: &str,
    headers: &[(String, String)],
    connection: &OcrConnection,
    native: bool,
    hooks: &Arc<dyn OcrHooks>,
) -> Result<DecodedOcrResponse<AzureDocumentIntelligenceOperation>, OcrError> {
    if response.status() != reqwest::StatusCode::ACCEPTED {
        let bytes =
            crate::ocr::client::read_response_bytes(response, connection.max_response_bytes)
                .await?;
        crate::ocr::handler::post_call(hooks, &bytes).await?;
        return Ok(crate::ocr::wire::decode_response(&bytes, native)?);
    }
    let location = response
        .headers()
        .get("operation-location")
        .and_then(|value| value.to_str().ok())
        .ok_or(OcrPollingError::PollLocation)?
        .to_string();
    let original = Url::parse(original_url).map_err(|_| OcrPollingError::PollOrigin)?;
    let operation = Url::parse(&location).map_err(|_| OcrPollingError::PollOrigin)?;
    if original.origin() != operation.origin()
        || !operation.username().is_empty()
        || operation.password().is_some()
    {
        return Err(OcrPollingError::PollOrigin.into());
    }
    let bytes =
        crate::ocr::client::read_response_bytes(response, connection.max_response_bytes).await?;
    crate::ocr::handler::post_call(hooks, &bytes).await?;
    poll_operation(http_client, operation, headers, connection, native, hooks).await
}

async fn poll_operation(
    http_client: &reqwest::Client,
    url: Url,
    headers: &[(String, String)],
    connection: &OcrConnection,
    native: bool,
    hooks: &Arc<dyn OcrHooks>,
) -> Result<DecodedOcrResponse<AzureDocumentIntelligenceOperation>, OcrError> {
    let deadline = Instant::now()
        .checked_add(connection.poll_timeout)
        .ok_or(OcrPollingError::PollTimeout)?;
    loop {
        let remaining = deadline
            .checked_duration_since(Instant::now())
            .filter(|remaining| !remaining.is_zero())
            .ok_or(OcrPollingError::PollTimeout)?;
        let builder = http_client
            .get(url.clone())
            .timeout(remaining.min(connection.timeout));
        let builder = crate::http_utils::with_headers(
            builder,
            headers,
            crate::http_utils::HeaderPolicy::Only(&[AZURE_DI_SUBSCRIPTION_HEADER, "authorization"]),
        );
        let response = tokio::time::timeout_at(deadline, crate::http_utils::http_request(builder))
            .await
            .map_err(|_| OcrPollingError::PollTimeout)?
            .map_err(crate::error::TransportError::from)?;
        let retry = response
            .headers()
            .get(reqwest::header::RETRY_AFTER)
            .and_then(|value| value.to_str().ok())
            .and_then(|value| value.parse::<u64>().ok())
            .unwrap_or(OCR_POLL_RETRY_SECS)
            .max(1);
        let decoded = tokio::time::timeout_at(
            deadline,
            read_json_response::<AzureDocumentIntelligenceOperation>(
                response,
                native,
                connection.max_response_bytes,
            ),
        )
        .await
        .map_err(|_| OcrPollingError::PollTimeout)??;
        match &decoded.data.status {
            Some(OperationStatus::Succeeded) => {
                crate::ocr::handler::post_call(hooks, decoded.text.as_bytes()).await?;
                return Ok(decoded);
            }
            Some(OperationStatus::Running | OperationStatus::NotStarted) => {
                tokio::time::timeout_at(deadline, tokio::time::sleep(Duration::from_secs(retry)))
                    .await
                    .map_err(|_| OcrPollingError::PollTimeout)?;
            }
            status => {
                return Err(OcrResponseError::OperationStatus(
                    status
                        .as_ref()
                        .map(ToString::to_string)
                        .unwrap_or_else(|| "None".into()),
                )
                .into());
            }
        }
    }
}
