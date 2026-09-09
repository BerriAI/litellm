use crate::ocr::error::OcrError;
use crate::ocr::error::OcrPollingError;
use crate::ocr::error::OcrResponseError;
use std::time::Duration;

use crate::constants::{AZURE_DI_SUBSCRIPTION_HEADER, OCR_POLL_RETRY_SECS};
use crate::ocr::client::read_json_response;
use crate::ocr::formats::document_intelligence::types::{
    AzureDocumentIntelligenceOperation, OperationStatus,
};
use crate::ocr::types::OcrConnection;
use crate::ocr::wire::DecodedOcrResponse;
use reqwest::Url;
use tokio::time::Instant;

pub(crate) async fn read_operation_response(
    http_client: &reqwest::Client,
    response: reqwest::Response,
    original_url: &str,
    headers: &[(String, String)],
    connection: &OcrConnection,
    native: bool,
) -> Result<DecodedOcrResponse<AzureDocumentIntelligenceOperation>, OcrError> {
    if response.status() != reqwest::StatusCode::ACCEPTED {
        return read_json_response(response, native).await;
    }
    let location = response
        .headers()
        .get("operation-location")
        .and_then(|value| value.to_str().ok())
        .ok_or(OcrPollingError::PollLocation)?;
    let original = Url::parse(original_url).map_err(|_| OcrPollingError::PollOrigin)?;
    let operation = Url::parse(location).map_err(|_| OcrPollingError::PollOrigin)?;
    if original.origin() != operation.origin()
        || !operation.username().is_empty()
        || operation.password().is_some()
    {
        return Err(OcrPollingError::PollOrigin.into());
    }
    poll_document_intelligence(http_client, operation, headers, connection, native).await
}

#[tracing::instrument(target = "litellm::function_trace", level = "trace", skip_all)]
async fn poll_document_intelligence(
    http_client: &reqwest::Client,
    url: Url,
    headers: &[(String, String)],
    connection: &OcrConnection,
    native: bool,
) -> Result<DecodedOcrResponse<AzureDocumentIntelligenceOperation>, OcrError> {
    let deadline = Instant::now()
        .checked_add(connection.poll_timeout)
        .ok_or(OcrPollingError::PollTimeout)?;
    loop {
        let remaining = deadline
            .checked_duration_since(Instant::now())
            .filter(|remaining| !remaining.is_zero())
            .ok_or(OcrPollingError::PollTimeout)?;
        let mut builder = http_client
            .get(url.clone())
            .timeout(remaining.min(connection.timeout));
        for (name, value) in headers {
            if name.eq_ignore_ascii_case(AZURE_DI_SUBSCRIPTION_HEADER)
                || name.eq_ignore_ascii_case("authorization")
            {
                builder = builder.header(name, value);
            }
        }
        let response = tokio::time::timeout_at(deadline, crate::http_utils::http_request(builder))
            .await
            .map_err(|_| OcrPollingError::PollTimeout)?
            .map_err(crate::error::TransportError::from)?;
        let retry = response
            .headers()
            .get(reqwest::header::RETRY_AFTER)
            .and_then(|value| value.to_str().ok())
            .and_then(|value| value.parse::<u64>().ok())
            .unwrap_or(OCR_POLL_RETRY_SECS);
        let decoded: DecodedOcrResponse<AzureDocumentIntelligenceOperation> =
            tokio::time::timeout_at(deadline, read_json_response(response, native))
                .await
                .map_err(|_| OcrPollingError::PollTimeout)??;
        match &decoded.data.status {
            Some(OperationStatus::Succeeded) => return Ok(decoded),
            Some(OperationStatus::Running | OperationStatus::NotStarted) => {
                tokio::time::timeout_at(deadline, tokio::time::sleep(Duration::from_secs(retry)))
                    .await
                    .map_err(|_| OcrPollingError::PollTimeout)?;
            }
            status => {
                return Err(OcrResponseError::OperationStatus(
                    status
                        .as_ref()
                        .map(|s| s.to_string())
                        .unwrap_or_else(|| "None".into()),
                )
                .into());
            }
        }
    }
}
