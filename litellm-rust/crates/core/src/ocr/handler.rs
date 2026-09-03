use super::client::http_client;
use super::transformation::OcrResponseHandling;
use super::types::ProviderOcrRequest;
use crate::Error;
use crate::constants::{AZURE_DOCUMENT_INTELLIGENCE_POLL_TIMEOUT_SECS, OCR_TIMEOUT_SECS};
use crate::http_utils::body::PreparedJsonBody;
use crate::http_utils::replay::{replay_client, send_json};
use crate::http_utils::truncate_error_body;
use serde_json::Value;
use std::time::{Duration, Instant};

pub async fn execute_ocr_provider_call(request: ProviderOcrRequest) -> Result<Value, Error> {
    let started = Instant::now();
    let body = PreparedJsonBody::new(request.body)?;
    let response = send_json(
        if body.is_streamed() {
            replay_client()?
        } else {
            http_client()?
        },
        &request.url,
        &body,
        &request.upstream_headers,
        request
            .timeout
            .unwrap_or(Duration::from_secs(OCR_TIMEOUT_SECS)),
        None,
    )
    .await?;
    let status = response.status();
    if request.config.response_handling() == OcrResponseHandling::AzureDocumentIntelligencePoll
        && status.as_u16() == 202
    {
        let operation_url = response
            .headers()
            .get("operation-location")
            .and_then(|value| value.to_str().ok())
            .map(str::to_string)
            .ok_or_else(|| {
                Error::InvalidResponse(
                    "Azure Document Intelligence returned 202 but no Operation-Location header found"
                        .to_string(),
                )
            })?;
        let response_json = poll_document_intelligence(
            &operation_url,
            &request.url,
            &request.upstream_headers,
            request
                .timeout
                .map(|timeout| timeout.saturating_sub(started.elapsed())),
        )
        .await?;
        return Ok(request
            .config
            .transform_ocr_response(&request.model, response_json)?
            .into_json());
    }

    let text = response
        .text()
        .await
        .map_err(|err| Error::Network(err.to_string()))?;

    if !status.is_success() {
        return Err(Error::Http {
            status: status.as_u16(),
            body: truncate_error_body(&text),
        });
    }

    let response_json: Value = serde_json::from_str(&text)
        .map_err(|err| Error::InvalidResponse(format!("invalid OCR response JSON: {err}")))?;

    Ok(request
        .config
        .transform_ocr_response(&request.model, response_json)?
        .into_json())
}

fn same_origin(left: &str, right: &str) -> bool {
    let Ok(left) = reqwest::Url::parse(left) else {
        return false;
    };
    let Ok(right) = reqwest::Url::parse(right) else {
        return false;
    };
    left.scheme() == right.scheme()
        && left.host_str() == right.host_str()
        && left.port_or_known_default() == right.port_or_known_default()
}

fn retry_after_secs(response: &reqwest::Response) -> u64 {
    response
        .headers()
        .get(reqwest::header::RETRY_AFTER)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or(2)
}

fn operation_status(response_json: &Value) -> Result<&str, Error> {
    let status = response_json
        .get("status")
        .and_then(Value::as_str)
        .ok_or(Error::MissingField("status"))?;
    match status {
        "succeeded" => Ok("succeeded"),
        "running" | "notStarted" => Ok("running"),
        "failed" => {
            let message = response_json
                .get("error")
                .and_then(|error| error.get("message"))
                .and_then(Value::as_str)
                .unwrap_or("Unknown error");
            Err(Error::InvalidResponse(format!(
                "Azure Document Intelligence analysis failed: {message}"
            )))
        }
        other => Err(Error::InvalidResponse(format!(
            "Unknown operation status: {other}"
        ))),
    }
}

async fn poll_document_intelligence(
    operation_url: &str,
    original_url: &str,
    headers: &[(String, String)],
    timeout: Option<Duration>,
) -> Result<Value, Error> {
    if !same_origin(operation_url, original_url) {
        return Err(Error::InvalidResponse(
            "Azure Document Intelligence: rejected cross-origin polling URL".to_string(),
        ));
    }

    let start = Instant::now();
    let timeout = timeout.unwrap_or(Duration::from_secs(
        AZURE_DOCUMENT_INTELLIGENCE_POLL_TIMEOUT_SECS,
    ));
    loop {
        if start.elapsed() > timeout {
            return Err(Error::Network(format!(
                "Azure Document Intelligence operation polling timed out after {} seconds",
                timeout.as_secs()
            )));
        }

        let mut request_builder = http_client()?.get(operation_url);
        for (key, value) in headers {
            if key.eq_ignore_ascii_case("ocp-apim-subscription-key") {
                request_builder = request_builder.header(key, value);
            }
        }
        let remaining = timeout
            .checked_sub(start.elapsed())
            .ok_or_else(|| Error::Network("OCR polling timed out".into()))?;
        let response = request_builder
            .timeout(remaining)
            .send()
            .await
            .map_err(|err| Error::Network(err.to_string()))?;
        let retry_after = retry_after_secs(&response);
        let status = response.status();
        let text = response
            .text()
            .await
            .map_err(|err| Error::Network(err.to_string()))?;
        if !status.is_success() {
            return Err(Error::Http {
                status: status.as_u16(),
                body: truncate_error_body(&text),
            });
        }
        let response_json: Value = serde_json::from_str(&text).map_err(|err| {
            Error::InvalidResponse(format!("invalid Azure DI poll response JSON: {err}"))
        })?;
        if operation_status(&response_json)? == "succeeded" {
            return Ok(response_json);
        }
        tokio::time::sleep(
            Duration::from_secs(retry_after).min(timeout.saturating_sub(start.elapsed())),
        )
        .await;
    }
}
