use std::sync::OnceLock;
use std::time::Duration;

use serde::de::DeserializeOwned;

use super::error::OcrError;
use super::handler::perform_ocr_request;
use super::types::{LiteLLMOcrRequest, LiteLLMOcrResponse};
use super::wire::{DecodedOcrResponse, decode_response};
use crate::Error;
use crate::constants::OCR_CONNECT_TIMEOUT_SECS;
use crate::error::TransportError;
use crate::media::MediaFetcher;

#[derive(Clone)]
pub struct OcrClient {
    provider_http: reqwest::Client,
    document_fetcher: MediaFetcher,
}

impl OcrClient {
    pub fn new(provider_http: reqwest::Client) -> Result<Self, TransportError> {
        let document_fetcher = MediaFetcher::new().map_err(TransportError::from)?;
        Ok(Self {
            provider_http,
            document_fetcher,
        })
    }

    #[tracing::instrument(
        name = "ocr",
        target = "litellm::function_trace",
        level = "trace",
        skip_all
    )]
    pub async fn perform(&self, request: LiteLLMOcrRequest) -> Result<LiteLLMOcrResponse, Error> {
        perform_ocr_request(self, request).await
    }

    pub(crate) fn provider_http(&self) -> &reqwest::Client {
        &self.provider_http
    }

    pub(crate) fn document_fetcher(&self) -> &MediaFetcher {
        &self.document_fetcher
    }

    #[cfg(test)]
    pub(crate) fn for_test(provider_http: reqwest::Client, document_http: reqwest::Client) -> Self {
        Self {
            provider_http,
            document_fetcher: MediaFetcher::for_test(document_http),
        }
    }
}

pub async fn ocr(request: LiteLLMOcrRequest) -> Result<LiteLLMOcrResponse, Error> {
    static CLIENT: OnceLock<Result<OcrClient, TransportError>> = OnceLock::new();
    let client = CLIENT
        .get_or_init(|| {
            reqwest::Client::builder()
                .connect_timeout(Duration::from_secs(OCR_CONNECT_TIMEOUT_SECS))
                .build()
                .map_err(TransportError::from)
                .and_then(OcrClient::new)
        })
        .clone()?;
    client.perform(request).await
}

pub async fn read_json_response<T: DeserializeOwned>(
    response: reqwest::Response,
    native: bool,
) -> Result<DecodedOcrResponse<T>, OcrError> {
    let status = response.status();
    let bytes = response
        .bytes()
        .await
        .map_err(crate::error::TransportError::from)?;
    if !status.is_success() {
        return Err(crate::error::TransportError::Http {
            status: status.as_u16(),
            body: crate::http_utils::truncate_error_body(&String::from_utf8_lossy(&bytes)),
        }
        .into());
    }
    Ok(decode_response(&bytes, native)?)
}
