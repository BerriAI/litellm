use std::sync::OnceLock;
use std::time::Duration;

use super::handler::perform_ocr_request;
use super::types::{OcrRequest, OcrResponseData};
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
    pub fn new(provider_http: reqwest::Client) -> Result<Self, Error> {
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
    pub async fn perform(&self, request: OcrRequest) -> Result<OcrResponseData, Error> {
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

pub async fn ocr(request: OcrRequest) -> Result<OcrResponseData, Error> {
    static CLIENT: OnceLock<Result<OcrClient, String>> = OnceLock::new();
    let client = CLIENT
        .get_or_init(|| {
            reqwest::Client::builder()
                .connect_timeout(Duration::from_secs(OCR_CONNECT_TIMEOUT_SECS))
                .build()
                .map_err(|error| error.to_string())
                .and_then(|provider_http| {
                    OcrClient::new(provider_http).map_err(|error| error.to_string())
                })
        })
        .as_ref()
        .map_err(|error| TransportError::Network(error.clone()))?;
    client.perform(request).await
}
