use std::sync::OnceLock;
use std::time::Duration;

use bytes::{Bytes, BytesMut};
use serde::de::DeserializeOwned;

use super::error::{OcrError, OcrResponseError};
use super::types::{LiteLLMOcrRequest, LiteLLMOcrResponse};
use super::wire::{DecodedOcrResponse, decode_response};
use crate::Error;
use crate::auth::vertex::VertexAuth;
use crate::constants::OCR_CONNECT_TIMEOUT_SECS;
use crate::error::TransportError;
use crate::media::MediaFetcher;

#[derive(Clone)]
pub struct OcrClient {
    provider_http: reqwest::Client,
    polling_http: reqwest::Client,
    document_fetcher: MediaFetcher,
    vertex_auth: VertexAuth,
}

impl OcrClient {
    pub fn new(provider_http: reqwest::Client) -> Result<Self, TransportError> {
        let document_fetcher = MediaFetcher::new().map_err(TransportError::from)?;
        Ok(Self {
            provider_http,
            polling_http: no_redirect_http()?,
            document_fetcher,
            vertex_auth: VertexAuth::default(),
        })
    }

    pub fn shared() -> Result<Self, Error> {
        shared_client()
    }

    #[tracing::instrument(
        name = "ocr",
        target = "litellm::function_trace",
        level = "trace",
        skip_all
    )]
    pub async fn perform(&self, request: LiteLLMOcrRequest) -> Result<LiteLLMOcrResponse, Error> {
        use super::{
            NativeOutcome, OcrAdmission, OcrCall, OcrCallStep, OcrHookHost, OcrHost,
            OcrHostOperation, OcrHostResult,
        };

        let host = OcrHookHost::new(request.hooks.clone());
        let mut request = Some(request);
        let NativeOutcome::Completed(mut call) = OcrCall::admit(self.clone(), OcrAdmission::all())
        else {
            return Err(Error::InvalidRequest(
                "native OCR host admission declined".into(),
            ));
        };
        let mut result = None;
        loop {
            match call.resume(result.take()).await? {
                OcrCallStep::Host(OcrHostOperation::ProjectRequest) => {
                    result = Some(OcrHostResult::Request(Ok((
                        Box::new(request.take().ok_or_else(|| {
                            Error::InvalidRequest("OCR request was already projected".into())
                        })?),
                        false,
                    ))))
                }
                OcrCallStep::Host(operation) => result = Some(host.invoke(operation).await),
                OcrCallStep::Complete(response) => return Ok(response),
            }
        }
    }

    pub(crate) fn provider_http(&self) -> &reqwest::Client {
        &self.provider_http
    }

    pub(crate) fn polling_http(&self) -> &reqwest::Client {
        &self.polling_http
    }

    pub(crate) fn document_fetcher(&self) -> &MediaFetcher {
        &self.document_fetcher
    }

    pub(crate) fn vertex_auth(&self) -> &VertexAuth {
        &self.vertex_auth
    }

    #[cfg(test)]
    pub(crate) fn for_test(provider_http: reqwest::Client, document_http: reqwest::Client) -> Self {
        Self {
            provider_http,
            polling_http: no_redirect_http().expect("test polling client builds"),
            document_fetcher: MediaFetcher::for_test(document_http),
            vertex_auth: VertexAuth::default(),
        }
    }
}

fn no_redirect_http() -> Result<reqwest::Client, TransportError> {
    reqwest::Client::builder()
        .connect_timeout(Duration::from_secs(OCR_CONNECT_TIMEOUT_SECS))
        .redirect(reqwest::redirect::Policy::none())
        .build()
        .map_err(TransportError::from)
}

pub(crate) fn shared_client() -> Result<OcrClient, Error> {
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
    Ok(client)
}

pub async fn ocr(request: LiteLLMOcrRequest) -> Result<LiteLLMOcrResponse, Error> {
    shared_client()?.perform(request).await
}

pub async fn read_json_response<T: DeserializeOwned>(
    response: reqwest::Response,
    native: bool,
    max_response_bytes: usize,
) -> Result<DecodedOcrResponse<T>, OcrError> {
    let bytes = read_response_bytes(response, max_response_bytes).await?;
    Ok(decode_response(&bytes, native)?)
}

pub(crate) async fn read_response_bytes(
    mut response: reqwest::Response,
    max_response_bytes: usize,
) -> Result<Bytes, OcrError> {
    let status = response.status();
    let limit = if status.is_success() {
        max_response_bytes
    } else {
        max_response_bytes.min(4 * (crate::constants::UPSTREAM_ERROR_BODY_MAX_CHARS + 1))
    };
    if status.is_success()
        && response
            .content_length()
            .is_some_and(|length| length > limit as u64)
    {
        return Err(OcrResponseError::TooLarge { limit }.into());
    }
    let mut bytes = BytesMut::new();
    while let Some(chunk) = response.chunk().await.map_err(transport_error)? {
        let remaining = limit.saturating_sub(bytes.len());
        if status.is_success() && chunk.len() > remaining {
            return Err(OcrResponseError::TooLarge { limit }.into());
        }
        bytes.extend_from_slice(&chunk[..chunk.len().min(remaining)]);
        if !status.is_success() && bytes.len() == limit {
            break;
        }
    }
    if !status.is_success() {
        return Err(crate::error::TransportError::Http {
            status: status.as_u16(),
            body: crate::http_utils::truncate_error_body(&String::from_utf8_lossy(&bytes)),
        }
        .into());
    }
    Ok(bytes.freeze())
}

pub(crate) fn transport_error(error: reqwest::Error) -> Error {
    if error.is_timeout() {
        return Error::Http {
            status: 408,
            body: "OCR request timed out".into(),
        };
    }
    crate::error::TransportError::from(error).into()
}
