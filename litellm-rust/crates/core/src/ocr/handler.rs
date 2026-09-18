use litellm_callbacks::event::{CallEvent, RawResponse};

use super::OcrClient;
use super::route::OcrHost;
use super::types::{LiteLLMOcrResponse, PreparedOcrRequest, ResolvedOcrRequest};
use crate::llms::base_llm::ocr::transformation::OcrResponseContext;

pub(crate) async fn perform_ocr_request(
    client: &OcrClient,
    request: ResolvedOcrRequest,
    host: &OcrHost,
    caller_document: bool,
) -> Result<LiteLLMOcrResponse, super::Error> {
    request.response_format()?;
    PreparedOcrCall::prepare(client.clone(), request, host, caller_document)
        .await?
        .execute()
        .await
}

pub(crate) struct PreparedOcrCall {
    client: OcrClient,
    request: PreparedOcrRequest,
    http: reqwest::Request,
}

impl PreparedOcrCall {
    pub(crate) async fn prepare(
        client: OcrClient,
        request: ResolvedOcrRequest,
        host: &OcrHost,
        caller_document: bool,
    ) -> Result<Self, super::Error> {
        let request = super::prepare::prepare_request(request, host.clone(), caller_document);
        let http = request.config.prepare_request(&request, &client).await?;
        Ok(Self {
            client,
            request,
            http,
        })
    }

    pub(crate) async fn execute(self) -> Result<LiteLLMOcrResponse, super::Error> {
        let url = self.http.url().to_string();
        let headers = request_headers(&self.http)?;
        let response =
            crate::http_utils::execute_http_request(self.client.provider_http(), self.http)
                .await
                .map_err(super::client::transport_error)?;
        if !response.status().is_success() {
            let headers = response
                .headers()
                .iter()
                .filter_map(|(name, value)| {
                    value
                        .to_str()
                        .ok()
                        .map(|value| (name.to_string(), value.to_string()))
                })
                .collect();
            return match super::client::read_response_bytes(
                response,
                self.request.connection.max_response_bytes,
            )
            .await
            {
                Err(super::Error::Transport(crate::transport::Error::Http { status, body })) => {
                    Err(self.request.config.get_error_class(body, status, headers))
                }
                Err(error) => Err(error),
                Ok(_) => unreachable!("non-success response produces an HTTP error"),
            };
        }
        let model = &self.request.model;
        let context = OcrResponseContext {
            client: &self.client,
            connection: &self.request.connection,
            host: &self.request.host,
            request_format: self.request.response_format()?,
            url: &url,
            headers: &headers,
        };
        self.request
            .config
            .async_transform_ocr_response(model, response, context)
            .await
    }
}

fn request_headers(request: &reqwest::Request) -> Result<Vec<(String, String)>, super::Error> {
    request
        .headers()
        .iter()
        .map(|(name, value)| {
            value
                .to_str()
                .map(|value| (name.to_string(), value.to_string()))
                .map_err(|_| super::Error::RequestField {
                    path: "headers".into(),
                })
        })
        .collect()
}

pub(crate) async fn emit_response_received(
    host: &OcrHost,
    bytes: &[u8],
) -> Result<(), super::Error> {
    host.emit(CallEvent::ResponseReceived {
        raw: RawResponse {
            body: String::from_utf8_lossy(bytes).into_owned(),
        },
    })
    .await
}
