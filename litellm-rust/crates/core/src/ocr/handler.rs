use std::sync::Arc;

use super::OcrClient;
use super::hooks::{OcrHooks, OcrLifecycleHooks, OcrPostCallRequest};
use super::types::{LiteLLMOcrResponse, PreparedOcrRequest, ResolvedOcrRequest};
use crate::call_lifecycle::{CallLifecycle, CallLifecycleContext};
use crate::llms::base_llm::ocr::transformation::OcrResponseContext;

pub(crate) async fn perform_ocr_request(
    client: &OcrClient,
    request: ResolvedOcrRequest,
) -> Result<LiteLLMOcrResponse, super::Error> {
    request.response_format()?;
    let context = CallLifecycleContext::new(
        "ocr",
        request.model.clone(),
        request.provider_name(),
        request
            .litellm_call_id
            .clone()
            .unwrap_or_else(|| format!("ocr-{:032x}", rand::random::<u128>())),
    );
    let hooks = OcrLifecycleHooks {
        hooks: request.hooks.clone(),
        provider_name: context.custom_llm_provider.clone(),
    };
    CallLifecycle::default()
        .run(context, request, &hooks, |request| async move {
            PreparedOcrCall::prepare(client.clone(), request)
                .await?
                .execute()
                .await
        })
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
    ) -> Result<Self, super::Error> {
        let request = super::prepare::prepare_request(request);
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
            hooks: &self.request.hooks,
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

pub(crate) async fn post_call(hooks: &Arc<dyn OcrHooks>, bytes: &[u8]) -> Result<(), super::Error> {
    let original_response = serde_json::Value::String(String::from_utf8_lossy(bytes).into_owned());
    hooks
        .post_call(OcrPostCallRequest { original_response })
        .await?;
    Ok(())
}
