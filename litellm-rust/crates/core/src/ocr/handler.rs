use std::sync::Arc;

use super::OcrClient;
use super::hooks::{OcrHooks, OcrLifecycleHooks, OcrPostCallRequest};
use super::provider_config::OcrConfigKind;
use super::types::{LiteLLMOcrRequest, LiteLLMOcrResponse};
use crate::call_lifecycle::{CallLifecycle, CallLifecycleContext};
use crate::llms::azure_ai::ocr::cohere_parse_transformation::AzureAICohereParseConfig;
use crate::llms::azure_ai::ocr::document_intelligence::transformation::AzureDocumentIntelligenceOCRConfig;
use crate::llms::azure_ai::ocr::transformation::AzureAIOCRConfig;
use crate::llms::base_llm::ocr::transformation::{BaseOcrConfig, OcrResponseContext};
use crate::llms::cohere::ocr::transformation::CohereParseConfig;
use crate::llms::mistral::ocr::transformation::MistralOCRConfig;
use crate::llms::reducto::ocr::transformation::{ReductoParseLegacyConfig, ReductoParseV3Config};
use crate::llms::vertex_ai::ocr::deepseek_transformation::VertexAIDeepSeekOCRConfig;
use crate::llms::vertex_ai::ocr::transformation::VertexAIOCRConfig;

pub(crate) async fn perform_ocr_request(
    client: &OcrClient,
    request: LiteLLMOcrRequest,
) -> Result<LiteLLMOcrResponse, super::Error> {
    request.response_format()?;
    let context = CallLifecycleContext::new(
        "ocr",
        request.model.clone(),
        request.config.provider().as_str(),
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
    request: LiteLLMOcrRequest,
    http: reqwest::Request,
}

impl PreparedOcrCall {
    pub(crate) async fn prepare(
        client: OcrClient,
        request: LiteLLMOcrRequest,
    ) -> Result<Self, super::Error> {
        let http = match request.config {
            OcrConfigKind::Cohere => CohereParseConfig.prepare_request(&request, &client).await?,
            OcrConfigKind::Mistral => MistralOCRConfig.prepare_request(&request, &client).await?,
            OcrConfigKind::AzureAi => AzureAIOCRConfig.prepare_request(&request, &client).await?,
            OcrConfigKind::AzureCohere => {
                AzureAICohereParseConfig
                    .prepare_request(&request, &client)
                    .await?
            }
            OcrConfigKind::AzureDocumentIntelligence => {
                AzureDocumentIntelligenceOCRConfig
                    .prepare_request(&request, &client)
                    .await?
            }
            OcrConfigKind::ReductoLegacy => {
                ReductoParseLegacyConfig
                    .prepare_request(&request, &client)
                    .await?
            }
            OcrConfigKind::ReductoV3 => {
                ReductoParseV3Config
                    .prepare_request(&request, &client)
                    .await?
            }
            OcrConfigKind::VertexAi => VertexAIOCRConfig.prepare_request(&request, &client).await?,
            OcrConfigKind::VertexDeepSeek => {
                VertexAIDeepSeekOCRConfig
                    .prepare_request(&request, &client)
                    .await?
            }
        };
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
        let model = &self.request.model;
        let context = OcrResponseContext {
            client: &self.client,
            connection: &self.request.connection,
            hooks: &self.request.hooks,
            request_format: self.request.response_format()?,
            url: &url,
            headers: &headers,
        };
        match self.request.config {
            OcrConfigKind::Cohere => {
                CohereParseConfig
                    .async_transform_ocr_response(model, response, context)
                    .await
            }
            OcrConfigKind::Mistral => {
                MistralOCRConfig
                    .async_transform_ocr_response(model, response, context)
                    .await
            }
            OcrConfigKind::AzureAi => {
                AzureAIOCRConfig
                    .async_transform_ocr_response(model, response, context)
                    .await
            }
            OcrConfigKind::AzureCohere => {
                AzureAICohereParseConfig
                    .async_transform_ocr_response(model, response, context)
                    .await
            }
            OcrConfigKind::AzureDocumentIntelligence => {
                AzureDocumentIntelligenceOCRConfig
                    .async_transform_ocr_response(model, response, context)
                    .await
            }
            OcrConfigKind::ReductoLegacy => {
                ReductoParseLegacyConfig
                    .async_transform_ocr_response(model, response, context)
                    .await
            }
            OcrConfigKind::ReductoV3 => {
                ReductoParseV3Config
                    .async_transform_ocr_response(model, response, context)
                    .await
            }
            OcrConfigKind::VertexAi => {
                VertexAIOCRConfig
                    .async_transform_ocr_response(model, response, context)
                    .await
            }
            OcrConfigKind::VertexDeepSeek => {
                VertexAIDeepSeekOCRConfig
                    .async_transform_ocr_response(model, response, context)
                    .await
            }
        }
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
