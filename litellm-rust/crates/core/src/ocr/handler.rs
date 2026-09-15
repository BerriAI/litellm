use std::sync::Arc;

use super::OcrClient;
use super::hooks::{OcrHooks, OcrLifecycleHooks, OcrPostCallRequest};
use super::provider_config::OcrConfigKind;
use super::types::{LiteLLMOcrRequest, LiteLLMOcrResponse};
use super::wire::DecodedOcrResponse;
use crate::call_lifecycle::{CallLifecycle, CallLifecycleContext};
use crate::llms::azure_ai::ocr::cohere_parse_transformation::AzureAICohereParseConfig;
use crate::llms::azure_ai::ocr::document_intelligence::AzureDocumentIntelligenceOperation;
use crate::llms::azure_ai::ocr::document_intelligence::transformation::AzureDocumentIntelligenceOCRConfig;
use crate::llms::azure_ai::ocr::transformation::AzureAIOCRConfig;
use crate::llms::base_llm::ocr::transformation::BaseOcrConfig;
use crate::llms::cohere::ocr::CohereResponse;
use crate::llms::cohere::ocr::transformation::CohereParseConfig;
use crate::llms::mistral::ocr::MistralOcrResponse;
use crate::llms::mistral::ocr::transformation::MistralOCRConfig;
use crate::llms::reducto::ocr::ReductoResponse;
use crate::llms::reducto::ocr::transformation::{ReductoParseLegacyConfig, ReductoParseV3Config};
use crate::llms::vertex_ai::ocr::deepseek_transformation::{
    DeepSeekOcrResponse, VertexAIDeepSeekOCRConfig,
};
use crate::llms::vertex_ai::ocr::transformation::VertexAIOCRConfig;
use crate::ocr::Error;

pub(crate) async fn perform_ocr_request(
    client: &OcrClient,
    request: LiteLLMOcrRequest,
) -> Result<LiteLLMOcrResponse, Error> {
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
                .await?
                .normalize()
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
    ) -> Result<Self, Error> {
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

    pub(crate) async fn execute(self) -> Result<OcrProviderResponse, Error> {
        let url = self.http.url().to_string();
        let headers = request_headers(&self.http)?;
        let response =
            crate::http_utils::execute_http_request(self.client.provider_http(), self.http)
                .await
                .map_err(super::client::transport_error)?;
        let data = match self.request.config {
            OcrConfigKind::Cohere => OcrProviderData::Cohere(
                CohereParseConfig
                    .read_response(&self.client, response, &url, &headers, &self.request)
                    .await?,
            ),
            OcrConfigKind::Mistral => OcrProviderData::Mistral(
                MistralOCRConfig
                    .read_response(&self.client, response, &url, &headers, &self.request)
                    .await?,
            ),
            OcrConfigKind::AzureAi => OcrProviderData::AzureAi(
                AzureAIOCRConfig
                    .read_response(&self.client, response, &url, &headers, &self.request)
                    .await?,
            ),
            OcrConfigKind::AzureCohere => OcrProviderData::AzureCohere(
                AzureAICohereParseConfig
                    .read_response(&self.client, response, &url, &headers, &self.request)
                    .await?,
            ),
            OcrConfigKind::AzureDocumentIntelligence => OcrProviderData::AzureDocumentIntelligence(
                AzureDocumentIntelligenceOCRConfig
                    .read_response(&self.client, response, &url, &headers, &self.request)
                    .await?,
            ),
            OcrConfigKind::ReductoLegacy => OcrProviderData::ReductoLegacy(
                ReductoParseLegacyConfig
                    .read_response(&self.client, response, &url, &headers, &self.request)
                    .await?,
            ),
            OcrConfigKind::ReductoV3 => OcrProviderData::ReductoV3(
                ReductoParseV3Config
                    .read_response(&self.client, response, &url, &headers, &self.request)
                    .await?,
            ),
            OcrConfigKind::VertexAi => OcrProviderData::VertexAi(
                VertexAIOCRConfig
                    .read_response(&self.client, response, &url, &headers, &self.request)
                    .await?,
            ),
            OcrConfigKind::VertexDeepSeek => OcrProviderData::VertexDeepSeek(
                VertexAIDeepSeekOCRConfig
                    .read_response(&self.client, response, &url, &headers, &self.request)
                    .await?,
            ),
        };
        Ok(OcrProviderResponse {
            request: self.request,
            data,
        })
    }
}

fn request_headers(request: &reqwest::Request) -> Result<Vec<(String, String)>, Error> {
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

enum OcrProviderData {
    Cohere(DecodedOcrResponse<CohereResponse>),
    Mistral(DecodedOcrResponse<MistralOcrResponse>),
    AzureAi(DecodedOcrResponse<MistralOcrResponse>),
    AzureCohere(DecodedOcrResponse<CohereResponse>),
    AzureDocumentIntelligence(DecodedOcrResponse<AzureDocumentIntelligenceOperation>),
    ReductoLegacy(DecodedOcrResponse<ReductoResponse>),
    ReductoV3(DecodedOcrResponse<ReductoResponse>),
    VertexAi(DecodedOcrResponse<MistralOcrResponse>),
    VertexDeepSeek(DecodedOcrResponse<DeepSeekOcrResponse>),
}

pub(crate) struct OcrProviderResponse {
    request: LiteLLMOcrRequest,
    data: OcrProviderData,
}

impl OcrProviderResponse {
    pub(crate) fn normalize(self) -> Result<LiteLLMOcrResponse, Error> {
        let (response, native) = match self.data {
            OcrProviderData::Cohere(decoded) => (
                CohereParseConfig.transform_ocr_response(&self.request, decoded.data)?,
                decoded.native,
            ),
            OcrProviderData::Mistral(decoded) => (
                MistralOCRConfig.transform_ocr_response(&self.request, decoded.data)?,
                decoded.native,
            ),
            OcrProviderData::AzureAi(decoded) => (
                AzureAIOCRConfig.transform_ocr_response(&self.request, decoded.data)?,
                decoded.native,
            ),
            OcrProviderData::AzureCohere(decoded) => (
                AzureAICohereParseConfig.transform_ocr_response(&self.request, decoded.data)?,
                decoded.native,
            ),
            OcrProviderData::AzureDocumentIntelligence(decoded) => (
                AzureDocumentIntelligenceOCRConfig
                    .transform_ocr_response(&self.request, decoded.data)?,
                decoded.native,
            ),
            OcrProviderData::ReductoLegacy(decoded) => (
                ReductoParseLegacyConfig.transform_ocr_response(&self.request, decoded.data)?,
                decoded.native,
            ),
            OcrProviderData::ReductoV3(decoded) => (
                ReductoParseV3Config.transform_ocr_response(&self.request, decoded.data)?,
                decoded.native,
            ),
            OcrProviderData::VertexAi(decoded) => (
                VertexAIOCRConfig.transform_ocr_response(&self.request, decoded.data)?,
                decoded.native,
            ),
            OcrProviderData::VertexDeepSeek(decoded) => (
                VertexAIDeepSeekOCRConfig.transform_ocr_response(&self.request, decoded.data)?,
                decoded.native,
            ),
        };
        Ok(LiteLLMOcrResponse {
            provider_native_response: native,
            ..response
        })
    }
}

pub(crate) async fn post_call(hooks: &Arc<dyn OcrHooks>, bytes: &[u8]) -> Result<(), Error> {
    let original_response = serde_json::Value::String(String::from_utf8_lossy(bytes).into_owned());
    hooks
        .post_call(OcrPostCallRequest { original_response })
        .await?;
    Ok(())
}
