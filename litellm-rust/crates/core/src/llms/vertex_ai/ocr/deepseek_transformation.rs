mod types {
    use serde::{Deserialize, Serialize};
    use serde_json::{Map, Value};

    use super::DeepSeekAi;
    use crate::routing_utils::model::ProviderModel;

    #[derive(Clone, Debug, Default, Serialize, Deserialize)]
    pub(crate) struct DeepSeekOcrParams {
        #[serde(skip_serializing_if = "Option::is_none")]
        pub stream: Option<bool>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub temperature: Option<f64>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub max_tokens: Option<i64>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub top_p: Option<f64>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub n: Option<i64>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub stop: Option<StopSequences>,
    }

    #[derive(Clone, Debug, Serialize, Deserialize)]
    #[serde(untagged)]
    pub(crate) enum StopSequences {
        One(String),
        Many(Vec<String>),
    }

    #[derive(Clone, Debug, Serialize, Deserialize)]
    pub(crate) struct DeepSeekOcrRequest {
        pub model: ProviderModel<DeepSeekAi>,
        pub messages: Vec<DeepSeekOcrMessage>,
        #[serde(flatten)]
        pub params: DeepSeekOcrParams,
    }

    #[derive(Clone, Debug, Serialize, Deserialize)]
    pub(crate) struct DeepSeekOcrMessage {
        pub role: UserRole,
        pub content: Vec<crate::ocr::types::OcrDocument>,
        #[serde(default, flatten)]
        pub extra: crate::params::OpaqueParams,
    }

    #[derive(Clone, Debug, Serialize, Deserialize)]
    #[serde(rename_all = "lowercase")]
    pub(crate) enum UserRole {
        User,
    }

    #[derive(Clone, Debug, Deserialize)]
    pub(crate) struct DeepSeekOcrResponse {
        #[serde(default)]
        pub choices: Vec<DeepSeekChoice>,
        pub usage: Option<Value>,
    }

    #[derive(Clone, Debug, Deserialize)]
    pub(crate) struct DeepSeekChoice {
        pub message: DeepSeekResponseMessage,
    }

    #[derive(Clone, Debug, Deserialize)]
    pub(crate) struct DeepSeekResponseMessage {
        pub content: Option<DeepSeekContent>,
    }

    #[derive(Clone, Debug, Deserialize)]
    #[serde(untagged)]
    pub(crate) enum DeepSeekContent {
        Text(String),
        Object(DeepSeekOcrResult),
    }

    #[derive(Clone, Debug, Default, Serialize, Deserialize)]
    pub(crate) struct DeepSeekOcrResult {
        #[serde(skip_serializing_if = "Option::is_none")]
        pub pages: Option<Vec<DeepSeekPage>>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub model: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub usage_info: Option<Value>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub document_annotation: Option<Value>,
        #[serde(flatten)]
        pub extra_fields: Map<String, Value>,
    }

    #[derive(Clone, Debug, Serialize, Deserialize)]
    pub(crate) struct DeepSeekPage {
        #[serde(default)]
        pub index: i64,
        #[serde(default)]
        pub markdown: String,
        pub images: Option<Value>,
        pub dimensions: Option<Value>,
        #[serde(flatten)]
        pub extra_fields: Map<String, Value>,
    }
}

pub(crate) use types::DeepSeekOcrParams;
pub(crate) use types::DeepSeekOcrResponse;

mod mapping {
    use serde::de::IntoDeserializer;
    use serde_json::{Value, json};

    use super::DeepSeekAi;
    use super::types::*;
    use crate::ocr::Error;
    use crate::ocr::types::{LiteLLMOcrResponse, OcrDocument};
    use crate::routing_utils::model::ProviderModel;

    #[tracing::instrument(target = "litellm::function_trace", level = "trace", skip_all)]
    pub(crate) fn transform_ocr_request(
        provider_model: ProviderModel<DeepSeekAi>,
        document: OcrDocument,
        params: &DeepSeekOcrParams,
    ) -> Result<DeepSeekOcrRequest, Error> {
        if document.source().is_empty() {
            return Err(Error::MissingDocumentUrl);
        }
        let content = OcrDocument::ImageUrl {
            image_url: document.source().to_string(),
            extra_fields: serde_json::Map::new(),
        };
        Ok(DeepSeekOcrRequest {
            model: provider_model,
            messages: vec![DeepSeekOcrMessage {
                role: UserRole::User,
                content: vec![content],
                extra: Default::default(),
            }],
            params: params.clone(),
        })
    }

    pub(crate) fn transform_ocr_response(
        model: &str,
        response: DeepSeekOcrResponse,
    ) -> Result<LiteLLMOcrResponse, Error> {
        let content = response
            .choices
            .into_iter()
            .next()
            .and_then(|choice| choice.message.content)
            .ok_or(Error::EmptyContent)?;
        let decoded = decode_content(content)?;
        let pages = match decoded.result.pages {
            Some(pages) if !pages.is_empty() => pages
                .into_iter()
                .map(|page| serde_json::to_value(page).expect("DeepSeek page serializes"))
                .collect(),
            _ => vec![json!({
                "index":0,
                "markdown":decoded.fallback_markdown,
                "images":null
            })],
        };
        Ok(LiteLLMOcrResponse {
            pages,
            model: decoded.result.model.unwrap_or_else(|| model.to_string()),
            document_annotation: decoded.result.document_annotation,
            usage_info: decoded.result.usage_info.or(response.usage),
            object: "ocr".into(),
            extra_fields: decoded.result.extra_fields,
            provider_native_response: None,
        })
    }

    struct DecodedContent {
        result: DeepSeekOcrResult,
        fallback_markdown: String,
    }

    fn decode_content(content: DeepSeekContent) -> Result<DecodedContent, Error> {
        let (result, fallback_markdown) = match content {
            DeepSeekContent::Text(text) if text.is_empty() => {
                return Err(Error::EmptyContent);
            }
            DeepSeekContent::Text(text) => (decode_json_content(&text)?, text),
            DeepSeekContent::Object(object) => {
                let fallback =
                    serde_json::to_string(&object).map_err(|_| Error::ResponseField {
                        path: "choices[0].message.content".into(),
                    })?;
                (Some(object), fallback)
            }
        };
        Ok(DecodedContent {
            result: result.unwrap_or_default(),
            fallback_markdown,
        })
    }

    fn decode_json_content(text: &str) -> Result<Option<DeepSeekOcrResult>, Error> {
        if !text.trim_start().starts_with('{') {
            return Ok(None);
        }
        let value = match serde_json::from_str::<Value>(text) {
            Ok(value) => value,
            Err(_) => return Ok(None),
        };
        serde_path_to_error::deserialize(value.into_deserializer())
            .map(Some)
            .map_err(|error| Error::ResponseField {
                path: format!("choices[0].message.content.{}", error.path()),
            })
    }
}

#[cfg(test)]
pub(crate) use mapping::{transform_ocr_request, transform_ocr_response};

use super::common_utils::validate_destination;
use crate::llms::base_llm::ocr::transformation::BaseOcrConfig;
use crate::ocr::Error;
use crate::ocr::OcrClient;
use crate::ocr::prepare::{
    _prepare_ocr_request, ParsedProviderParams, credential_env, transform_request_body,
};
use crate::ocr::types::{LiteLLMOcrRequest, LiteLLMOcrResponse};
use crate::routing_utils::model::{ModelNamespace, ProviderModel, RoutedModel};
use crate::url_utils::ApiUrl;
use litellm_auth_gcp::{self as vertex, VertexConfig};
const DEFAULT_API_BASE: &str = "https://aiplatform.googleapis.com";
const MODEL_NAMESPACE: &str = "deepseek-ai";
const DEFAULT_LOCATION: &str = "us-central1";

#[derive(Clone, Debug)]
pub(crate) struct DeepSeekAi;

impl ModelNamespace for DeepSeekAi {
    const NAME: &'static str = MODEL_NAMESPACE;
}

#[derive(Clone, Debug)]
pub(crate) struct VertexAIDeepSeekOCRConfig;

impl BaseOcrConfig for VertexAIDeepSeekOCRConfig {
    type ProviderResponse = DeepSeekOcrResponse;

    fn get_supported_ocr_params(&self, _model: &str) -> &'static [&'static str] {
        &["stream", "temperature", "max_tokens", "top_p", "n", "stop"]
    }

    async fn prepare_request(
        &self,
        request: &LiteLLMOcrRequest,
        client: &OcrClient,
    ) -> Result<reqwest::Request, Error> {
        validate_destination(&request.connection)?;
        let ParsedProviderParams {
            known: params,
            extra_params: _extra_params,
        } = _prepare_ocr_request::<DeepSeekOcrParams>(request)?;
        let config = VertexConfig::from_sourced_optional_params(
            &request.optional_params,
            &request.input_sources,
        )
        .map_err(Error::from)?;
        let authentication = client
            .vertex_auth()
            .validate_environment(
                request.connection.extra_headers.clone(),
                request.connection.api_key.as_deref(),
                &config,
                &credential_env,
            )
            .await
            .map_err(Error::from)?;
        let location = vertex::get_vertex_ai_location(&config, &credential_env)
            .unwrap_or_else(|| DEFAULT_LOCATION.to_string());
        let url = get_complete_url(
            request.connection.api_base.as_deref(),
            &authentication.project_id,
            &location,
        )?;
        let document = request.document.clone();
        let body =
            mapping::transform_ocr_request(provider_model(&request.model)?, document, &params)?;
        transform_request_body(
            client,
            request,
            &url,
            &authentication.headers,
            false,
            body,
            |_| Ok(()),
        )
        .await
    }

    fn transform_ocr_response(
        &self,
        request: &LiteLLMOcrRequest,
        response: DeepSeekOcrResponse,
    ) -> Result<LiteLLMOcrResponse, Error> {
        mapping::transform_ocr_response(&request.model, response)
    }
}

pub(crate) fn provider_model(model: &str) -> Result<ProviderModel<DeepSeekAi>, Error> {
    RoutedModel::new(model)
        .and_then(RoutedModel::into_provider::<DeepSeekAi>)
        .map_err(|_| Error::RequestField {
            path: "model".into(),
        })
}

fn get_complete_url(
    api_base: Option<&str>,
    project: &str,
    location: &str,
) -> Result<String, Error> {
    let base = api_base
        .map(str::trim)
        .filter(|base| !base.is_empty())
        .unwrap_or(DEFAULT_API_BASE);
    ApiUrl::parse(base)
        .and_then(|url| {
            url.complete_path(&[
                "v1",
                "projects",
                project,
                "locations",
                location,
                "endpoints",
                "openapi",
                "chat",
                "completions",
            ])
        })
        .map(|url| url.into_string())
        .map_err(|_| Error::RequestField {
            path: "api_base".into(),
        })
}

#[cfg(test)]
mod tests {
    use super::{get_complete_url, provider_model};

    #[test]
    fn config_owns_model_namespace_and_endpoint() {
        assert_eq!(
            provider_model("deepseek-ocr-maas").unwrap().as_str(),
            "deepseek-ai/deepseek-ocr-maas"
        );
        assert_eq!(
            provider_model("deepseek-ai/deepseek-ocr-maas")
                .unwrap()
                .as_str(),
            "deepseek-ai/deepseek-ocr-maas"
        );
        assert_eq!(
            get_complete_url(None, "proj-1", "europe-west4").unwrap(),
            "https://aiplatform.googleapis.com/v1/projects/proj-1/locations/europe-west4/endpoints/openapi/chat/completions"
        );
    }
}
