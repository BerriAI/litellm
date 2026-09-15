use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use super::transformation::VertexAIOCRConfig;
use crate::llms::base_llm::ocr::transformation::{BaseOcrConfig, OcrRequestContext};
use crate::ocr::OcrClient;
use crate::ocr::prepare::{credential_env, transform_request_body};
use crate::ocr::types::{
    LiteLLMOcrRequest, LiteLLMOcrResponse, OcrDocument, OcrPage, OcrPageDimensions, OcrPageImage,
    OcrUsageInfo,
};
use crate::params::OpaqueParams;
use crate::routing_utils::model::{ModelNamespace, ProviderModel, RoutedModel};
use crate::url_utils::ApiUrl;
use litellm_auth_gcp::{self as vertex, VertexConfig};

const DEFAULT_API_BASE: &str = "https://aiplatform.googleapis.com";
const MODEL_NAMESPACE: &str = "deepseek-ai";
const DEFAULT_LOCATION: &str = "us-central1";
const DEEPSEEK_OCR_PARAMS: &[&str] = &["stream", "temperature", "max_tokens", "top_p", "n", "stop"];

pub(crate) type DeepSeekOcrParams = OpaqueParams;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) struct DeepSeekOcrRequest {
    pub model: ProviderModel<DeepSeekAi>,
    pub messages: Vec<DeepSeekOcrMessage>,
    #[serde(flatten)]
    pub params: OpaqueParams,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) struct DeepSeekOcrMessage {
    pub role: UserRole,
    pub content: Vec<DeepSeekDocument>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "type")]
pub(crate) enum DeepSeekDocument {
    #[serde(rename = "image_url")]
    ImageUrl { image_url: String },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub(crate) enum UserRole {
    User,
}

#[derive(Clone, Debug, Deserialize)]
pub(crate) struct DeepSeekOcrResponse {
    #[serde(default)]
    choices: Vec<DeepSeekChoice>,
    #[serde(default = "empty_object")]
    usage: Value,
}

#[derive(Clone, Debug, Deserialize)]
struct DeepSeekChoice {
    #[serde(default)]
    message: DeepSeekResponseMessage,
}

#[derive(Clone, Debug, Default, Deserialize)]
struct DeepSeekResponseMessage {
    content: Option<DeepSeekContent>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(untagged)]
enum DeepSeekContent {
    Text(String),
    Object(Map<String, Value>),
}

#[serde_with::serde_as]
#[derive(Deserialize)]
struct DeepSeekPage {
    #[serde(default)]
    #[serde_as(deserialize_as = "crate::serde_compat::LaxI64")]
    index: i64,
    #[serde(default)]
    markdown: String,
    images: Option<Vec<OcrPageImage>>,
    dimensions: Option<OcrPageDimensions>,
}

#[derive(Clone, Debug)]
pub(crate) struct DeepSeekAi;

impl ModelNamespace for DeepSeekAi {
    const NAME: &'static str = MODEL_NAMESPACE;
}

#[derive(Clone, Debug)]
pub(crate) struct VertexAIDeepSeekOCRConfig;

impl BaseOcrConfig for VertexAIDeepSeekOCRConfig {
    type OcrParams = DeepSeekOcrParams;
    type ProviderRequest = DeepSeekOcrRequest;
    type ProviderResponse = DeepSeekOcrResponse;

    async fn async_transform_ocr_request(
        &self,
        model: &str,
        document: OcrDocument,
        optional_params: &DeepSeekOcrParams,
        headers: &[(String, String)],
        _context: OcrRequestContext<'_>,
    ) -> Result<DeepSeekOcrRequest, crate::ocr::Error> {
        self.transform_ocr_request(model, document, optional_params, headers)
    }

    fn normalize_response(
        &self,
        model: &str,
        response: DeepSeekOcrResponse,
    ) -> Result<LiteLLMOcrResponse, crate::ocr::Error> {
        normalize_response(model, response)
    }
}

impl VertexAIDeepSeekOCRConfig {
    pub(crate) fn transform_ocr_request(
        &self,
        model: &str,
        document: OcrDocument,
        optional_params: &DeepSeekOcrParams,
        _headers: &[(String, String)],
    ) -> Result<DeepSeekOcrRequest, crate::ocr::Error> {
        if document.source().is_empty() {
            return Err(crate::ocr::Error::MissingDocumentUrl);
        }
        Ok(DeepSeekOcrRequest {
            model: provider_model(model)?,
            messages: vec![DeepSeekOcrMessage {
                role: UserRole::User,
                content: vec![DeepSeekDocument::ImageUrl {
                    image_url: document.source().to_string(),
                }],
            }],
            params: optional_params
                .iter()
                .filter(|(name, _)| DEEPSEEK_OCR_PARAMS.contains(&name.as_str()))
                .map(|(name, value)| (name.clone(), value.clone()))
                .collect(),
        })
    }

    pub(crate) async fn prepare_request(
        &self,
        request: &LiteLLMOcrRequest,
        client: &OcrClient,
    ) -> Result<reqwest::Request, crate::ocr::Error> {
        let params = self.map_ocr_params(
            &OpaqueParams::default(),
            &request.optional_params,
            &request.model,
        )?;
        let config = VertexConfig::from_sourced_optional_params(
            &request.optional_params,
            &request.input_sources,
        )
        .map_err(crate::ocr::Error::from)?;
        let authentication = VertexAIOCRConfig
            .validate_environment(&request.connection, &config, client)
            .await?;
        let location = vertex::get_vertex_ai_location(&config, &credential_env)
            .unwrap_or_else(|| DEFAULT_LOCATION.to_string());
        let url = self.get_complete_url(
            request.connection.api_base.as_deref(),
            &authentication.project_id,
            &location,
        )?;
        let body = self
            .async_transform_ocr_request(
                &request.model,
                request.document.clone(),
                &params,
                &authentication.headers,
                OcrRequestContext {
                    client,
                    connection: &request.connection,
                },
            )
            .await?;
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
}

pub(crate) fn normalize_response(
    model: &str,
    response: DeepSeekOcrResponse,
) -> Result<LiteLLMOcrResponse, crate::ocr::Error> {
    let content = response
        .choices
        .into_iter()
        .next()
        .and_then(|choice| choice.message.content)
        .ok_or(crate::ocr::Error::EmptyContent)?;
    let (ocr_data, fallback_markdown) = match content {
        DeepSeekContent::Text(text) if text.is_empty() => {
            return Err(crate::ocr::Error::EmptyContent);
        }
        DeepSeekContent::Text(text) => {
            let parsed = text
                .trim_start()
                .starts_with('{')
                .then(|| serde_json::from_str::<Map<String, Value>>(&text).ok())
                .flatten();
            (parsed.unwrap_or_default(), text)
        }
        DeepSeekContent::Object(data) if data.is_empty() => {
            return Err(crate::ocr::Error::EmptyContent);
        }
        DeepSeekContent::Object(data) => {
            let fallback = if data.contains_key("pages") {
                String::new()
            } else {
                let mut output = Vec::new();
                data.serialize(&mut serde_json::Serializer::with_formatter(
                    &mut output,
                    PythonJsonFormatter,
                ))
                .map_err(|_| response_field("content"))?;
                String::from_utf8(output).map_err(|_| response_field("content"))?
            };
            (data, fallback)
        }
    };
    let has_pages = ocr_data.contains_key("pages");
    let pages = match ocr_data.get("pages") {
        Some(Value::Array(pages)) => pages
            .iter()
            .enumerate()
            .filter(|(_, page)| page.is_object())
            .map(|(position, page)| {
                let page: DeepSeekPage = crate::ocr::wire::decode_response_value(
                    page.clone(),
                    &format!("choices[0].message.content.pages[{position}]"),
                )?;
                Ok(OcrPage {
                    index: page.index,
                    markdown: page.markdown,
                    images: page.images,
                    dimensions: page.dimensions,
                    ..Default::default()
                })
            })
            .collect::<Result<Vec<_>, crate::ocr::Error>>()?,
        Some(_) => return Err(response_field("pages")),
        None => Vec::new(),
    };
    let usage = ocr_data
        .get("usage_info")
        .or_else(|| (!has_pages).then_some(&response.usage));
    let usage_info: Option<OcrUsageInfo> = usage
        .filter(|usage| usage.is_object())
        .map(|usage| crate::ocr::wire::decode_response_value(usage.clone(), "usage_info"))
        .transpose()?;
    let model = match ocr_data.get("model") {
        Some(Value::String(model)) => model.clone(),
        Some(_) => return Err(response_field("model")),
        None => model.to_string(),
    };
    Ok(LiteLLMOcrResponse {
        document_annotation: has_pages
            .then(|| ocr_data.get("document_annotation").cloned())
            .flatten(),
        usage_info,
        ..LiteLLMOcrResponse::new(
            model,
            if pages.is_empty() {
                vec![OcrPage {
                    markdown: fallback_markdown,
                    ..Default::default()
                }]
            } else {
                pages
            },
        )
    })
}

fn empty_object() -> Value {
    Value::Object(Map::new())
}

struct PythonJsonFormatter;

impl serde_json::ser::Formatter for PythonJsonFormatter {
    fn begin_array_value<W: std::io::Write + ?Sized>(
        &mut self,
        writer: &mut W,
        first: bool,
    ) -> std::io::Result<()> {
        if first {
            Ok(())
        } else {
            writer.write_all(b", ")
        }
    }

    fn begin_object_key<W: std::io::Write + ?Sized>(
        &mut self,
        writer: &mut W,
        first: bool,
    ) -> std::io::Result<()> {
        if first {
            Ok(())
        } else {
            writer.write_all(b", ")
        }
    }

    fn begin_object_value<W: std::io::Write + ?Sized>(
        &mut self,
        writer: &mut W,
    ) -> std::io::Result<()> {
        writer.write_all(b": ")
    }

    fn write_string_fragment<W: std::io::Write + ?Sized>(
        &mut self,
        writer: &mut W,
        fragment: &str,
    ) -> std::io::Result<()> {
        for character in fragment.chars() {
            if character.is_ascii() && character != '\u{7f}' {
                writer.write_all(&[character as u8])?;
            } else {
                for unit in character.encode_utf16(&mut [0; 2]) {
                    write!(writer, "\\u{unit:04x}")?;
                }
            }
        }
        Ok(())
    }
}

fn response_field(field: &str) -> crate::ocr::Error {
    crate::ocr::Error::ResponseField {
        path: format!("choices[0].message.content.{field}"),
    }
}

pub(crate) fn provider_model(model: &str) -> Result<ProviderModel<DeepSeekAi>, crate::ocr::Error> {
    RoutedModel::new(model)
        .and_then(RoutedModel::into_provider::<DeepSeekAi>)
        .map_err(|_| crate::ocr::Error::RequestField {
            path: "model".into(),
        })
}

impl VertexAIDeepSeekOCRConfig {
    fn get_complete_url(
        &self,
        api_base: Option<&str>,
        project: &str,
        location: &str,
    ) -> Result<String, crate::ocr::Error> {
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
            .map_err(|_| crate::ocr::Error::RequestField {
                path: "api_base".into(),
            })
    }
}

#[cfg(test)]
mod tests {
    use super::{VertexAIDeepSeekOCRConfig, provider_model};

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
            VertexAIDeepSeekOCRConfig
                .get_complete_url(None, "proj-1", "europe-west4")
                .unwrap(),
            "https://aiplatform.googleapis.com/v1/projects/proj-1/locations/europe-west4/endpoints/openapi/chat/completions"
        );
    }
}
