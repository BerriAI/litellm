use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use litellm_auth_gcp::{self as vertex, VertexConfig};

use super::transformation::VertexAIOCRConfig;
use crate::call_arguments::CallArguments;
use crate::llms::base_llm::ocr::transformation::{BaseOcrConfig, OcrRequestContext};
use crate::ocr::OcrClient;
use crate::ocr::prepare::credential_env;
use crate::ocr::types::{
    LiteLLMOcrResponse, OcrDocument, OcrPage, OcrPageDimensions, OcrPageImage, OcrUsageInfo,
    PreparedOcrRequest,
};
use crate::params::OpaqueParams;
use crate::providers::model::{ModelNamespace, ProviderModel, RoutedModel};
use crate::url_utils::ApiUrl;

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
    type Environment = vertex::VertexEnvironment;

    fn get_api_key_env_var(&self) -> Option<&'static str> {
        VertexAIOCRConfig.get_api_key_env_var()
    }

    fn map_ocr_params(
        &self,
        _arguments: &CallArguments,
        _model: &str,
    ) -> Result<DeepSeekOcrParams, crate::ocr::Error> {
        Ok(DeepSeekOcrParams::default())
    }

    async fn validate_environment(
        &self,
        request: &PreparedOcrRequest,
        client: &OcrClient,
    ) -> Result<Self::Environment, crate::ocr::Error> {
        BaseOcrConfig::validate_environment(&VertexAIOCRConfig, request, client).await
    }

    fn get_complete_url(
        &self,
        request: &PreparedOcrRequest,
        _params: &Self::OcrParams,
        environment: &Self::Environment,
    ) -> Result<String, crate::ocr::Error> {
        let config = VertexConfig::from_sourced_optional_params(
            &request.optional_params,
            &request.input_sources,
        )?;
        let location = vertex::get_vertex_ai_location(&config, &credential_env)
            .unwrap_or_else(|| DEFAULT_LOCATION.to_string());
        self.get_complete_url(
            request.connection.api_base.as_deref(),
            &environment.project_id,
            &location,
        )
    }

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

    fn transform_ocr_response(
        &self,
        model: &str,
        raw_response: &[u8],
        request_format: crate::ocr::types::OcrResponseFormat,
    ) -> Result<LiteLLMOcrResponse, crate::ocr::Error> {
        crate::llms::base_llm::ocr::transformation::decode_and_normalize_response(
            model,
            raw_response,
            request_format,
            normalize_response,
        )
    }

    fn transform_ocr_request(
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
                let page: DeepSeekPage = crate::ocr::json::decode_response_value(
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
        .map(|usage| crate::ocr::json::decode_response_value(usage.clone(), "usage_info"))
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
    use super::{
        DeepSeekOcrParams, DeepSeekOcrResponse, VertexAIDeepSeekOCRConfig, normalize_response,
        provider_model,
    };
    use serde_json::{Value, json};

    #[test]
    fn unconsumed_options_remain_available_for_body_composition() {
        use crate::llms::base_llm::ocr::transformation::BaseOcrConfig;
        use serde_json::json;

        let arguments =
            serde_json::from_value(json!({"temperature":0.5,"extension":null})).unwrap();
        assert_eq!(
            serde_json::to_value(
                VertexAIDeepSeekOCRConfig
                    .map_ocr_params(&arguments, "deepseek-ocr")
                    .unwrap()
            )
            .unwrap(),
            json!({})
        );
        assert_eq!(
            crate::call_arguments::compose_body(&arguments, &json!({"model":"deepseek-ocr"}), &[])
                .unwrap(),
            json!({"model":"deepseek-ocr","temperature":0.5,"extension":null})
        );
    }

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

    use rstest::rstest;

    use crate::llms::base_llm::ocr::transformation::BaseOcrConfig;
    use crate::ocr::types::OcrDocument;

    fn document() -> OcrDocument {
        serde_json::from_value(json!({"type":"image_url","image_url":"gs://bucket/a.png"})).unwrap()
    }

    #[rstest]
    #[case("stream", json!(true))]
    #[case("temperature", json!(0.1))]
    #[case("max_tokens", json!(1024))]
    #[case("top_p", json!(0.9))]
    #[case("n", json!(2))]
    #[case("stop", json!("done"))]
    #[case("stop", json!(["done", "stop"]))]
    #[case("temperature", json!(null))]
    fn request_mapping_matches_python(#[case] name: &str, #[case] value: Value) {
        let params: DeepSeekOcrParams =
            serde_json::from_value(json!({name: value.clone(), "ignored": true})).unwrap();
        let result = serde_json::to_value(
            VertexAIDeepSeekOCRConfig
                .transform_ocr_request("deepseek-ai/deepseek-ocr-maas", document(), &params, &[])
                .unwrap(),
        )
        .unwrap();
        assert_eq!(result["model"], "deepseek-ai/deepseek-ocr-maas");
        assert_eq!(
            result["messages"][0]["content"][0],
            json!({"type":"image_url","image_url":"gs://bucket/a.png"})
        );
        assert_eq!(result[name], value);
        assert!(result.get("ignored").is_none());
    }

    #[rstest]
    #[case(json!({"type":"image_url","image_url":"data:image/png;base64,AA=="}))]
    #[case(json!({"type":"document_url","document_url":"data:application/pdf;base64,AA=="}))]
    fn request_maps_both_document_types_to_image_content(#[case] document: Value) {
        let source = document
            .get("image_url")
            .or_else(|| document.get("document_url"))
            .unwrap()
            .clone();
        let request = VertexAIDeepSeekOCRConfig
            .transform_ocr_request(
                "deepseek-ai/deepseek-ocr-maas",
                serde_json::from_value(document).unwrap(),
                &DeepSeekOcrParams::default(),
                &[],
            )
            .unwrap();
        let result = serde_json::to_value(request).unwrap();
        assert_eq!(
            result["messages"][0]["content"][0],
            json!({"type":"image_url","image_url":source})
        );
    }

    #[rstest]
    #[case(json!("# hello"), "# hello")]
    #[case(json!("{broken"), "{broken")]
    #[case(json!(" {\"pages\":[]} "), " {\"pages\":[]} ")]
    #[case(json!({"pages":[]}), "")]
    #[case(json!("[]"), "[]")]
    #[case(json!("{\"pages\":[{\"markdown\":\"json text\"}]}"), "json text")]
    #[case(json!({"pages":[{"markdown":"object"}]}), "object")]
    fn response_transform_handles_text_json_and_objects(
        #[case] content: Value,
        #[case] expected: &str,
    ) {
        let has_pages = content
            .as_object()
            .is_some_and(|data| data.contains_key("pages"))
            || content
                .as_str()
                .is_some_and(|text| text.contains("\"pages\""));
        let response: DeepSeekOcrResponse = serde_json::from_value(
            json!({"choices":[{"message":{"content":content}}],"usage":{"prompt_tokens":1}}),
        )
        .unwrap();
        let result = normalize_response("model", response).unwrap().into_json();
        assert_eq!(result["pages"][0]["markdown"], expected);
        assert_eq!(result["pages"][0]["index"], 0);
        if has_pages {
            assert!(result["usage_info"].is_null());
        } else {
            assert_eq!(result["usage_info"]["prompt_tokens"], 1);
        }
    }

    #[test]
    fn structured_result_maps_pages_usage_model_and_annotation() {
        let response: DeepSeekOcrResponse = serde_json::from_value(json!({
            "choices":[{"message":{"content":{
                "pages":[{"index":2,"markdown":"page","images":[{"id":"one"}],"dimensions":{"width":10}}],
                "model":"provider-model",
                "usage_info":{"pages_processed":1},
                "document_annotation":{"language":"en"},
                "future":"kept"
            }}}]
        }))
        .unwrap();
        let result = normalize_response("requested", response)
            .unwrap()
            .into_json();
        assert_eq!(result["pages"][0]["index"], 2);
        assert_eq!(result["pages"][0]["images"][0]["id"], "one");
        assert_eq!(result["model"], "provider-model");
        assert_eq!(result["usage_info"]["pages_processed"], 1);
        assert_eq!(result["document_annotation"]["language"], "en");
        assert!(result.get("future").is_none());
    }

    #[test]
    fn response_transform_rejects_missing_empty_and_malformed_content() {
        for value in [
            json!({"choices":[]}),
            json!({"choices":[{"message":{"content":{}}}]}),
            json!({"choices":[{"message":{"content":""}}]}),
            json!({"choices":[{"message":{"content":"{\"pages\":[{\"markdown\":42}]}"}}]}),
            json!({"choices":[{"message":{"content":{"pages":[{"markdown":42}]}}}]}),
        ] {
            let result = serde_json::from_value::<DeepSeekOcrResponse>(value)
                .map_err(|_| ())
                .and_then(|response| normalize_response("model", response).map_err(|_| ()));
            assert!(result.is_err());
        }
    }

    #[test]
    fn structured_content_preserves_usage_presence_and_shared_page_defaults() {
        for (usage, expected) in [(json!(null), None), (json!({"pages_processed":2}), Some(2))] {
            let response = serde_json::from_value(json!({
                "choices":[{"message":{"content":{
                    "pages":[42, {"index":"2", "images":[{"id":"kept"}], "ignored":true}],
                    "usage_info":usage
                }}}],
                "usage":{"pages_processed":99}
            }))
            .unwrap();
            let normalized = normalize_response("model", response).unwrap();
            assert_eq!(normalized.pages.len(), 1);
            assert_eq!(normalized.pages[0].index, 2);
            assert_eq!(normalized.pages[0].markdown, "");
            assert!(normalized.pages[0].extra_fields.is_empty());
            assert_eq!(
                normalized
                    .usage_info
                    .and_then(|usage| usage.pages_processed),
                expected
            );
        }
    }

    use crate::ocr::test_support::{MockResponse, mock_server, perform_ocr, wire_request};
    use litellm_auth::InputSource;

    fn request_body(request: &str) -> Value {
        serde_json::from_str(request.split_once("\r\n\r\n").unwrap().1).unwrap()
    }

    #[tokio::test]
    async fn facade_executes_vertex_deepseek_at_the_openai_endpoint() {
        let (base, seen, server) = mock_server(vec![MockResponse::json(json!({
            "choices":[{"message":{"content":"recognized"}}],
            "usage":{"prompt_tokens":1}
        }))])
        .await;
        let request = wire_request(
            "vertex_ai/deepseek-ocr-maas",
            &base,
            json!({
                "vertex_project":"project-1",
                "vertex_location":"europe-west4",
                "temperature":0.1,
                "future_ocr_option":true,
                "extra_body":{"provider_option":"value"}
            }),
        );
        let request = crate::ocr::test_support::with_source(request, "gs://bucket/document.pdf");

        let response = perform_ocr(request).await.unwrap();
        server.await.unwrap();
        assert_eq!(response.pages[0].markdown, "recognized");
        assert_eq!(
            response.usage_info.unwrap().extra_fields["prompt_tokens"],
            1
        );
        let requests = seen.lock().unwrap();
        assert!(requests[0].starts_with(
            "POST /v1/projects/project-1/locations/europe-west4/endpoints/openapi/chat/completions "
        ));
        assert!(
            requests[0]
                .to_ascii_lowercase()
                .contains("authorization: bearer test-key")
        );
        let body = request_body(&requests[0]);
        assert_eq!(body["model"], "deepseek-ai/deepseek-ocr-maas");
        assert_eq!(body["temperature"], 0.1);
        assert_eq!(body["future_ocr_option"], true);
        assert_eq!(body["provider_option"], "value");
        assert!(body.get("vertex_project").is_none());
        assert!(body.get("extra_body").is_none());
        assert_eq!(
            body["messages"][0]["content"][0],
            json!({"type":"image_url","image_url":"gs://bucket/document.pdf"})
        );
    }

    #[test]
    fn host_registration_selects_deepseek_without_affecting_mistral() {
        assert!(crate::ocr::is_supported_request(
            "deepseek-ocr-maas",
            Some("vertex_ai")
        ));
        assert!(crate::ocr::is_supported_request(
            "mistral-ocr-maas",
            Some("vertex_ai")
        ));
    }

    #[tokio::test]
    async fn request_controlled_api_base_is_rejected_before_vertex_auth() {
        let mut request = wire_request(
            "vertex_ai/deepseek-ocr-maas",
            "https://caller.example",
            json!({"vertex_project":"project-1"}),
        );
        request.credentials.api_base = Some(litellm_auth::Sourced::new(
            "https://caller.example".into(),
            InputSource::Request,
        ));

        let error = perform_ocr(request).await.unwrap_err();
        assert!(
            error
                .to_string()
                .contains("request-controlled Vertex AI endpoint")
        );
    }
}
