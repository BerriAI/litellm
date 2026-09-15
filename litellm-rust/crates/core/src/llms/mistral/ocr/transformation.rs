use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::Error;
use crate::constants::MISTRAL_OCR_API_BASE;
use crate::llms::base_llm::ocr::transformation::BaseOcrConfig;
use crate::ocr::OcrClient;
use crate::ocr::error::{OcrError, OcrRequestError, OcrResponseError};
use crate::ocr::prepare::{credential_env, transform_request_body};
use crate::ocr::types::{LiteLLMOcrRequest, LiteLLMOcrResponse, OcrConnection, OcrDocument};
use crate::params::OpaqueParams;
use crate::url_utils::ApiUrl;

const MISTRAL_API_KEY_ENV: &str = "MISTRAL_API_KEY";

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) struct MistralOcrRequest {
    pub model: String,
    pub document: OcrDocument,
    #[serde(flatten)]
    pub params: OpaqueParams,
}

#[derive(Clone, Debug, Default, Deserialize)]
pub(crate) struct MistralOcrResponse {
    #[serde(default)]
    pub pages: Vec<Value>,
    pub model: Option<String>,
    pub document_annotation: Option<Value>,
    pub usage_info: Option<Value>,
    #[serde(flatten)]
    pub extra_fields: Map<String, Value>,
}
pub(crate) fn transform_ocr_response(
    model: &str,
    response: MistralOcrResponse,
) -> Result<LiteLLMOcrResponse, OcrResponseError> {
    Ok(LiteLLMOcrResponse {
        pages: response.pages,
        model: response.model.unwrap_or_else(|| model.to_string()),
        document_annotation: response.document_annotation,
        usage_info: response.usage_info,
        object: "ocr".to_string(),
        extra_fields: response.extra_fields,
        provider_native_response: None,
    })
}

#[derive(Clone, Debug, Default)]
pub(crate) struct MistralOCRConfig;

impl MistralOCRConfig {
    #[tracing::instrument(target = "litellm::function_trace", level = "trace", skip_all)]
    pub(crate) fn transform_ocr_request(
        &self,
        model: &str,
        document: OcrDocument,
        params: &OpaqueParams,
    ) -> Result<MistralOcrRequest, OcrRequestError> {
        Ok(MistralOcrRequest {
            model: model.to_string(),
            document,
            params: params.clone(),
        })
    }
}

impl BaseOcrConfig for MistralOCRConfig {
    type ProviderResponse = MistralOcrResponse;

    fn get_supported_ocr_params(&self, _model: &str) -> &'static [&'static str] {
        &[
            "pages",
            "include_image_base64",
            "image_limit",
            "image_min_size",
            "bbox_annotation_format",
            "document_annotation_format",
            "document_annotation_prompt",
            "extract_header",
            "extract_footer",
            "table_format",
            "confidence_scores_granularity",
            "include_blocks",
            "id",
        ]
    }

    async fn prepare_request(
        &self,
        request: &LiteLLMOcrRequest,
        client: &OcrClient,
    ) -> Result<reqwest::Request, OcrError> {
        let params = self.map_ocr_params(&request.model, &request.optional_params);
        let headers = validate_environment(&request.connection, &credential_env)?;
        let url = get_complete_url(request.connection.api_base.as_deref())?;
        let body = self.transform_ocr_request(&request.model, request.document.clone(), &params)?;
        transform_request_body(client, request, &url, &headers, true, body, |_| Ok(())).await
    }

    fn transform_ocr_response(
        &self,
        request: &LiteLLMOcrRequest,
        response: MistralOcrResponse,
    ) -> Result<LiteLLMOcrResponse, OcrResponseError> {
        transform_ocr_response(&request.model, response)
    }
}

pub(crate) fn get_complete_url(api_base: Option<&str>) -> Result<String, OcrError> {
    let base = api_base
        .map(str::trim)
        .filter(|base| !base.is_empty())
        .unwrap_or(MISTRAL_OCR_API_BASE);
    ApiUrl::parse(base)
        .and_then(|url| url.complete_path(&["v1", "ocr"]))
        .map(|url| url.into_string())
        .map_err(|_| {
            OcrRequestError::RequestField {
                path: "api_base".into(),
            }
            .into()
        })
}

fn validate_environment(
    connection: &OcrConnection,
    env_lookup: &(dyn Fn(&str) -> Option<String> + Sync),
) -> Result<Vec<(String, String)>, OcrError> {
    if crate::http_utils::has_header(&connection.extra_headers, "authorization") {
        return Ok(connection.extra_headers.clone());
    }
    let api_key = connection
        .api_key
        .as_deref()
        .map(str::trim)
        .filter(|key| !key.is_empty())
        .map(str::to_string)
        .or_else(|| env_lookup(MISTRAL_API_KEY_ENV).filter(|key| !key.trim().is_empty()))
        .ok_or(Error::MissingApiKey {
            provider: "Mistral",
        })?;
    Ok(
        std::iter::once(("Authorization".into(), format!("Bearer {api_key}")))
            .chain(connection.extra_headers.clone())
            .collect(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::rstest;
    use serde_json::{Value, json};

    fn mapped_params(value: Value) -> Value {
        let params = serde_json::from_value::<OpaqueParams>(value).unwrap();
        serde_json::to_value(MistralOCRConfig.map_ocr_params("model", &params)).unwrap()
    }

    fn document() -> OcrDocument {
        serde_json::from_value(
            json!({"type":"document_url","document_url":"https://example.com/a.pdf"}),
        )
        .unwrap()
    }

    #[rstest]
    fn extract_header_is_a_supported_ocr_param() {
        assert_eq!(
            mapped_params(json!({"extract_header":true}))["extract_header"],
            true
        );
    }

    #[rstest]
    fn extract_footer_is_a_supported_ocr_param() {
        assert_eq!(
            mapped_params(json!({"extract_footer":false}))["extract_footer"],
            false
        );
    }

    #[rstest]
    fn existing_ocr_params_remain_supported() {
        let mapped = mapped_params(json!({
            "pages":[0,2],
            "include_image_base64":true,
            "image_limit":2,
            "image_min_size":100,
            "bbox_annotation_format":{"type":"json_schema"},
            "document_annotation_format":{"type":"json_schema"}
        }));
        assert_eq!(mapped["pages"], json!([0, 2]));
        assert_eq!(mapped["include_image_base64"], true);
        assert_eq!(mapped["image_limit"], 2);
        assert_eq!(mapped["image_min_size"], 100);
        assert_eq!(mapped["bbox_annotation_format"]["type"], "json_schema");
        assert_eq!(mapped["document_annotation_format"]["type"], "json_schema");
    }

    #[rstest]
    fn map_ocr_params_forwards_extract_header() {
        assert_eq!(
            mapped_params(json!({"extract_header":true}))["extract_header"],
            true
        );
    }

    #[rstest]
    fn map_ocr_params_forwards_extract_footer() {
        assert_eq!(
            mapped_params(json!({"extract_footer":true}))["extract_footer"],
            true
        );
    }

    #[rstest]
    fn map_ocr_params_forwards_extract_header_and_footer() {
        let mapped = mapped_params(json!({"extract_header":true,"extract_footer":false}));
        assert_eq!(mapped["extract_header"], true);
        assert_eq!(mapped["extract_footer"], false);
    }

    #[rstest]
    fn map_ocr_params_preserves_unknown_params() {
        let mapped = mapped_params(json!({"extract_header":true,"unsupported_param":"value"}));
        assert_eq!(mapped["extract_header"], true);
        assert_eq!(mapped["unsupported_param"], "value");
    }

    #[rstest]
    fn map_ocr_params_preserves_unvalidated_values_and_explicit_null() {
        let mapped = mapped_params(json!({
            "pages":{"future":"shape"},
            "include_image_base64":null
        }));
        assert_eq!(mapped["pages"], json!({"future":"shape"}));
        assert!(mapped.get("include_image_base64").unwrap().is_null());
    }

    #[rstest]
    #[case("table_format", json!("html"))]
    #[case("confidence_scores_granularity", json!("word"))]
    #[case("confidence_scores_granularity", json!("block"))]
    #[case("document_annotation_prompt", json!("extract"))]
    #[case("include_blocks", json!(true))]
    #[case("id", json!("req-123"))]
    fn new_ocr_params_are_supported(#[case] name: &str, #[case] value: Value) {
        assert_eq!(mapped_params(json!({name:value.clone()}))[name], value);
    }

    #[rstest]
    #[case("table_format", json!("html"))]
    #[case("confidence_scores_granularity", json!("word"))]
    #[case("document_annotation_prompt", json!("extract"))]
    #[case("include_blocks", json!(true))]
    #[case("id", json!("req-123"))]
    fn map_ocr_params_forwards_new_ocr_params(#[case] name: &str, #[case] value: Value) {
        assert_eq!(mapped_params(json!({name:value.clone()}))[name], value);
    }

    #[rstest]
    #[case("pages", json!([0, 2]))]
    #[case("pages", json!("0,2-4"))]
    #[case("include_image_base64", json!(true))]
    #[case("image_limit", json!(2))]
    #[case("image_min_size", json!(100))]
    #[case("bbox_annotation_format", json!({"type":"json_schema"}))]
    #[case("document_annotation_format", json!({"type":"json_schema"}))]
    #[case("document_annotation_prompt", json!("extract"))]
    #[case("extract_header", json!(true))]
    #[case("extract_footer", json!(false))]
    #[case("table_format", json!("html"))]
    #[case("confidence_scores_granularity", json!("word"))]
    #[case("include_blocks", json!(true))]
    #[case("id", json!("req-123"))]
    fn request_mapping_matches_python(#[case] name: &str, #[case] value: Value) {
        let params: OpaqueParams = serde_json::from_value(json!({name: value.clone()})).unwrap();
        let result = serde_json::to_value(
            MistralOCRConfig
                .transform_ocr_request("model", document(), &params)
                .unwrap(),
        )
        .unwrap();
        assert_eq!(result["model"], "model");
        assert_eq!(result[name], value);
    }

    #[rstest]
    #[case("table_format", json!("html"))]
    #[case("confidence_scores_granularity", json!("word"))]
    #[case("document_annotation_prompt", json!("extract"))]
    #[case("id", json!("req-123"))]
    #[case("extract_header", json!(true))]
    #[case("include_blocks", json!(true))]
    #[case("pages", json!([0,1]))]
    fn transform_ocr_request_includes_each_optional_param(
        #[case] name: &str,
        #[case] value: Value,
    ) {
        let params: OpaqueParams = serde_json::from_value(json!({name:value.clone()})).unwrap();
        let result = serde_json::to_value(
            MistralOCRConfig
                .transform_ocr_request("mistral-ocr-latest", document(), &params)
                .unwrap(),
        )
        .unwrap();
        assert_eq!(result[name], value);
        assert_eq!(result["model"], "mistral-ocr-latest");
    }

    #[rstest]
    fn transform_ocr_request_includes_multiple_new_params() {
        let params: OpaqueParams = serde_json::from_value(json!({
            "table_format":"html",
            "confidence_scores_granularity":"page",
            "extract_header":true
        }))
        .unwrap();
        let result = serde_json::to_value(
            MistralOCRConfig
                .transform_ocr_request("mistral-ocr-latest", document(), &params)
                .unwrap(),
        )
        .unwrap();
        assert_eq!(result["table_format"], "html");
        assert_eq!(result["confidence_scores_granularity"], "page");
        assert_eq!(result["extract_header"], true);
    }

    #[rstest]
    fn transform_ocr_response_preserves_blocks_and_confidence_scores() {
        let response: MistralOcrResponse = serde_json::from_value(json!({
                "pages":[{
                    "index":0,
                    "markdown":"hello",
                    "images":[{"id":"img-0","image_base64":"data:image/png;base64,AA=="}],
                    "dimensions":{"width":612,"height":792,"dpi":72},
                    "blocks":[{"type":"title","bbox":{"x":1},"confidence_scores":{"mean":0.98}}],
                    "confidence_scores":{"average_page_confidence_score":0.99,"minimum_page_confidence_score":0.97}
                }],
                "model":"returned-model",
                "document_annotation":"{\"language\":\"en\"}",
                "usage_info":{"pages_processed":1}
            }))
            .unwrap();
        let result = transform_ocr_response("model", response)
            .unwrap()
            .into_json();
        assert_eq!(result["pages"][0]["blocks"][0]["type"], "title");
        assert_eq!(result["pages"][0]["blocks"][0]["bbox"]["x"], 1);
        assert_eq!(
            result["pages"][0]["blocks"][0]["confidence_scores"]["mean"],
            0.98
        );
        assert_eq!(
            result["pages"][0]["confidence_scores"]["average_page_confidence_score"],
            0.99
        );
        assert_eq!(result["pages"][0]["images"][0]["id"], "img-0");
        assert_eq!(result["pages"][0]["dimensions"]["dpi"], 72);
        assert_eq!(result["model"], "returned-model");
        assert_eq!(result["document_annotation"], "{\"language\":\"en\"}");
        assert_eq!(result["usage_info"]["pages_processed"], 1);
    }

    #[rstest]
    fn transform_ocr_response_preserves_ocr4_page_fields() {
        let page = json!({
            "index":0,
            "markdown":"table page",
            "tables":[{"rows":2,"cols":3}],
            "hyperlinks":["https://example.com"],
            "header":"header",
            "footer":"footer"
        });
        let response: MistralOcrResponse =
            serde_json::from_value(json!({"pages":[page.clone()]})).unwrap();
        let result = transform_ocr_response("model", response)
            .unwrap()
            .into_json();
        assert_eq!(result["pages"][0], page);
    }

    #[test]
    fn complete_url_defaults_and_dedupes_v1() {
        assert_eq!(
            get_complete_url(None).unwrap(),
            "https://api.mistral.ai/v1/ocr"
        );
        assert_eq!(
            get_complete_url(Some("https://example.com/v1?tenant=a")).unwrap(),
            "https://example.com/v1/ocr?tenant=a"
        );
        assert_eq!(
            get_complete_url(Some("https://example.com/v1/ocr?tenant=a")).unwrap(),
            "https://example.com/v1/ocr?tenant=a"
        );
    }

    #[test]
    fn environment_prefers_explicit_key_then_environment() {
        let explicit = OcrConnection {
            api_key: Some("explicit".into()),
            ..OcrConnection::default()
        };
        assert_eq!(
            validate_environment(&explicit, &|_| Some("environment".into())).unwrap()[0],
            ("Authorization".into(), "Bearer explicit".into())
        );

        assert_eq!(
            validate_environment(&OcrConnection::default(), &|_| Some("environment".into()))
                .unwrap()[0],
            ("Authorization".into(), "Bearer environment".into())
        );
    }

    #[test]
    fn environment_preserves_forwarded_authorization() {
        let connection = OcrConnection {
            extra_headers: vec![("authorization".into(), "Bearer forwarded".into())],
            ..OcrConnection::default()
        };
        assert_eq!(
            validate_environment(&connection, &|_| None).unwrap(),
            connection.extra_headers
        );
    }

    #[test]
    fn environment_rejects_missing_key() {
        assert!(matches!(
            validate_environment(&OcrConnection::default(), &|_| None),
            Err(OcrError::Public(Error::MissingApiKey {
                provider: "Mistral",
            }))
        ));
    }
}
