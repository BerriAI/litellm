use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::call_arguments::CallArguments;
use crate::constants::MISTRAL_OCR_API_BASE;
use crate::llms::base_llm::ocr::transformation::{BaseOcrConfig, OcrRequestContext};
use crate::ocr::OcrClient;
use crate::ocr::prepare::credential_env;
use crate::ocr::types::{
    LiteLLMOcrResponse, OcrConnection, OcrDocument, OcrPage, OcrUsageInfo, PreparedOcrRequest,
};
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
    #[serde(flatten)]
    pub extra_fields: serde_json::Map<String, Value>,
    #[serde(default)]
    pub pages: Vec<OcrPage>,
    #[serde(
        default,
        deserialize_with = "serde_with::rust::double_option::deserialize"
    )]
    pub model: Option<Option<String>>,
    pub document_annotation: Option<Value>,
    pub usage_info: Option<OcrUsageInfo>,
}

#[derive(Clone, Debug, Default)]
pub(crate) struct MistralOCRConfig;

impl BaseOcrConfig for MistralOCRConfig {
    type OcrParams = OpaqueParams;
    type ProviderRequest = MistralOcrRequest;
    type Environment = Vec<(String, String)>;

    fn get_api_key_env_var(&self) -> Option<&'static str> {
        Some(MISTRAL_API_KEY_ENV)
    }

    async fn validate_environment(
        &self,
        request: &PreparedOcrRequest,
        _client: &OcrClient,
    ) -> Result<Self::Environment, crate::ocr::Error> {
        self.validate_environment(&request.connection, &credential_env)
    }

    fn get_complete_url(
        &self,
        request: &PreparedOcrRequest,
        _params: &Self::OcrParams,
        _environment: &Self::Environment,
    ) -> Result<String, crate::ocr::Error> {
        self.get_complete_url(request.connection.api_base.as_deref())
    }

    fn transform_ocr_request(
        &self,
        model: &str,
        document: OcrDocument,
        optional_params: &OpaqueParams,
        _headers: &[(String, String)],
    ) -> Result<MistralOcrRequest, crate::ocr::Error> {
        Ok(MistralOcrRequest {
            model: model.to_string(),
            document,
            params: optional_params.clone(),
        })
    }

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

    fn map_ocr_params(
        &self,
        arguments: &CallArguments,
        model: &str,
    ) -> Result<OpaqueParams, crate::ocr::Error> {
        Ok(arguments
            .select(self.get_supported_ocr_params(model))
            .into())
    }

    async fn async_transform_ocr_request(
        &self,
        model: &str,
        document: OcrDocument,
        optional_params: &OpaqueParams,
        headers: &[(String, String)],
        _context: OcrRequestContext<'_>,
    ) -> Result<MistralOcrRequest, crate::ocr::Error> {
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
}

pub(crate) fn normalize_response(
    model: &str,
    response: MistralOcrResponse,
) -> Result<LiteLLMOcrResponse, crate::ocr::Error> {
    let model = match response.model {
        Some(Some(model)) => model,
        Some(None) => {
            return Err(crate::ocr::Error::ResponseField {
                path: "model".into(),
            });
        }
        None => model.to_string(),
    };
    Ok(LiteLLMOcrResponse {
        extra_fields: response.extra_fields,
        document_annotation: response.document_annotation,
        usage_info: response.usage_info,
        ..LiteLLMOcrResponse::new(model, response.pages)
    })
}

impl MistralOCRConfig {
    fn get_complete_url(&self, api_base: Option<&str>) -> Result<String, crate::ocr::Error> {
        let base = api_base
            .map(str::trim)
            .filter(|base| !base.is_empty())
            .unwrap_or(MISTRAL_OCR_API_BASE);
        ApiUrl::parse(base)
            .and_then(|url| url.complete_path(&["v1", "ocr"]))
            .map(|url| url.into_string())
            .map_err(|_| crate::ocr::Error::RequestField {
                path: "api_base".into(),
            })
    }

    fn validate_environment(
        &self,
        connection: &OcrConnection,
        env_lookup: &(dyn Fn(&str) -> Option<String> + Sync),
    ) -> Result<Vec<(String, String)>, crate::ocr::Error> {
        if crate::http_utils::has_header(&connection.extra_headers, "authorization") {
            return Ok(connection.extra_headers.clone());
        }
        let api_key = connection
            .api_key
            .as_deref()
            .map(str::trim)
            .filter(|key| !key.is_empty())
            .map(str::to_string)
            .or_else(|| {
                self.get_api_key_env_var()
                    .and_then(env_lookup)
                    .filter(|key| !key.trim().is_empty())
            })
            .ok_or(litellm_auth::Error::MissingApiKey {
                provider: "Mistral",
                environment_variable: MISTRAL_API_KEY_ENV,
            })?;
        Ok(
            std::iter::once(("Authorization".into(), format!("Bearer {api_key}")))
                .chain(connection.extra_headers.clone())
                .collect(),
        )
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use serde_json::{Value, json};

    use super::*;

    #[test]
    fn explicit_null_model_does_not_use_the_missing_model_default() {
        let response = serde_json::from_value(json!({"model":null})).unwrap();
        assert!(matches!(
            normalize_response("fallback", response).unwrap_err(),
            crate::ocr::Error::ResponseField { path } if path == "model"
        ));
    }

    #[test]
    fn response_validates_normalized_shapes_at_the_provider_boundary() {
        for (payload, path) in [
            (json!({"pages":[42]}), "pages[0]"),
            (json!({"pages":[{"index":0}]}), "pages[0]"),
            (
                json!({"pages":[{"index":0,"markdown":42}]}),
                "pages[0].markdown",
            ),
            (
                json!({"pages":[{"index":0,"markdown":"","images":[42]}]}),
                "pages[0].images[0]",
            ),
            (
                json!({"pages":[{"index":0,"markdown":"","dimensions":{"width":1.5}}]}),
                "pages[0].dimensions.width",
            ),
            (
                json!({"usage_info":{"pages_processed":"bad"}}),
                "usage_info.pages_processed",
            ),
        ] {
            let error = crate::ocr::json::decode_response::<MistralOcrResponse>(
                &serde_json::to_vec(&payload).unwrap(),
                false,
            )
            .unwrap_err();
            assert!(matches!(
                error,
                crate::ocr::Error::ResponseField { path: actual } if actual == path
            ));
        }
    }

    #[test]
    fn response_normalizes_python_numeric_inputs_and_shared_defaults() {
        let response = serde_json::from_value(json!({
            "pages":[{"index":"2","markdown":"text","dimensions":{"width":1.0},"extension":false}],
            "usage_info":{"pages_processed":true,"credits":"1.5","custom":0},
            "extra":"ignored"
        }))
        .unwrap();
        let response = normalize_response("model", response).unwrap();
        assert_eq!(response.pages[0].index, 2);
        assert_eq!(
            response.pages[0].dimensions.as_ref().unwrap().width,
            Some(1)
        );
        assert_eq!(
            response.usage_info.as_ref().unwrap().pages_processed,
            Some(1)
        );
        assert_eq!(response.usage_info.as_ref().unwrap().credits, Some(1.5));
        let serialized = response.into_json();
        assert_eq!(serialized["pages"][0]["extension"], false);
        assert!(serialized["pages"][0]["images"].is_null());
        assert!(serialized["usage_info"]["doc_size_bytes"].is_null());
        assert_eq!(serialized["usage_info"]["custom"], 0);
        assert!(serialized["content"].is_null());
        assert_eq!(serialized["extra"], "ignored");
    }

    #[test]
    fn map_ocr_params_selects_known_fields_without_changing_arguments() {
        let input =
            serde_json::from_value(json!({"pages":null,"extract_header":false,"unknown":true}))
                .unwrap();
        let params = MistralOCRConfig.map_ocr_params(&input, "model").unwrap();
        assert_eq!(
            serde_json::to_value(params).unwrap(),
            json!({"pages":null,"extract_header":false})
        );
        assert_eq!(input["unknown"], true);
        assert_eq!(input.get("pages"), Some(&Value::Null));
    }

    #[test]
    fn request_transform_uses_already_mapped_params_without_filtering_again() {
        let params = serde_json::from_value(json!({"extension":{"nested":null}})).unwrap();
        let body = MistralOCRConfig
            .transform_ocr_request("model", document(), &params, &[])
            .unwrap();
        assert_eq!(
            serde_json::to_value(body).unwrap()["extension"],
            json!({"nested":null})
        );
    }

    #[test]
    fn raw_response_transform_keeps_native_payload_separate_from_typed_normalization() {
        let raw = br#"{"pages":[{"index":"2","markdown":"text"}],"provider_extension":false}"#;
        let response = MistralOCRConfig
            .transform_ocr_response("model", raw, crate::ocr::types::OcrResponseFormat::Native)
            .unwrap();
        assert_eq!(response.pages[0].index, 2);
        let native = response.provider_native_response.unwrap();
        assert_eq!(native["pages"][0]["index"], "2");
        assert_eq!(native["provider_extension"], false);
        assert_eq!(response.extra_fields["provider_extension"], false);
        assert!(
            MistralOCRConfig
                .transform_ocr_response(
                    "model",
                    br#"{"pages":[{"index":0}]}"#,
                    crate::ocr::types::OcrResponseFormat::Litellm
                )
                .is_err()
        );
    }

    fn mapped_params(value: Value) -> Value {
        let params = serde_json::from_value(value).unwrap();
        serde_json::to_value(MistralOCRConfig.map_ocr_params(&params, "model").unwrap()).unwrap()
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
    fn map_ocr_params_excludes_extensions_from_the_provider_options() {
        let mapped = mapped_params(json!({"extract_header":true,"unsupported_param":"value"}));
        assert_eq!(mapped["extract_header"], true);
        assert!(mapped.get("unsupported_param").is_none());
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
                .transform_ocr_request("model", document(), &params, &[])
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
                .transform_ocr_request("mistral-ocr-latest", document(), &params, &[])
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
                .transform_ocr_request("mistral-ocr-latest", document(), &params, &[])
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
        let result = normalize_response("model", response).unwrap().into_json();
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
        let result = normalize_response("model", response).unwrap().into_json();
        assert_eq!(result["pages"][0]["tables"], page["tables"]);
        assert_eq!(result["pages"][0]["hyperlinks"], page["hyperlinks"]);
        assert_eq!(result["pages"][0]["header"], page["header"]);
        assert_eq!(result["pages"][0]["footer"], page["footer"]);
        assert!(result["pages"][0]["images"].is_null());
        assert!(result["pages"][0]["dimensions"].is_null());
    }

    #[test]
    fn complete_url_defaults_and_dedupes_v1() {
        assert_eq!(
            MistralOCRConfig.get_complete_url(None).unwrap(),
            "https://api.mistral.ai/v1/ocr"
        );
        assert_eq!(
            MistralOCRConfig
                .get_complete_url(Some("https://example.com/v1?tenant=a"))
                .unwrap(),
            "https://example.com/v1/ocr?tenant=a"
        );
        assert_eq!(
            MistralOCRConfig
                .get_complete_url(Some("https://example.com/v1/ocr?tenant=a"))
                .unwrap(),
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
            MistralOCRConfig
                .validate_environment(&explicit, &|_| Some("environment".into()))
                .unwrap()[0],
            ("Authorization".into(), "Bearer explicit".into())
        );

        assert_eq!(
            MistralOCRConfig
                .validate_environment(&OcrConnection::default(), &|_| Some("environment".into()))
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
            MistralOCRConfig
                .validate_environment(&connection, &|_| None)
                .unwrap(),
            connection.extra_headers
        );
    }

    #[test]
    fn environment_rejects_missing_key() {
        assert!(matches!(
            MistralOCRConfig.validate_environment(&OcrConnection::default(), &|_| None),
            Err(crate::ocr::Error::Auth(
                litellm_auth::Error::MissingApiKey {
                    provider: "Mistral",
                    environment_variable: MISTRAL_API_KEY_ENV,
                }
            ))
        ));
    }
}
