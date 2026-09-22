use litellm_core_utils::{call_arguments::CallArguments, params::OpaqueParams, url_utils::ApiUrl};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::base_llm::ocr::{
    error::Error,
    handler::OcrClient,
    transformation::{
        BaseOcrConfig, LiteLLMOcrResponse, OcrConnection, OcrDocument, OcrPage, OcrResponseFormat,
        OcrUsageInfo, PreparedOcrRequest, decode_and_normalize_response,
    },
};

const MISTRAL_OCR_API_BASE: &str = "https://api.mistral.ai/v1";

const MISTRAL_OCR_API_KEY_ENV_VAR: &str = "MISTRAL_API_KEY";

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MistralOcrRequest {
    pub model: String,
    pub document: OcrDocument,
    #[serde(flatten)]
    pub params: OpaqueParams,
}

#[derive(Clone, Debug, Default, Deserialize)]
pub struct MistralOcrResponse {
    #[serde(default)]
    pub pages: Vec<OcrPage>,
    #[serde(
        default,
        deserialize_with = "serde_with::rust::double_option::deserialize"
    )]
    pub model: Option<Option<String>>,
    pub document_annotation: Option<Value>,
    pub usage_info: Option<OcrUsageInfo>,

    #[serde(flatten)]
    pub extra_fields: serde_json::Map<String, Value>,
}

#[derive(Clone, Debug, Default)]
pub struct MistralOcrConfig;

impl BaseOcrConfig for MistralOcrConfig {
    type OcrParams = OpaqueParams;
    type ProviderRequest = MistralOcrRequest;
    type Environment = Vec<(String, String)>;

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

    fn get_api_key_env_var(&self) -> Option<&'static str> {
        Some(MISTRAL_OCR_API_KEY_ENV_VAR)
    }

    fn secret_names(&self) -> Vec<&'static str> {
        vec![
            MISTRAL_OCR_API_KEY_ENV_VAR,
            "MISTRAL_AZURE_API_KEY",
            "MISTRAL_AZURE_API_BASE",
        ]
    }

    fn map_ocr_params(
        &self,
        non_default_params: &CallArguments,
        model: &str,
    ) -> Result<OpaqueParams, Error> {
        Ok(non_default_params
            .select(self.get_supported_ocr_params(model))
            .into())
    }

    async fn validate_environment(
        &self,
        request: &PreparedOcrRequest,
        _client: &OcrClient,
    ) -> Result<Self::Environment, Error> {
        self.resolve_headers(&request.connection, &|name: &str| {
            request.connection.secret(name)
        })
    }

    fn get_complete_url(
        &self,
        request: &PreparedOcrRequest,
        _optional_params: &Self::OcrParams,
        _environment: &Self::Environment,
    ) -> Result<String, Error> {
        self.build_ocr_url(request.connection.api_base.as_deref())
    }

    fn transform_ocr_request(
        &self,
        model: &str,
        document: OcrDocument,
        optional_params: &OpaqueParams,
        _headers: &[(String, String)],
    ) -> Result<MistralOcrRequest, Error> {
        Ok(MistralOcrRequest {
            model: model.to_string(),
            document,
            params: optional_params.clone(),
        })
    }

    fn transform_ocr_response(
        &self,
        model: &str,
        raw_response: &[u8],
        request_format: OcrResponseFormat,
    ) -> Result<LiteLLMOcrResponse, Error> {
        decode_and_normalize_response(model, raw_response, request_format, normalize_response)
    }
}

impl MistralOcrConfig {
    fn resolve_headers(
        &self,
        connection: &OcrConnection,
        env_lookup: &(dyn Fn(&str) -> Option<String> + Sync),
    ) -> Result<Vec<(String, String)>, Error> {
        if litellm_http::request::has_header(&connection.extra_headers, "authorization") {
            return Ok(connection.extra_headers.clone());
        }
        let api_key = connection
            .api_key
            .as_ref()
            .map(|key| key.expose().trim())
            .filter(|key| !key.is_empty())
            .map(str::to_string)
            .or_else(|| {
                self.get_api_key_env_var()
                    .and_then(env_lookup)
                    .filter(|key| !key.trim().is_empty())
            })
            .ok_or(litellm_auth::Error::MissingApiKey {
                provider: "Mistral",
                environment_variable: MISTRAL_OCR_API_KEY_ENV_VAR,
            })?;
        Ok(
            std::iter::once(("Authorization".into(), format!("Bearer {api_key}")))
                .chain(connection.extra_headers.clone())
                .collect(),
        )
    }

    fn build_ocr_url(&self, api_base: Option<&str>) -> Result<String, Error> {
        let base = api_base
            .map(str::trim)
            .filter(|base| !base.is_empty())
            .unwrap_or(MISTRAL_OCR_API_BASE);
        ApiUrl::parse(base)
            .and_then(|url| url.complete_path(&["v1", "ocr"]))
            .map(|url| url.into_string())
            .map_err(|_| Error::RequestField {
                path: "api_base".into(),
            })
    }
}

pub fn normalize_response(
    model: &str,
    response: MistralOcrResponse,
) -> Result<LiteLLMOcrResponse, Error> {
    let model = match response.model {
        Some(Some(model)) => model,
        Some(None) => {
            return Err(Error::ResponseField {
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

#[cfg(test)]
mod tests {
    use rstest::{fixture, rstest};
    use serde_json::{Value, json};

    use super::*;
    use crate::base_llm::ocr::transformation::decode_response;

    #[fixture]
    fn document() -> OcrDocument {
        serde_json::from_value(
            json!({"type":"document_url","document_url":"https://example.com/a.pdf"}),
        )
        .unwrap()
    }

    #[fixture]
    fn connection(
        #[default(None)] api_key: Option<&str>,
        #[default(vec![])] extra_headers: Vec<(String, String)>,
    ) -> OcrConnection {
        OcrConnection {
            api_key: api_key.map(litellm_auth::SecretValue::new),
            extra_headers,
            ..OcrConnection::default()
        }
    }

    #[test]
    fn explicit_null_model_does_not_use_the_missing_model_default() {
        let response = serde_json::from_value(json!({"model":null})).unwrap();
        assert!(matches!(
            normalize_response("fallback", response).unwrap_err(),
            Error::ResponseField { path } if path == "model"
        ));
    }

    #[rstest]
    #[case::non_object_page(json!({"pages":[42]}), "pages[0]")]
    #[case::missing_markdown(json!({"pages":[{"index":0}]}), "pages[0]")]
    #[case::non_string_markdown(
        json!({"pages":[{"index":0,"markdown":42}]}),
        "pages[0].markdown"
    )]
    #[case::non_object_image(
        json!({"pages":[{"index":0,"markdown":"","images":[42]}]}),
        "pages[0].images[0]"
    )]
    #[case::fractional_width(
        json!({"pages":[{"index":0,"markdown":"","dimensions":{"width":1.5}}]}),
        "pages[0].dimensions.width"
    )]
    #[case::invalid_page_count(
        json!({"usage_info":{"pages_processed":"bad"}}),
        "usage_info.pages_processed"
    )]
    fn response_validates_normalized_shapes_at_the_provider_boundary(
        #[case] payload: Value,
        #[case] path: &str,
    ) {
        let error =
            decode_response::<MistralOcrResponse>(&serde_json::to_vec(&payload).unwrap(), false)
                .unwrap_err();
        assert!(matches!(
            error,
            Error::ResponseField { path: actual } if actual == path
        ));
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
        let params = MistralOcrConfig.map_ocr_params(&input, "model").unwrap();
        assert_eq!(
            serde_json::to_value(params).unwrap(),
            json!({"pages":null,"extract_header":false})
        );
        assert_eq!(input["unknown"], true);
        assert_eq!(input.get("pages"), Some(&Value::Null));
    }

    #[rstest]
    fn request_transform_uses_already_mapped_params_without_filtering_again(document: OcrDocument) {
        let params = serde_json::from_value(json!({"extension":{"nested":null}})).unwrap();
        let body = MistralOcrConfig
            .transform_ocr_request("model", document, &params, &[])
            .unwrap();
        assert_eq!(
            serde_json::to_value(body).unwrap()["extension"],
            json!({"nested":null})
        );
    }

    #[test]
    fn raw_response_transform_keeps_native_payload_separate_from_typed_normalization() {
        let raw = br#"{"pages":[{"index":"2","markdown":"text"}],"provider_extension":false}"#;
        let response = MistralOcrConfig
            .transform_ocr_response(
                "model",
                raw,
                crate::base_llm::ocr::transformation::OcrResponseFormat::Native,
            )
            .unwrap();
        assert_eq!(response.pages[0].index, 2);
        let native = response.provider_native_response.unwrap();
        assert_eq!(native["pages"][0]["index"], "2");
        assert_eq!(native["provider_extension"], false);
        assert_eq!(response.extra_fields["provider_extension"], false);
    }

    #[rstest]
    fn raw_response_transform_rejects_invalid_page(
        #[values(OcrResponseFormat::Litellm, OcrResponseFormat::Native)]
        request_format: OcrResponseFormat,
    ) {
        assert!(
            MistralOcrConfig
                .transform_ocr_response("model", br#"{"pages":[{"index":0}]}"#, request_format)
                .is_err()
        );
    }

    fn mapped_params(value: Value) -> Value {
        let params = serde_json::from_value(value).unwrap();
        serde_json::to_value(MistralOcrConfig.map_ocr_params(&params, "model").unwrap()).unwrap()
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
    #[case("table_format", json!("markdown"))]
    #[case("confidence_scores_granularity", json!("word"))]
    #[case("confidence_scores_granularity", json!("page"))]
    #[case("document_annotation_prompt", json!("extract"))]
    #[case("include_blocks", json!(true))]
    #[case("id", json!("req-123"))]
    fn map_ocr_params_forwards_new_ocr_params(#[case] name: &str, #[case] value: Value) {
        assert_eq!(mapped_params(json!({name:value.clone()}))[name], value);
    }

    #[rstest]
    #[case("pages", json!([0, 2]))]
    #[case("pages", json!("0,2-4"))]
    #[case("pages", Value::Null)]
    #[case("include_image_base64", json!(true))]
    #[case("include_image_base64", json!(false))]
    #[case("image_limit", json!(2))]
    #[case("image_min_size", json!(100))]
    #[case("bbox_annotation_format", json!({"type":"json_schema"}))]
    #[case("document_annotation_format", json!({"type":"json_schema"}))]
    #[case("document_annotation_prompt", json!("extract"))]
    #[case("extract_header", json!(true))]
    #[case("extract_footer", json!(false))]
    #[case("table_format", json!("html"))]
    #[case("table_format", json!("markdown"))]
    #[case("confidence_scores_granularity", json!("word"))]
    #[case("confidence_scores_granularity", json!("page"))]
    #[case("confidence_scores_granularity", json!("block"))]
    #[case("include_blocks", json!(true))]
    #[case("include_blocks", json!(false))]
    #[case("id", json!("req-123"))]
    fn request_mapping_preserves_supplied_options(
        document: OcrDocument,
        #[case] name: &str,
        #[case] value: Value,
    ) {
        let arguments = serde_json::from_value(json!({name: value.clone()})).unwrap();
        let params = MistralOcrConfig
            .map_ocr_params(&arguments, "model")
            .unwrap();
        let result = serde_json::to_value(
            MistralOcrConfig
                .transform_ocr_request("model", document.clone(), &params, &[])
                .unwrap(),
        )
        .unwrap();
        assert_eq!(
            result,
            json!({"model":"model", "document":document, name:value})
        );
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
        document: OcrDocument,
        #[case] name: &str,
        #[case] value: Value,
    ) {
        let params: OpaqueParams = serde_json::from_value(json!({name:value.clone()})).unwrap();
        let result = serde_json::to_value(
            MistralOcrConfig
                .transform_ocr_request("mistral-ocr-latest", document, &params, &[])
                .unwrap(),
        )
        .unwrap();
        assert_eq!(result[name], value);
        assert_eq!(result["model"], "mistral-ocr-latest");
    }

    #[rstest]
    fn transform_ocr_request_includes_multiple_new_params(document: OcrDocument) {
        let params: OpaqueParams = serde_json::from_value(json!({
            "table_format":"html",
            "confidence_scores_granularity":"page",
            "extract_header":true
        }))
        .unwrap();
        let result = serde_json::to_value(
            MistralOcrConfig
                .transform_ocr_request("mistral-ocr-latest", document, &params, &[])
                .unwrap(),
        )
        .unwrap();
        assert_eq!(result["table_format"], "html");
        assert_eq!(result["confidence_scores_granularity"], "page");
        assert_eq!(result["extract_header"], true);
    }

    #[rstest]
    fn transform_ocr_response_preserves_blocks_and_confidence_scores() {
        let payload = json!({
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
        });
        let response: MistralOcrResponse = serde_json::from_value(payload.clone()).unwrap();
        let result = normalize_response("model", response).unwrap().into_json();
        assert_eq!(result["pages"][0]["blocks"], payload["pages"][0]["blocks"]);
        assert_eq!(
            result["pages"][0]["confidence_scores"],
            payload["pages"][0]["confidence_scores"]
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

    #[rstest]
    #[case::default_base(None, "https://api.mistral.ai/v1/ocr")]
    #[case::versioned_base(
        Some("https://example.com/v1?tenant=a"),
        "https://example.com/v1/ocr?tenant=a"
    )]
    #[case::complete_endpoint(
        Some("https://example.com/v1/ocr?tenant=a"),
        "https://example.com/v1/ocr?tenant=a"
    )]
    fn complete_url_defaults_and_dedupes_v1(
        #[case] api_base: Option<&str>,
        #[case] expected: &str,
    ) {
        assert_eq!(MistralOcrConfig.build_ocr_url(api_base).unwrap(), expected);
    }

    #[rstest]
    #[case::explicit_key(Some("explicit"), "Bearer explicit")]
    #[case::environment_fallback(None, "Bearer environment")]
    fn environment_prefers_explicit_key_then_environment(
        #[case] _api_key: Option<&str>,
        #[case] expected: &str,
        #[with(_api_key)] connection: OcrConnection,
    ) {
        assert_eq!(
            MistralOcrConfig
                .resolve_headers(&connection, &|_| Some("environment".into()))
                .unwrap()[0],
            ("Authorization".into(), expected.into())
        );
    }

    #[rstest]
    fn environment_preserves_forwarded_authorization(
        #[with(None, vec![("authorization".into(), "Bearer forwarded".into())])]
        connection: OcrConnection,
    ) {
        assert_eq!(
            MistralOcrConfig
                .resolve_headers(&connection, &|_| None)
                .unwrap(),
            connection.extra_headers
        );
    }

    #[rstest]
    fn environment_keeps_extra_headers_after_the_bearer_key(
        #[with(Some("explicit"), vec![("X-Trace".into(), "trace-1".into())])]
        connection: OcrConnection,
    ) {
        assert_eq!(
            MistralOcrConfig
                .resolve_headers(&connection, &|_| None)
                .unwrap(),
            [
                ("Authorization".to_string(), "Bearer explicit".to_string()),
                ("X-Trace".to_string(), "trace-1".to_string()),
            ]
        );
    }

    #[rstest]
    fn environment_rejects_missing_key(connection: OcrConnection) {
        assert!(matches!(
            MistralOcrConfig.resolve_headers(&connection, &|_| None),
            Err(Error::Auth(litellm_auth::Error::MissingApiKey {
                provider: "Mistral",
                environment_variable: MISTRAL_OCR_API_KEY_ENV_VAR,
            }))
        ));
    }
}
