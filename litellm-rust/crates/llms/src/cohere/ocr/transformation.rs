use litellm_core_utils::{
    call_arguments::{CallArguments, parse_options},
    serde_compat::LaxI64,
    url_utils::ApiUrl,
};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use serde_with::serde_as;

use crate::{
    base_llm::ocr::{
        document::InlineDocument,
        error::Error,
        transformation::{
            BaseOcrConfig, LiteLLMOcrResponse, OCR_INLINE_MAX_BYTES, OcrConnection, OcrDocument,
            OcrPage, OcrPageImage, OcrResponseFormat, OcrUsageInfo, PreparedOcrRequest,
            credential_env, decode_and_normalize_response, decode_response_value,
        },
    },
    custom_httpx::llm_http_handler::OcrClient,
};

const COHERE_PARSE_API_BASE: &str = "https://api.cohere.com";
const COHERE_API_KEY_ENV: &str = "COHERE_API_KEY";

const COHERE_PARSE_HEALTH_CHECK_IMAGE_DATA_URI: &str = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4//8/AAX+Av4N70a4AAAAAElFTkSuQmCC";

#[derive(Clone, Copy, Debug, Default, Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum OutputFormat {
    #[default]
    Markdown,
    Blocks,
}

#[derive(Default, Deserialize, Serialize)]
pub struct CohereOptions {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_format: Option<OutputFormat>,
}

#[derive(Deserialize, Serialize)]
pub struct CohereRequest {
    pub model: String,
    pub document: CohereParseDocument,
    pub output_format: String,
}

#[derive(Deserialize, Serialize)]
#[serde(tag = "type")]
pub enum CohereParseDocument {
    #[serde(rename = "image_url")]
    ImageUrl { image_url: String },
}

#[derive(Deserialize)]
pub struct CohereResponse {
    #[serde(default)]
    pages: Vec<CoherePage>,
    meta: Option<CohereMeta>,
}

#[serde_as]
#[derive(Deserialize)]
struct CoherePage {
    #[serde_as(deserialize_as = "Option<LaxI64>")]
    index: Option<i64>,
    markdown: Option<CohereMarkdown>,
    blocks: Option<Vec<Map<String, Value>>>,
}

#[derive(Deserialize, Serialize)]
struct CohereMarkdown {
    #[serde(default)]
    content: String,
    images: Option<Vec<Map<String, Value>>>,
}

#[derive(Deserialize)]
struct CohereMeta {
    billed_units: Option<CohereBilledUnits>,
}

#[serde_as]
#[derive(Deserialize)]
struct CohereBilledUnits {
    #[serde_as(deserialize_as = "Option<LaxI64>")]
    pages: Option<i64>,
}

#[derive(Default)]
pub struct CohereParseConfig;

impl BaseOcrConfig for CohereParseConfig {
    type OcrParams = CohereOptions;
    type ProviderRequest = CohereRequest;
    type Environment = Vec<(String, String)>;

    fn get_supported_ocr_params(&self, _model: &str) -> &'static [&'static str] {
        &["output_format", "req_format"]
    }

    fn get_api_key_env_var(&self) -> Option<&'static str> {
        Some(COHERE_API_KEY_ENV)
    }

    fn get_health_check_document(&self) -> OcrDocument {
        OcrDocument::ImageUrl {
            image_url: COHERE_PARSE_HEALTH_CHECK_IMAGE_DATA_URI.into(),
            extra_fields: Default::default(),
        }
    }

    fn map_ocr_params(
        &self,
        non_default_params: &CallArguments,
        _model: &str,
    ) -> Result<CohereOptions, Error> {
        Ok(parse_options(non_default_params)?)
    }

    async fn validate_environment(
        &self,
        request: &PreparedOcrRequest,
        _client: &OcrClient,
    ) -> Result<Self::Environment, Error> {
        self.resolve_headers(&request.connection, &credential_env)
    }

    fn get_complete_url(
        &self,
        request: &PreparedOcrRequest,
        _optional_params: &Self::OcrParams,
        _environment: &Self::Environment,
    ) -> Result<String, Error> {
        self.build_ocr_url(
            request
                .connection
                .api_base
                .as_deref()
                .unwrap_or(COHERE_PARSE_API_BASE),
        )
    }

    fn transform_ocr_request(
        &self,
        model: &str,
        document: OcrDocument,
        optional_params: &CohereOptions,
        _headers: &[(String, String)],
    ) -> Result<CohereRequest, Error> {
        let image_url = image_url(document)?;
        Ok(build_request(model, image_url, optional_params))
    }

    fn transform_ocr_response(
        &self,
        model: &str,
        raw_response: &[u8],
        request_format: OcrResponseFormat,
    ) -> Result<LiteLLMOcrResponse, Error> {
        decode_and_normalize_response(model, raw_response, request_format, normalize_response)
    }

    fn validate_request_body(&self, body: &Value) -> Result<(), Error> {
        validate_document(&crate::custom_httpx::llm_http_handler::body_document(body)?)
    }
}

impl CohereParseConfig {
    fn resolve_headers(
        &self,
        connection: &OcrConnection,
        env_lookup: &(dyn Fn(&str) -> Option<String> + Sync),
    ) -> Result<Vec<(String, String)>, Error> {
        if crate::custom_httpx::http_handler::has_header(&connection.extra_headers, "authorization")
        {
            return Ok(connection.extra_headers.clone());
        }
        let key = connection
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
            .ok_or_else(|| {
                Error::Auth(litellm_auth::Error::ProviderAuthentication(
                    "Missing COHERE_API_KEY - set it in the environment or pass api_key".into(),
                ))
            })?;
        Ok(
            std::iter::once(("Authorization".into(), format!("Bearer {key}")))
                .chain(connection.extra_headers.clone())
                .collect(),
        )
    }

    fn build_ocr_url(&self, api_base: &str) -> Result<String, Error> {
        let parsed = reqwest::Url::parse(api_base).map_err(|_| invalid_api_base())?;
        if !matches!(parsed.scheme(), "http" | "https") {
            return Err(invalid_api_base());
        }
        ApiUrl::parse(api_base)
            .and_then(|url| url.complete_path(&["v2", "parse"]))
            .map(|url| url.into_string())
            .map_err(|_| invalid_api_base())
    }
}

pub fn validate_document(document: &OcrDocument) -> Result<(), Error> {
    let OcrDocument::ImageUrl { image_url, .. } = document else {
        return Err(Error::CohereImageOnly);
    };
    if image_url.is_empty() {
        return Err(Error::CohereImageOnly);
    }
    if let Some(inline) = InlineDocument::parse(image_url)? {
        if !inline.mime_type().type_.eq_ignore_ascii_case("image") {
            return Err(Error::CohereImageOnly);
        }
        inline.decode(OCR_INLINE_MAX_BYTES)?;
    }
    Ok(())
}

pub fn normalize_response(
    model: &str,
    response: CohereResponse,
) -> Result<LiteLLMOcrResponse, Error> {
    let pages_processed = billed_pages(&response).map(Ok).unwrap_or_else(|| {
        i64::try_from(response.pages.len()).map_err(|_| Error::NumericRange("pages"))
    })?;
    let pages = response
        .pages
        .into_iter()
        .enumerate()
        .map(|(position, page)| normalize_page(page, position))
        .collect::<Result<Vec<_>, Error>>()?;
    Ok(LiteLLMOcrResponse {
        usage_info: Some(OcrUsageInfo {
            pages_processed: Some(pages_processed),
            ..Default::default()
        }),
        ..LiteLLMOcrResponse::new(model, pages)
    })
}

fn image_url(document: OcrDocument) -> Result<String, Error> {
    validate_document(&document)?;
    let OcrDocument::ImageUrl { image_url, .. } = document else {
        return Err(Error::CohereImageOnly);
    };
    Ok(image_url)
}

fn build_request(model: &str, image_url: String, params: &CohereOptions) -> CohereRequest {
    CohereRequest {
        model: model.into(),
        document: CohereParseDocument::ImageUrl { image_url },
        output_format: match params.output_format.unwrap_or_default() {
            OutputFormat::Markdown => "markdown",
            OutputFormat::Blocks => "blocks",
        }
        .into(),
    }
}

fn page_image(mut image: Map<String, Value>, path: &str) -> Result<OcrPageImage, Error> {
    if let Some(Value::Object(bbox)) = image.get("bounding_box") {
        image.insert("bbox".into(), Value::Object(bbox.clone()));
    }
    decode_response_value(Value::Object(image), path)
}

fn normalize_page(page: CoherePage, position: usize) -> Result<OcrPage, Error> {
    let index = page.index.map(Ok).unwrap_or_else(|| {
        i64::try_from(position).map_err(|_| Error::NumericRange("page index"))
    })?;
    let (markdown, images) = match page.markdown {
        Some(markdown) => {
            let images = markdown
                .images
                .filter(|images| !images.is_empty())
                .map(|images| {
                    images
                        .into_iter()
                        .enumerate()
                        .map(|(image_index, image)| {
                            page_image(
                                image,
                                &format!("pages[{position}].markdown.images[{image_index}]"),
                            )
                        })
                        .collect::<Result<Vec<_>, _>>()
                })
                .transpose()?;
            (markdown.content, images)
        }
        None => (String::new(), None),
    };
    let extra_fields = page
        .blocks
        .map(|blocks| {
            (
                "blocks".into(),
                Value::Array(blocks.into_iter().map(Value::Object).collect()),
            )
        })
        .into_iter()
        .collect();
    Ok(OcrPage {
        index,
        markdown,
        images,
        extra_fields,
        ..Default::default()
    })
}

fn billed_pages(response: &CohereResponse) -> Option<i64> {
    response.meta.as_ref()?.billed_units.as_ref()?.pages
}

fn invalid_api_base() -> Error {
    Error::RequestField {
        path: "api_base".into(),
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use serde_json::json;

    use super::*;
    use crate::azure_ai::ocr::cohere_parse_transformation::AzureAICohereParseConfig;

    #[rstest]
    #[case::cohere(false)]
    #[case::azure(true)]
    fn options_read_known_fields_without_changing_arguments(#[case] azure: bool) {
        let arguments = serde_json::from_value(json!({
            "output_format":"blocks", "req_format":"native", "extension":false
        }))
        .unwrap();
        let mapped = if azure {
            AzureAICohereParseConfig.map_ocr_params(&arguments, "parse")
        } else {
            CohereParseConfig.map_ocr_params(&arguments, "parse")
        }
        .unwrap();
        assert_eq!(
            serde_json::to_value(mapped).unwrap(),
            json!({"output_format":"blocks"})
        );
        assert_eq!(arguments["req_format"], "native");
        assert_eq!(arguments["extension"], false);
    }

    #[test]
    fn options_reject_invalid_output_format() {
        let invalid = serde_json::from_value(json!({"output_format":"html"})).unwrap();
        assert!(matches!(
            CohereParseConfig.map_ocr_params(&invalid, "parse"),
            Err(Error::RequestField { path })
                if path == "optional_params.output_format"
        ));
    }

    #[test]
    fn billed_pages_accept_integral_doubles() {
        let response = serde_json::from_str::<CohereResponse>(
            r#"{"pages":[],"meta":{"billed_units":{"pages":1.0}}}"#,
        )
        .unwrap();
        let normalized = normalize_response("parse", response).unwrap();
        assert_eq!(normalized.usage_info.unwrap().pages_processed, Some(1));
    }

    #[test]
    fn billed_pages_reject_fractional_counts() {
        assert!(
            serde_json::from_str::<CohereResponse>(
                r#"{"pages":[],"meta":{"billed_units":{"pages":1.5}}}"#,
            )
            .is_err()
        );
    }

    #[test]
    fn response_preserves_python_mapping_shapes_and_extensions() {
        let blocks = json!([
            {"type":"text", "text":"Total Due: $4.00"},
            {"type":"future", "payload":{"nested":[null,false,0]}}
        ]);
        let response = serde_json::from_value(json!({
            "pages":[{
                "index":"2",
                "markdown":{"content":"receipt", "images":[
                    {"bounding_box":{"x":1}, "bbox":"replaced", "category":"future", "extension":null},
                    {"image_base64":"encoded"}
                ]},
                "blocks":blocks
            }],
            "meta":{"billed_units":{"pages":0}}
        })).unwrap();
        let response = normalize_response("parse", response).unwrap();
        assert_eq!(response.pages[0].index, 2);
        assert_eq!(response.usage_info.unwrap().pages_processed, Some(0));
        assert_eq!(response.pages[0].extra_fields["blocks"], blocks);
        let images = response.pages[0].images.as_ref().unwrap();
        assert_eq!(images[0].bbox.as_ref().unwrap()["x"], 1);
        assert_eq!(images[0].extra_fields["category"], "future");
        assert_eq!(images[0].extra_fields.get("extension"), Some(&Value::Null));
        assert_eq!(images[1].image_base64.as_deref(), Some("encoded"));
        assert!(images[1].bbox.is_none());
    }

    #[test]
    fn malformed_normalized_image_fields_report_the_original_path() {
        let response = serde_json::from_value(json!({
            "pages":[{"markdown":{"images":[{"image_base64":42}]}}]
        }))
        .unwrap();
        assert!(matches!(
            normalize_response("parse", response).unwrap_err(),
            Error::ResponseField { path }
                if path == "pages[0].markdown.images[0].image_base64"
        ));
    }

    #[rstest]
    fn provider_options_exclude_response_controls_and_extensions(
        #[values("markdown", "blocks")] output_format: &str,
        #[values("https://example.com/a.png", "data:image/png;base64,YWJj")] source: &str,
    ) {
        let arguments = serde_json::from_value(
            json!({"output_format":output_format,"req_format":"native","unknown":true}),
        )
        .unwrap();
        let params = CohereParseConfig
            .map_ocr_params(&arguments, "parse")
            .unwrap();
        assert_eq!(
            serde_json::to_value(&params).unwrap(),
            json!({"output_format":output_format})
        );
        let document = serde_json::from_value(
            json!({"type":"image_url","image_url":source,"ignored":"field"}),
        )
        .unwrap();
        let body = CohereParseConfig
            .transform_ocr_request("parse", document, &params, &[])
            .unwrap();
        assert_eq!(
            serde_json::to_value(body).unwrap(),
            json!({
                "model":"parse", "document":{"type":"image_url","image_url":source}, "output_format":output_format
            })
        );
    }

    #[rstest]
    fn response_normalizes_markdown_images_blocks_and_billed_pages() {
        let payload = json!({
            "pages": [
                {
                    "type":"markdown",
                    "index":4,
                    "markdown":{
                        "content":"receipt",
                        "images":[{
                            "id":"image",
                            "bounding_box":{
                                "top_left_x":1,
                                "top_left_y":2,
                                "bottom_right_x":48,
                                "bottom_right_y":49
                            },
                            "bounding_box_normalized":{
                                "top_left_x":0.04,
                                "top_left_y":0.05,
                                "bottom_right_x":0.15,
                                "bottom_right_y":0.16
                            },
                            "description":"scan",
                            "category":"logo",
                            "provider_extension":"preserved"
                        }]
                    }
                },
                {"type":"blocks","blocks":[{"type":"text","text":{"content":"total"}}]}
            ],
            "meta":{"api_version":{"version":"2"},"billed_units":{"pages":3}}
        });
        let response = serde_json::from_value(payload.clone()).unwrap();
        let normalized = normalize_response("parse-v5.0", response).unwrap();
        assert_eq!(normalized.pages[0].index, 4);
        assert_eq!(normalized.pages[0].markdown, "receipt");
        let image = &normalized.pages[0].images.as_ref().unwrap()[0];
        let original_image = &payload["pages"][0]["markdown"]["images"][0];
        assert_eq!(
            serde_json::to_value(&image.bbox).unwrap(),
            original_image["bounding_box"]
        );
        assert_eq!(
            image.extra_fields["bounding_box_normalized"],
            original_image["bounding_box_normalized"]
        );
        assert_eq!(image.extra_fields["id"], original_image["id"]);
        assert_eq!(image.extra_fields["description"], "scan");
        assert_eq!(image.extra_fields["category"], "logo");
        assert_eq!(image.extra_fields["provider_extension"], "preserved");
        assert_eq!(normalized.pages[1].index, 1);
        assert_eq!(normalized.pages[1].markdown, "");
        assert_eq!(
            normalized.pages[1].extra_fields["blocks"][0]["text"]["content"],
            "total"
        );
        assert_eq!(normalized.usage_info.unwrap().pages_processed, Some(3));
    }

    #[rstest]
    #[case::empty(json!({}))]
    #[case::null_meta(json!({"meta":null}))]
    #[case::null_billed_units(json!({"pages":[],"meta":{"billed_units":null}}))]
    fn response_defaults(#[case] value: Value) {
        let normalized =
            normalize_response("parse", serde_json::from_value(value).unwrap()).unwrap();
        assert!(normalized.pages.is_empty());
        assert_eq!(normalized.usage_info.unwrap().pages_processed, Some(0));
    }

    #[rstest]
    #[case::null_pages(json!({"pages":null}))]
    #[case::invalid_markdown(json!({"pages":[{"markdown":"text"}]}))]
    #[case::invalid_index(json!({"pages":[{"index":"bad"}]}))]
    fn response_rejects_invalid_fields(#[case] value: Value) {
        assert!(serde_json::from_value::<CohereResponse>(value).is_err());
    }

    #[test]
    fn null_markdown_uses_page_defaults() {
        let normalized = normalize_response(
            "parse",
            serde_json::from_value(json!({"pages":[{"markdown":null}]})).unwrap(),
        )
        .unwrap();
        assert_eq!(normalized.usage_info.unwrap().pages_processed, Some(1));
        assert!(normalized.pages[0].images.is_none());
    }

    #[rstest]
    fn response_types_documented_block_variants(
        #[values(
            crate::base_llm::ocr::transformation::OcrResponseFormat::Litellm,
            crate::base_llm::ocr::transformation::OcrResponseFormat::Native
        )]
        response_format: crate::base_llm::ocr::transformation::OcrResponseFormat,
    ) {
        let payload = json!({
            "pages": [{
                "type": "blocks",
                "index": 0,
                "blocks": [
                    {"type": "text", "text": {"content": "hello"}},
                    {
                        "type": "image",
                        "image": {
                            "id": "img-0",
                            "description": "logo",
                            "category": "logo",
                            "bounding_box": {
                                "top_left_x": 1,
                                "top_left_y": 2,
                                "bottom_right_x": 3,
                                "bottom_right_y": 4
                            },
                            "bounding_box_normalized": {
                                "top_left_x": 0.1,
                                "top_left_y": 0.2,
                                "bottom_right_x": 0.3,
                                "bottom_right_y": 0.4
                            }
                        }
                    },
                    {
                        "type": "table",
                        "table": {
                            "type": "html",
                            "html": "<table></table>",
                            "bounding_box": {
                                "top_left_x": 5,
                                "top_left_y": 6,
                                "bottom_right_x": 7,
                                "bottom_right_y": 8
                            },
                            "bounding_box_normalized": {
                                "top_left_x": 0.5,
                                "top_left_y": 0.6,
                                "bottom_right_x": 0.7,
                                "bottom_right_y": 0.8
                            },
                            "title": "Totals",
                            "description": "Invoice totals"
                        }
                    }
                ]
            }]
        });
        let normalized = CohereParseConfig
            .transform_ocr_response(
                "parse-v5.0",
                &serde_json::to_vec(&payload).unwrap(),
                response_format,
            )
            .unwrap();
        assert_eq!(
            normalized.pages[0].extra_fields["blocks"],
            payload["pages"][0]["blocks"]
        );
        assert_eq!(normalized.pages[0].markdown, "");
        assert_eq!(normalized.pages[0].index, 0);
        assert_eq!(
            normalized.usage_info.as_ref().unwrap().pages_processed,
            Some(1)
        );
        match response_format {
            crate::base_llm::ocr::transformation::OcrResponseFormat::Litellm => {
                assert!(normalized.provider_native_response.is_none());
            }
            crate::base_llm::ocr::transformation::OcrResponseFormat::Native => {
                assert_eq!(
                    normalized.provider_native_response.as_ref(),
                    payload.as_object()
                );
            }
        }
        assert_eq!(
            normalized.into_json()["pages"][0]["blocks"],
            payload["pages"][0]["blocks"]
        );
    }

    #[rstest]
    #[case::document_url(json!({"type":"document_url","document_url":"https://example.com/a.pdf"}))]
    #[case::empty_image_url(json!({"type":"image_url","image_url":""}))]
    #[case::pdf_data_uri(json!({"type":"image_url","image_url":"data:application/pdf;base64,YQ=="}))]
    fn request_requires_image(#[case] value: Value) {
        assert!(matches!(
            validate_document(&serde_json::from_value(value).unwrap()),
            Err(Error::CohereImageOnly)
        ));
    }

    #[rstest]
    #[case::markdown("markdown", true)]
    #[case::blocks("blocks", true)]
    #[case::unsupported("html", false)]
    fn request_requires_supported_output_format(#[case] format: &str, #[case] valid: bool) {
        assert_eq!(
            serde_json::from_value::<CohereOptions>(json!({"output_format":format})).is_ok(),
            valid
        );
    }

    #[test]
    fn request_defaults_to_markdown() {
        let request = CohereParseConfig
            .transform_ocr_request(
                "parse-v5.0",
                serde_json::from_value(json!({
                    "type":"image_url",
                    "image_url":"https://example.com/image.png"
                }))
                .unwrap(),
                &serde_json::from_value(json!({})).unwrap(),
                &[],
            )
            .unwrap();
        assert_eq!(
            serde_json::to_value(request).unwrap()["output_format"],
            "markdown"
        );
    }

    #[rstest]
    #[case::base("", "/v2/parse")]
    #[case::version("/v2", "/v2/parse")]
    #[case::complete("/v2/parse", "/v2/parse")]
    #[case::proxy_prefix("/cohere/", "/cohere/v2/parse")]
    fn completes_provider_urls_without_duplicate_paths_and_preserves_queries(
        #[case] suffix: &str,
        #[case] path: &str,
    ) {
        assert_eq!(
            CohereParseConfig
                .build_ocr_url(&format!("https://example.com{suffix}?tenant=a"))
                .unwrap(),
            format!("https://example.com{path}?tenant=a")
        );
    }

    #[rstest]
    #[case::relative("relative/path")]
    #[case::unsupported_scheme("ftp://example.com")]
    fn rejects_invalid_urls(#[case] api_base: &str) {
        assert!(CohereParseConfig.build_ocr_url(api_base).is_err());
    }

    #[test]
    fn rejects_blank_keys() {
        assert!(matches!(
            CohereParseConfig.resolve_headers(
                &OcrConnection {
                    api_key: Some("  ".into()),
                    ..Default::default()
                },
                &|_| None,
            ),
            Err(Error::Auth(_))
        ));
    }

    #[test]
    fn environment_key_becomes_the_bearer() {
        let headers = CohereParseConfig
            .resolve_headers(&OcrConnection::default(), &|name| {
                (name == COHERE_API_KEY_ENV).then(|| "env-key".to_string())
            })
            .unwrap();

        assert_eq!(
            headers,
            [("Authorization".to_string(), "Bearer env-key".to_string())]
        );
    }

    #[test]
    fn missing_key_names_the_environment_variable() {
        let error = CohereParseConfig
            .resolve_headers(&OcrConnection::default(), &|_| None)
            .unwrap_err();

        assert!(error.to_string().contains(COHERE_API_KEY_ENV), "{error}");
    }
}
