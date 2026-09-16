use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use serde_with::serde_as;

use crate::call_arguments::{CallArguments, parse_options};
use crate::constants::{COHERE_API_KEY_ENV, COHERE_PARSE_API_BASE};
use crate::llms::base_llm::ocr::transformation::{BaseOcrConfig, OcrRequestContext};
use crate::ocr::OcrClient;
use crate::ocr::document::InlineDocument;
use crate::ocr::prepare::credential_env;
use crate::ocr::types::{
    LiteLLMOcrResponse, OcrConnection, OcrDocument, OcrPage, OcrPageImage, OcrUsageInfo,
    PreparedOcrRequest,
};
use crate::serde_compat::LaxI64;
use crate::url_utils::ApiUrl;

const COHERE_PARSE_HEALTH_CHECK_IMAGE_DATA_URI: &str = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4//8/AAX+Av4N70a4AAAAAElFTkSuQmCC";

#[derive(Clone, Copy, Debug, Default, Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
pub(crate) enum OutputFormat {
    #[default]
    Markdown,
    Blocks,
}

#[derive(Default, Deserialize, Serialize)]
pub(crate) struct CohereOptions {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_format: Option<OutputFormat>,
}

#[derive(Deserialize, Serialize)]
pub(crate) struct CohereRequest {
    pub model: String,
    pub document: CohereParseDocument,
    pub output_format: String,
}

#[derive(Deserialize, Serialize)]
#[serde(tag = "type")]
pub(crate) enum CohereParseDocument {
    #[serde(rename = "image_url")]
    ImageUrl { image_url: String },
}

#[derive(Deserialize)]
pub(crate) struct CohereResponse {
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
pub(crate) struct CohereParseConfig;

impl BaseOcrConfig for CohereParseConfig {
    type OcrParams = CohereOptions;
    type ProviderRequest = CohereRequest;
    type Environment = Vec<(String, String)>;

    fn get_api_key_env_var(&self) -> Option<&'static str> {
        Some(COHERE_API_KEY_ENV)
    }

    fn get_health_check_document(&self) -> OcrDocument {
        OcrDocument::ImageUrl {
            image_url: COHERE_PARSE_HEALTH_CHECK_IMAGE_DATA_URI.into(),
            extra_fields: Default::default(),
        }
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
        self.get_complete_url(
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
    ) -> Result<CohereRequest, crate::ocr::Error> {
        let image_url = image_url(document)?;
        Ok(build_request(model, image_url, optional_params))
    }

    fn get_supported_ocr_params(&self, _model: &str) -> &'static [&'static str] {
        &["output_format", "req_format"]
    }

    fn map_ocr_params(
        &self,
        arguments: &CallArguments,
        _model: &str,
    ) -> Result<CohereOptions, crate::ocr::Error> {
        Ok(parse_options(arguments)?)
    }

    async fn async_transform_ocr_request(
        &self,
        model: &str,
        document: OcrDocument,
        optional_params: &CohereOptions,
        headers: &[(String, String)],
        _context: OcrRequestContext<'_>,
    ) -> Result<CohereRequest, crate::ocr::Error> {
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

    fn validate_request_body(&self, body: &Value) -> Result<(), crate::ocr::Error> {
        validate_document(&crate::ocr::prepare::body_document(body)?)
    }
}

pub(crate) fn validate_document(document: &OcrDocument) -> Result<(), crate::ocr::Error> {
    let OcrDocument::ImageUrl { image_url, .. } = document else {
        return Err(crate::ocr::Error::CohereImageOnly);
    };
    if image_url.is_empty() {
        return Err(crate::ocr::Error::CohereImageOnly);
    }
    if let Some(inline) = InlineDocument::parse(image_url)? {
        if !inline.mime_type().type_.eq_ignore_ascii_case("image") {
            return Err(crate::ocr::Error::CohereImageOnly);
        }
        inline.decode(crate::constants::OCR_INLINE_MAX_BYTES)?;
    }
    Ok(())
}

pub(crate) fn normalize_response(
    model: &str,
    response: CohereResponse,
) -> Result<LiteLLMOcrResponse, crate::ocr::Error> {
    let pages_processed = billed_pages(&response).map(Ok).unwrap_or_else(|| {
        i64::try_from(response.pages.len()).map_err(|_| crate::ocr::Error::NumericRange("pages"))
    })?;
    let pages = response
        .pages
        .into_iter()
        .enumerate()
        .map(|(position, page)| normalize_page(page, position))
        .collect::<Result<Vec<_>, crate::ocr::Error>>()?;
    Ok(LiteLLMOcrResponse {
        usage_info: Some(OcrUsageInfo {
            pages_processed: Some(pages_processed),
            ..Default::default()
        }),
        ..LiteLLMOcrResponse::new(model, pages)
    })
}

fn image_url(document: OcrDocument) -> Result<String, crate::ocr::Error> {
    validate_document(&document)?;
    let OcrDocument::ImageUrl { image_url, .. } = document else {
        return Err(crate::ocr::Error::CohereImageOnly);
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

fn page_image(
    mut image: Map<String, Value>,
    path: &str,
) -> Result<OcrPageImage, crate::ocr::Error> {
    if let Some(Value::Object(bbox)) = image.get("bounding_box") {
        image.insert("bbox".into(), Value::Object(bbox.clone()));
    }
    crate::ocr::json::decode_response_value(Value::Object(image), path)
}

fn normalize_page(page: CoherePage, position: usize) -> Result<OcrPage, crate::ocr::Error> {
    let index = page.index.map(Ok).unwrap_or_else(|| {
        i64::try_from(position).map_err(|_| crate::ocr::Error::NumericRange("page index"))
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

impl CohereParseConfig {
    fn get_complete_url(&self, base: &str) -> Result<String, crate::ocr::Error> {
        let parsed = reqwest::Url::parse(base).map_err(|_| invalid_api_base())?;
        if !matches!(parsed.scheme(), "http" | "https") {
            return Err(invalid_api_base());
        }
        ApiUrl::parse(base)
            .and_then(|url| url.complete_path(&["v2", "parse"]))
            .map(|url| url.into_string())
            .map_err(|_| invalid_api_base())
    }

    fn validate_environment(
        &self,
        connection: &OcrConnection,
        env_lookup: &(dyn Fn(&str) -> Option<String> + Sync),
    ) -> Result<Vec<(String, String)>, crate::ocr::Error> {
        if crate::http_utils::has_header(&connection.extra_headers, "authorization") {
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
                crate::ocr::Error::Auth(litellm_auth::Error::ProviderAuthentication(
                    "Missing COHERE_API_KEY - set it in the environment or pass api_key".into(),
                ))
            })?;
        Ok(
            std::iter::once(("Authorization".into(), format!("Bearer {key}")))
                .chain(connection.extra_headers.clone())
                .collect(),
        )
    }
}

fn invalid_api_base() -> crate::ocr::Error {
    crate::ocr::Error::RequestField {
        path: "api_base".into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[tokio::test]
    async fn composed_body_preserves_native_document_fields_and_untyped_overrides() {
        let mut request = crate::ocr::test_support::wire_request(
            "cohere/parse",
            "https://example.com",
            json!({
                "output_format":"markdown", "metadata":{"host":true},
                "extra_body":{
                    "output_format": {"future":true},
                    "document":{"type":"image_url","image_url":"https://example.com/a.png",
                        "provider_options":{"nested":[false,0,null]}}
                }
            }),
        );
        request.document = serde_json::from_value(json!({
            "type":"image_url","image_url":"https://example.com/original.png"
        }))
        .unwrap();
        let request = crate::ocr::prepare::prepare_request(request);
        let http = CohereParseConfig
            .prepare_request(&request, &crate::ocr::test_support::ocr_client())
            .await
            .unwrap();
        let body: Value = serde_json::from_slice(http.body().unwrap().as_bytes().unwrap()).unwrap();
        assert_eq!(
            body,
            json!({
                "model":"parse", "output_format":{"future":true},
                "document":{"type":"image_url","image_url":"https://example.com/a.png",
                    "provider_options":{"nested":[false,0,null]}}
            })
        );
    }

    #[test]
    fn options_read_known_fields_without_changing_arguments() {
        let arguments = serde_json::from_value(json!({
            "output_format":"blocks", "req_format":"native", "extension":false
        }))
        .unwrap();
        for config in [false, true] {
            let mapped = if config {
                crate::llms::azure_ai::ocr::cohere_parse_transformation::AzureAICohereParseConfig
                    .map_ocr_params(&arguments, "parse")
            } else {
                CohereParseConfig.map_ocr_params(&arguments, "parse")
            }
            .unwrap();
            assert_eq!(
                serde_json::to_value(mapped).unwrap(),
                json!({"output_format":"blocks"})
            );
        }
        assert_eq!(arguments["req_format"], "native");
        assert_eq!(arguments["extension"], false);
        let invalid = serde_json::from_value(json!({"output_format":"html"})).unwrap();
        assert!(matches!(
            CohereParseConfig.map_ocr_params(&invalid, "parse"),
            Err(crate::ocr::Error::RequestField { path })
                if path == "optional_params.output_format"
        ));
    }

    #[test]
    fn billed_pages_accept_integral_doubles_and_reject_fractional_counts() {
        let response = serde_json::from_str::<CohereResponse>(
            r#"{"pages":[],"meta":{"billed_units":{"pages":1.0}}}"#,
        )
        .unwrap();
        let normalized = normalize_response("parse", response).unwrap();
        assert_eq!(normalized.usage_info.unwrap().pages_processed, Some(1));
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
        assert_eq!(
            normalize_response("parse", response).unwrap_err(),
            crate::ocr::Error::ResponseField {
                path: "pages[0].markdown.images[0].image_base64".into()
            }
        );
    }

    #[test]
    fn provider_options_exclude_response_controls_and_extensions() {
        let arguments = serde_json::from_value(
            json!({"output_format":"blocks","req_format":"native","unknown":true}),
        )
        .unwrap();
        let params = CohereParseConfig
            .map_ocr_params(&arguments, "parse")
            .unwrap();
        assert_eq!(
            serde_json::to_value(&params).unwrap(),
            json!({"output_format":"blocks"})
        );
        let document = serde_json::from_value(
            json!({"type":"image_url","image_url":"https://example.com/a.png","ignored":"field"}),
        )
        .unwrap();
        let body = CohereParseConfig
            .transform_ocr_request("parse", document, &params, &[])
            .unwrap();
        assert_eq!(
            serde_json::to_value(body).unwrap(),
            json!({
                "model":"parse", "document":{"type":"image_url","image_url":"https://example.com/a.png"}, "output_format":"blocks"
            })
        );
    }

    #[tokio::test]
    async fn explicit_null_options_use_defaults_before_http() {
        let request = crate::ocr::test_support::wire_request(
            "cohere/parse",
            "https://example.com",
            json!({"output_format":null,"req_format":null}),
        );
        let request = crate::ocr::types::LiteLLMOcrRequest {
            document: serde_json::from_value(
                json!({"type":"image_url","image_url":"https://example.com/a.png"}),
            )
            .unwrap(),
            ..request
        };
        assert_eq!(
            request.response_format().unwrap(),
            crate::ocr::types::OcrResponseFormat::Litellm
        );
        let request = crate::ocr::prepare::prepare_request(request);
        let http = CohereParseConfig
            .prepare_request(&request, &crate::ocr::test_support::ocr_client())
            .await
            .unwrap();
        let body: Value = serde_json::from_slice(http.body().unwrap().as_bytes().unwrap()).unwrap();
        assert_eq!(body["output_format"], "markdown");
        assert!(body.get("req_format").is_none());
    }

    #[test]
    fn response_normalizes_markdown_images_blocks_and_billed_pages() {
        let response = serde_json::from_value(json!({
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
        }))
        .unwrap();
        let normalized = normalize_response("parse-v5.0", response).unwrap();
        assert_eq!(normalized.pages[0].index, 4);
        assert_eq!(normalized.pages[0].markdown, "receipt");
        let image = &normalized.pages[0].images.as_ref().unwrap()[0];
        assert_eq!(image.bbox.as_ref().unwrap()["top_left_x"], 1);
        assert_eq!(
            image.extra_fields["bounding_box_normalized"]["bottom_right_x"],
            0.15
        );
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

    #[test]
    fn response_defaults_and_invalid_fields() {
        for value in [
            json!({}),
            json!({"meta":null}),
            json!({"pages":[],"meta":{"billed_units":null}}),
        ] {
            let normalized =
                normalize_response("parse", serde_json::from_value(value).unwrap()).unwrap();
            assert!(normalized.pages.is_empty());
            assert_eq!(normalized.usage_info.unwrap().pages_processed, Some(0));
        }
        for value in [
            json!({"pages":null}),
            json!({"pages":[{"markdown":"text"}]}),
            json!({"pages":[{"index":"bad"}]}),
        ] {
            assert!(serde_json::from_value::<CohereResponse>(value).is_err());
        }
        let normalized = normalize_response(
            "parse",
            serde_json::from_value(json!({"pages":[{"markdown":null}]})).unwrap(),
        )
        .unwrap();
        assert_eq!(normalized.usage_info.unwrap().pages_processed, Some(1));
        assert!(normalized.pages[0].images.is_none());
    }

    #[test]
    fn response_types_documented_block_variants() {
        let response = serde_json::from_value(json!({
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
                            "title": "Totals"
                        }
                    }
                ]
            }]
        }))
        .unwrap();
        let normalized = normalize_response("parse-v5.0", response).unwrap();
        let blocks = normalized.pages[0].extra_fields["blocks"]
            .as_array()
            .unwrap();
        assert_eq!(blocks[0]["text"]["content"], "hello");
        assert_eq!(blocks[1]["image"]["category"], "logo");
        assert_eq!(blocks[2]["table"]["type"], "html");
        assert_eq!(blocks[2]["table"]["title"], "Totals");
    }

    #[test]
    fn request_requires_image_and_supported_output_format() {
        for value in [
            json!({"type":"document_url","document_url":"https://example.com/a.pdf"}),
            json!({"type":"image_url","image_url":""}),
            json!({"type":"image_url","image_url":"data:application/pdf;base64,YQ=="}),
        ] {
            assert_eq!(
                validate_document(&serde_json::from_value(value).unwrap()),
                Err(crate::ocr::Error::CohereImageOnly)
            );
        }
        assert!(serde_json::from_value::<CohereOptions>(json!({"output_format":"html"})).is_err());
        for format in ["markdown", "blocks"] {
            assert!(
                serde_json::from_value::<CohereOptions>(json!({"output_format":format})).is_ok()
            );
        }
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

    #[test]
    fn completes_provider_urls_without_duplicate_paths_and_preserves_queries() {
        for suffix in ["", "/v2", "/v2/parse"] {
            assert_eq!(
                CohereParseConfig
                    .get_complete_url(&format!("https://example.com{suffix}?tenant=a"))
                    .unwrap(),
                "https://example.com/v2/parse?tenant=a"
            );
        }
    }

    #[test]
    fn rejects_invalid_urls_and_blank_keys() {
        assert!(CohereParseConfig.get_complete_url("relative/path").is_err());
        assert!(
            CohereParseConfig
                .get_complete_url("ftp://example.com")
                .is_err()
        );
        assert!(matches!(
            CohereParseConfig.validate_environment(
                &OcrConnection {
                    api_key: Some("  ".into()),
                    ..Default::default()
                },
                &|_| None,
            ),
            Err(crate::ocr::Error::Auth(_))
        ));
    }
}
