use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};

use crate::ocr::document::InlineDocument;
use crate::ocr::error::{OcrRequestError, OcrResponseError};
use crate::ocr::types::{LiteLLMOcrResponse, OcrDocument};

#[derive(Clone, Copy, Debug, Default, Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
pub(crate) enum OutputFormat {
    #[default]
    Markdown,
    Blocks,
}

#[derive(Deserialize)]
pub(crate) struct CohereParams {
    #[serde(default)]
    pub output_format: OutputFormat,
}

#[derive(Deserialize, Serialize)]
pub(crate) struct CohereRequest {
    pub model: String,
    pub document: OcrDocument,
    pub output_format: OutputFormat,
}

pub(crate) fn validate_document(document: &OcrDocument) -> Result<(), OcrRequestError> {
    let OcrDocument::ImageUrl { image_url, .. } = document else {
        return Err(OcrRequestError::CohereImageOnly);
    };
    if image_url.is_empty() {
        return Err(OcrRequestError::CohereImageOnly);
    }
    if let Some(inline) = InlineDocument::parse(image_url)? {
        if !inline.mime_type().type_.eq_ignore_ascii_case("image") {
            return Err(OcrRequestError::CohereImageOnly);
        }
        inline.decode(crate::constants::OCR_INLINE_MAX_BYTES)?;
    }
    Ok(())
}

#[derive(Deserialize)]
pub(crate) struct CohereResponse {
    #[serde(default)]
    pages: Vec<CoherePage>,
    meta: Option<CohereMeta>,
}

#[derive(Deserialize)]
struct CoherePage {
    index: Option<i64>,
    markdown: Option<CohereMarkdown>,
    blocks: Option<Vec<Map<String, Value>>>,
}

#[derive(Deserialize)]
struct CohereMarkdown {
    #[serde(default)]
    content: String,
    images: Option<Vec<Map<String, Value>>>,
}

#[derive(Deserialize)]
struct CohereMeta {
    billed_units: Option<CohereBilledUnits>,
}

#[derive(Deserialize)]
struct CohereBilledUnits {
    pages: Option<i64>,
}

pub(crate) fn transform_response(
    model: &str,
    response: CohereResponse,
) -> Result<LiteLLMOcrResponse, OcrResponseError> {
    let pages_processed = response
        .meta
        .and_then(|meta| meta.billed_units)
        .and_then(|units| units.pages)
        .map(Ok)
        .unwrap_or_else(|| {
            i64::try_from(response.pages.len()).map_err(|_| OcrResponseError::NumericRange("pages"))
        })?;
    let pages = response
        .pages
        .into_iter()
        .enumerate()
        .map(|(position, page)| {
            let index = page.index.map(Ok).unwrap_or_else(|| {
                i64::try_from(position).map_err(|_| OcrResponseError::NumericRange("page index"))
            })?;
            let (content, images) = page
                .markdown
                .map(|markdown| {
                    let images =
                        markdown
                            .images
                            .filter(|images| !images.is_empty())
                            .map(|images| {
                                images
                                    .into_iter()
                                    .map(|mut image| {
                                        if let Some(Value::Object(bbox)) =
                                            image.get("bounding_box").cloned()
                                        {
                                            image.insert("bbox".into(), Value::Object(bbox));
                                        }
                                        Value::Object(image)
                                    })
                                    .collect::<Vec<_>>()
                            });
                    (markdown.content, images)
                })
                .unwrap_or_default();
            let mut normalized = json!({"index": index, "markdown": content, "images": images});
            if let Some(blocks) = page.blocks {
                normalized["blocks"] = json!(blocks);
            }
            Ok(normalized)
        })
        .collect::<Result<Vec<_>, OcrResponseError>>()?;
    Ok(LiteLLMOcrResponse {
        pages,
        model: model.into(),
        document_annotation: None,
        usage_info: Some(json!({"pages_processed": pages_processed})),
        object: "ocr".into(),
        extra_fields: Map::new(),
        provider_native_response: None,
    })
}

pub(crate) fn transform_request(
    model: &str,
    document: OcrDocument,
    params: CohereParams,
) -> Result<CohereRequest, OcrRequestError> {
    validate_document(&document)?;
    Ok(CohereRequest {
        model: model.into(),
        document,
        output_format: params.output_format,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn response_normalizes_markdown_images_blocks_and_billed_pages() {
        let response = serde_json::from_value(json!({
            "pages": [
                {"index": 4, "markdown": {"content": "receipt", "images": [{"id":"image", "bounding_box":{"top_left_x":1}, "description":"scan"}]}},
                {"blocks": [{"type":"text","text":"total"}]}
            ],
            "meta": {"billed_units":{"pages":3}}
        })).unwrap();
        let normalized = transform_response("parse-v5.0", response).unwrap();
        assert_eq!(normalized.pages[0]["index"], 4);
        assert_eq!(normalized.pages[0]["markdown"], "receipt");
        assert_eq!(normalized.pages[0]["images"][0]["bbox"]["top_left_x"], 1);
        assert_eq!(normalized.pages[0]["images"][0]["description"], "scan");
        assert_eq!(normalized.pages[1]["index"], 1);
        assert_eq!(normalized.pages[1]["markdown"], "");
        assert_eq!(normalized.pages[1]["blocks"][0]["text"], "total");
        assert_eq!(normalized.usage_info.unwrap()["pages_processed"], 3);
    }

    #[test]
    fn response_defaults_and_invalid_fields() {
        for value in [
            json!({}),
            json!({"meta":null}),
            json!({"pages":[],"meta":{"billed_units":null}}),
        ] {
            let normalized =
                transform_response("parse", serde_json::from_value(value).unwrap()).unwrap();
            assert!(normalized.pages.is_empty());
            assert_eq!(normalized.usage_info.unwrap()["pages_processed"], 0);
        }
        for value in [
            json!({"pages":null}),
            json!({"pages":[{"markdown":"text"}]}),
            json!({"pages":[{"index":"bad"}]}),
        ] {
            assert!(serde_json::from_value::<CohereResponse>(value).is_err());
        }
        let normalized = transform_response(
            "parse",
            serde_json::from_value(json!({"pages":[{"markdown":null}]})).unwrap(),
        )
        .unwrap();
        assert_eq!(normalized.usage_info.unwrap()["pages_processed"], 1);
        assert!(normalized.pages[0]["images"].is_null());
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
                Err(OcrRequestError::CohereImageOnly)
            );
        }
        assert!(serde_json::from_value::<CohereParams>(json!({"output_format":"html"})).is_err());
        for format in ["markdown", "blocks"] {
            assert!(
                serde_json::from_value::<CohereParams>(json!({"output_format":format})).is_ok()
            );
        }
        let request = transform_request(
            "parse-v5.0",
            serde_json::from_value(json!({
                "type":"image_url",
                "image_url":"https://example.com/image.png"
            }))
            .unwrap(),
            serde_json::from_value(json!({})).unwrap(),
        )
        .unwrap();
        assert_eq!(
            serde_json::to_value(request).unwrap()["output_format"],
            "markdown"
        );
    }
}
