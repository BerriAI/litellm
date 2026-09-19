use litellm_core_utils::call_arguments::CallArguments;
use serde::{Deserialize, Serialize};

use super::common_utils::{
    Block, DocumentMetadata, HEALTH_CHECK_IMAGE_DATA_URI, TextractDocument, TextractEnvironment,
    document_bytes, endpoint, environment, error_class, inline_document, lines_by_page,
};
use crate::base_llm::ocr::{
    error::Error,
    handler::OcrClient,
    transformation::{
        BaseOcrConfig, LiteLLMOcrResponse, OcrDocument, OcrPage, OcrRequestContext,
        OcrResponseFormat, OcrUsageInfo, PreparedOcrRequest, decode_and_normalize_response,
    },
};

const DETECT_DOCUMENT_TEXT_TARGET: &str = "Textract.DetectDocumentText";

#[derive(Debug, Deserialize, Serialize)]
pub struct DetectDocumentTextRequest {
    #[serde(rename = "Document")]
    pub document: TextractDocument,
}

#[derive(Deserialize)]
#[serde(rename_all = "PascalCase")]
pub struct DetectDocumentTextResponse {
    #[serde(default)]
    blocks: Vec<Block>,
    document_metadata: Option<DocumentMetadata>,
}

/// Synchronous `DetectDocumentText`: plain lines from one image or single-page document.
#[derive(Clone, Copy, Debug, Default)]
pub struct TextractDetectTextConfig;

impl BaseOcrConfig for TextractDetectTextConfig {
    type OcrParams = ();
    type ProviderRequest = DetectDocumentTextRequest;
    type Environment = TextractEnvironment;

    fn get_health_check_document(&self) -> OcrDocument {
        OcrDocument::ImageUrl {
            image_url: HEALTH_CHECK_IMAGE_DATA_URI.into(),
            extra_fields: Default::default(),
        }
    }

    fn map_ocr_params(
        &self,
        _non_default_params: &CallArguments,
        _model: &str,
    ) -> Result<(), Error> {
        Ok(())
    }

    async fn validate_environment(
        &self,
        request: &PreparedOcrRequest,
        _client: &OcrClient,
    ) -> Result<TextractEnvironment, Error> {
        environment(request, DETECT_DOCUMENT_TEXT_TARGET).await
    }

    fn get_complete_url(
        &self,
        request: &PreparedOcrRequest,
        _optional_params: &(),
        environment: &TextractEnvironment,
    ) -> Result<String, Error> {
        Ok(endpoint(request, environment))
    }

    fn transform_ocr_request(
        &self,
        _model: &str,
        document: OcrDocument,
        _optional_params: &(),
        _headers: &[(String, String)],
    ) -> Result<DetectDocumentTextRequest, Error> {
        Ok(DetectDocumentTextRequest {
            document: document_bytes(&document)?,
        })
    }

    async fn async_transform_ocr_request(
        &self,
        model: &str,
        document: OcrDocument,
        optional_params: &(),
        headers: &[(String, String)],
        context: OcrRequestContext<'_>,
    ) -> Result<DetectDocumentTextRequest, Error> {
        let document = inline_document(document, context).await?;
        self.transform_ocr_request(model, document, optional_params, headers)
    }

    fn transform_ocr_response(
        &self,
        model: &str,
        raw_response: &[u8],
        request_format: OcrResponseFormat,
    ) -> Result<LiteLLMOcrResponse, Error> {
        decode_and_normalize_response(model, raw_response, request_format, normalize_response)
    }

    fn get_error_class(
        &self,
        error_message: String,
        status_code: u16,
        headers: Vec<(String, String)>,
    ) -> Error {
        error_class(error_message, status_code, headers)
    }
}

fn normalize_response(
    model: &str,
    response: DetectDocumentTextResponse,
) -> Result<LiteLLMOcrResponse, Error> {
    let pages: Vec<OcrPage> = lines_by_page(&response.blocks)
        .into_iter()
        .map(|(page, markdown)| OcrPage {
            index: page - 1,
            markdown,
            ..Default::default()
        })
        .collect();
    let pages_processed = response
        .document_metadata
        .and_then(|metadata| metadata.pages)
        .or_else(|| i64::try_from(pages.len()).ok());
    Ok(LiteLLMOcrResponse {
        usage_info: Some(OcrUsageInfo {
            pages_processed,
            ..Default::default()
        }),
        ..LiteLLMOcrResponse::new(model, pages)
    })
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    fn normalize(response: serde_json::Value) -> LiteLLMOcrResponse {
        TextractDetectTextConfig
            .transform_ocr_response(
                "detect-document-text",
                &serde_json::to_vec(&response).unwrap(),
                OcrResponseFormat::Litellm,
            )
            .unwrap()
    }

    #[test]
    fn lines_become_one_markdown_page_and_words_are_not_repeated() {
        let response = normalize(json!({
            "DocumentMetadata": {"Pages": 1},
            "Blocks": [
                {"BlockType": "PAGE"},
                {"BlockType": "LINE", "Text": "Invoice 12345"},
                {"BlockType": "WORD", "Text": "Invoice"},
                {"BlockType": "WORD", "Text": "12345"},
                {"BlockType": "LINE", "Text": "total 67.89"}
            ]
        }));

        assert_eq!(response.pages.len(), 1);
        assert_eq!(response.pages[0].index, 0);
        assert_eq!(response.pages[0].markdown, "Invoice 12345\ntotal 67.89");
        assert_eq!(response.usage_info.unwrap().pages_processed, Some(1));
    }

    #[test]
    fn lines_are_grouped_by_their_page_in_page_order() {
        let response = normalize(json!({
            "DocumentMetadata": {"Pages": 2},
            "Blocks": [
                {"BlockType": "LINE", "Text": "second", "Page": 2},
                {"BlockType": "LINE", "Text": "first", "Page": 1},
                {"BlockType": "LINE", "Text": "also second", "Page": 2}
            ]
        }));

        let pages: Vec<(i64, &str)> = response
            .pages
            .iter()
            .map(|page| (page.index, page.markdown.as_str()))
            .collect();
        assert_eq!(pages, vec![(0, "first"), (1, "second\nalso second")]);
    }

    #[test]
    fn a_multi_page_rejection_is_explained_to_the_caller() {
        let error = TextractDetectTextConfig.get_error_class(
            r#"{"__type":"UnsupportedDocumentException","Message":"Request has unsupported document format"}"#.into(),
            400,
            Vec::new(),
        );

        assert!(
            error
                .to_string()
                .contains("multi-page documents are not supported")
        );
    }

    #[test]
    fn the_request_carries_the_document_bytes_without_the_data_uri_envelope() {
        let request = TextractDetectTextConfig
            .transform_ocr_request(
                "detect-document-text",
                OcrDocument::ImageUrl {
                    image_url: "data:image/png;base64,aGVsbG8=".into(),
                    extra_fields: Default::default(),
                },
                &(),
                &[],
            )
            .unwrap();

        assert_eq!(
            serde_json::to_value(request).unwrap(),
            json!({"Document": {"Bytes": "aGVsbG8="}})
        );
    }

    #[test]
    fn a_remote_url_is_refused_by_the_sync_transform() {
        let error = TextractDetectTextConfig
            .transform_ocr_request(
                "detect-document-text",
                OcrDocument::DocumentUrl {
                    document_url: "https://example.com/a.pdf".into(),
                    extra_fields: Default::default(),
                },
                &(),
                &[],
            )
            .unwrap_err();

        assert!(matches!(error, Error::InvalidDataUri));
    }
}
