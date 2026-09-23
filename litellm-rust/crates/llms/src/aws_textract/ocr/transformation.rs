use litellm_core_utils::call_arguments::CallArguments;
use serde::{Deserialize, Serialize};

use super::common_utils::{
    TextractDocument, TextractEnvironment, TextractOperation, TextractResponse, document_bytes,
    endpoint, environment, error_class, health_check_document, inline_document, lines_by_page,
    ocr_response,
};
use crate::base_llm::ocr::{
    error::Error,
    handler::OcrClient,
    transformation::{
        BaseOcrConfig, LiteLLMOcrResponse, OcrDocument, OcrRequestContext, OcrResponseFormat,
        PreparedOcrRequest, decode_and_normalize_response,
    },
};

#[derive(Debug, Deserialize, Serialize)]
pub struct DetectDocumentTextRequest {
    #[serde(rename = "Document")]
    pub document: TextractDocument,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct TextractDetectTextConfig;

impl BaseOcrConfig for TextractDetectTextConfig {
    type OcrParams = ();
    type ProviderRequest = DetectDocumentTextRequest;
    type Environment = TextractEnvironment;

    fn secret_names(&self) -> Vec<&'static str> {
        litellm_auth_aws::constants::SECRET_NAMES.to_vec()
    }

    fn get_health_check_document(&self) -> OcrDocument {
        health_check_document()
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
        environment(request, TextractOperation::DetectDocumentText).await
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
    response: TextractResponse,
) -> Result<LiteLLMOcrResponse, Error> {
    Ok(ocr_response(
        model,
        lines_by_page(&response.blocks),
        response.document_metadata,
    ))
}

#[cfg(test)]
mod tests {
    use rstest::{fixture, rstest};
    use serde_json::{Value, json};

    use super::*;

    const MODEL: &str = "detect-document-text";

    #[fixture]
    fn document(#[default("data:image/png;base64,aGVsbG8=")] source: &str) -> OcrDocument {
        OcrDocument::DocumentUrl {
            document_url: source.into(),
            extra_fields: Default::default(),
        }
    }

    #[rstest]
    #[case::one_page_without_page_numbers(
        json!({
            "DetectDocumentTextModelVersion": "1.0",
            "DocumentMetadata": {"Pages": 1},
            "Blocks": [
                {"BlockType": "PAGE"},
                {"BlockType": "LINE", "Text": "Invoice 12345"},
                {"BlockType": "WORD", "Text": "Invoice"},
                {"BlockType": "WORD", "Text": "12345"},
                {"BlockType": "LINE", "Text": "total 67.89"}
            ]
        }),
        vec![(0, "Invoice 12345\ntotal 67.89")],
        Some(1)
    )]
    #[case::pages_out_of_order(
        json!({
            "DocumentMetadata": {"Pages": 2},
            "Blocks": [
                {"BlockType": "LINE", "Text": "second", "Page": 2},
                {"BlockType": "LINE", "Text": "first", "Page": 1},
                {"BlockType": "LINE", "Text": "also second", "Page": 2}
            ]
        }),
        vec![(0, "first"), (1, "second\nalso second")],
        Some(2)
    )]
    #[case::missing_metadata_counts_the_pages_with_text(
        json!({"Blocks": [{"BlockType": "LINE", "Text": "only"}]}),
        vec![(0, "only")],
        Some(1)
    )]
    #[case::blank_document(json!({"DocumentMetadata": {"Pages": 1}}), vec![], Some(1))]
    fn response_lines_become_one_markdown_page_per_document_page(
        #[case] raw_response: Value,
        #[case] expected_pages: Vec<(i64, &str)>,
        #[case] expected_pages_processed: Option<i64>,
    ) {
        let response = TextractDetectTextConfig
            .transform_ocr_response(
                MODEL,
                &serde_json::to_vec(&raw_response).unwrap(),
                OcrResponseFormat::Litellm,
            )
            .unwrap();

        let pages: Vec<(i64, &str)> = response
            .pages
            .iter()
            .map(|page| (page.index, page.markdown.as_str()))
            .collect();
        assert_eq!(pages, expected_pages);
        assert_eq!(response.model, MODEL);
        assert_eq!(
            response.usage_info.unwrap().pages_processed,
            expected_pages_processed
        );
    }

    #[rstest]
    fn the_request_is_only_the_document_bytes(document: OcrDocument) {
        let request = TextractDetectTextConfig
            .transform_ocr_request(MODEL, document, &(), &[])
            .unwrap();

        assert_eq!(
            serde_json::to_value(request).unwrap(),
            json!({"Document": {"Bytes": "aGVsbG8="}})
        );
    }

    #[rstest]
    fn a_remote_url_is_refused_by_the_sync_transform(
        #[with("https://example.com/a.pdf")] document: OcrDocument,
    ) {
        let error = TextractDetectTextConfig
            .transform_ocr_request(MODEL, document, &(), &[])
            .unwrap_err();

        assert!(matches!(error, Error::InvalidDataUri));
    }

    #[rstest]
    fn the_health_check_document_is_an_inline_image_the_request_accepts() {
        let document = TextractDetectTextConfig.get_health_check_document();

        assert!(
            TextractDetectTextConfig
                .transform_ocr_request(MODEL, document, &(), &[])
                .is_ok()
        );
    }

    #[rstest]
    fn provider_errors_go_through_the_shared_textract_error_class() {
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
}
