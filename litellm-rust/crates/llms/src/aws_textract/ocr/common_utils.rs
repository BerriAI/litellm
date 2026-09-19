use base64::{Engine, engine::general_purpose::STANDARD};
use litellm_auth_aws::{SigV4Signer, resolve_aws_region};
use litellm_http::outbound::RequestSigner;
use serde::{Deserialize, Serialize};
use strum::{EnumString, IntoStaticStr, VariantNames};

use crate::base_llm::ocr::{
    document::{InlineDocument, inline_remote_document},
    error::Error,
    transformation::{
        LiteLLMOcrResponse, OcrDocument, OcrEnvironment, OcrPage, OcrRequestContext, OcrUsageInfo,
        PreparedOcrRequest,
    },
};

const TEXTRACT_SERVICE: &str = "textract";
const AWS_JSON_CONTENT_TYPE: &str = "application/x-amz-json-1.1";
const TARGET_HEADER: &str = "X-Amz-Target";
const CONTENT_TYPE_HEADER: &str = "Content-Type";
const UNSUPPORTED_DOCUMENT: &str = "UnsupportedDocumentException";
const SYNC_DOCUMENT_MAX_BYTES: usize = 10 * 1024 * 1024;

const HEALTH_CHECK_IMAGE_DATA_URI: &str = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4//8/AAX+Av4N70a4AAAAAElFTkSuQmCC";

/// Textract has operations rather than models; the model slot of
/// `aws_textract/<model>` names the one to call.
#[derive(Clone, Copy, Debug, EnumString, IntoStaticStr, VariantNames, PartialEq, Eq)]
#[strum(serialize_all = "kebab-case", ascii_case_insensitive)]
pub enum TextractOperation {
    DetectDocumentText,
    AnalyzeDocument,
}

impl TextractOperation {
    pub const PROVIDER: &'static str = "aws_textract";

    pub fn from_model(model: &str) -> Result<Self, Error> {
        model.parse().map_err(|_| Error::InvalidModel {
            provider: Self::PROVIDER,
            model: model.to_string(),
            supported: Self::VARIANTS,
        })
    }

    fn target(self) -> &'static str {
        match self {
            Self::DetectDocumentText => "Textract.DetectDocumentText",
            Self::AnalyzeDocument => "Textract.AnalyzeDocument",
        }
    }
}

#[derive(Debug, Deserialize, Serialize)]
pub struct TextractDocument {
    #[serde(rename = "Bytes")]
    pub bytes: String,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum FeatureType {
    Tables,
    Forms,
    Queries,
    Signatures,
    Layout,
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub(super) enum BlockType {
    KeyValueSet,
    Page,
    Line,
    Word,
    Table,
    Cell,
    SelectionElement,
    MergedCell,
    Title,
    Query,
    QueryResult,
    Signature,
    TableTitle,
    TableFooter,
    LayoutText,
    LayoutTitle,
    LayoutHeader,
    LayoutFooter,
    LayoutSectionHeader,
    LayoutPageNumber,
    LayoutList,
    LayoutFigure,
    LayoutTable,
    LayoutKeyValue,
    #[serde(other)]
    Unknown,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum LayoutType {
    Text,
    Title,
    Header,
    Footer,
    SectionHeader,
    PageNumber,
    List,
    Figure,
    Table,
    KeyValue,
}

impl BlockType {
    pub fn layout(self) -> Option<LayoutType> {
        match self {
            Self::LayoutText => Some(LayoutType::Text),
            Self::LayoutTitle => Some(LayoutType::Title),
            Self::LayoutHeader => Some(LayoutType::Header),
            Self::LayoutFooter => Some(LayoutType::Footer),
            Self::LayoutSectionHeader => Some(LayoutType::SectionHeader),
            Self::LayoutPageNumber => Some(LayoutType::PageNumber),
            Self::LayoutList => Some(LayoutType::List),
            Self::LayoutFigure => Some(LayoutType::Figure),
            Self::LayoutTable => Some(LayoutType::Table),
            Self::LayoutKeyValue => Some(LayoutType::KeyValue),
            Self::KeyValueSet
            | Self::Page
            | Self::Line
            | Self::Word
            | Self::Table
            | Self::Cell
            | Self::SelectionElement
            | Self::MergedCell
            | Self::Title
            | Self::Query
            | Self::QueryResult
            | Self::Signature
            | Self::TableTitle
            | Self::TableFooter
            | Self::Unknown => None,
        }
    }
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub(super) enum RelationshipType {
    Value,
    Child,
    ComplexFeatures,
    MergedCell,
    Title,
    Answer,
    Table,
    TableTitle,
    TableFooter,
    #[serde(other)]
    Unknown,
}

#[derive(Deserialize)]
#[serde(rename_all = "PascalCase")]
pub(super) struct Block {
    #[serde(default)]
    pub id: String,
    pub block_type: BlockType,
    pub text: Option<String>,
    pub page: Option<i64>,
    pub row_index: Option<usize>,
    pub column_index: Option<usize>,
    #[serde(default)]
    pub relationships: Vec<Relationship>,
}

#[derive(Deserialize)]
#[serde(rename_all = "PascalCase")]
pub(super) struct Relationship {
    pub r#type: RelationshipType,
    #[serde(default)]
    pub ids: Vec<String>,
}

impl Block {
    pub fn page(&self) -> i64 {
        self.page.unwrap_or(1)
    }

    pub fn children(&self) -> impl Iterator<Item = &str> {
        self.relationships
            .iter()
            .filter(|relationship| relationship.r#type == RelationshipType::Child)
            .flat_map(|relationship| relationship.ids.iter().map(String::as_str))
    }
}

#[derive(Deserialize)]
#[serde(rename_all = "PascalCase")]
pub(super) struct DocumentMetadata {
    pub pages: Option<i64>,
}

#[derive(Deserialize)]
#[serde(rename_all = "PascalCase")]
pub struct TextractResponse {
    #[serde(default)]
    pub(super) blocks: Vec<Block>,
    pub(super) document_metadata: Option<DocumentMetadata>,
}

pub struct TextractEnvironment {
    headers: Vec<(String, String)>,
    region: String,
    signer: SigV4Signer,
}

impl OcrEnvironment for TextractEnvironment {
    fn headers(&self) -> &[(String, String)] {
        &self.headers
    }

    fn signer(&self) -> Option<&dyn RequestSigner> {
        Some(&self.signer)
    }
}

pub(super) fn health_check_document() -> OcrDocument {
    OcrDocument::ImageUrl {
        image_url: HEALTH_CHECK_IMAGE_DATA_URI.into(),
        extra_fields: Default::default(),
    }
}

pub(super) async fn environment(
    request: &PreparedOcrRequest,
    operation: TextractOperation,
) -> Result<TextractEnvironment, Error> {
    let env_lookup = |name: &str| request.connection.secret(name);
    let region =
        resolve_aws_region(None, &request.optional_params, &env_lookup).ok_or_else(|| {
            Error::InvalidRequest(
                "Missing AWS region - pass aws_region_name or set AWS_REGION_NAME or AWS_REGION"
                    .into(),
            )
        })?;
    let signer = SigV4Signer::resolve(
        region.clone(),
        TEXTRACT_SERVICE,
        &request.optional_params,
        &env_lookup,
    )
    .await
    .map_err(litellm_auth::Error::from)?;
    Ok(TextractEnvironment {
        headers: operation_headers(&request.connection.extra_headers, operation),
        region,
        signer,
    })
}

/// A caller's copy of an operation header would reach the wire next to ours
/// while the signature covers only one value, which Textract rejects.
fn operation_headers(
    extra_headers: &[(String, String)],
    operation: TextractOperation,
) -> Vec<(String, String)> {
    let operation = [
        (TARGET_HEADER, operation.target()),
        (CONTENT_TYPE_HEADER, AWS_JSON_CONTENT_TYPE),
    ];
    extra_headers
        .iter()
        .filter(|(name, _)| {
            !operation
                .iter()
                .any(|(operation_name, _)| name.eq_ignore_ascii_case(operation_name))
        })
        .cloned()
        .chain(
            operation
                .iter()
                .map(|(name, value)| (name.to_string(), value.to_string())),
        )
        .collect()
}

pub(super) fn endpoint(request: &PreparedOcrRequest, environment: &TextractEnvironment) -> String {
    request
        .connection
        .api_base
        .clone()
        .unwrap_or_else(|| format!("https://textract.{}.amazonaws.com/", environment.region))
}

pub(super) fn document_bytes(document: &OcrDocument) -> Result<TextractDocument, Error> {
    let inline = InlineDocument::parse(document.source())?.ok_or(Error::InvalidDataUri)?;
    Ok(TextractDocument {
        bytes: STANDARD.encode(inline.decode(SYNC_DOCUMENT_MAX_BYTES)?),
    })
}

pub(super) async fn inline_document(
    document: OcrDocument,
    context: OcrRequestContext<'_>,
) -> Result<OcrDocument, Error> {
    inline_remote_document(
        context.client.document_fetcher(),
        document,
        context.connection,
    )
    .await
}

#[derive(Deserialize)]
struct AwsError {
    #[serde(rename = "__type", default)]
    kind: String,
    #[serde(rename = "Message", alias = "message", default)]
    message: String,
}

/// Textract answers both an unsupported format and a multi-page PDF or TIFF
/// with a bare "unsupported document format", which reads like a corrupt file.
/// Say what the synchronous API accepts.
pub(super) fn error_class(body: String, status: u16, headers: Vec<(String, String)>) -> Error {
    let unsupported = serde_json::from_str::<AwsError>(&body)
        .ok()
        .filter(|error| error.kind.ends_with(UNSUPPORTED_DOCUMENT));
    Error::Provider {
        status,
        body: match unsupported {
            Some(error) => format!(
                "{UNSUPPORTED_DOCUMENT}: {}. aws_textract uses Textract's synchronous API, which reads a JPEG, PNG, or a single-page PDF or TIFF; other formats and multi-page documents are not supported",
                error.message
            ),
            None => body,
        },
        headers,
    }
}

pub(super) fn lines_by_page(blocks: &[Block]) -> Vec<(i64, String)> {
    let pages: std::collections::BTreeSet<i64> = blocks.iter().map(Block::page).collect();
    pages
        .into_iter()
        .map(|page| {
            let lines: Vec<&str> = blocks
                .iter()
                .filter(|block| block.block_type == BlockType::Line && block.page() == page)
                .filter_map(|block| block.text.as_deref())
                .collect();
            (page, lines.join("\n"))
        })
        .filter(|(_, markdown)| !markdown.is_empty())
        .collect()
}

pub(super) fn ocr_response(
    model: &str,
    page_markdown: Vec<(i64, String)>,
    document_metadata: Option<DocumentMetadata>,
) -> LiteLLMOcrResponse {
    let pages: Vec<OcrPage> = page_markdown
        .into_iter()
        .map(|(page, markdown)| OcrPage {
            index: page - 1,
            markdown,
            ..Default::default()
        })
        .collect();
    let pages_processed = document_metadata
        .and_then(|metadata| metadata.pages)
        .or_else(|| i64::try_from(pages.len()).ok());
    LiteLLMOcrResponse {
        usage_info: Some(OcrUsageInfo {
            pages_processed,
            ..Default::default()
        }),
        ..LiteLLMOcrResponse::new(model, pages)
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use serde_json::{Value, json};

    use super::*;

    const HINT: &str = "other formats and multi-page documents are not supported";

    fn blocks(value: Value) -> Vec<Block> {
        serde_json::from_value(value).unwrap()
    }

    #[rstest]
    #[case::detect("detect-document-text", TextractOperation::DetectDocumentText)]
    #[case::analyze("analyze-document", TextractOperation::AnalyzeDocument)]
    #[case::any_case("Analyze-Document", TextractOperation::AnalyzeDocument)]
    fn a_model_names_its_operation(#[case] model: &str, #[case] expected: TextractOperation) {
        assert_eq!(TextractOperation::from_model(model).unwrap(), expected);
    }

    #[rstest]
    #[case::misspelled("analyse-document")]
    #[case::operation_name_from_the_api("AnalyzeDocument")]
    #[case::operation_litellm_does_not_call("analyze-expense")]
    #[case::empty("")]
    fn a_model_outside_the_operations_is_refused_with_the_supported_names(#[case] model: &str) {
        let error = TextractOperation::from_model(model).unwrap_err();

        assert_eq!(
            error.to_string(),
            format!(
                "invalid model: aws_textract has no model {model:?} - use one of: detect-document-text, analyze-document"
            )
        );
        assert_eq!(error.http_status_code(), Some(400));
    }

    #[rstest]
    #[case::line("LINE", BlockType::Line)]
    #[case::key_value_set("KEY_VALUE_SET", BlockType::KeyValueSet)]
    #[case::layout_section_header("LAYOUT_SECTION_HEADER", BlockType::LayoutSectionHeader)]
    #[case::layout_key_value("LAYOUT_KEY_VALUE", BlockType::LayoutKeyValue)]
    #[case::added_by_textract_later("LAYOUT_SIDEBAR", BlockType::Unknown)]
    fn block_type_reads_the_documented_names(#[case] wire: &str, #[case] expected: BlockType) {
        let block: Block = serde_json::from_value(json!({"BlockType": wire})).unwrap();

        assert_eq!(block.block_type, expected);
    }

    #[rstest]
    #[case::layout_title(BlockType::LayoutTitle, Some(LayoutType::Title))]
    #[case::layout_table(BlockType::LayoutTable, Some(LayoutType::Table))]
    #[case::table_is_not_layout(BlockType::Table, None)]
    #[case::title_is_not_layout(BlockType::Title, None)]
    #[case::unknown_is_not_layout(BlockType::Unknown, None)]
    fn only_layout_block_types_have_a_layout_type(
        #[case] block_type: BlockType,
        #[case] expected: Option<LayoutType>,
    ) {
        assert_eq!(block_type.layout(), expected);
    }

    #[rstest]
    #[case::child_only(json!([{"Type": "CHILD", "Ids": ["a", "b"]}]), vec!["a", "b"])]
    #[case::other_relationships_are_skipped(
        json!([
            {"Type": "TABLE_TITLE", "Ids": ["t"]},
            {"Type": "CHILD", "Ids": ["a"]},
            {"Type": "MERGED_CELL", "Ids": ["m"]},
            {"Type": "ADDED_LATER", "Ids": ["x"]},
            {"Type": "CHILD", "Ids": ["b"]}
        ]),
        vec!["a", "b"]
    )]
    #[case::no_relationships(json!([]), vec![])]
    fn children_are_the_ids_of_child_relationships(
        #[case] relationships: Value,
        #[case] expected: Vec<&str>,
    ) {
        let block: Block =
            serde_json::from_value(json!({"BlockType": "LINE", "Relationships": relationships}))
                .unwrap();

        assert_eq!(block.children().collect::<Vec<_>>(), expected);
    }

    #[rstest]
    #[case::tables("TABLES", Some(FeatureType::Tables))]
    #[case::forms("FORMS", Some(FeatureType::Forms))]
    #[case::queries("QUERIES", Some(FeatureType::Queries))]
    #[case::signatures("SIGNATURES", Some(FeatureType::Signatures))]
    #[case::layout("LAYOUT", Some(FeatureType::Layout))]
    #[case::lowercase_is_not_a_feature("layout", None)]
    #[case::undocumented("HANDWRITING", None)]
    fn feature_type_accepts_only_the_documented_values(
        #[case] wire: &str,
        #[case] expected: Option<FeatureType>,
    ) {
        assert_eq!(
            serde_json::from_value::<FeatureType>(json!(wire)).ok(),
            expected
        );
        if let Some(feature) = expected {
            assert_eq!(serde_json::to_value(feature).unwrap(), json!(wire));
        }
    }

    #[rstest]
    #[case::image_url(
        OcrDocument::ImageUrl {
            image_url: "data:image/png;base64,aGVsbG8=".into(),
            extra_fields: Default::default(),
        },
        "aGVsbG8="
    )]
    #[case::document_url(
        OcrDocument::DocumentUrl {
            document_url: "data:application/pdf;base64,YWJj".into(),
            extra_fields: Default::default(),
        },
        "YWJj"
    )]
    #[case::percent_encoded_data_uri_is_re_encoded_as_base64(
        OcrDocument::DocumentUrl {
            document_url: "data:,abc".into(),
            extra_fields: Default::default(),
        },
        "YWJj"
    )]
    fn document_bytes_are_the_base64_payload_without_the_data_uri_envelope(
        #[case] document: OcrDocument,
        #[case] expected: &str,
    ) {
        assert_eq!(document_bytes(&document).unwrap().bytes, expected);
    }

    #[rstest]
    #[case::remote_url("https://example.com/a.pdf".to_string(), Error::InvalidDataUri)]
    #[case::invalid_base64("data:image/png;base64,@@@".to_string(), Error::InvalidDataUri)]
    #[case::over_the_sync_limit(
        format!("data:,{}", "a".repeat(SYNC_DOCUMENT_MAX_BYTES + 1)),
        Error::InlineDocumentTooLarge
    )]
    fn document_bytes_refuse_what_the_sync_api_cannot_take(
        #[case] document_url: String,
        #[case] expected: Error,
    ) {
        let error = document_bytes(&OcrDocument::DocumentUrl {
            document_url,
            extra_fields: Default::default(),
        })
        .unwrap_err();

        assert_eq!(
            std::mem::discriminant(&error),
            std::mem::discriminant(&expected)
        );
    }

    #[rstest]
    #[case::bare_type(
        r#"{"__type":"UnsupportedDocumentException","Message":"Request has unsupported document format"}"#,
        Some("Request has unsupported document format")
    )]
    #[case::namespaced_type(
        r#"{"__type":"com.amazonaws.textract#UnsupportedDocumentException","Message":"bad"}"#,
        Some("bad")
    )]
    #[case::lowercase_message(
        r#"{"__type":"UnsupportedDocumentException","message":"bad"}"#,
        Some("bad")
    )]
    #[case::other_exception(r#"{"__type":"AccessDeniedException","Message":"no"}"#, None)]
    #[case::json_without_a_type(r#"{"Message":"no"}"#, None)]
    #[case::not_json("<html>bad gateway</html>", None)]
    fn only_an_unsupported_document_gains_the_sync_api_hint(
        #[case] body: &str,
        #[case] hinted_message: Option<&str>,
    ) {
        let response_headers = vec![("x-amzn-requestid".to_string(), "abc".to_string())];

        let Error::Provider {
            status,
            body: reported,
            headers,
        } = error_class(body.into(), 400, response_headers.clone())
        else {
            panic!("expected a provider error");
        };

        assert_eq!(status, 400);
        assert_eq!(headers, response_headers);
        match hinted_message {
            Some(message) => {
                assert!(reported.contains(message), "{reported}");
                assert!(reported.contains(HINT), "{reported}");
            }
            None => assert_eq!(reported, body),
        }
    }

    #[rstest]
    #[case::no_caller_headers(vec![], vec![])]
    #[case::unrelated_headers_are_kept(vec![("x-trace", "1")], vec![("x-trace", "1")])]
    #[case::a_caller_content_type_is_replaced(
        vec![("content-type", "application/json"), ("x-trace", "1")],
        vec![("x-trace", "1")]
    )]
    #[case::a_caller_target_is_replaced(
        vec![("X-AMZ-TARGET", "Textract.AnalyzeDocument")],
        vec![]
    )]
    fn operation_headers_are_sent_once(
        #[case] extra_headers: Vec<(&str, &str)>,
        #[case] kept: Vec<(&str, &str)>,
    ) {
        let owned = |headers: Vec<(&str, &str)>| -> Vec<(String, String)> {
            headers
                .into_iter()
                .map(|(name, value)| (name.to_string(), value.to_string()))
                .collect()
        };

        let headers =
            operation_headers(&owned(extra_headers), TextractOperation::DetectDocumentText);

        let mut expected = owned(kept);
        expected.extend(owned(vec![
            ("X-Amz-Target", "Textract.DetectDocumentText"),
            ("Content-Type", "application/x-amz-json-1.1"),
        ]));
        assert_eq!(headers, expected);
    }

    #[rstest]
    #[case::words_are_not_repeated(
        json!([
            {"BlockType": "PAGE"},
            {"BlockType": "LINE", "Text": "Invoice 12345"},
            {"BlockType": "WORD", "Text": "Invoice"},
            {"BlockType": "WORD", "Text": "12345"},
            {"BlockType": "LINE", "Text": "total 67.89"}
        ]),
        vec![(1, "Invoice 12345\ntotal 67.89")]
    )]
    #[case::pages_are_sorted_and_keep_line_order(
        json!([
            {"BlockType": "LINE", "Text": "second", "Page": 2},
            {"BlockType": "LINE", "Text": "first", "Page": 1},
            {"BlockType": "LINE", "Text": "also second", "Page": 2}
        ]),
        vec![(1, "first"), (2, "second\nalso second")]
    )]
    #[case::a_page_without_lines_is_dropped(
        json!([
            {"BlockType": "PAGE", "Page": 1},
            {"BlockType": "LINE", "Text": "only", "Page": 2}
        ]),
        vec![(2, "only")]
    )]
    #[case::no_blocks(json!([]), vec![])]
    fn lines_are_grouped_by_page(#[case] input: Value, #[case] expected: Vec<(i64, &str)>) {
        let pages = lines_by_page(&blocks(input));

        let pages: Vec<(i64, &str)> = pages
            .iter()
            .map(|(page, markdown)| (*page, markdown.as_str()))
            .collect();
        assert_eq!(pages, expected);
    }

    #[rstest]
    #[case::metadata_wins(Some(3), Some(3))]
    #[case::metadata_without_pages_falls_back_to_the_page_count(None, Some(2))]
    fn pages_are_zero_indexed_and_usage_reports_pages_processed(
        #[case] metadata_pages: Option<i64>,
        #[case] expected: Option<i64>,
    ) {
        let response = ocr_response(
            "detect-document-text",
            vec![(1, "first".into()), (3, "third".into())],
            Some(DocumentMetadata {
                pages: metadata_pages,
            }),
        );

        let pages: Vec<(i64, &str)> = response
            .pages
            .iter()
            .map(|page| (page.index, page.markdown.as_str()))
            .collect();
        assert_eq!(pages, vec![(0, "first"), (2, "third")]);
        assert_eq!(response.usage_info.unwrap().pages_processed, expected);
    }
}
