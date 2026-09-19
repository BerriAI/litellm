use std::collections::{BTreeMap, BTreeSet, HashMap};

use litellm_core_utils::call_arguments::{CallArguments, parse_options};
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

const ANALYZE_DOCUMENT_TARGET: &str = "Textract.AnalyzeDocument";
const DEFAULT_FEATURE_TYPES: [&str; 2] = ["LAYOUT", "TABLES"];

#[derive(Default, Deserialize)]
pub struct AnalyzeDocumentOptions {
    pub feature_types: Option<Vec<String>>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct AnalyzeDocumentRequest {
    #[serde(rename = "Document")]
    pub document: TextractDocument,
    #[serde(rename = "FeatureTypes")]
    pub feature_types: Vec<String>,
}

#[derive(Deserialize)]
#[serde(rename_all = "PascalCase")]
pub struct AnalyzeDocumentResponse {
    #[serde(default)]
    blocks: Vec<Block>,
    document_metadata: Option<DocumentMetadata>,
}

/// Synchronous `AnalyzeDocument`: layout and tables rendered as markdown.
#[derive(Clone, Copy, Debug, Default)]
pub struct TextractAnalyzeDocumentConfig;

impl BaseOcrConfig for TextractAnalyzeDocumentConfig {
    type OcrParams = AnalyzeDocumentOptions;
    type ProviderRequest = AnalyzeDocumentRequest;
    type Environment = TextractEnvironment;

    fn get_supported_ocr_params(&self, _model: &str) -> &'static [&'static str] {
        &["feature_types"]
    }

    fn get_health_check_document(&self) -> OcrDocument {
        OcrDocument::ImageUrl {
            image_url: HEALTH_CHECK_IMAGE_DATA_URI.into(),
            extra_fields: Default::default(),
        }
    }

    fn map_ocr_params(
        &self,
        non_default_params: &CallArguments,
        _model: &str,
    ) -> Result<AnalyzeDocumentOptions, Error> {
        Ok(parse_options(non_default_params)?)
    }

    async fn validate_environment(
        &self,
        request: &PreparedOcrRequest,
        _client: &OcrClient,
    ) -> Result<TextractEnvironment, Error> {
        environment(request, ANALYZE_DOCUMENT_TARGET).await
    }

    fn get_complete_url(
        &self,
        request: &PreparedOcrRequest,
        _optional_params: &AnalyzeDocumentOptions,
        environment: &TextractEnvironment,
    ) -> Result<String, Error> {
        Ok(endpoint(request, environment))
    }

    fn transform_ocr_request(
        &self,
        _model: &str,
        document: OcrDocument,
        optional_params: &AnalyzeDocumentOptions,
        _headers: &[(String, String)],
    ) -> Result<AnalyzeDocumentRequest, Error> {
        Ok(AnalyzeDocumentRequest {
            document: document_bytes(&document)?,
            feature_types: optional_params.feature_types.clone().unwrap_or_else(|| {
                DEFAULT_FEATURE_TYPES
                    .iter()
                    .map(|feature| feature.to_string())
                    .collect()
            }),
        })
    }

    async fn async_transform_ocr_request(
        &self,
        model: &str,
        document: OcrDocument,
        optional_params: &AnalyzeDocumentOptions,
        headers: &[(String, String)],
        context: OcrRequestContext<'_>,
    ) -> Result<AnalyzeDocumentRequest, Error> {
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
    response: AnalyzeDocumentResponse,
) -> Result<LiteLLMOcrResponse, Error> {
    let blocks = &response.blocks;
    let has_layout = blocks.iter().any(is_layout);
    let page_markdown: Vec<(i64, String)> = if has_layout {
        let by_id: HashMap<&str, &Block> = blocks
            .iter()
            .map(|block| (block.id.as_str(), block))
            .collect();
        let pages: BTreeSet<i64> = blocks.iter().map(Block::page).collect();
        pages
            .into_iter()
            .map(|page| (page, layout_markdown(blocks, page, &by_id)))
            .filter(|(_, markdown)| !markdown.is_empty())
            .collect()
    } else {
        lines_by_page(blocks)
    };
    let pages: Vec<OcrPage> = page_markdown
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

fn is_layout(block: &Block) -> bool {
    block.block_type.starts_with("LAYOUT_")
}

/// Layout blocks arrive in reading order. A list's items are repeated as
/// top-level `LAYOUT_TEXT` blocks, and a `LAYOUT_TABLE` only links to the
/// table's lines, so the nth layout table on a page takes the nth `TABLE`.
fn layout_markdown(blocks: &[Block], page: i64, by_id: &HashMap<&str, &Block>) -> String {
    let on_page = || blocks.iter().filter(move |block| block.page() == page);
    let list_items: BTreeSet<&str> = on_page()
        .filter(|block| block.block_type == "LAYOUT_LIST")
        .flat_map(Block::children)
        .collect();
    let tables: Vec<&Block> = on_page()
        .filter(|block| block.block_type == "TABLE")
        .collect();
    let table_ordinal: HashMap<&str, usize> = on_page()
        .filter(|block| block.block_type == "LAYOUT_TABLE")
        .enumerate()
        .map(|(ordinal, block)| (block.id.as_str(), ordinal))
        .collect();
    let sections: Vec<String> = on_page()
        .filter(|block| is_layout(block) && !list_items.contains(block.id.as_str()))
        .map(|block| match block.block_type.as_str() {
            "LAYOUT_TITLE" => format!("# {}", text_of(block, by_id, " ")),
            "LAYOUT_SECTION_HEADER" => format!("## {}", text_of(block, by_id, " ")),
            "LAYOUT_LIST" => block
                .children()
                .filter_map(|id| by_id.get(id))
                .map(|item| format!("- {}", strip_bullet(&text_of(item, by_id, " "))))
                .collect::<Vec<_>>()
                .join("\n"),
            "LAYOUT_TABLE" => table_ordinal
                .get(block.id.as_str())
                .and_then(|ordinal| tables.get(*ordinal))
                .map(|table| table_markdown(table, by_id))
                .unwrap_or_else(|| text_of(block, by_id, "\n")),
            _ => text_of(block, by_id, " "),
        })
        .filter(|section| !section.trim().is_empty())
        .collect();
    sections.join("\n\n")
}

fn text_of(block: &Block, by_id: &HashMap<&str, &Block>, separator: &str) -> String {
    match &block.text {
        Some(text) => text.clone(),
        None => block
            .children()
            .filter_map(|id| by_id.get(id))
            .map(|child| text_of(child, by_id, separator))
            .filter(|text| !text.is_empty())
            .collect::<Vec<_>>()
            .join(separator),
    }
}

fn strip_bullet(item: &str) -> &str {
    item.trim_start_matches(['-', '*', '\u{2022}', '\u{00b7}'])
        .trim_start()
}

fn table_markdown(table: &Block, by_id: &HashMap<&str, &Block>) -> String {
    let cells: BTreeMap<(usize, usize), String> = table
        .children()
        .filter_map(|id| by_id.get(id))
        .filter(|cell| cell.block_type == "CELL")
        .filter_map(|cell| {
            Some((
                (cell.row_index?, cell.column_index?),
                text_of(cell, by_id, " ").replace('|', "\\|"),
            ))
        })
        .collect();
    let columns = cells.keys().map(|(_, column)| *column).max().unwrap_or(0);
    let rows: BTreeSet<usize> = cells.keys().map(|(row, _)| *row).collect();
    let render = |row: usize| {
        let values: Vec<&str> = (1..=columns)
            .map(|column| cells.get(&(row, column)).map_or("", String::as_str))
            .collect();
        format!("| {} |", values.join(" | "))
    };
    let divider = format!("|{}", " --- |".repeat(columns));
    rows.iter()
        .enumerate()
        .flat_map(|(position, row)| {
            std::iter::once(render(*row)).chain((position == 0).then(|| divider.clone()))
        })
        .collect::<Vec<_>>()
        .join("\n")
}

#[cfg(test)]
mod tests {
    use serde_json::{Value, json};

    use super::*;

    fn markdown(blocks: Value) -> Vec<(i64, String)> {
        TextractAnalyzeDocumentConfig
            .transform_ocr_response(
                "analyze-document",
                &serde_json::to_vec(&json!({"DocumentMetadata": {"Pages": 1}, "Blocks": blocks}))
                    .unwrap(),
                OcrResponseFormat::Litellm,
            )
            .unwrap()
            .pages
            .into_iter()
            .map(|page| (page.index, page.markdown))
            .collect()
    }

    fn child(ids: &[&str]) -> Value {
        json!([{"Type": "CHILD", "Ids": ids}])
    }

    fn line(id: &str, text: &str) -> Value {
        json!({"Id": id, "BlockType": "LINE", "Text": text})
    }

    fn word(id: &str, text: &str) -> Value {
        json!({"Id": id, "BlockType": "WORD", "Text": text})
    }

    fn cell(id: &str, row: usize, column: usize, words: &[&str]) -> Value {
        json!({"Id": id, "BlockType": "CELL", "RowIndex": row, "ColumnIndex": column,
            "Relationships": child(words)})
    }

    #[test]
    fn layout_becomes_headings_paragraphs_and_a_list_without_repeating_its_items() {
        let pages = markdown(json!([
            line("l1", "Quarterly Report"),
            line("l2", "This report lists"),
            line("l3", "the invoices."),
            line("l4", "Line items"),
            line("l5", "- Pay within 30 days"),
            line("l6", "\u{2022} Quote the number"),
            {"Id": "t", "BlockType": "LAYOUT_TITLE", "Relationships": child(&["l1"])},
            {"Id": "p", "BlockType": "LAYOUT_TEXT", "Relationships": child(&["l2", "l3"])},
            {"Id": "h", "BlockType": "LAYOUT_SECTION_HEADER", "Relationships": child(&["l4"])},
            {"Id": "ul", "BlockType": "LAYOUT_LIST", "Relationships": child(&["i1", "i2"])},
            {"Id": "i1", "BlockType": "LAYOUT_TEXT", "Relationships": child(&["l5"])},
            {"Id": "i2", "BlockType": "LAYOUT_TEXT", "Relationships": child(&["l6"])}
        ]));

        assert_eq!(
            pages,
            vec![(
                0,
                "# Quarterly Report\n\nThis report lists the invoices.\n\n## Line items\n\n- Pay within 30 days\n- Quote the number".to_string()
            )]
        );
    }

    #[test]
    fn a_layout_table_is_rendered_from_the_table_cells_in_row_and_column_order() {
        let pages = markdown(json!([
            line("l1", "Invoice"), line("l2", "Total"), line("l3", "12345"), line("l4", "a|b"),
            word("w1", "Invoice"), word("w2", "Total"), word("w3", "12345"), word("w4", "a|b"),
            {"Id": "tb", "BlockType": "TABLE", "Relationships": [
                {"Type": "CHILD", "Ids": ["c4", "c1", "c3", "c2"]},
                {"Type": "TABLE_TITLE", "Ids": ["title"]}
            ]},
            cell("c1", 1, 1, &["w1"]), cell("c2", 1, 2, &["w2"]),
            cell("c3", 2, 1, &["w3"]), cell("c4", 2, 2, &["w4"]),
            {"Id": "lt", "BlockType": "LAYOUT_TABLE", "Relationships": child(&["l1", "l2", "l3", "l4"])}
        ]));

        assert_eq!(
            pages,
            vec![(
                0,
                "| Invoice | Total |\n| --- | --- |\n| 12345 | a\\|b |".to_string()
            )]
        );
    }

    #[test]
    fn a_layout_table_without_table_blocks_keeps_its_lines() {
        let pages = markdown(json!([
            line("l1", "Invoice Total"),
            line("l2", "12345 67.89"),
            {"Id": "lt", "BlockType": "LAYOUT_TABLE", "Relationships": child(&["l1", "l2"])}
        ]));

        assert_eq!(pages, vec![(0, "Invoice Total\n12345 67.89".to_string())]);
    }

    #[test]
    fn a_response_without_layout_blocks_falls_back_to_lines() {
        let pages = markdown(json!([
            line("l1", "first"),
            word("w1", "first"),
            line("l2", "second")
        ]));

        assert_eq!(pages, vec![(0, "first\nsecond".to_string())]);
    }

    #[test]
    fn each_page_gets_its_own_markdown_and_its_own_tables() {
        let pages = markdown(json!([
            {"Id": "a", "BlockType": "LINE", "Text": "one", "Page": 1},
            {"Id": "b", "BlockType": "LINE", "Text": "two", "Page": 2},
            {"Id": "w", "BlockType": "WORD", "Text": "cell", "Page": 2},
            {"Id": "t1", "BlockType": "LAYOUT_TEXT", "Page": 1, "Relationships": child(&["a"])},
            {"Id": "tb", "BlockType": "TABLE", "Page": 2, "Relationships": child(&["c"])},
            {"Id": "c", "BlockType": "CELL", "Page": 2, "RowIndex": 1, "ColumnIndex": 1,
                "Relationships": child(&["w"])},
            {"Id": "lt", "BlockType": "LAYOUT_TABLE", "Page": 2, "Relationships": child(&["b"])}
        ]));

        assert_eq!(
            pages,
            vec![(0, "one".to_string()), (1, "| cell |\n| --- |".to_string())]
        );
    }

    #[test]
    fn feature_types_default_to_layout_and_tables_and_can_be_overridden() {
        let document = || OcrDocument::ImageUrl {
            image_url: "data:image/png;base64,aGk=".into(),
            extra_fields: Default::default(),
        };
        let request = |options: Value| {
            let arguments: CallArguments = serde_json::from_value(options).unwrap();
            let params = TextractAnalyzeDocumentConfig
                .map_ocr_params(&arguments, "analyze-document")
                .unwrap();
            serde_json::to_value(
                TextractAnalyzeDocumentConfig
                    .transform_ocr_request("analyze-document", document(), &params, &[])
                    .unwrap(),
            )
            .unwrap()
        };

        assert_eq!(
            request(json!({})),
            json!({"Document": {"Bytes": "aGk="}, "FeatureTypes": ["LAYOUT", "TABLES"]})
        );
        assert_eq!(
            request(json!({"feature_types": ["FORMS"]}))["FeatureTypes"],
            json!(["FORMS"])
        );
    }
}
