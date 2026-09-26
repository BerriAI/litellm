use std::collections::{BTreeMap, BTreeSet, HashMap};

use litellm_core_utils::call_arguments::{CallArguments, parse_options};
use serde::{Deserialize, Serialize};

use super::common_utils::{
    Block, BlockType, FeatureType, LayoutType, TextractDocument, TextractEnvironment,
    TextractOperation, TextractResponse, document_bytes, endpoint, environment, error_class,
    health_check_document, inline_document, lines_by_page, ocr_response,
};
use crate::base_llm::ocr::{
    error::Error,
    handler::OcrClient,
    transformation::{
        BaseOcrConfig, LiteLLMOcrResponse, OcrDocument, OcrRequestContext, OcrResponseFormat,
        PreparedOcrRequest, decode_and_normalize_response,
    },
};

const DEFAULT_FEATURE_TYPES: [FeatureType; 2] = [FeatureType::Layout, FeatureType::Tables];

#[derive(Default, Deserialize)]
pub struct AnalyzeDocumentOptions {
    pub feature_types: Option<Vec<FeatureType>>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct AnalyzeDocumentRequest {
    #[serde(rename = "Document")]
    pub document: TextractDocument,
    #[serde(rename = "FeatureTypes")]
    pub feature_types: Vec<FeatureType>,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct TextractAnalyzeDocumentConfig;

impl BaseOcrConfig for TextractAnalyzeDocumentConfig {
    type OcrParams = AnalyzeDocumentOptions;
    type ProviderRequest = AnalyzeDocumentRequest;
    type Environment = TextractEnvironment;

    fn secret_names(&self) -> Vec<&'static str> {
        litellm_auth_aws::constants::SECRET_NAMES.to_vec()
    }

    fn get_supported_ocr_params(&self, _model: &str) -> &'static [&'static str] {
        &["feature_types"]
    }

    fn get_health_check_document(&self) -> OcrDocument {
        health_check_document()
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
        client: &OcrClient,
    ) -> Result<TextractEnvironment, Error> {
        environment(
            &client.auth().aws,
            request,
            TextractOperation::AnalyzeDocument,
        )
        .await
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
            feature_types: optional_params
                .feature_types
                .clone()
                .unwrap_or_else(|| DEFAULT_FEATURE_TYPES.to_vec()),
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
    response: TextractResponse,
) -> Result<LiteLLMOcrResponse, Error> {
    let blocks = &response.blocks;
    let has_layout = blocks
        .iter()
        .any(|block| block.block_type.layout().is_some());
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
    Ok(ocr_response(
        model,
        page_markdown,
        response.document_metadata,
    ))
}

/// Layout blocks arrive in reading order. A list's items are repeated as
/// top-level `LAYOUT_TEXT` blocks. A `LAYOUT_TABLE` that links to its `TABLE`
/// renders it; one that only links to the table's lines takes the `TABLE` at
/// the same position on the page.
fn layout_markdown(blocks: &[Block], page: i64, by_id: &HashMap<&str, &Block>) -> String {
    let on_page = || blocks.iter().filter(move |block| block.page() == page);
    let list_items: BTreeSet<&str> = on_page()
        .filter(|block| block.block_type == BlockType::LayoutList)
        .flat_map(Block::children)
        .collect();
    let tables: Vec<&Block> = on_page()
        .filter(|block| block.block_type == BlockType::Table)
        .collect();
    let table_ordinal: HashMap<&str, usize> = on_page()
        .filter(|block| block.block_type == BlockType::LayoutTable)
        .enumerate()
        .map(|(ordinal, block)| (block.id.as_str(), ordinal))
        .collect();
    let table_of = |layout_table: &Block| {
        layout_table
            .children()
            .filter_map(|id| by_id.get(id).copied())
            .find(|child| child.block_type == BlockType::Table)
            .or_else(|| {
                table_ordinal
                    .get(layout_table.id.as_str())
                    .and_then(|ordinal| tables.get(*ordinal).copied())
            })
    };
    let sections: Vec<String> = on_page()
        .filter(|block| !list_items.contains(block.id.as_str()))
        .filter_map(|block| Some((block, block.block_type.layout()?)))
        .map(|(block, layout)| match layout {
            LayoutType::Title => format!("# {}", text_of(block, by_id, " ")),
            LayoutType::SectionHeader => format!("## {}", text_of(block, by_id, " ")),
            LayoutType::List => block
                .children()
                .filter_map(|id| by_id.get(id))
                .map(|item| format!("- {}", strip_bullet(&text_of(item, by_id, " "))))
                .collect::<Vec<_>>()
                .join("\n"),
            LayoutType::Table => match table_of(block) {
                Some(table) => table_markdown(table, by_id),
                None => text_of(block, by_id, "\n"),
            },
            LayoutType::KeyValue => text_of(block, by_id, "\n"),
            LayoutType::Figure => String::new(),
            LayoutType::Text | LayoutType::Header | LayoutType::Footer | LayoutType::PageNumber => {
                text_of(block, by_id, " ")
            }
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
        .filter(|cell| cell.block_type == BlockType::Cell)
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
    use rstest::{fixture, rstest};
    use serde_json::{Value, json};

    use super::*;

    const MODEL: &str = "analyze-document";

    #[fixture]
    fn document() -> OcrDocument {
        OcrDocument::ImageUrl {
            image_url: "data:image/png;base64,aGk=".into(),
            extra_fields: Default::default(),
        }
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

    fn layout(id: &str, block_type: &str, children: &[&str]) -> Value {
        json!({"Id": id, "BlockType": block_type, "Relationships": child(children)})
    }

    fn table(id: &str, cells: &[&str]) -> Value {
        json!({"Id": id, "BlockType": "TABLE", "Relationships": child(cells)})
    }

    fn cell(id: &str, row: usize, column: usize, words: &[&str]) -> Value {
        json!({"Id": id, "BlockType": "CELL", "RowIndex": row, "ColumnIndex": column,
            "Relationships": child(words)})
    }

    fn on_page(page: i64, mut block: Value) -> Value {
        block["Page"] = json!(page);
        block
    }

    #[rstest]
    #[case::headings_paragraphs_and_a_list_without_repeating_its_items(
        json!([
            line("l1", "Quarterly Report"),
            line("l2", "This report lists"),
            line("l3", "the invoices."),
            line("l4", "Line items"),
            line("l5", "- Pay within 30 days"),
            line("l6", "\u{2022} Quote the number"),
            layout("t", "LAYOUT_TITLE", &["l1"]),
            layout("p", "LAYOUT_TEXT", &["l2", "l3"]),
            layout("h", "LAYOUT_SECTION_HEADER", &["l4"]),
            layout("ul", "LAYOUT_LIST", &["i1", "i2"]),
            layout("i1", "LAYOUT_TEXT", &["l5"]),
            layout("i2", "LAYOUT_TEXT", &["l6"])
        ]),
        vec![(
            0,
            "# Quarterly Report\n\nThis report lists the invoices.\n\n## Line items\n\n- Pay within 30 days\n- Quote the number"
        )]
    )]
    #[case::header_footer_and_page_number_stay_in_reading_order(
        json!([
            line("l1", "ACME Corp"), line("l2", "Body"), line("l3", "Confidential"), line("l4", "3"),
            layout("hd", "LAYOUT_HEADER", &["l1"]),
            layout("p", "LAYOUT_TEXT", &["l2"]),
            layout("ft", "LAYOUT_FOOTER", &["l3"]),
            layout("pn", "LAYOUT_PAGE_NUMBER", &["l4"])
        ]),
        vec![(0, "ACME Corp\n\nBody\n\nConfidential\n\n3")]
    )]
    #[case::a_table_is_rendered_from_its_cells_in_row_and_column_order(
        json!([
            line("l1", "Invoice"), line("l2", "Total"), line("l3", "12345"), line("l4", "a|b"),
            word("w1", "Invoice"), word("w2", "Total"), word("w3", "12345"), word("w4", "a|b"),
            {"Id": "tb", "BlockType": "TABLE", "Relationships": [
                {"Type": "CHILD", "Ids": ["c4", "c1", "c3", "c2"]},
                {"Type": "TABLE_TITLE", "Ids": ["title"]}
            ]},
            cell("c1", 1, 1, &["w1"]), cell("c2", 1, 2, &["w2"]),
            cell("c3", 2, 1, &["w3"]), cell("c4", 2, 2, &["w4"]),
            layout("lt", "LAYOUT_TABLE", &["l1", "l2", "l3", "l4"])
        ]),
        vec![(0, "| Invoice | Total |\n| --- | --- |\n| 12345 | a\\|b |")]
    )]
    #[case::a_layout_table_that_links_its_table_renders_that_one(
        json!([
            word("w1", "first"), word("w2", "second"),
            table("tb1", &["c1"]), cell("c1", 1, 1, &["w1"]),
            table("tb2", &["c2"]), cell("c2", 1, 1, &["w2"]),
            layout("lt", "LAYOUT_TABLE", &["tb2"])
        ]),
        vec![(0, "| second |\n| --- |")]
    )]
    #[case::a_missing_cell_leaves_an_empty_column(
        json!([
            word("w1", "a"), word("w2", "b"), word("w3", "c"),
            table("tb", &["c1", "c2", "c3"]),
            cell("c1", 1, 1, &["w1"]), cell("c2", 1, 2, &["w2"]), cell("c3", 2, 2, &["w3"]),
            layout("lt", "LAYOUT_TABLE", &[])
        ]),
        vec![(0, "| a | b |\n| --- | --- |\n|  | c |")]
    )]
    #[case::a_layout_table_without_table_blocks_keeps_its_lines(
        json!([
            line("l1", "Invoice Total"),
            line("l2", "12345 67.89"),
            layout("lt", "LAYOUT_TABLE", &["l1", "l2"])
        ]),
        vec![(0, "Invoice Total\n12345 67.89")]
    )]
    #[case::key_values_keep_one_line_each(
        json!([
            line("l1", "Name: Ana"),
            line("l2", "Date: 2024-01-01"),
            layout("kv", "LAYOUT_KEY_VALUE", &["l1", "l2"])
        ]),
        vec![(0, "Name: Ana\nDate: 2024-01-01")]
    )]
    #[case::a_figure_has_no_markdown(
        json!([
            line("l1", "Caption"),
            layout("f", "LAYOUT_FIGURE", &[]),
            layout("p", "LAYOUT_TEXT", &["l1"])
        ]),
        vec![(0, "Caption")]
    )]
    #[case::a_block_type_added_later_is_ignored(
        json!([
            line("l1", "Body"),
            layout("new", "LAYOUT_SIDEBAR", &["l1"]),
            layout("p", "LAYOUT_TEXT", &["l1"])
        ]),
        vec![(0, "Body")]
    )]
    #[case::without_layout_blocks_lines_are_used(
        json!([line("l1", "first"), word("w1", "first"), line("l2", "second")]),
        vec![(0, "first\nsecond")]
    )]
    #[case::each_page_gets_its_own_markdown_and_its_own_tables(
        json!([
            on_page(1, line("a", "one")),
            on_page(2, line("b", "two")),
            on_page(2, word("w", "cell")),
            on_page(1, layout("t1", "LAYOUT_TEXT", &["a"])),
            on_page(2, table("tb", &["c"])),
            on_page(2, cell("c", 1, 1, &["w"])),
            on_page(2, layout("lt", "LAYOUT_TABLE", &["b"]))
        ]),
        vec![(0, "one"), (1, "| cell |\n| --- |")]
    )]
    fn blocks_become_markdown_pages(#[case] blocks: Value, #[case] expected: Vec<(i64, &str)>) {
        let response = TextractAnalyzeDocumentConfig
            .transform_ocr_response(
                MODEL,
                &serde_json::to_vec(&json!({"DocumentMetadata": {"Pages": 1}, "Blocks": blocks}))
                    .unwrap(),
                OcrResponseFormat::Litellm,
            )
            .unwrap();

        let pages: Vec<(i64, &str)> = response
            .pages
            .iter()
            .map(|page| (page.index, page.markdown.as_str()))
            .collect();
        assert_eq!(pages, expected);
    }

    #[rstest]
    #[case::hyphen("- item", "item")]
    #[case::asterisk("* item", "item")]
    #[case::bullet("\u{2022} item", "item")]
    #[case::middle_dot("\u{00b7}item", "item")]
    #[case::no_bullet("item - with a dash", "item - with a dash")]
    fn list_items_lose_their_own_bullet(#[case] item: &str, #[case] expected: &str) {
        assert_eq!(strip_bullet(item), expected);
    }

    #[rstest]
    #[case::defaults_to_layout_and_tables(json!({}), json!(["LAYOUT", "TABLES"]))]
    #[case::overridden(json!({"feature_types": ["FORMS", "SIGNATURES"]}), json!(["FORMS", "SIGNATURES"]))]
    #[case::explicit_null_uses_the_default(json!({"feature_types": null}), json!(["LAYOUT", "TABLES"]))]
    fn feature_types_reach_the_request(
        document: OcrDocument,
        #[case] arguments: Value,
        #[case] expected: Value,
    ) {
        let arguments: CallArguments = serde_json::from_value(arguments).unwrap();
        let params = TextractAnalyzeDocumentConfig
            .map_ocr_params(&arguments, MODEL)
            .unwrap();

        let request = TextractAnalyzeDocumentConfig
            .transform_ocr_request(MODEL, document, &params, &[])
            .unwrap();

        assert_eq!(
            serde_json::to_value(request).unwrap(),
            json!({"Document": {"Bytes": "aGk="}, "FeatureTypes": expected})
        );
    }

    #[rstest]
    #[case::undocumented_feature(json!({"feature_types": ["HANDWRITING"]}))]
    #[case::lowercase_feature(json!({"feature_types": ["layout"]}))]
    #[case::not_a_list(json!({"feature_types": "LAYOUT"}))]
    fn feature_types_outside_the_documented_values_are_refused(#[case] arguments: Value) {
        let arguments: CallArguments = serde_json::from_value(arguments).unwrap();

        assert!(
            TextractAnalyzeDocumentConfig
                .map_ocr_params(&arguments, MODEL)
                .is_err()
        );
    }
}
