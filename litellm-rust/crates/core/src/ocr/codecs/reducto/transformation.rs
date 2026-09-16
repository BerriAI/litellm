use std::collections::BTreeMap;

use serde_json::{Value, json};

use super::types::*;
use crate::ocr::error::{OcrRequestError, OcrResponseError};
use crate::ocr::types::{LiteLLMOcrResponse, OcrDocument};

pub(crate) fn transform_v3_ocr_request(
    _model: &str,
    document: OcrDocument,
    params: &ReductoV3Params,
) -> Result<ReductoV3Request, OcrRequestError> {
    Ok(ReductoV3Request {
        input: document.source().to_string(),
        params: params.clone(),
    })
}

pub(crate) fn transform_legacy_ocr_request(
    _model: &str,
    document: OcrDocument,
    params: &ReductoLegacyParams,
) -> Result<ReductoLegacyRequest, OcrRequestError> {
    Ok(ReductoLegacyRequest {
        document_url: document.source().to_string(),
        options: params.enhance.as_ref().map(|_| params.clone()),
    })
}

pub(crate) fn transform_ocr_response(
    model: &str,
    response: ReductoResponse,
) -> Result<LiteLLMOcrResponse, OcrResponseError> {
    let result = match response.result {
        Some(result) => result.unwrap_or_default(),
        None => ReductoResult {
            chunks: response.chunks,
        },
    };
    let usage = response.usage.unwrap_or_default();
    Ok(LiteLLMOcrResponse {
        pages: build_pages(result.chunks.unwrap_or_default()),
        model: model.to_string(),
        document_annotation: None,
        usage_info: Some(json!({
            "pages_processed": usage.num_pages,
            "credits": usage.credits,
        })),
        object: "ocr".to_string(),
        extra_fields: serde_json::Map::new(),
        provider_native_response: None,
    })
}

fn build_pages(chunks: Vec<ReductoChunk>) -> Vec<Value> {
    let blocks_by_page = chunks
        .iter()
        .flat_map(|chunk| chunk.blocks.iter().flatten())
        .filter_map(|block| block.bbox.as_ref()?.page.map(|page| (page, block)))
        .fold(
            BTreeMap::<i64, Vec<&ReductoBlock>>::new(),
            |mut pages, (page, block)| {
                pages.entry(page).or_default().push(block);
                pages
            },
        );
    if blocks_by_page.is_empty() {
        let markdown = join_content(chunks.iter().map(|chunk| chunk.content.as_deref()));
        return if markdown.is_empty() {
            Vec::new()
        } else {
            vec![page(0, markdown, None)]
        };
    }
    blocks_by_page
        .into_iter()
        .map(|(index, blocks)| {
            let markdown = join_content(blocks.iter().map(|block| block.content.as_deref()));
            page(
                index.saturating_sub(1).max(0),
                markdown,
                Some(json!(blocks)),
            )
        })
        .collect()
}

fn join_content<'a>(content: impl Iterator<Item = Option<&'a str>>) -> String {
    content
        .flatten()
        .filter(|text| !text.is_empty())
        .collect::<Vec<_>>()
        .join("\n\n")
}

fn page(index: i64, markdown: String, blocks: Option<Value>) -> Value {
    let mut result = json!({"index":index,"markdown":markdown,"images":null});
    if let (Value::Object(fields), Some(blocks)) = (&mut result, blocks) {
        fields.insert("blocks".into(), blocks);
    }
    result
}
