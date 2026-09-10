use std::collections::BTreeMap;

use super::types::*;
use crate::ocr::error::{OcrRequestError, OcrResponseError};
use crate::ocr::types::{OcrPage, OcrResponseData, OcrUsageInfo};

fn join_content<'a>(content: impl Iterator<Item = Option<&'a str>>) -> String {
    content
        .flatten()
        .filter(|text| !text.is_empty())
        .collect::<Vec<_>>()
        .join("\n\n")
}

fn build_pages(chunks: Vec<ReductoChunk>) -> Vec<OcrPage> {
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
            vec![OcrPage::text(0, markdown)]
        };
    }
    blocks_by_page
        .into_iter()
        .map(|(page, blocks)| {
            let markdown = join_content(blocks.iter().map(|block| block.content.as_deref()));
            OcrPage {
                extra_fields: serde_json::Map::from_iter([(
                    "blocks".into(),
                    serde_json::json!(blocks),
                )]),
                ..OcrPage::text(page.saturating_sub(1).max(0), markdown)
            }
        })
        .collect()
}

pub(super) fn response<P>(
    model: &str,
    response: ReductoResponse,
    _params: &P,
) -> Result<OcrResponseData, OcrResponseError> {
    let result = match response.result {
        Some(result) => result.unwrap_or_default(),
        None => ReductoResult {
            chunks: response.chunks,
        },
    };
    let usage = response.usage.unwrap_or_default();
    Ok(OcrResponseData {
        usage_info: Some(OcrUsageInfo {
            pages_processed: usage.num_pages,
            credits: usage.credits,
            ..Default::default()
        }),
        ..OcrResponseData::new(
            model.to_string(),
            build_pages(result.chunks.unwrap_or_default()),
        )
    })
}

pub(super) fn v3_request(
    _model: &str,
    document: ReductoFileId,
    params: &ReductoV3Params,
) -> Result<ReductoV3Request, OcrRequestError> {
    Ok(ReductoV3Request {
        input: document.0,
        params: params.clone(),
    })
}

pub(super) fn legacy_request(
    _model: &str,
    document: ReductoFileId,
    params: &ReductoLegacyParams,
) -> Result<ReductoLegacyRequest, OcrRequestError> {
    Ok(ReductoLegacyRequest {
        document_url: document.0,
        options: params.enhance.as_ref().map(|_| params.clone()),
    })
}
