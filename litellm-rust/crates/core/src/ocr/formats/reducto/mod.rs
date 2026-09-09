use crate::ocr::error::OcrRequestError;
use crate::ocr::error::OcrResponseError;
use std::collections::BTreeMap;

pub mod types;

use self::types::*;
use crate::ocr::formats::OcrFormat;
use crate::ocr::types::{OcrPage, OcrResponseData, OcrUsageInfo};

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
        let markdown = chunks
            .iter()
            .filter_map(|chunk| chunk.content.as_deref())
            .filter(|s| !s.is_empty())
            .collect::<Vec<_>>()
            .join("\n\n");
        return if markdown.is_empty() {
            Vec::new()
        } else {
            vec![OcrPage::text(0, markdown)]
        };
    }
    blocks_by_page
        .into_iter()
        .map(|(page, blocks)| {
            let markdown = blocks
                .iter()
                .filter_map(|block| block.content.as_deref())
                .filter(|s| !s.is_empty())
                .collect::<Vec<_>>()
                .join("\n\n");
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

fn transform_reducto_response(model: &str, response: ReductoResponse) -> OcrResponseData {
    let result = match response.result {
        Some(result) => result.unwrap_or_default(),
        None => ReductoResult {
            chunks: response.chunks,
        },
    };
    let usage = response.usage.unwrap_or_default();
    OcrResponseData {
        usage_info: Some(OcrUsageInfo {
            pages_processed: usage.num_pages,
            credits: usage.credits,
            ..Default::default()
        }),
        ..OcrResponseData::new(
            model.to_string(),
            build_pages(result.chunks.unwrap_or_default()),
        )
    }
}

pub struct ReductoParseV3Format;

impl OcrFormat for ReductoParseV3Format {
    type InputParams = ReductoV3Params;
    type MappedParams = ReductoV3Params;
    type PreparedDocument = ReductoFileId;
    type RequestBody = ReductoV3Request;
    type ResponseBody = ReductoResponse;

    #[tracing::instrument(target = "litellm::function_trace", level = "trace", skip_all)]
    fn map_ocr_params(
        &self,
        params: Self::InputParams,
    ) -> Result<Self::MappedParams, OcrRequestError> {
        Ok(params)
    }

    #[tracing::instrument(target = "litellm::function_trace", level = "trace", skip_all)]
    fn transform_ocr_request(
        &self,
        _model: &str,
        document: ReductoFileId,
        params: &Self::MappedParams,
    ) -> Result<Self::RequestBody, OcrRequestError> {
        Ok(ReductoV3Request {
            input: document.0,
            params: params.clone(),
        })
    }

    #[tracing::instrument(target = "litellm::function_trace", level = "trace", skip_all)]
    fn transform_ocr_response(
        &self,
        model: &str,
        response: Self::ResponseBody,
        _params: &Self::MappedParams,
    ) -> Result<OcrResponseData, OcrResponseError> {
        Ok(transform_reducto_response(model, response))
    }
}

pub struct ReductoParseLegacyFormat;

impl OcrFormat for ReductoParseLegacyFormat {
    type InputParams = ReductoLegacyParams;
    type MappedParams = ReductoLegacyParams;
    type PreparedDocument = ReductoFileId;
    type RequestBody = ReductoLegacyRequest;
    type ResponseBody = ReductoResponse;

    #[tracing::instrument(target = "litellm::function_trace", level = "trace", skip_all)]
    fn map_ocr_params(
        &self,
        params: Self::InputParams,
    ) -> Result<Self::MappedParams, OcrRequestError> {
        Ok(params)
    }

    #[tracing::instrument(target = "litellm::function_trace", level = "trace", skip_all)]
    fn transform_ocr_request(
        &self,
        _model: &str,
        document: ReductoFileId,
        params: &Self::MappedParams,
    ) -> Result<Self::RequestBody, OcrRequestError> {
        Ok(ReductoLegacyRequest {
            document_url: document.0,
            options: params.enhance.as_ref().map(|_| params.clone()),
        })
    }

    #[tracing::instrument(target = "litellm::function_trace", level = "trace", skip_all)]
    fn transform_ocr_response(
        &self,
        model: &str,
        response: Self::ResponseBody,
        _params: &Self::MappedParams,
    ) -> Result<OcrResponseData, OcrResponseError> {
        Ok(transform_reducto_response(model, response))
    }
}
