use crate::auth::AuthError;
use crate::ocr::error::OcrError;
use crate::ocr::error::OcrRequestError;
use crate::ocr::error::OcrResponseError;
use std::collections::BTreeMap;

use super::types::*;
use crate::constants::{REDUCTO_ID_PREFIX, REDUCTO_OCR_API_BASE};
use crate::ocr::transformation::OcrProviderConfig;
use crate::ocr::types::{OcrConnection, OcrDocument, OcrPage, OcrResponseData, OcrUsageInfo};
use crate::providers::reducto::auth;
use base64::Engine;
use base64::engine::general_purpose::STANDARD as BASE64_STANDARD;

pub struct ReductoParseV3Config;
pub struct ReductoParseLegacyConfig;
pub const REDUCTO_PARSE_V3_CONFIG: ReductoParseV3Config = ReductoParseV3Config;
pub const REDUCTO_PARSE_LEGACY_CONFIG: ReductoParseLegacyConfig = ReductoParseLegacyConfig;

pub fn normalize_api_base(api_base: Option<&str>) -> &str {
    api_base
        .map(str::trim)
        .filter(|base| !base.is_empty())
        .unwrap_or(REDUCTO_OCR_API_BASE)
        .trim_end_matches('/')
}

async fn prepare_reducto_document(
    document: OcrDocument,
    connection: &OcrConnection,
    headers: &[(String, String)],
) -> Result<ReductoFileId, OcrError> {
    let source = document.source();
    if source.starts_with(REDUCTO_ID_PREFIX) {
        return Ok(ReductoFileId(source.to_string()));
    }
    if !source.starts_with("data:") {
        return Err(OcrRequestError::ReductoSource.into());
    }
    let (header, encoded) = source
        .split_once(',')
        .ok_or(OcrRequestError::ReductoDataUri)?;
    if !header.split(';').any(|part| part == "base64") {
        return Err(OcrRequestError::ReductoDataUri.into());
    }
    let mime = header
        .strip_prefix("data:")
        .and_then(|header| header.split(';').next())
        .filter(|mime| !mime.is_empty())
        .unwrap_or("application/octet-stream");
    let bytes = BASE64_STANDARD
        .decode(encoded)
        .map_err(|_| OcrRequestError::ReductoDataUri)?;
    let part = reqwest::multipart::Part::bytes(bytes)
        .file_name("document")
        .mime_str(mime)
        .map_err(|_| OcrRequestError::ReductoDataUri)?;
    let mut builder = crate::ocr::client::http_client()?
        .post(format!(
            "{}/upload",
            normalize_api_base(connection.api_base.as_deref())
        ))
        .multipart(reqwest::multipart::Form::new().part("file", part))
        .timeout(connection.timeout);
    for (name, value) in headers {
        if !name.eq_ignore_ascii_case("content-type")
            && !name.eq_ignore_ascii_case("content-length")
        {
            builder = builder.header(name, value);
        }
    }
    let response = crate::http_utils::http_request(builder)
        .await
        .map_err(crate::ocr::client::network_error)?;
    let response = crate::ocr::wire::read_json_response::<ReductoUploadResponse>(response, false)
        .await?
        .data;
    if response.file_id.is_empty() {
        return Err(OcrResponseError::ResponseField {
            path: "file_id".into(),
        }
        .into());
    }
    Ok(ReductoFileId(response.file_id))
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

impl OcrProviderConfig for ReductoParseV3Config {
    type InputParams = ReductoV3Params;
    type MappedParams = ReductoV3Params;
    type PreparedDocument = ReductoFileId;
    type RequestBody = ReductoV3Request;
    type ResponseBody = ReductoResponse;
    crate::ocr_provider_hooks!(ReductoV3, ReductoV3);

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

    #[tracing::instrument(target = "litellm::function_trace", level = "trace", skip_all)]
    fn complete_url(
        &self,
        connection: &OcrConnection,
        _model: &str,
        _params: &Self::MappedParams,
    ) -> Result<String, OcrError> {
        Ok(format!(
            "{}/parse",
            normalize_api_base(connection.api_base.as_deref())
        ))
    }

    async fn prepare_document(
        &self,
        document: OcrDocument,
        connection: &OcrConnection,
        headers: &[(String, String)],
    ) -> Result<ReductoFileId, OcrError> {
        prepare_reducto_document(document, connection, headers).await
    }

    fn guard_document_before_preparation(&self) -> bool {
        true
    }
    fn preserve_native_response(&self, _params: &Self::MappedParams) -> bool {
        true
    }

    async fn authenticate(
        &self,
        connection: &OcrConnection,
    ) -> Result<Vec<(String, String)>, AuthError> {
        Ok(auth::validate_environment(
            connection.extra_headers.clone(),
            connection.api_key.as_deref(),
            &|name| std::env::var(name).ok(),
        )?)
    }
}

impl OcrProviderConfig for ReductoParseLegacyConfig {
    type InputParams = ReductoLegacyParams;
    type MappedParams = ReductoLegacyParams;
    type PreparedDocument = ReductoFileId;
    type RequestBody = ReductoLegacyRequest;
    type ResponseBody = ReductoResponse;
    crate::ocr_provider_hooks!(ReductoLegacy, ReductoLegacy);

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

    #[tracing::instrument(target = "litellm::function_trace", level = "trace", skip_all)]
    fn complete_url(
        &self,
        connection: &OcrConnection,
        _model: &str,
        _params: &Self::MappedParams,
    ) -> Result<String, OcrError> {
        Ok(format!(
            "{}/parse",
            normalize_api_base(connection.api_base.as_deref())
        ))
    }

    async fn prepare_document(
        &self,
        document: OcrDocument,
        connection: &OcrConnection,
        headers: &[(String, String)],
    ) -> Result<ReductoFileId, OcrError> {
        prepare_reducto_document(document, connection, headers).await
    }

    fn guard_document_before_preparation(&self) -> bool {
        true
    }
    fn preserve_native_response(&self, _params: &Self::MappedParams) -> bool {
        true
    }

    async fn authenticate(
        &self,
        connection: &OcrConnection,
    ) -> Result<Vec<(String, String)>, AuthError> {
        Ok(auth::validate_environment(
            connection.extra_headers.clone(),
            connection.api_key.as_deref(),
            &|name| std::env::var(name).ok(),
        )?)
    }
}
