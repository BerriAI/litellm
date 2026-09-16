use std::collections::BTreeMap;

use serde::{Deserialize, Deserializer, Serialize};
use serde_json::{Map, Value, json};

use crate::constants::{REDUCTO_API_BASE, REDUCTO_API_KEY_ENV, REDUCTO_ID_PREFIX};
use crate::llms::base_llm::ocr::transformation::{BaseOcrConfig, OcrRequestContext};
use crate::ocr::OcrArguments;
use crate::ocr::OcrClient;
use crate::ocr::document::InlineDocument;
use crate::ocr::prepare::{build_http_request, credential_env, guardrail_document};
use crate::ocr::types::{
    LiteLLMOcrRequest, LiteLLMOcrResponse, OcrConnection, OcrDocument, OcrPage, OcrUsageInfo,
};
use crate::params::OpaqueParams;
use crate::url_utils::ApiUrl;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(transparent)]
pub(crate) struct ReductoFileId(String);

pub(crate) type ReductoV3Params = OpaqueParams;
pub(crate) type ReductoLegacyParams = OpaqueParams;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) struct ReductoV3Request {
    pub input: ReductoFileId,
    #[serde(flatten)]
    pub params: ReductoV3Params,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) struct ReductoLegacyRequest {
    pub document_url: ReductoFileId,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub options: Option<ReductoLegacyOptions>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) struct ReductoLegacyOptions {
    pub enhance: Value,
}

#[derive(Deserialize)]
struct ReductoUploadResponse {
    pub file_id: Option<String>,
}

#[derive(Clone, Debug, Deserialize)]
pub(crate) struct ReductoResponse {
    #[serde(default, deserialize_with = "present_nullable")]
    result: Option<Option<ReductoResult>>,
    usage: Option<ReductoUsage>,
    #[serde(default)]
    chunks: Option<Vec<ReductoChunk>>,
}

#[derive(Clone, Debug, Default, Deserialize)]
struct ReductoResult {
    pub chunks: Option<Vec<ReductoChunk>>,
}

#[serde_with::serde_as]
#[derive(Clone, Debug, Default, Deserialize)]
struct ReductoUsage {
    #[serde_as(deserialize_as = "Option<crate::serde_compat::LaxI64>")]
    pub num_pages: Option<i64>,
    #[serde_as(deserialize_as = "Option<crate::serde_compat::FiniteF64>")]
    pub credits: Option<f64>,
}

#[derive(Clone, Debug, Deserialize)]
struct ReductoChunk {
    pub content: Option<String>,
    pub blocks: Option<Vec<Map<String, Value>>>,
}

#[derive(Clone, Debug)]
pub(crate) struct ReductoParseV3Config;

impl BaseOcrConfig for ReductoParseV3Config {
    type OcrParams = ReductoV3Params;
    type ProviderRequest = ReductoV3Request;
    type ProviderResponse = ReductoResponse;

    fn get_supported_ocr_params(&self, _model: &str) -> &'static [&'static str] {
        &["formatting", "retrieval", "settings"]
    }

    fn map_ocr_params(
        &self,
        non_default_params: &OcrArguments,
        optional_params: &OcrArguments,
        model: &str,
    ) -> Result<OcrArguments, crate::ocr::Error> {
        Ok(map_ocr_params(
            non_default_params,
            optional_params,
            self.get_supported_ocr_params(model),
        ))
    }

    #[tracing::instrument(
        name = "async_transform_ocr_request",
        target = "litellm::function_trace",
        level = "trace",
        skip_all
    )]
    async fn async_transform_ocr_request(
        &self,
        _model: &str,
        document: OcrDocument,
        optional_params: &ReductoV3Params,
        headers: &[(String, String)],
        context: OcrRequestContext<'_>,
    ) -> Result<ReductoV3Request, crate::ocr::Error> {
        let file_id = ensure_file_id_async(document, headers, context).await?;
        Ok(ReductoV3Request {
            input: file_id,
            params: optional_params.clone(),
        })
    }

    fn normalize_response(
        &self,
        model: &str,
        response: ReductoResponse,
    ) -> Result<LiteLLMOcrResponse, crate::ocr::Error> {
        normalize_response(model, response)
    }
}

impl ReductoParseV3Config {
    pub(crate) async fn prepare_request(
        &self,
        request: &LiteLLMOcrRequest,
        client: &OcrClient,
    ) -> Result<reqwest::Request, crate::ocr::Error> {
        let params = self.parse_options(&request.optional_params, &request.model)?;
        let headers = validate_environment(&request.connection, &credential_env)?;
        let url = get_complete_url(request.connection.api_base.as_deref())?;
        let (document, headers) = guardrail_document(request, &url, &headers).await?;
        let body = self
            .async_transform_ocr_request(
                &request.model,
                document,
                &params,
                &headers,
                OcrRequestContext {
                    client,
                    connection: &request.connection,
                },
            )
            .await?;
        let body = request
            .optional_params
            .compose_body(&body, self.get_supported_ocr_params(&request.model))?;
        build_http_request(client, request, &url, &headers, &body)
    }
}

#[derive(Clone, Debug)]
pub(crate) struct ReductoParseLegacyConfig;

impl BaseOcrConfig for ReductoParseLegacyConfig {
    type OcrParams = ReductoLegacyParams;
    type ProviderRequest = ReductoLegacyRequest;
    type ProviderResponse = ReductoResponse;

    fn get_supported_ocr_params(&self, _model: &str) -> &'static [&'static str] {
        &["enhance"]
    }

    fn map_ocr_params(
        &self,
        non_default_params: &OcrArguments,
        optional_params: &OcrArguments,
        model: &str,
    ) -> Result<OcrArguments, crate::ocr::Error> {
        Ok(map_ocr_params(
            non_default_params,
            optional_params,
            self.get_supported_ocr_params(model),
        ))
    }

    #[tracing::instrument(
        name = "async_transform_ocr_request",
        target = "litellm::function_trace",
        level = "trace",
        skip_all
    )]
    async fn async_transform_ocr_request(
        &self,
        _model: &str,
        document: OcrDocument,
        optional_params: &ReductoLegacyParams,
        headers: &[(String, String)],
        context: OcrRequestContext<'_>,
    ) -> Result<ReductoLegacyRequest, crate::ocr::Error> {
        let file_id = ensure_file_id_async(document, headers, context).await?;
        Ok(build_legacy_body(file_id, optional_params))
    }

    fn normalize_response(
        &self,
        model: &str,
        response: ReductoResponse,
    ) -> Result<LiteLLMOcrResponse, crate::ocr::Error> {
        normalize_response(model, response)
    }
}

impl ReductoParseLegacyConfig {
    pub(crate) async fn prepare_request(
        &self,
        request: &LiteLLMOcrRequest,
        client: &OcrClient,
    ) -> Result<reqwest::Request, crate::ocr::Error> {
        let params = self.parse_options(&request.optional_params, &request.model)?;
        let headers = validate_environment(&request.connection, &credential_env)?;
        let url = get_complete_url(request.connection.api_base.as_deref())?;
        let (document, headers) = guardrail_document(request, &url, &headers).await?;
        let body = self
            .async_transform_ocr_request(
                &request.model,
                document,
                &params,
                &headers,
                OcrRequestContext {
                    client,
                    connection: &request.connection,
                },
            )
            .await?;
        let body = request
            .optional_params
            .compose_body(&body, self.get_supported_ocr_params(&request.model))?;
        build_http_request(client, request, &url, &headers, &body)
    }
}

fn map_ocr_params(
    non_default_params: &OcrArguments,
    optional_params: &OcrArguments,
    supported_params: &[&str],
) -> OcrArguments {
    optional_params
        .iter()
        .chain(
            non_default_params
                .iter()
                .filter(|(name, _)| supported_params.contains(&name.as_str())),
        )
        .map(|(name, value)| (name.clone(), value.clone()))
        .collect()
}

fn present_nullable<'de, D: Deserializer<'de>, T: Deserialize<'de>>(
    deserializer: D,
) -> Result<Option<Option<T>>, D::Error> {
    Option::<T>::deserialize(deserializer).map(Some)
}

fn block_page_number(value: &Value) -> Option<i64> {
    match value {
        Value::Number(number) => number
            .as_i64()
            .or_else(|| number.as_f64().and_then(checked_truncated_i64)),
        Value::String(value) => value.trim().parse::<i64>().ok(),
        Value::Bool(value) => Some(i64::from(*value)),
        _ => None,
    }
}

fn checked_truncated_i64(value: f64) -> Option<i64> {
    (value.is_finite() && value >= i64::MIN as f64 && value <= i64::MAX as f64)
        .then(|| value.trunc() as i64)
}

pub(crate) fn normalize_response(
    model: &str,
    response: ReductoResponse,
) -> Result<LiteLLMOcrResponse, crate::ocr::Error> {
    let result = match response.result {
        Some(result) => result.unwrap_or_default(),
        None => ReductoResult {
            chunks: response.chunks,
        },
    };
    let usage = response.usage.unwrap_or_default();
    Ok(LiteLLMOcrResponse {
        usage_info: Some(OcrUsageInfo {
            pages_processed: usage.num_pages,
            credits: usage.credits,
            ..Default::default()
        }),
        ..LiteLLMOcrResponse::new(
            model,
            build_pages_from_reducto(result.chunks.unwrap_or_default())?,
        )
    })
}

fn build_pages_from_reducto(chunks: Vec<ReductoChunk>) -> Result<Vec<OcrPage>, crate::ocr::Error> {
    let blocks_by_page = chunks
        .iter()
        .flat_map(|chunk| chunk.blocks.iter().flatten())
        .filter_map(|block| {
            block_page_number(block.get("bbox")?.get("page")?).map(|page| (page, block))
        })
        .fold(
            BTreeMap::<i64, Vec<&Map<String, Value>>>::new(),
            |mut pages, (page, block)| {
                pages.entry(page).or_default().push(block);
                pages
            },
        );
    if blocks_by_page.is_empty() {
        let markdown = join_content(chunks.iter().map(|chunk| chunk.content.as_deref()));
        return Ok(if markdown.is_empty() {
            Vec::new()
        } else {
            vec![page(0, markdown, None)]
        });
    }
    blocks_by_page
        .into_iter()
        .map(|(index, blocks)| {
            let content = blocks
                .iter()
                .map(|block| match block.get("content") {
                    None | Some(Value::Null) => Ok(None),
                    Some(Value::String(content)) => Ok(Some(content.as_str())),
                    Some(_) => Err(crate::ocr::Error::ResponseField {
                        path: "result.chunks.blocks.content".into(),
                    }),
                })
                .collect::<Result<Vec<_>, _>>()?;
            let markdown = join_content(content.into_iter());
            Ok(page(
                index.saturating_sub(1).max(0),
                markdown,
                Some(json!(blocks)),
            ))
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

fn page(index: i64, markdown: String, blocks: Option<Value>) -> OcrPage {
    OcrPage {
        index,
        markdown,
        extra_fields: blocks
            .map(|blocks| ("blocks".into(), blocks))
            .into_iter()
            .collect(),
        ..Default::default()
    }
}
fn get_complete_url(api_base: Option<&str>) -> Result<String, crate::ocr::Error> {
    complete_endpoint_url(api_base, "parse")
}

fn complete_endpoint_url(api_base: Option<&str>, path: &str) -> Result<String, crate::ocr::Error> {
    let base = api_base
        .map(str::trim)
        .filter(|base| !base.is_empty())
        .unwrap_or(REDUCTO_API_BASE);
    ApiUrl::parse(base)
        .and_then(|url| url.complete_path(&[path]))
        .map(|url| url.into_string())
        .map_err(|_| crate::ocr::Error::RequestField {
            path: "api_base".into(),
        })
}

fn validate_environment(
    connection: &OcrConnection,
    env_lookup: &(dyn Fn(&str) -> Option<String> + Sync),
) -> Result<Vec<(String, String)>, crate::ocr::Error> {
    if crate::http_utils::has_header(&connection.extra_headers, "authorization") {
        return Ok(connection.extra_headers.clone());
    }
    let api_key = connection
        .api_key
        .as_deref()
        .map(str::trim)
        .filter(|key| !key.is_empty())
        .map(str::to_string)
        .or_else(|| {
            env_lookup(REDUCTO_API_KEY_ENV)
                .map(|key| key.trim().to_string())
                .filter(|key| !key.is_empty())
        })
        .ok_or(crate::ocr::Error::MissingReductoApiKey)?;
    Ok(
        std::iter::once(("Authorization".into(), format!("Bearer {api_key}")))
            .chain(connection.extra_headers.clone())
            .collect(),
    )
}

fn build_legacy_body(
    file_id: ReductoFileId,
    optional_params: &ReductoLegacyParams,
) -> ReductoLegacyRequest {
    ReductoLegacyRequest {
        document_url: file_id,
        options: optional_params
            .get("enhance")
            .filter(|value| !value.is_null())
            .map(|enhance| ReductoLegacyOptions {
                enhance: enhance.clone(),
            }),
    }
}

async fn ensure_file_id_async(
    document: OcrDocument,
    headers: &[(String, String)],
    context: OcrRequestContext<'_>,
) -> Result<ReductoFileId, crate::ocr::Error> {
    if document.source().starts_with(REDUCTO_ID_PREFIX) {
        if document.source()[REDUCTO_ID_PREFIX.len()..]
            .trim()
            .is_empty()
        {
            return Err(crate::ocr::Error::RequestField {
                path: "document file id".into(),
            });
        }
        return Ok(ReductoFileId(document.source().to_string()));
    }
    let inline =
        InlineDocument::parse(document.source())?.ok_or(crate::ocr::Error::ReductoSource)?;
    let mime = inline.mime_type().to_string();
    let bytes = inline.decode(crate::constants::OCR_INLINE_MAX_BYTES)?;
    upload_bytes_async(bytes, &mime, headers, context).await
}

async fn upload_bytes_async(
    bytes: Vec<u8>,
    mime: &str,
    headers: &[(String, String)],
    context: OcrRequestContext<'_>,
) -> Result<ReductoFileId, crate::ocr::Error> {
    let OcrRequestContext { client, connection } = context;
    let part = reqwest::multipart::Part::bytes(bytes)
        .file_name("document")
        .mime_str(mime)
        .map_err(|_| crate::ocr::Error::InvalidDataUri)?;
    let builder = client
        .provider_http()
        .post(complete_endpoint_url(
            connection.api_base.as_deref(),
            "upload",
        )?)
        .multipart(reqwest::multipart::Form::new().part("file", part))
        .timeout(connection.timeout);
    let builder = crate::http_utils::with_headers(
        builder,
        headers,
        crate::http_utils::HeaderPolicy::Except(&["content-type", "content-length"]),
    );
    let response = crate::http_utils::http_request(builder)
        .await
        .map_err(crate::transport::Error::from)?;
    let uploaded = crate::ocr::client::read_json_response::<ReductoUploadResponse>(
        response,
        false,
        connection.max_response_bytes,
    )
    .await?
    .data;
    let file_id = uploaded
        .file_id
        .as_deref()
        .map(str::trim)
        .filter(|id| !id.is_empty());
    let Some(file_id) = file_id else {
        return Err(crate::ocr::Error::ResponseField {
            path: "file_id".into(),
        });
    };
    Ok(ReductoFileId(file_id.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mapping_merges_supported_overrides_including_null_with_supplied_options() {
        let supplied = serde_json::from_value(json!({
            "formatting":{"old":true}, "enhance":true, "extension":false
        }))
        .unwrap();
        let overrides = serde_json::from_value(json!({
            "formatting":null, "enhance":null, "ignored":true
        }))
        .unwrap();
        let v3 = ReductoParseV3Config
            .map_ocr_params(&overrides, &supplied, "parse-v3")
            .unwrap();
        assert_eq!(
            serde_json::to_value(v3).unwrap(),
            json!({
                "formatting":null, "enhance":true, "extension":false
            })
        );
        let legacy = ReductoParseLegacyConfig
            .map_ocr_params(&overrides, &supplied, "parse-legacy")
            .unwrap();
        assert_eq!(
            serde_json::to_value(legacy).unwrap(),
            json!({
                "formatting":{"old":true}, "enhance":null, "extension":false
            })
        );
    }

    #[test]
    fn usage_uses_shared_validation_while_block_page_numbers_are_best_effort() {
        for usage in [
            json!({"num_pages":1.5}),
            json!({"num_pages":[]}),
            json!({"credits":{}}),
        ] {
            assert!(serde_json::from_value::<ReductoResponse>(json!({"usage":usage})).is_err());
        }
        let response = serde_json::from_value(json!({"result":{"chunks":[{"blocks":[
            {"content":"ignored", "bbox":{"page":"invalid"}},
            {"content":"kept", "bbox":{"page":2.5}, "extra":null}
        ]}]}, "usage":{"num_pages":2.0, "credits":true}}))
        .unwrap();
        let normalized = normalize_response("model", response).unwrap();
        assert_eq!(normalized.pages[0].index, 1);
        assert_eq!(normalized.pages[0].markdown, "kept");
        assert_eq!(
            normalized.pages[0].extra_fields["blocks"][0]["bbox"]["page"],
            2.5
        );
        assert_eq!(normalized.usage_info.unwrap().credits, Some(1.0));
    }

    #[tokio::test]
    async fn v3_options_preserve_explicit_null() {
        let overrides =
            serde_json::from_value(json!({"formatting":null,"settings":{},"unknown":true}))
                .unwrap();
        let params = ReductoParseV3Config
            .parse_options(&overrides, "parse-v3")
            .unwrap();
        let client = crate::ocr::test_support::ocr_client();
        let connection = OcrConnection::default();
        let document = serde_json::from_value(
            json!({"type":"document_url","document_url":"reducto://ready.pdf"}),
        )
        .unwrap();
        let body = ReductoParseV3Config
            .async_transform_ocr_request(
                "parse-v3",
                document,
                &params,
                &[],
                OcrRequestContext {
                    client: &client,
                    connection: &connection,
                },
            )
            .await
            .unwrap();
        assert_eq!(
            serde_json::to_value(body).unwrap(),
            json!({
                "input":"reducto://ready.pdf", "formatting":null, "settings":{}
            })
        );
        let absent = ReductoParseV3Config
            .parse_options(&OcrArguments::default(), "parse-v3")
            .unwrap();
        assert_eq!(serde_json::to_value(absent).unwrap(), json!({}));
    }

    #[test]
    fn legacy_body_omits_null_enhance_and_wraps_mapped_options() {
        for (value, expected) in [
            (json!(null), json!({"document_url":"reducto://ready.pdf"})),
            (
                json!({}),
                json!({"document_url":"reducto://ready.pdf","options":{"enhance":{}}}),
            ),
        ] {
            let overrides =
                serde_json::from_value(json!({"enhance":value,"unknown":true})).unwrap();
            let params = ReductoParseLegacyConfig
                .parse_options(&overrides, "parse-legacy")
                .unwrap();
            assert_eq!(
                serde_json::to_value(build_legacy_body(
                    ReductoFileId("reducto://ready.pdf".into()),
                    &params
                ))
                .unwrap(),
                expected
            );
        }
    }

    #[test]
    fn explicit_key_precedes_environment_key() {
        let connection = OcrConnection {
            api_key: Some("passed-key".into()),
            ..Default::default()
        };
        let headers = validate_environment(&connection, &|_| Some("env-key".into())).unwrap();
        assert_eq!(headers[0].1, "Bearer passed-key");
    }

    #[test]
    fn blank_explicit_key_uses_environment_key() {
        let connection = OcrConnection {
            api_key: Some(" ".into()),
            ..Default::default()
        };
        let headers = validate_environment(&connection, &|_| Some(" env-key ".into())).unwrap();
        assert_eq!(headers[0].1, "Bearer env-key");
    }

    #[test]
    fn existing_authorization_skips_key_lookup() {
        let connection = OcrConnection {
            extra_headers: vec![("authorization".into(), "Bearer existing".into())],
            ..Default::default()
        };
        assert_eq!(
            validate_environment(&connection, &|_| None).unwrap(),
            connection.extra_headers
        );
    }
}
