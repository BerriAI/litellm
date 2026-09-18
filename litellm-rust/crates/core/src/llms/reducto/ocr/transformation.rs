use std::collections::BTreeMap;

use serde::{Deserialize, Deserializer, Serialize};
use serde_json::{Map, Value, json};

use crate::call_arguments::{CallArguments, compose_body};
use crate::constants::{REDUCTO_API_BASE, REDUCTO_API_KEY_ENV, REDUCTO_ID_PREFIX};
use crate::llms::base_llm::ocr::transformation::{
    BaseOcrConfig, OcrRequestContext, decode_and_normalize_response,
};
use crate::ocr::OcrClient;
use crate::ocr::document::InlineDocument;
use crate::ocr::prepare::{build_http_request, credential_env, guardrail_document};
use crate::ocr::types::{
    LiteLLMOcrResponse, OcrConnection, OcrDocument, OcrPage, OcrResponseFormat, OcrUsageInfo,
    PreparedOcrRequest,
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
    type Environment = Vec<(String, String)>;

    fn get_supported_ocr_params(&self, _model: &str) -> &'static [&'static str] {
        &["formatting", "retrieval", "settings"]
    }

    fn map_ocr_params(
        &self,
        non_default_params: &CallArguments,
        model: &str,
    ) -> Result<ReductoV3Params, crate::ocr::Error> {
        Ok(non_default_params
            .select(self.get_supported_ocr_params(model))
            .into())
    }

    async fn validate_environment(
        &self,
        request: &PreparedOcrRequest,
        _client: &OcrClient,
    ) -> Result<Self::Environment, crate::ocr::Error> {
        resolve_headers(&request.connection, &credential_env)
    }

    fn get_complete_url(
        &self,
        request: &PreparedOcrRequest,
        _optional_params: &Self::OcrParams,
        _environment: &Self::Environment,
    ) -> Result<String, crate::ocr::Error> {
        build_ocr_url(request.connection.api_base.as_deref())
    }

    fn transform_ocr_request(
        &self,
        _model: &str,
        document: OcrDocument,
        optional_params: &Self::OcrParams,
        _headers: &[(String, String)],
    ) -> Result<Self::ProviderRequest, crate::ocr::Error> {
        Ok(ReductoV3Request {
            input: uploaded_file_id(document)?,
            params: optional_params.clone(),
        })
    }

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

    fn transform_ocr_response(
        &self,
        model: &str,
        raw_response: &[u8],
        request_format: OcrResponseFormat,
    ) -> Result<LiteLLMOcrResponse, crate::ocr::Error> {
        decode_and_normalize_response(model, raw_response, request_format, normalize_response)
    }

    async fn prepare_request(
        &self,
        request: &PreparedOcrRequest,
        client: &OcrClient,
    ) -> Result<reqwest::Request, crate::ocr::Error> {
        prepare_upload_request(self, request, client).await
    }
}

#[derive(Clone, Debug)]
pub(crate) struct ReductoParseLegacyConfig;

impl BaseOcrConfig for ReductoParseLegacyConfig {
    type OcrParams = ReductoLegacyParams;
    type ProviderRequest = ReductoLegacyRequest;
    type Environment = Vec<(String, String)>;

    fn get_supported_ocr_params(&self, _model: &str) -> &'static [&'static str] {
        &["enhance"]
    }

    fn map_ocr_params(
        &self,
        non_default_params: &CallArguments,
        model: &str,
    ) -> Result<ReductoLegacyParams, crate::ocr::Error> {
        Ok(non_default_params
            .select(self.get_supported_ocr_params(model))
            .into())
    }

    async fn validate_environment(
        &self,
        request: &PreparedOcrRequest,
        client: &OcrClient,
    ) -> Result<Self::Environment, crate::ocr::Error> {
        ReductoParseV3Config
            .validate_environment(request, client)
            .await
    }

    fn get_complete_url(
        &self,
        request: &PreparedOcrRequest,
        optional_params: &Self::OcrParams,
        environment: &Self::Environment,
    ) -> Result<String, crate::ocr::Error> {
        ReductoParseV3Config.get_complete_url(request, optional_params, environment)
    }

    fn transform_ocr_request(
        &self,
        _model: &str,
        document: OcrDocument,
        optional_params: &Self::OcrParams,
        _headers: &[(String, String)],
    ) -> Result<Self::ProviderRequest, crate::ocr::Error> {
        Ok(build_legacy_body(
            uploaded_file_id(document)?,
            optional_params,
        ))
    }

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

    fn transform_ocr_response(
        &self,
        model: &str,
        raw_response: &[u8],
        request_format: OcrResponseFormat,
    ) -> Result<LiteLLMOcrResponse, crate::ocr::Error> {
        ReductoParseV3Config.transform_ocr_response(model, raw_response, request_format)
    }

    async fn prepare_request(
        &self,
        request: &PreparedOcrRequest,
        client: &OcrClient,
    ) -> Result<reqwest::Request, crate::ocr::Error> {
        prepare_upload_request(self, request, client).await
    }
}

/// Reducto differs from the shared `BaseOcrConfig::prepare_request` flow:
/// guardrails see the *source* document before it is uploaded, because the
/// final body only carries the opaque Reducto file id.
async fn prepare_upload_request<C: BaseOcrConfig<Environment = Vec<(String, String)>>>(
    config: &C,
    request: &PreparedOcrRequest,
    client: &OcrClient,
) -> Result<reqwest::Request, crate::ocr::Error> {
    let params = config.map_ocr_params(&request.optional_params, &request.model)?;
    let headers = config.validate_environment(request, client).await?;
    let url = config.get_complete_url(request, &params, &headers)?;
    let (document, headers) = guardrail_document(request, &url, &headers).await?;
    let body = config
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
    let body = compose_body(
        &request.optional_params,
        &body,
        config.get_supported_ocr_params(&request.model),
    )?;
    build_http_request(client, request, &url, &headers, &body)
}

fn uploaded_file_id(document: OcrDocument) -> Result<ReductoFileId, crate::ocr::Error> {
    if !document.source().starts_with(REDUCTO_ID_PREFIX) {
        return Err(crate::ocr::Error::ReductoSource);
    }
    if document.source()[REDUCTO_ID_PREFIX.len()..]
        .trim()
        .is_empty()
    {
        return Err(crate::ocr::Error::RequestField {
            path: "document file id".into(),
        });
    }
    Ok(ReductoFileId(document.source().into()))
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
fn build_ocr_url(api_base: Option<&str>) -> Result<String, crate::ocr::Error> {
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

fn resolve_headers(
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
    fn options_preserve_null_and_select_the_provider_fields() {
        let overrides = serde_json::from_value(json!({
            "formatting":null, "enhance":null, "ignored":true
        }))
        .unwrap();
        let v3 = ReductoParseV3Config
            .map_ocr_params(&overrides, "parse-v3")
            .unwrap();
        assert_eq!(
            serde_json::to_value(v3).unwrap(),
            json!({
                "formatting":null
            })
        );
        let legacy = ReductoParseLegacyConfig
            .map_ocr_params(&overrides, "parse-legacy")
            .unwrap();
        assert_eq!(
            serde_json::to_value(legacy).unwrap(),
            json!({
                "enhance":null
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
            .map_ocr_params(&overrides, "parse-v3")
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
            .map_ocr_params(&crate::call_arguments::CallArguments::default(), "parse-v3")
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
                .map_ocr_params(&overrides, "parse-legacy")
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
        let headers = resolve_headers(&connection, &|_| Some("env-key".into())).unwrap();
        assert_eq!(headers[0].1, "Bearer passed-key");
    }

    #[test]
    fn blank_explicit_key_uses_environment_key() {
        let connection = OcrConnection {
            api_key: Some(" ".into()),
            ..Default::default()
        };
        let headers = resolve_headers(&connection, &|_| Some(" env-key ".into())).unwrap();
        assert_eq!(headers[0].1, "Bearer env-key");
    }

    #[test]
    fn existing_authorization_skips_key_lookup() {
        let connection = OcrConnection {
            extra_headers: vec![("authorization".into(), "Bearer existing".into())],
            ..Default::default()
        };
        assert_eq!(
            resolve_headers(&connection, &|_| None).unwrap(),
            connection.extra_headers
        );
    }

    use litellm_callbacks::event::{CallEvent, WireRequest};
    use rstest::rstest;

    use crate::ocr::LocalOcrHost;
    use crate::ocr::test_support::{
        MockResponse, mock_server, perform_ocr, perform_ocr_with, wire_request,
    };

    fn request_body(request: &str) -> Value {
        serde_json::from_str(request.split_once("\r\n\r\n").unwrap().1).unwrap()
    }

    #[rstest]
    #[case(
        "reducto/parse-v3",
        json!({
            "formatting":{"table_output_format":"html"},
            "retrieval":{"chunk_mode":"section"},
            "settings":{"ocr_system":"standard"},
            "future_ocr_option":true,
            "extra_body":{"provider_option":"value"}
        }),
        "reducto://already.pdf",
        json!({
            "input":"reducto://already.pdf",
            "formatting":{"table_output_format":"html"},
            "retrieval":{"chunk_mode":"section"},
            "settings":{"ocr_system":"standard"},
            "future_ocr_option":true,
            "provider_option":"value"
        })
    )]
    #[case(
        "reducto/parse-legacy",
        json!({
            "enhance":{"agentic":[{"type":"table"}]},
            "future_ocr_option":true,
            "extra_body":{"provider_option":"value"}
        }),
        "reducto://legacy.pdf",
        json!({
            "document_url":"reducto://legacy.pdf",
            "options":{"enhance":{"agentic":[{"type":"table"}]}},
            "future_ocr_option":true,
            "provider_option":"value"
        })
    )]
    #[tokio::test]
    async fn request_mapping_matches_python(
        #[case] model: &str,
        #[case] options: Value,
        #[case] source: &str,
        #[case] expected: Value,
    ) {
        let (base, seen, server) = mock_server(vec![MockResponse::json(json!({
            "result":{"chunks":[]}
        }))])
        .await;
        let request =
            crate::ocr::test_support::with_source(wire_request(model, &base, options), source);

        perform_ocr(request).await.unwrap();
        server.await.unwrap();
        let requests = seen.lock().unwrap();
        assert_eq!(requests.len(), 1);
        assert!(requests[0].starts_with("POST /parse "));
        assert_eq!(request_body(&requests[0]), expected);
    }

    #[rstest]
    #[case("parse-v3")]
    #[case("parse-legacy")]
    #[tokio::test]
    async fn data_uri_upload_preserves_multipart_headers(#[case] model: &str) {
        let (base, seen, server) = mock_server(vec![
            MockResponse::json(json!({"file_id":"reducto://uploaded.pdf"})),
            MockResponse::json(json!({"result":{"chunks":[{"content":"hello"}]}})),
        ])
        .await;
        let mut request = wire_request(&format!("reducto/{model}"), &base, json!({}));
        request.transport.extra_headers = vec![
            ("Content-Type".into(), "application/json".into()),
            ("X-Trace".into(), "upload-test".into()),
        ];

        let response = perform_ocr(request).await.unwrap();
        server.await.unwrap();
        assert_eq!(response.pages[0].markdown, "hello");
        let requests = seen.lock().unwrap();
        assert_eq!(requests.len(), 2);
        assert!(requests[0].starts_with("POST /upload "));
        assert!(
            requests[0]
                .to_ascii_lowercase()
                .contains("content-type: multipart/form-data; boundary=")
        );
        assert!(requests[0].contains("x-trace: upload-test"));
        assert!(requests[0].contains("application/pdf"));
        assert!(requests[0].contains("abc"));
        assert!(requests[1].starts_with("POST /parse "));
    }

    #[tokio::test]
    async fn response_received_stays_after_reducto_upload_and_parse() {
        let (base, seen, server) = mock_server(vec![
            MockResponse::json(json!({"file_id":"reducto://uploaded.pdf"})),
            MockResponse::json(json!({"result":{"chunks":[]}})),
        ])
        .await;
        let request_count = seen.clone();
        let host = LocalOcrHost::new(wire_request("reducto/parse-v3", &base, json!({})))
            .with_observer(move |event| {
                if let CallEvent::ResponseReceived { raw } = event {
                    assert_eq!(request_count.lock().unwrap().len(), 2);
                    assert_eq!(raw.body, r#"{"result":{"chunks":[]}}"#);
                }
            });

        perform_ocr_with(host).await.unwrap();
        server.await.unwrap();
        assert_eq!(seen.lock().unwrap().len(), 2);
    }

    #[rstest]
    #[case(json!({"file_id":""}))]
    #[case(json!({}))]
    #[case(json!({"file_id":null}))]
    #[tokio::test]
    async fn invalid_upload_ids_stop_before_parse(#[case] response: Value) {
        let (base, seen, server) = mock_server(vec![MockResponse::json(response)]).await;
        let error = perform_ocr(wire_request("reducto/parse-v3", &base, json!({})))
            .await
            .unwrap_err();
        server.await.unwrap();
        assert!(error.to_string().contains("file_id"));
        assert_eq!(seen.lock().unwrap().len(), 1);
    }

    #[tokio::test]
    async fn upload_failure_stops_before_parse() {
        let (base, seen, server) = mock_server(vec![MockResponse {
            status: 503,
            headers: vec![],
            body: json!({"error":"unavailable"}),
        }])
        .await;
        assert!(
            perform_ocr(wire_request("reducto/parse-v3", &base, json!({})))
                .await
                .is_err()
        );
        server.await.unwrap();
        assert_eq!(seen.lock().unwrap().len(), 1);
    }

    #[rstest]
    #[case("https://example.com/a.pdf")]
    #[case("reducto://")]
    #[case("data:application/pdf;base64")]
    #[case("data:application/pdf;base64,INVALID!")]
    #[tokio::test]
    async fn rejects_invalid_document_sources_before_network(#[case] source: &str) {
        let request = crate::ocr::test_support::with_source(
            wire_request("reducto/parse-v3", "http://127.0.0.1:1", json!({})),
            source,
        );
        assert!(perform_ocr(request).await.is_err());
    }

    #[test]
    fn response_normalization_groups_blocks_and_distinguishes_null_result() {
        use crate::llms::reducto::ocr::transformation::{ReductoResponse, normalize_response};

        let raw = json!({"usage":{"num_pages":"2","credits":"3"},"result":{"type":"full","chunks":[
            {"blocks":[{
                "type":"Table",
                "content":"B",
                "bbox":{"left":0.1,"top":0.2,"width":0.8,"height":0.3,"page":2,"original_page":4},
                "confidence":"high",
                "granular_confidence":{"parse_confidence":0.95,"extract_confidence":null},
                "image_url":null
            }]},
            {"blocks":[{"content":"A","bbox":{"page":1},"type":"Text"},{"content":"C","bbox":{"page":1}}]}
        ]}});
        let response: ReductoResponse = serde_json::from_value(raw).unwrap();
        let normalized = normalize_response("parse-v3", response)
            .unwrap()
            .into_json();
        assert_eq!(normalized["pages"][0]["markdown"], "A\n\nC");
        assert_eq!(normalized["pages"][1]["markdown"], "B");
        assert_eq!(normalized["pages"][1]["blocks"][0]["type"], "Table");
        assert_eq!(
            normalized["pages"][1]["blocks"][0]["bbox"],
            json!({"left":0.1,"top":0.2,"width":0.8,"height":0.3,"page":2,"original_page":4})
        );
        assert_eq!(normalized["pages"][1]["blocks"][0]["confidence"], "high");
        assert_eq!(
            normalized["pages"][1]["blocks"][0]["granular_confidence"]["parse_confidence"],
            0.95
        );
        assert!(normalized["pages"][1]["blocks"][0]["image_url"].is_null());
        assert_eq!(normalized["usage_info"]["pages_processed"], 2);
        assert_eq!(normalized["usage_info"]["credits"], 3.0);

        let missing: ReductoResponse =
            serde_json::from_value(json!({"chunks":[{"content":"text"}]})).unwrap();
        let missing = normalize_response("parse-v3", missing).unwrap();
        assert_eq!(missing.pages[0].markdown, "text");
        let null: ReductoResponse = serde_json::from_value(
            json!({"result":null,"chunks":[{"content":"ignored"}],"usage":null}),
        )
        .unwrap();
        let null = normalize_response("parse-v3", null).unwrap();
        assert!(null.pages.is_empty());
    }

    #[tokio::test]
    async fn facade_omits_native_response_by_default_and_preserves_auth_priority() {
        let raw = json!({"job_id":"job-1","result":{"chunks":[]}});
        let (base, seen, server) = mock_server(vec![MockResponse::json(raw)]).await;
        let mut request = crate::ocr::test_support::with_source(
            wire_request("reducto/parse-v3", &base, json!({})),
            "reducto://ready.pdf",
        );
        request.transport.extra_headers = vec![("authorization".into(), "Bearer existing".into())];

        let response = perform_ocr(request).await.unwrap();
        server.await.unwrap();
        assert_eq!(response.provider_native_response, None);
        assert!(
            seen.lock().unwrap()[0]
                .to_ascii_lowercase()
                .contains("authorization: bearer existing")
        );
    }

    #[rstest]
    #[case("reducto/parse-v3")]
    #[case("reducto/parse-legacy")]
    #[tokio::test]
    async fn guardrail_headers_reach_upload_and_parse(#[case] model: &str) {
        let (base, seen, server) = mock_server(vec![
            MockResponse::json(json!({"file_id":"reducto://uploaded.pdf"})),
            MockResponse::json(json!({"result":{"chunks":[]}})),
        ])
        .await;
        let mut request = wire_request(model, &base, json!({}));
        request.transport.extra_headers = vec![("authorization".into(), "Bearer original".into())];
        let host = LocalOcrHost::new(request).with_before_send(|wire, _| {
            Ok(WireRequest {
                headers: vec![("authorization".into(), "Bearer guarded".into())],
                ..wire
            })
        });

        perform_ocr_with(host).await.unwrap();
        server.await.unwrap();
        let requests = seen.lock().unwrap();
        assert_eq!(requests.len(), 2);
        assert!(requests[0].starts_with("POST /upload "));
        assert!(requests[1].starts_with("POST /parse "));
        for request in requests.iter() {
            assert!(request.contains("authorization: Bearer guarded"));
            assert!(!request.contains("Bearer original"));
        }
    }

    #[tokio::test]
    async fn guardrail_rewrites_document_before_upload() {
        let (base, seen, server) =
            mock_server(vec![MockResponse::json(json!({"result":{"chunks":[]}}))]).await;
        let host = LocalOcrHost::new(wire_request("reducto/parse-v3", &base, json!({})))
            .with_before_send(|wire, _| {
                assert_eq!(
                    wire.body["document_url"],
                    "data:application/pdf;base64,YWJj"
                );
                Ok(WireRequest {
                    body: json!({"type":"document_url","document_url":"reducto://guarded.pdf"}),
                    ..wire
                })
            });

        perform_ocr_with(host).await.unwrap();
        server.await.unwrap();
        let requests = seen.lock().unwrap();
        assert_eq!(requests.len(), 1);
        assert!(requests[0].starts_with("POST /parse "));
        assert!(requests[0].contains("reducto://guarded.pdf"));
    }
}
