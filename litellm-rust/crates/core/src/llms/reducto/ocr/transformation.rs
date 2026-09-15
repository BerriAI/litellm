mod types {
    use serde::{Deserialize, Deserializer, Serialize};
    use serde_json::{Map, Value};

    #[derive(Clone, Debug, Default, Serialize, Deserialize)]
    pub(crate) struct ReductoV3Params {
        #[serde(skip_serializing_if = "Option::is_none")]
        pub formatting: Option<Map<String, Value>>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub retrieval: Option<Map<String, Value>>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub settings: Option<Map<String, Value>>,
    }

    #[derive(Clone, Debug, Default, Serialize, Deserialize)]
    pub(crate) struct ReductoLegacyParams {
        #[serde(skip_serializing_if = "Option::is_none")]
        pub enhance: Option<Map<String, Value>>,
    }

    #[derive(Clone, Debug, Serialize, Deserialize)]
    pub(crate) struct ReductoV3Request {
        pub input: String,
        #[serde(flatten)]
        pub params: ReductoV3Params,
    }

    #[derive(Clone, Debug, Serialize, Deserialize)]
    pub(crate) struct ReductoLegacyRequest {
        pub document_url: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub options: Option<ReductoLegacyParams>,
    }

    #[derive(Deserialize)]
    pub(crate) struct ReductoUploadResponse {
        pub file_id: Option<String>,
    }

    #[derive(Clone, Debug, Deserialize)]
    pub(crate) struct ReductoResponse {
        #[serde(default, deserialize_with = "present_nullable")]
        pub result: Option<Option<ReductoResult>>,
        pub usage: Option<ReductoUsage>,
        #[serde(default)]
        pub chunks: Option<Vec<ReductoChunk>>,
    }

    fn present_nullable<'de, D: Deserializer<'de>, T: Deserialize<'de>>(
        deserializer: D,
    ) -> Result<Option<Option<T>>, D::Error> {
        Option::<T>::deserialize(deserializer).map(Some)
    }

    #[derive(Clone, Debug, Default, Deserialize)]
    pub(crate) struct ReductoResult {
        pub chunks: Option<Vec<ReductoChunk>>,
    }

    #[derive(Clone, Debug, Default, Deserialize)]
    pub(crate) struct ReductoUsage {
        #[serde(default, deserialize_with = "optional_i64")]
        pub num_pages: Option<i64>,
        #[serde(default, deserialize_with = "optional_f64")]
        pub credits: Option<f64>,
    }

    #[derive(Clone, Debug, Deserialize)]
    pub(crate) struct ReductoChunk {
        pub content: Option<String>,
        pub blocks: Option<Vec<ReductoBlock>>,
    }

    #[derive(Clone, Debug, Serialize, Deserialize)]
    pub(crate) struct ReductoBlock {
        #[serde(skip_serializing_if = "Option::is_none")]
        pub content: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub bbox: Option<ReductoBoundingBox>,
        #[serde(flatten)]
        pub extra_fields: Map<String, Value>,
    }

    #[derive(Clone, Debug, Serialize, Deserialize)]
    pub(crate) struct ReductoBoundingBox {
        #[serde(default, deserialize_with = "optional_i64")]
        pub page: Option<i64>,
        #[serde(flatten)]
        pub extra_fields: Map<String, Value>,
    }

    fn optional_i64<'de, D: Deserializer<'de>>(deserializer: D) -> Result<Option<i64>, D::Error> {
        match Option::<Value>::deserialize(deserializer)? {
            None | Some(Value::Null) => Ok(None),
            Some(Value::Number(number)) => number
                .as_i64()
                .or_else(|| number.as_f64().and_then(checked_truncated_i64))
                .map(Some)
                .ok_or_else(|| serde::de::Error::custom("expected an integer")),
            Some(Value::String(value)) => value
                .trim()
                .parse::<i64>()
                .map(Some)
                .map_err(|_| serde::de::Error::custom("expected an integer")),
            Some(Value::Bool(value)) => Ok(Some(i64::from(value))),
            Some(_) => Ok(None),
        }
    }

    fn optional_f64<'de, D: Deserializer<'de>>(deserializer: D) -> Result<Option<f64>, D::Error> {
        match Option::<Value>::deserialize(deserializer)? {
            None | Some(Value::Null) => Ok(None),
            Some(Value::Number(number)) => number
                .as_f64()
                .map(Some)
                .ok_or_else(|| serde::de::Error::custom("expected a number")),
            Some(Value::String(value)) => value
                .trim()
                .parse::<f64>()
                .map(Some)
                .map_err(|_| serde::de::Error::custom("expected a number")),
            Some(_) => Ok(None),
        }
    }

    fn checked_truncated_i64(value: f64) -> Option<i64> {
        (value.is_finite() && value >= i64::MIN as f64 && value <= i64::MAX as f64)
            .then(|| value.trunc() as i64)
    }
}

#[cfg(test)]
pub(crate) use mapping::transform_ocr_response;
pub(crate) use types::ReductoResponse;

mod mapping {
    use std::collections::BTreeMap;

    use serde_json::{Value, json};

    use super::types::*;
    use crate::ocr::error::{OcrRequestError, OcrResponseError};
    use crate::ocr::types::{LiteLLMOcrResponse, OcrDocument};

    #[tracing::instrument(
        name = "transform_ocr_request",
        target = "litellm::function_trace",
        level = "trace",
        skip_all
    )]
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

    #[tracing::instrument(
        name = "transform_ocr_request",
        target = "litellm::function_trace",
        level = "trace",
        skip_all
    )]
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
}

mod common {
    use crate::Error;
    use crate::constants::{REDUCTO_API_BASE, REDUCTO_API_KEY_ENV, REDUCTO_ID_PREFIX};
    use crate::llms::base_llm::ocr::transformation::BaseOcrConfig;
    use crate::ocr::document::InlineDocument;
    use crate::ocr::error::{OcrError, OcrRequestError, OcrResponseError};
    use crate::ocr::types::{OcrConnection, OcrDocument};
    use crate::url_utils::ApiUrl;

    pub(crate) trait BaseReductoOcrConfig: BaseOcrConfig {
        fn get_complete_url(&self, api_base: Option<&str>, path: &str) -> Result<String, OcrError> {
            get_complete_url(api_base, path)
        }

        fn validate_environment(
            &self,
            connection: &OcrConnection,
            env_lookup: &(dyn Fn(&str) -> Option<String> + Sync),
        ) -> Result<Vec<(String, String)>, OcrError> {
            validate_environment(connection, env_lookup)
        }

        async fn prepare_document(
            &self,
            client: &crate::ocr::OcrClient,
            document: OcrDocument,
            connection: &OcrConnection,
            headers: &[(String, String)],
        ) -> Result<OcrDocument, OcrError> {
            prepare_document(client, document, connection, headers).await
        }
    }

    pub(super) fn get_complete_url(api_base: Option<&str>, path: &str) -> Result<String, OcrError> {
        let base = api_base
            .map(str::trim)
            .filter(|base| !base.is_empty())
            .unwrap_or(REDUCTO_API_BASE);
        ApiUrl::parse(base)
            .and_then(|url| url.complete_path(&[path]))
            .map(|url| url.into_string())
            .map_err(|_| {
                OcrRequestError::RequestField {
                    path: "api_base".into(),
                }
                .into()
            })
    }

    pub(super) fn validate_environment(
        connection: &OcrConnection,
        env_lookup: &(dyn Fn(&str) -> Option<String> + Sync),
    ) -> Result<Vec<(String, String)>, OcrError> {
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
            .ok_or(Error::MissingReductoApiKey)?;
        Ok(
            std::iter::once(("Authorization".into(), format!("Bearer {api_key}")))
                .chain(connection.extra_headers.clone())
                .collect(),
        )
    }

    pub(super) async fn prepare_document(
        client: &crate::ocr::OcrClient,
        document: OcrDocument,
        connection: &OcrConnection,
        headers: &[(String, String)],
    ) -> Result<OcrDocument, OcrError> {
        if document.source().starts_with(REDUCTO_ID_PREFIX) {
            if document.source()[REDUCTO_ID_PREFIX.len()..]
                .trim()
                .is_empty()
            {
                return Err(OcrRequestError::RequestField {
                    path: "document file id".into(),
                }
                .into());
            }
            return Ok(document);
        }
        let inline =
            InlineDocument::parse(document.source())?.ok_or(OcrRequestError::ReductoSource)?;
        let mime = inline.mime_type().to_string();
        let bytes = inline.decode(crate::constants::OCR_INLINE_MAX_BYTES)?;
        let part = reqwest::multipart::Part::bytes(bytes)
            .file_name("document")
            .mime_str(&mime)
            .map_err(|_| OcrRequestError::InvalidDataUri)?;
        let builder = client
            .provider_http()
            .post(get_complete_url(connection.api_base.as_deref(), "upload")?)
            .multipart(reqwest::multipart::Form::new().part("file", part))
            .timeout(connection.timeout);
        let builder = crate::http_utils::with_headers(
            builder,
            headers,
            crate::http_utils::HeaderPolicy::Except(&["content-type", "content-length"]),
        );
        let response = crate::http_utils::http_request(builder)
            .await
            .map_err(crate::error::TransportError::from)?;
        let uploaded =
            crate::ocr::client::read_json_response::<super::types::ReductoUploadResponse>(
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
            return Err(OcrResponseError::ResponseField {
                path: "file_id".into(),
            }
            .into());
        };
        Ok(document.with_source(file_id.to_string()))
    }

    #[cfg(test)]
    mod tests {
        use super::*;

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
}

mod legacy {
    use super::common::BaseReductoOcrConfig;
    use super::mapping as reducto;
    use super::types::{ReductoLegacyParams, ReductoResponse};
    use crate::llms::base_llm::ocr::transformation::BaseOcrConfig;
    use crate::ocr::OcrClient;
    use crate::ocr::error::{OcrError, OcrResponseError};
    use crate::ocr::prepare::{
        _prepare_ocr_request, ParsedProviderParams, build_http_request, credential_env,
        guardrail_document, merge_extra_params,
    };
    use crate::ocr::types::{LiteLLMOcrRequest, LiteLLMOcrResponse};

    #[derive(Clone, Debug)]
    pub(crate) struct ReductoParseLegacyConfig;

    impl BaseReductoOcrConfig for ReductoParseLegacyConfig {}

    impl BaseOcrConfig for ReductoParseLegacyConfig {
        type ProviderResponse = ReductoResponse;

        async fn prepare_request(
            &self,
            request: &LiteLLMOcrRequest,
            client: &OcrClient,
        ) -> Result<reqwest::Request, OcrError> {
            let ParsedProviderParams {
                known: params,
                extra_params,
            } = _prepare_ocr_request::<ReductoLegacyParams>(request)?;
            let headers = self.validate_environment(&request.connection, &credential_env)?;
            let url = self.get_complete_url(request.connection.api_base.as_deref(), "parse")?;
            let (document, headers) = guardrail_document(request, &url, &headers).await?;
            let document = self
                .prepare_document(client, document, &request.connection, &headers)
                .await?;
            let body = reducto::transform_legacy_ocr_request(&request.model, document, &params)?;
            let body = merge_extra_params(&body, extra_params)?;
            build_http_request(client, request, &url, &headers, &body)
        }

        fn transform_ocr_response(
            &self,
            request: &LiteLLMOcrRequest,
            response: ReductoResponse,
        ) -> Result<LiteLLMOcrResponse, OcrResponseError> {
            reducto::transform_ocr_response(&request.model, response)
        }
    }
}

pub(crate) use legacy::ReductoParseLegacyConfig;

mod v3 {
    use super::common::BaseReductoOcrConfig;
    use super::mapping as reducto;
    use super::types::{ReductoResponse, ReductoV3Params};
    use crate::llms::base_llm::ocr::transformation::BaseOcrConfig;
    use crate::ocr::OcrClient;
    use crate::ocr::error::{OcrError, OcrResponseError};
    use crate::ocr::prepare::{
        _prepare_ocr_request, ParsedProviderParams, build_http_request, credential_env,
        guardrail_document, merge_extra_params,
    };
    use crate::ocr::types::{LiteLLMOcrRequest, LiteLLMOcrResponse};

    #[derive(Clone, Debug)]
    pub(crate) struct ReductoParseV3Config;

    impl BaseReductoOcrConfig for ReductoParseV3Config {}

    impl BaseOcrConfig for ReductoParseV3Config {
        type ProviderResponse = ReductoResponse;

        async fn prepare_request(
            &self,
            request: &LiteLLMOcrRequest,
            client: &OcrClient,
        ) -> Result<reqwest::Request, OcrError> {
            let ParsedProviderParams {
                known: params,
                extra_params,
            } = _prepare_ocr_request::<ReductoV3Params>(request)?;
            let headers = self.validate_environment(&request.connection, &credential_env)?;
            let url = self.get_complete_url(request.connection.api_base.as_deref(), "parse")?;
            let (document, headers) = guardrail_document(request, &url, &headers).await?;
            let document = self
                .prepare_document(client, document, &request.connection, &headers)
                .await?;
            let body = reducto::transform_v3_ocr_request(&request.model, document, &params)?;
            let body = merge_extra_params(&body, extra_params)?;
            build_http_request(client, request, &url, &headers, &body)
        }

        fn transform_ocr_response(
            &self,
            request: &LiteLLMOcrRequest,
            response: ReductoResponse,
        ) -> Result<LiteLLMOcrResponse, OcrResponseError> {
            reducto::transform_ocr_response(&request.model, response)
        }
    }
}

pub(crate) use v3::ReductoParseV3Config;
