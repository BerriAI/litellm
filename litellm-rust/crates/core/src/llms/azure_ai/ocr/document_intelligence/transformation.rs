use std::collections::BTreeSet;
use std::sync::Arc;
use std::time::Duration;

use base64::{Engine, engine::general_purpose::STANDARD};
use reqwest::Url;
use serde::{Deserialize, Deserializer, Serialize};
use serde_json::{Map, Value, json};
use tokio::time::Instant;

use litellm_auth::{InputSource, Sourced};
use litellm_auth_azure::AzureAuthInputs;

use crate::constants::{
    AZURE_DI_API_VERSION, AZURE_DI_DEFAULT_DPI, AZURE_DI_DEFAULT_HEIGHT, AZURE_DI_DEFAULT_WIDTH,
    AZURE_DI_SUBSCRIPTION_HEADER, OCR_POLL_RETRY_SECS,
};
use crate::llms::base_llm::ocr::transformation::BaseOcrConfig;
use crate::ocr::OcrClient;
use crate::ocr::client::read_json_response;
use crate::ocr::document::InlineDocument;
use crate::ocr::hooks::OcrHooks;
use crate::ocr::prepare::{ParsedProviderParams, credential_env, transform_request_body};
use crate::ocr::types::{
    LiteLLMOcrRequest, LiteLLMOcrResponse, OcrConnection, OcrDocument, OcrResponseFormat,
};
use crate::ocr::wire::DecodedOcrResponse;
use crate::url_utils::ApiUrl;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
enum PagesInput {
    ZeroBasedIndices(Vec<i64>),
    NativeTokens(Vec<String>),
    NativeRange(String),
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
enum FeaturesInput {
    Names(Vec<String>),
    CommaSeparated(String),
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
struct DocumentIntelligenceInputParams {
    pub pages: Option<PagesInput>,
    pub features: Option<FeaturesInput>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
struct DocumentIntelligenceParams {
    pub pages: Option<String>,
    pub features: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(untagged)]
enum DocumentIntelligenceRequest {
    UrlSource {
        #[serde(rename = "urlSource")]
        url_source: String,
    },
    Base64Source {
        #[serde(rename = "base64Source")]
        base64_source: String,
    },
}

#[derive(Clone, Debug, PartialEq)]
enum OperationStatus {
    Succeeded,
    Running,
    NotStarted,
    Failed,
    Unknown(String),
}

impl<'de> Deserialize<'de> for OperationStatus {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        Ok(match String::deserialize(deserializer)?.as_str() {
            "succeeded" => Self::Succeeded,
            "running" => Self::Running,
            "notStarted" => Self::NotStarted,
            "failed" => Self::Failed,
            value => Self::Unknown(value.to_string()),
        })
    }
}

impl std::fmt::Display for OperationStatus {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(match self {
            Self::Succeeded => "succeeded",
            Self::Running => "running",
            Self::NotStarted => "notStarted",
            Self::Failed => "failed",
            Self::Unknown(value) => value,
        })
    }
}

#[derive(Clone, Debug, Deserialize)]
pub(crate) struct AzureDocumentIntelligenceOperation {
    status: Option<OperationStatus>,
    #[serde(rename = "analyzeResult")]
    analyze_result: Option<AzureDocumentIntelligenceAnalyzeResult>,
}

#[derive(Clone, Debug, Default, Deserialize)]
struct AzureDocumentIntelligenceAnalyzeResult {
    pub content: Option<String>,
    #[serde(default)]
    pub pages: Vec<AzureDocumentIntelligencePage>,
    pub tables: Option<Vec<Map<String, Value>>>,
    #[serde(rename = "keyValuePairs")]
    pub key_value_pairs: Option<Vec<Map<String, Value>>>,
}

#[derive(Clone, Debug, Deserialize)]
struct AzureDocumentIntelligencePage {
    #[serde(rename = "pageNumber", default, deserialize_with = "optional_i64")]
    pub page_number: Option<i64>,
    #[serde(default, deserialize_with = "optional_f64")]
    pub width: Option<f64>,
    #[serde(default, deserialize_with = "optional_f64")]
    pub height: Option<f64>,
    pub unit: Option<String>,
    #[serde(default)]
    pub lines: Vec<AzureDocumentIntelligenceLine>,
}

#[derive(Clone, Debug, Deserialize)]
struct AzureDocumentIntelligenceLine {
    pub content: Option<String>,
}

fn optional_i64<'de, D: Deserializer<'de>>(deserializer: D) -> Result<Option<i64>, D::Error> {
    match Option::<Value>::deserialize(deserializer)? {
        None | Some(Value::Null) => Ok(None),
        Some(Value::Number(number)) => number
            .as_i64()
            .map(Some)
            .ok_or_else(|| serde::de::Error::custom("expected an integer")),
        Some(Value::String(value)) => value
            .parse::<i64>()
            .map(Some)
            .map_err(|_| serde::de::Error::custom("expected an integer")),
        Some(_) => Err(serde::de::Error::custom("expected an integer")),
    }
}

fn optional_f64<'de, D: Deserializer<'de>>(deserializer: D) -> Result<Option<f64>, D::Error> {
    match Option::<Value>::deserialize(deserializer)? {
        None | Some(Value::Null) => Ok(None),
        Some(Value::Number(number)) => number
            .as_f64()
            .filter(|value| value.is_finite())
            .map(Some)
            .ok_or_else(|| serde::de::Error::custom("expected a finite number")),
        Some(Value::String(value)) => value
            .parse::<f64>()
            .ok()
            .filter(|value| value.is_finite())
            .map(Some)
            .ok_or_else(|| serde::de::Error::custom("expected a finite number")),
        Some(_) => Err(serde::de::Error::custom("expected a number")),
    }
}

fn decode_input_params(
    params: Map<String, Value>,
    prefix: &str,
) -> Result<ParsedProviderParams<DocumentIntelligenceInputParams>, crate::ocr::Error> {
    if let Some(Value::Array(pages)) = params.get("pages") {
        if pages.iter().any(Value::is_boolean) {
            return Err(crate::ocr::Error::Pages("boolean page index".into()));
        }
        if pages
            .iter()
            .any(|page| page.is_number() && page.as_i64().is_none())
        {
            return Err(crate::ocr::Error::Pages(
                "page index is out of range".into(),
            ));
        }
        if !pages.iter().all(Value::is_i64) && !pages.iter().all(Value::is_string) {
            return Err(crate::ocr::Error::Pages("mixed page element types".into()));
        }
    }
    crate::ocr::wire::decode_request_value(Value::Object(params), prefix)
}

fn normalize_ocr_params(
    params: DocumentIntelligenceInputParams,
) -> Result<DocumentIntelligenceParams, crate::ocr::Error> {
    Ok(DocumentIntelligenceParams {
        pages: params.pages.map(normalize_pages).transpose()?.flatten(),
        features: params
            .features
            .map(normalize_features)
            .transpose()?
            .flatten(),
    })
}

fn normalize_pages(pages: PagesInput) -> Result<Option<String>, crate::ocr::Error> {
    let normalized = match pages {
        PagesInput::ZeroBasedIndices(indices) => {
            if indices.is_empty() {
                return Ok(None);
            }
            indices
                .into_iter()
                .map(|page| {
                    if page < 0 {
                        return Err(crate::ocr::Error::Pages("negative page index".into()));
                    }
                    page.checked_add(1).ok_or_else(|| {
                        crate::ocr::Error::Pages("page index is out of range".into())
                    })
                })
                .collect::<Result<BTreeSet<_>, _>>()?
                .into_iter()
                .map(|page| page.to_string())
                .collect::<Vec<_>>()
                .join(",")
        }
        PagesInput::NativeTokens(tokens) => {
            if tokens.is_empty() {
                return Ok(None);
            }
            tokens
                .iter()
                .map(|token| token.trim())
                .collect::<Vec<_>>()
                .join(",")
        }
        PagesInput::NativeRange(range) => range
            .split(',')
            .map(str::trim)
            .collect::<Vec<_>>()
            .join(","),
    };
    if !normalized.split(',').all(valid_page_token) {
        return Err(crate::ocr::Error::Pages("invalid native page range".into()));
    }
    Ok(Some(normalized))
}

fn valid_page_token(token: &str) -> bool {
    let mut parts = token.split('-');
    let start = parts.next().unwrap_or_default();
    if start.is_empty() || !start.chars().all(|character| character.is_ascii_digit()) {
        return false;
    }
    match parts.next() {
        None => true,
        Some(end) => {
            !end.is_empty()
                && end.chars().all(|character| character.is_ascii_digit())
                && parts.next().is_none()
        }
    }
}

fn normalize_features(features: FeaturesInput) -> Result<Option<String>, crate::ocr::Error> {
    let tokens = match features {
        FeaturesInput::Names(names) => names,
        FeaturesInput::CommaSeparated(names) => names.split(',').map(str::to_string).collect(),
    };
    if tokens.is_empty() {
        return Ok(None);
    }
    let normalized = tokens.iter().map(|token| token.trim()).collect::<Vec<_>>();
    if !normalized.iter().all(|token| {
        let Some((first, rest)) = token.as_bytes().split_first() else {
            return false;
        };
        first.is_ascii_alphabetic() && rest.iter().all(u8::is_ascii_alphanumeric)
    }) {
        return Err(crate::ocr::Error::Features);
    }
    Ok(Some(normalized.join(",")))
}

#[tracing::instrument(target = "litellm::function_trace", level = "trace", skip_all)]
fn transform_ocr_request(
    document: OcrDocument,
) -> Result<DocumentIntelligenceRequest, crate::ocr::Error> {
    let source = document.source();
    if source.is_empty() {
        return Err(crate::ocr::Error::MissingDocumentUrl);
    }
    Ok(if let Some(document) = InlineDocument::parse(source)? {
        DocumentIntelligenceRequest::Base64Source {
            base64_source: STANDARD
                .encode(document.decode(crate::constants::OCR_INLINE_MAX_BYTES)?),
        }
    } else {
        DocumentIntelligenceRequest::UrlSource {
            url_source: source.to_string(),
        }
    })
}

fn transform_ocr_response(
    model: &str,
    response: AzureDocumentIntelligenceOperation,
) -> Result<LiteLLMOcrResponse, crate::ocr::Error> {
    if response.status != Some(OperationStatus::Succeeded) {
        return Err(crate::ocr::Error::OperationStatus(
            response
                .status
                .map(|status| status.to_string())
                .unwrap_or_else(|| "None".into()),
        ));
    }
    let result = response.analyze_result.unwrap_or_default();
    let pages = result
        .pages
        .into_iter()
        .map(normalize_page)
        .collect::<Result<Vec<_>, _>>()?;
    let pages_processed = pages.len();
    let mut extra_fields = Map::new();
    extra_fields.insert("content".into(), option_value(result.content));
    extra_fields.insert("tables".into(), option_value(result.tables));
    extra_fields.insert("keyValuePairs".into(), option_value(result.key_value_pairs));
    Ok(LiteLLMOcrResponse {
        pages,
        model: model.into(),
        document_annotation: None,
        usage_info: Some(json!({"pages_processed":pages_processed})),
        object: "ocr".into(),
        extra_fields,
        provider_native_response: None,
    })
}

fn normalize_page(page: AzureDocumentIntelligencePage) -> Result<Value, crate::ocr::Error> {
    let index = page
        .page_number
        .unwrap_or(1)
        .checked_sub(1)
        .ok_or(crate::ocr::Error::NumericRange("page.pageNumber"))?;
    let scale = if page.unit.as_deref().unwrap_or("inch") == "inch" {
        AZURE_DI_DEFAULT_DPI as f64
    } else {
        1.0
    };
    let width = pixel_dimension(
        page.width.unwrap_or(AZURE_DI_DEFAULT_WIDTH),
        scale,
        "page.width",
    )?;
    let height = pixel_dimension(
        page.height.unwrap_or(AZURE_DI_DEFAULT_HEIGHT),
        scale,
        "page.height",
    )?;
    let markdown = page
        .lines
        .iter()
        .map(|line| line.content.as_deref().unwrap_or_default())
        .collect::<Vec<_>>()
        .join("\n");
    Ok(json!({
        "index":index,
        "markdown":markdown,
        "images":null,
        "dimensions":{"width":width,"height":height,"dpi":AZURE_DI_DEFAULT_DPI}
    }))
}

fn pixel_dimension(value: f64, scale: f64, field: &'static str) -> Result<i64, crate::ocr::Error> {
    let value = value * scale;
    if !value.is_finite() || value < i64::MIN as f64 || value > i64::MAX as f64 {
        return Err(crate::ocr::Error::NumericRange(field));
    }
    Ok(value.trunc() as i64)
}

fn option_value<T: serde::Serialize>(value: Option<T>) -> Value {
    value
        .and_then(|value| serde_json::to_value(value).ok())
        .unwrap_or(Value::Null)
}
async fn read_operation_response(
    http_client: &reqwest::Client,
    response: reqwest::Response,
    original_url: &str,
    headers: &[(String, String)],
    connection: &OcrConnection,
    native: bool,
    hooks: &Arc<dyn OcrHooks>,
) -> Result<DecodedOcrResponse<AzureDocumentIntelligenceOperation>, crate::ocr::Error> {
    if response.status() != reqwest::StatusCode::ACCEPTED {
        let bytes =
            crate::ocr::client::read_response_bytes(response, connection.max_response_bytes)
                .await?;
        crate::ocr::handler::post_call(hooks, &bytes).await?;
        return crate::ocr::wire::decode_response(&bytes, native);
    }
    let location = response
        .headers()
        .get("operation-location")
        .and_then(|value| value.to_str().ok())
        .ok_or(crate::ocr::Error::PollLocation)?
        .to_string();
    let original = Url::parse(original_url).map_err(|_| crate::ocr::Error::PollOrigin)?;
    let operation = Url::parse(&location).map_err(|_| crate::ocr::Error::PollOrigin)?;
    if original.origin() != operation.origin()
        || !operation.username().is_empty()
        || operation.password().is_some()
    {
        return Err(crate::ocr::Error::PollOrigin);
    }
    let bytes =
        crate::ocr::client::read_response_bytes(response, connection.max_response_bytes).await?;
    crate::ocr::handler::post_call(hooks, &bytes).await?;
    poll_operation(http_client, operation, headers, connection, native).await
}

async fn poll_operation(
    http_client: &reqwest::Client,
    url: Url,
    headers: &[(String, String)],
    connection: &OcrConnection,
    native: bool,
) -> Result<DecodedOcrResponse<AzureDocumentIntelligenceOperation>, crate::ocr::Error> {
    let deadline = Instant::now()
        .checked_add(connection.poll_timeout)
        .ok_or(crate::ocr::Error::PollTimeout)?;

    loop {
        let remaining = deadline
            .checked_duration_since(Instant::now())
            .filter(|remaining| !remaining.is_zero())
            .ok_or(crate::ocr::Error::PollTimeout)?;
        let builder = http_client
            .get(url.clone())
            .timeout(remaining.min(connection.timeout));
        let builder = crate::http_utils::with_headers(
            builder,
            headers,
            crate::http_utils::HeaderPolicy::Only(&[AZURE_DI_SUBSCRIPTION_HEADER, "authorization"]),
        );
        let response = tokio::time::timeout_at(deadline, crate::http_utils::http_request(builder))
            .await
            .map_err(|_| crate::ocr::Error::PollTimeout)?
            .map_err(crate::transport::Error::from)?;
        let retry = response
            .headers()
            .get(reqwest::header::RETRY_AFTER)
            .and_then(|value| value.to_str().ok())
            .and_then(|value| value.parse::<u64>().ok())
            .unwrap_or(OCR_POLL_RETRY_SECS)
            .max(1);
        let decoded = tokio::time::timeout_at(
            deadline,
            read_json_response::<AzureDocumentIntelligenceOperation>(
                response,
                native,
                connection.max_response_bytes,
            ),
        )
        .await
        .map_err(|_| crate::ocr::Error::PollTimeout)??;
        match &decoded.data.status {
            Some(OperationStatus::Succeeded) => return Ok(decoded),
            Some(OperationStatus::Running | OperationStatus::NotStarted) => {
                tokio::time::timeout_at(deadline, tokio::time::sleep(Duration::from_secs(retry)))
                    .await
                    .map_err(|_| crate::ocr::Error::PollTimeout)?;
            }
            status => {
                return Err(crate::ocr::Error::OperationStatus(
                    status
                        .as_ref()
                        .map(ToString::to_string)
                        .unwrap_or_else(|| "None".into()),
                ));
            }
        }
    }
}

const AZURE_DI_API_KEY_ENV: &str = "AZURE_DOCUMENT_INTELLIGENCE_API_KEY";
const AZURE_DI_ENDPOINT_ENV: &str = "AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT";

#[derive(Clone, Debug)]
pub(crate) struct AzureDocumentIntelligenceOCRConfig;

impl BaseOcrConfig for AzureDocumentIntelligenceOCRConfig {
    type ProviderResponse = AzureDocumentIntelligenceOperation;

    fn get_supported_ocr_params(&self, _model: &str) -> &'static [&'static str] {
        &["pages", "features"]
    }

    async fn prepare_request(
        &self,
        request: &LiteLLMOcrRequest,
        client: &OcrClient,
    ) -> Result<reqwest::Request, crate::ocr::Error> {
        let params = map_ocr_params(request)?;
        let config = AzureAuthInputs {
            azure_ad_token_provider: request.azure_ad_token_provider.clone(),
            ..AzureAuthInputs::from_sourced_optional_params(
                &request.optional_params,
                &request.input_sources,
            )
            .map_err(crate::ocr::Error::from)?
        };
        let headers = validate_environment(&request.connection, &config, &credential_env).await?;
        let endpoint = nonblank(request.connection.api_base.clone())
            .or_else(|| nonblank(credential_env(AZURE_DI_ENDPOINT_ENV)))
            .ok_or_else(|| crate::ocr::Error::Auth(litellm_auth::Error::ProviderAuthentication("Missing Azure Document Intelligence API Base - Set AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT or pass api_base".into())))?;
        let url = get_complete_url(&endpoint, &request.model, &params)?;
        let body = transform_ocr_request(request.document.clone())?;
        transform_request_body(client, request, &url, &headers, false, body, |_| Ok(())).await
    }

    fn transform_ocr_response(
        &self,
        request: &LiteLLMOcrRequest,
        response: AzureDocumentIntelligenceOperation,
    ) -> Result<LiteLLMOcrResponse, crate::ocr::Error> {
        transform_ocr_response(&request.model, response)
    }

    async fn read_response(
        &self,
        client: &OcrClient,
        response: reqwest::Response,
        url: &str,
        headers: &[(String, String)],
        request: &LiteLLMOcrRequest,
    ) -> Result<
        crate::ocr::wire::DecodedOcrResponse<AzureDocumentIntelligenceOperation>,
        crate::ocr::Error,
    > {
        read_operation_response(
            client.polling_http(),
            response,
            url,
            headers,
            &request.connection,
            request.response_format()? == OcrResponseFormat::Native,
            &request.hooks,
        )
        .await
    }
}

#[tracing::instrument(target = "litellm::function_trace", level = "trace", skip_all)]
fn map_ocr_params(
    request: &LiteLLMOcrRequest,
) -> Result<DocumentIntelligenceParams, crate::ocr::Error> {
    let params = decode_input_params(request.optional_params.clone().into(), "optional_params")?;
    let crate::ocr::prepare::ParsedProviderParams {
        known: params,
        extra_params: _extra_params,
    } = params;
    normalize_ocr_params(params)
}

fn get_complete_url(
    endpoint: &str,
    model: &str,
    params: &DocumentIntelligenceParams,
) -> Result<String, crate::ocr::Error> {
    let model = format!("{}:analyze", model_id(model)?);
    ApiUrl::parse(endpoint)
        .and_then(|url| url.complete_path(&["documentintelligence", "documentModels", &model]))
        .map(|url| {
            url.append_query_pairs(
                [("api-version", AZURE_DI_API_VERSION)]
                    .into_iter()
                    .chain(params.pages.iter().map(|pages| ("pages", pages.as_str())))
                    .chain(
                        params
                            .features
                            .iter()
                            .map(|features| ("features", features.as_str())),
                    ),
            )
            .into_string()
        })
        .map_err(|_| crate::ocr::Error::RequestField {
            path: "api_base".into(),
        })
}

async fn validate_environment(
    connection: &OcrConnection,
    config: &AzureAuthInputs,
    env_lookup: &(dyn Fn(&str) -> Option<String> + Sync),
) -> Result<Vec<(String, String)>, crate::ocr::Error> {
    if crate::http_utils::has_header(&connection.extra_headers, "authorization")
        || crate::http_utils::has_header(&connection.extra_headers, AZURE_DI_SUBSCRIPTION_HEADER)
    {
        super::super::common_utils::validate_destination(
            connection,
            connection.extra_headers_source,
        )?;
        return Ok(connection.extra_headers.clone());
    }
    let key = nonblank(connection.api_key.clone())
        .map(|value| Sourced::new(value, connection.api_key_source))
        .or_else(|| {
            nonblank(env_lookup(AZURE_DI_API_KEY_ENV))
                .map(|value| Sourced::new(value, InputSource::Environment))
        });
    if let Some(key) = key {
        super::super::common_utils::validate_destination(connection, key.source())?;
        return Ok(
            std::iter::once((AZURE_DI_SUBSCRIPTION_HEADER.into(), key.into_value()))
                .chain(connection.extra_headers.clone())
                .collect(),
        );
    }
    let token = super::super::common_utils::resolve_entra(config, env_lookup)
        .await?
        .ok_or(crate::ocr::Error::MissingAzureDocumentIntelligenceCredentials)?;
    super::super::common_utils::validate_destination(connection, token.source())?;
    Ok(
        std::iter::once(("Authorization".into(), format!("Bearer {}", token.value())))
            .chain(connection.extra_headers.clone())
            .collect(),
    )
}

fn model_id(model: &str) -> Result<&str, crate::ocr::Error> {
    let model = model.rsplit('/').next().unwrap_or(model);
    if matches!(model, "." | "..") {
        return Err(crate::ocr::Error::DotModel);
    }
    Ok(model)
}

fn nonblank(value: Option<String>) -> Option<String> {
    value
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::rstest;
    use serde_json::{Value, json};

    fn map(value: Value) -> Result<DocumentIntelligenceParams, crate::ocr::Error> {
        let fields = value.as_object().unwrap().clone();
        normalize_ocr_params(decode_input_params(fields, "optional_params")?.known)
    }

    #[test]
    fn input_params_retain_unknown_fields() {
        let parsed = decode_input_params(
            json!({
                "pages": [0],
                "future_ocr_option": true,
                "extra_body": {"provider_option": "value"}
            })
            .as_object()
            .unwrap()
            .clone(),
            "optional_params",
        )
        .unwrap();

        assert_eq!(
            parsed.known.pages,
            Some(PagesInput::ZeroBasedIndices(vec![0]))
        );
        assert_eq!(parsed.extra_params["future_ocr_option"], true);
        assert_eq!(
            parsed.extra_params["extra_body"],
            json!({"provider_option": "value"})
        );
        assert_eq!(
            serde_json::to_value(normalize_ocr_params(parsed.known).unwrap()).unwrap(),
            json!({"pages": "1", "features": null})
        );
    }

    #[rstest]
    #[case(json!([0, 1, 2]), Some("1,2,3"))]
    #[case(json!([2, 0, 0, 1]), Some("1,2,3"))]
    #[case(json!([]), None)]
    #[case(json!("3-9"), Some("3-9"))]
    #[case(json!("1-3, 5"), Some("1-3,5"))]
    #[case(json!(["1", "3-5"]), Some("1,3-5"))]
    fn page_mapping_matches_python(#[case] input: Value, #[case] expected: Option<&str>) {
        assert_eq!(
            map(json!({"pages": input})).unwrap().pages.as_deref(),
            expected
        );
    }

    #[rstest]
    #[case(json!("a,b"))]
    #[case(json!([-1]))]
    #[case(json!([true, false]))]
    #[case(json!([1, "2"]))]
    #[case(json!(5))]
    fn invalid_page_mapping_matches_python(#[case] input: Value) {
        assert!(map(json!({"pages": input})).is_err());
    }

    #[rstest]
    #[case(json!(["keyValuePairs"]), "keyValuePairs")]
    #[case(json!(["keyValuePairs", "languages"]), "keyValuePairs,languages")]
    #[case(json!("keyValuePairs"), "keyValuePairs")]
    #[case(json!("keyValuePairs,languages"), "keyValuePairs,languages")]
    #[case(json!("keyValuePairs, languages"), "keyValuePairs,languages")]
    fn feature_mapping_matches_python(#[case] input: Value, #[case] expected: &str) {
        assert_eq!(
            map(json!({"features": input})).unwrap().features.as_deref(),
            Some(expected)
        );
    }

    #[rstest]
    #[case(json!("keyValuePairs&pages=9"))]
    #[case(json!("key value pairs"))]
    #[case(json!(""))]
    #[case(json!([1, 2]))]
    #[case(json!([["keyValuePairs"]]))]
    #[case(json!({"feature":"keyValuePairs"}))]
    #[case(json!(5))]
    fn invalid_feature_mapping_matches_python(#[case] input: Value) {
        assert!(map(json!({"features": input})).is_err());
    }

    #[test]
    fn empty_feature_list_is_omitted() {
        assert_eq!(map(json!({"features": []})).unwrap().features, None);
    }

    #[tokio::test]
    async fn request_endpoint_cannot_receive_environment_key() {
        let connection = OcrConnection {
            api_base: Some("https://request.example".into()),
            api_base_source: InputSource::Request,
            ..Default::default()
        };

        let error = validate_environment(&connection, &Default::default(), &|name| {
            (name == AZURE_DI_API_KEY_ENV).then(|| "environment-key".into())
        })
        .await
        .unwrap_err();

        assert!(
            error
                .to_string()
                .contains("request-controlled Azure endpoint")
        );
    }

    #[tokio::test]
    async fn request_endpoint_accepts_request_owned_key() {
        let connection = OcrConnection {
            api_key: Some("request-key".into()),
            api_key_source: InputSource::Request,
            api_base: Some("https://request.example".into()),
            api_base_source: InputSource::Request,
            ..Default::default()
        };

        let headers = validate_environment(&connection, &Default::default(), &|_| None)
            .await
            .unwrap();

        assert_eq!(
            headers[0],
            (AZURE_DI_SUBSCRIPTION_HEADER.into(), "request-key".into())
        );
    }
}
