use std::collections::BTreeSet;
use std::sync::Arc;
use std::time::Duration;

use base64::{Engine, engine::general_purpose::STANDARD};
use litellm_auth::{InputSource, Sourced};
use litellm_auth_azure::AzureAuthInputs;
use reqwest::Url;
use serde::{Deserialize, Deserializer, Serialize};
use serde_json::{Map, Value};
use serde_with::serde_as;
use tokio::time::Instant;

use crate::call_arguments::CallArguments;
use crate::constants::{
    AZURE_DI_API_VERSION, AZURE_DI_DEFAULT_DPI, AZURE_DI_DEFAULT_HEIGHT, AZURE_DI_DEFAULT_WIDTH,
    AZURE_DI_SUBSCRIPTION_HEADER, OCR_POLL_RETRY_SECS,
};
use crate::llms::base_llm::ocr::transformation::{
    BaseOcrConfig, OcrResponseContext, decode_and_normalize_response,
};
use crate::ocr::OcrClient;
use crate::ocr::client::read_json_response;
use crate::ocr::document::InlineDocument;
use crate::ocr::hooks::OcrHooks;
use crate::ocr::json::DecodedOcrResponse;
use crate::ocr::prepare::credential_env;
use crate::ocr::types::{
    LiteLLMOcrResponse, OcrConnection, OcrCredentialInputs, OcrDocument, OcrPage,
    OcrPageDimensions, OcrResponseFormat, OcrUsageInfo, PreparedOcrRequest, ResolvedOcrCredentials,
};
use crate::serde_compat::{FiniteF64, LaxI64};
use crate::url_utils::ApiUrl;

const AZURE_DI_API_KEY_ENV: &str = "AZURE_DOCUMENT_INTELLIGENCE_API_KEY";
const AZURE_DI_ENDPOINT_ENV: &str = "AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT";

#[derive(Clone, Debug, PartialEq, Serialize)]
pub(crate) struct DocumentIntelligenceParams {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pages: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub features: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(untagged)]
pub(crate) enum DocumentIntelligenceRequest {
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

#[serde_as]
#[derive(Clone, Debug, Deserialize)]
struct AzureDocumentIntelligencePage {
    #[serde(rename = "pageNumber")]
    #[serde_as(deserialize_as = "Option<LaxI64>")]
    pub page_number: Option<i64>,
    #[serde_as(deserialize_as = "Option<FiniteF64>")]
    pub width: Option<f64>,
    #[serde_as(deserialize_as = "Option<FiniteF64>")]
    pub height: Option<f64>,
    pub unit: Option<String>,
    #[serde(default)]
    pub lines: Vec<AzureDocumentIntelligenceLine>,
}

#[derive(Clone, Debug, Deserialize)]
struct AzureDocumentIntelligenceLine {
    pub content: Option<String>,
}

#[derive(Clone, Debug)]
pub(crate) struct AzureDocumentIntelligenceOcrConfig;

impl BaseOcrConfig for AzureDocumentIntelligenceOcrConfig {
    type OcrParams = DocumentIntelligenceParams;
    type ProviderRequest = DocumentIntelligenceRequest;
    type Environment = Vec<(String, String)>;

    fn get_supported_ocr_params(&self, _model: &str) -> &'static [&'static str] {
        &["pages", "features", "req_format"]
    }

    fn get_api_key_env_var(&self) -> Option<&'static str> {
        Some(AZURE_DI_API_KEY_ENV)
    }

    fn resolve_connection_params(&self, inputs: OcrCredentialInputs) -> ResolvedOcrCredentials {
        ResolvedOcrCredentials {
            api_key: inputs.api_key.and_then(|key| {
                inputs
                    .dynamic_api_key
                    .filter(|value| !value.value().is_empty())
                    .or(Some(key))
            }),
            api_base: inputs.api_base.and_then(|base| {
                inputs
                    .dynamic_api_base
                    .filter(|value| !value.value().is_empty())
                    .or(Some(base))
            }),
        }
    }

    fn map_ocr_params(
        &self,
        non_default_params: &CallArguments,
        _model: &str,
    ) -> Result<DocumentIntelligenceParams, crate::ocr::Error> {
        Ok(DocumentIntelligenceParams {
            pages: normalize_pages_param(non_default_params.get("pages"))?,
            features: normalize_features_param(non_default_params.get("features"))?,
        })
    }

    async fn validate_environment(
        &self,
        request: &PreparedOcrRequest,
        _client: &OcrClient,
    ) -> Result<Self::Environment, crate::ocr::Error> {
        let config = AzureAuthInputs {
            azure_ad_token_provider: request.azure_ad_token_provider.clone(),
            ..AzureAuthInputs::from_sourced_optional_params(
                &request.optional_params,
                &request.input_sources,
            )?
        };
        self.resolve_headers(&request.connection, &config, &credential_env)
            .await
    }

    fn get_complete_url(
        &self,
        request: &PreparedOcrRequest,
        optional_params: &Self::OcrParams,
        _environment: &Self::Environment,
    ) -> Result<String, crate::ocr::Error> {
        let endpoint = nonblank(request.connection.api_base.clone())
            .or_else(|| nonblank(credential_env(AZURE_DI_ENDPOINT_ENV)))
            .ok_or_else(|| crate::ocr::Error::Auth(litellm_auth::Error::ProviderAuthentication("Missing Azure Document Intelligence API Base - Set AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT or pass api_base".into())))?;
        self.build_ocr_url(&endpoint, &request.model, optional_params)
    }

    fn transform_ocr_request(
        &self,
        _model: &str,
        document: OcrDocument,
        _optional_params: &DocumentIntelligenceParams,
        _headers: &[(String, String)],
    ) -> Result<DocumentIntelligenceRequest, crate::ocr::Error> {
        build_request(document)
    }

    fn transform_ocr_response(
        &self,
        model: &str,
        raw_response: &[u8],
        request_format: OcrResponseFormat,
    ) -> Result<LiteLLMOcrResponse, crate::ocr::Error> {
        decode_and_normalize_response(
            model,
            raw_response,
            request_format,
            transform_completed_response,
        )
    }

    async fn async_transform_ocr_response(
        &self,
        model: &str,
        raw_response: reqwest::Response,
        context: OcrResponseContext<'_>,
    ) -> Result<LiteLLMOcrResponse, crate::ocr::Error> {
        let decoded = read_operation_response(
            context.client.polling_http(),
            raw_response,
            context.url,
            context.headers,
            context.connection,
            context.request_format == OcrResponseFormat::Native,
            context.hooks,
        )
        .await?;
        Ok(LiteLLMOcrResponse {
            provider_native_response: decoded.native,
            ..transform_completed_response(model, decoded.data)?
        })
    }
}

fn normalize_pages_param(pages: Option<&Value>) -> Result<Option<String>, crate::ocr::Error> {
    let normalized = match pages {
        None | Some(Value::Null) => return Ok(None),
        Some(Value::Array(pages)) if pages.is_empty() => return Ok(None),
        Some(Value::Array(pages)) if pages.iter().all(Value::is_number) => pages
            .iter()
            .map(|page| {
                let page = page
                    .as_i64()
                    .ok_or_else(|| crate::ocr::Error::Pages("page index is out of range".into()))?;
                if page < 0 {
                    return Err(crate::ocr::Error::Pages("negative page index".into()));
                }
                page.checked_add(1)
                    .ok_or_else(|| crate::ocr::Error::Pages("page index is out of range".into()))
            })
            .collect::<Result<BTreeSet<_>, _>>()?
            .into_iter()
            .map(|page| page.to_string())
            .collect::<Vec<_>>()
            .join(","),
        Some(Value::Array(tokens)) => tokens
            .iter()
            .map(|token| {
                token.as_str().map(str::trim).ok_or_else(|| {
                    crate::ocr::Error::Pages("expected only integers or only strings".into())
                })
            })
            .collect::<Result<Vec<_>, _>>()?
            .join(","),
        Some(Value::String(range)) => range
            .split(',')
            .map(str::trim)
            .collect::<Vec<_>>()
            .join(","),
        Some(_) => {
            return Err(crate::ocr::Error::Pages(
                "expected an array of integers or strings, or a native page range".into(),
            ));
        }
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

fn normalize_features_param(features: Option<&Value>) -> Result<Option<String>, crate::ocr::Error> {
    let tokens = match features {
        None | Some(Value::Null) => return Ok(None),
        Some(Value::Array(names)) => names
            .iter()
            .map(|name| name.as_str().ok_or(crate::ocr::Error::Features))
            .collect::<Result<Vec<_>, _>>()?,
        Some(Value::String(names)) => names.split(',').collect(),
        Some(_) => return Err(crate::ocr::Error::Features),
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

fn build_request(document: OcrDocument) -> Result<DocumentIntelligenceRequest, crate::ocr::Error> {
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

fn transform_completed_response(
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
        .map(transform_azure_page)
        .collect::<Result<Vec<_>, _>>()?;
    let pages_processed =
        i64::try_from(pages.len()).map_err(|_| crate::ocr::Error::NumericRange("pages"))?;
    Ok(LiteLLMOcrResponse {
        content: result.content,
        tables: result.tables,
        key_value_pairs: result.key_value_pairs,
        usage_info: Some(OcrUsageInfo {
            pages_processed: Some(pages_processed),
            ..Default::default()
        }),
        ..LiteLLMOcrResponse::new(model, pages)
    })
}

fn transform_azure_page(page: AzureDocumentIntelligencePage) -> Result<OcrPage, crate::ocr::Error> {
    let index = page
        .page_number
        .unwrap_or(1)
        .checked_sub(1)
        .ok_or(crate::ocr::Error::NumericRange("page.pageNumber"))?;
    let dimensions = convert_dimensions(
        page.width.unwrap_or(AZURE_DI_DEFAULT_WIDTH),
        page.height.unwrap_or(AZURE_DI_DEFAULT_HEIGHT),
        page.unit.as_deref().unwrap_or("inch"),
    )?;
    let markdown = page
        .lines
        .iter()
        .map(|line| line.content.as_deref().unwrap_or_default())
        .collect::<Vec<_>>()
        .join("\n");
    Ok(OcrPage {
        index,
        markdown,
        dimensions: Some(dimensions),
        ..Default::default()
    })
}

fn convert_dimensions(
    width: f64,
    height: f64,
    unit: &str,
) -> Result<OcrPageDimensions, crate::ocr::Error> {
    let scale = if unit == "inch" {
        AZURE_DI_DEFAULT_DPI as f64
    } else {
        1.0
    };
    Ok(OcrPageDimensions {
        width: Some(pixel_dimension(width, scale, "page.width")?),
        height: Some(pixel_dimension(height, scale, "page.height")?),
        dpi: Some(AZURE_DI_DEFAULT_DPI),
    })
}

fn pixel_dimension(value: f64, scale: f64, field: &'static str) -> Result<i64, crate::ocr::Error> {
    let value = value * scale;
    if !value.is_finite() || value < i64::MIN as f64 || value >= -(i64::MIN as f64) {
        return Err(crate::ocr::Error::NumericRange(field));
    }
    Ok(value.trunc() as i64)
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
        return crate::ocr::json::decode_response(&bytes, native);
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
    poll_operation(http_client, operation, headers, connection, native, hooks).await
}

async fn poll_operation(
    http_client: &reqwest::Client,
    url: Url,
    headers: &[(String, String)],
    connection: &OcrConnection,
    native: bool,
    hooks: &Arc<dyn OcrHooks>,
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
            Some(OperationStatus::Succeeded) => {
                crate::ocr::handler::post_call(hooks, decoded.text.as_bytes()).await?;
                return Ok(decoded);
            }
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

impl AzureDocumentIntelligenceOcrConfig {
    fn build_ocr_url(
        &self,
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

    async fn resolve_headers(
        &self,
        connection: &OcrConnection,
        config: &AzureAuthInputs,
        env_lookup: &(dyn Fn(&str) -> Option<String> + Sync),
    ) -> Result<Vec<(String, String)>, crate::ocr::Error> {
        if crate::http_utils::has_header(&connection.extra_headers, "authorization")
            || crate::http_utils::has_header(
                &connection.extra_headers,
                AZURE_DI_SUBSCRIPTION_HEADER,
            )
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
                nonblank(self.get_api_key_env_var().and_then(env_lookup))
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
    use rstest::rstest;
    use serde_json::{Value, json};

    use super::*;

    fn map(value: Value) -> Result<DocumentIntelligenceParams, crate::ocr::Error> {
        let arguments = serde_json::from_value(value).unwrap();
        AzureDocumentIntelligenceOcrConfig.map_ocr_params(&arguments, "model")
    }

    #[test]
    fn empty_options_do_not_create_query_fields() {
        let overrides =
            serde_json::from_value(json!({"pages":[], "features":null, "req_format":"native"}))
                .unwrap();
        let mapped = AzureDocumentIntelligenceOcrConfig
            .map_ocr_params(&overrides, "model")
            .unwrap();
        assert_eq!(serde_json::to_value(mapped).unwrap(), json!({}));
    }

    #[test]
    fn input_params_retain_unknown_fields() {
        let arguments = serde_json::from_value(json!({
            "pages": [0],
            "future_ocr_option": true,
            "extra_body": {"provider_option": "value"}
        }))
        .unwrap();
        let mapped = AzureDocumentIntelligenceOcrConfig
            .map_ocr_params(&arguments, "model")
            .unwrap();
        assert_eq!(mapped.pages.as_deref(), Some("1"));
        assert_eq!(mapped.features, None);
        assert_eq!(arguments["pages"], json!([0]));
        assert_eq!(arguments["future_ocr_option"], true);
        assert_eq!(arguments["extra_body"], json!({"provider_option": "value"}));
    }

    #[test]
    fn options_normalize_query_fields_without_consuming_extensions() {
        let arguments = serde_json::from_value(json!({
            "pages":"4", "features":"languages", "extension":true
        }))
        .unwrap();
        let mapped = AzureDocumentIntelligenceOcrConfig
            .map_ocr_params(&arguments, "model")
            .unwrap();
        assert_eq!(
            serde_json::to_value(mapped).unwrap(),
            json!({
                "pages":"4", "features":"languages"
            })
        );
        assert_eq!(arguments["extension"], true);
    }

    #[test]
    fn response_numbers_follow_python_validation_before_dimension_conversion() {
        let response = AzureDocumentIntelligenceOcrConfig.transform_ocr_response(
            "model",
            br#"{"status":"succeeded","analyzeResult":{"pages":[{"pageNumber":2.0,"width":" 8.5 ","height":true}]}}"#,
            OcrResponseFormat::Litellm,
        ).unwrap();
        assert_eq!(response.pages[0].index, 1);
        let dimensions = response.pages[0].dimensions.as_ref().unwrap();
        assert_eq!(dimensions.width, Some(816));
        assert_eq!(dimensions.height, Some(96));
    }

    #[test]
    fn pixel_dimension_rejects_out_of_range_value() {
        assert!(pixel_dimension(9_223_372_036_854_775_808.0, 1.0, "width").is_err());
    }

    #[rstest]
    #[case(json!([0, 1, 2]), Some("1,2,3"))]
    #[case(json!([2, 0, 0, 1]), Some("1,2,3"))]
    #[case(json!([]), None)]
    #[case(Value::Null, None)]
    #[case(json!([i64::MAX - 1]), Some("9223372036854775807"))]
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
    #[case(json!(["1", 2]))]
    #[case(json!([1.0]))]
    #[case(json!([i64::MAX]))]
    #[case(json!([u64::MAX]))]
    #[case(json!([null]))]
    #[case(json!([[1]]))]
    #[case(json!(5))]
    fn page_mapping_rejects_invalid_shapes_and_overflow(#[case] input: Value) {
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

        let error = AzureDocumentIntelligenceOcrConfig
            .resolve_headers(&connection, &Default::default(), &|name| {
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

        let headers = AzureDocumentIntelligenceOcrConfig
            .resolve_headers(&connection, &Default::default(), &|_| None)
            .await
            .unwrap();

        assert_eq!(
            headers[0],
            (AZURE_DI_SUBSCRIPTION_HEADER.into(), "request-key".into())
        );
    }

    use std::sync::{Arc, Mutex};

    use crate::ocr::test_support::{MockResponse, mock_server, perform_ocr, wire_request};

    fn query_value(url: &str, key: &str) -> Option<String> {
        url::Url::parse(url)
            .unwrap()
            .query_pairs()
            .find_map(|(name, value)| (name == key).then(|| value.into_owned()))
    }

    #[tokio::test]
    async fn facade_maps_pages_features_and_url_document() {
        let (base, seen, server) = mock_server(vec![MockResponse::json(json!({
            "status":"succeeded",
            "analyzeResult":{"pages":[]}
        }))])
        .await;
        let mut request = wire_request(
            "azure_ai/doc-intelligence/prebuilt-read",
            &base,
            json!({"pages":[2,0,0,1],"features":["keyValuePairs","languages"], "future_option": {"nested":null}, "extra_body":{"provider_option":false}}),
        );
        request.document = serde_json::from_value::<OcrDocument>(json!({
            "type":"document_url",
            "document_url":"https://example.com/document.pdf"
        }))
        .unwrap()
        .into();

        perform_ocr(request).await.unwrap();
        server.await.unwrap();
        let request = &seen.lock().unwrap()[0];
        let target = request.split_whitespace().nth(1).unwrap();
        let url = format!("{base}{target}");
        assert_eq!(query_value(&url, "pages").as_deref(), Some("1,2,3"));
        assert_eq!(
            query_value(&url, "features").as_deref(),
            Some("keyValuePairs,languages")
        );
        let body: Value = serde_json::from_str(request.split_once("\r\n\r\n").unwrap().1).unwrap();
        assert_eq!(
            body,
            json!({"urlSource":"https://example.com/document.pdf", "future_option":{"nested":null}, "provider_option":false})
        );
    }

    #[tokio::test]
    async fn rejects_invalid_pages_features_and_format() {
        for options in [
            json!({"pages":[true]}),
            json!({"pages":[1,"2"]}),
            json!({"pages":[-1]}),
            json!({"pages":"1&&features=bad"}),
            json!({"features":"languages&pages=1"}),
            json!({"req_format":"azure"}),
        ] {
            let request = wire_request(
                "azure_ai/doc-intelligence/prebuilt-read",
                "http://127.0.0.1:1",
                options.clone(),
            );
            let rejected = perform_ocr(request).await.is_err();
            assert!(rejected, "accepted {options}");
        }
    }

    #[tokio::test]
    async fn inline_document_decodes_to_base64_source() {
        let (base, seen, server) = mock_server(vec![MockResponse::json(json!({
            "status":"succeeded"
        }))])
        .await;
        let request = wire_request("azure_ai/doc-intelligence/prebuilt-read", &base, json!({}));

        perform_ocr(request).await.unwrap();
        server.await.unwrap();
        let request = &seen.lock().unwrap()[0];
        let body: Value = serde_json::from_str(request.split_once("\r\n\r\n").unwrap().1).unwrap();
        assert_eq!(body, json!({"base64Source":"YWJj"}));
    }

    #[tokio::test]
    async fn immediate_response_normalizes_pages_and_preserves_native() {
        let operation = json!({
            "status":"succeeded",
            "operationExtension":42,
            "analyzeResult":{
                "content":"A\n\nB",
                "tables":[{"cells":[]}],
                "keyValuePairs":[{"key":{"content":"A"}}],
                "pages":[{
                    "pageNumber":"2",
                    "width":"8.5",
                    "height":11,
                    "unit":"inch",
                    "lines":[{"content":"A"},{"content":null},{"content":"B"}]
                }]
            }
        });
        let (base, _, server) = mock_server(vec![MockResponse::json(operation.clone())]).await;
        let result = perform_ocr(wire_request(
            "azure_ai/doc-intelligence/prebuilt-read",
            &base,
            json!({"req_format":"native"}),
        ))
        .await
        .unwrap();
        server.await.unwrap();

        assert_eq!(result.pages[0].index, 1);
        assert_eq!(result.pages[0].markdown, "A\n\nB");
        assert_eq!(
            serde_json::to_value(&result.pages[0].dimensions).unwrap(),
            json!({"width":816,"height":1056,"dpi":96})
        );
        assert_eq!(result.usage_info.as_ref().unwrap().pages_processed, Some(1));
        let serialized = result.clone().into_json();
        assert_eq!(serialized["content"], "A\n\nB");
        assert_eq!(serialized["tables"], json!([{"cells":[]}]));
        assert_eq!(
            serialized["keyValuePairs"],
            json!([{"key":{"content":"A"}}])
        );
        assert!(serialized.get("key_value_pairs").is_none());
        assert_eq!(
            result.provider_native_response.as_ref(),
            operation.as_object()
        );
    }

    #[tokio::test]
    async fn accepted_response_polls_to_success_with_only_credentials() {
        let operation = json!({"status":"succeeded","analyzeResult":{"pages":[]}});
        let (base, seen, server) = mock_server(vec![
            MockResponse {
                status: 202,
                headers: vec![("Operation-Location", "{base}/operation".into())],
                body: json!({}),
            },
            MockResponse {
                status: 200,
                headers: vec![("Retry-After", "0".into())],
                body: json!({"status":"running"}),
            },
            MockResponse::json(operation.clone()),
        ])
        .await;
        let mut request = wire_request(
            "azure_ai/doc-intelligence/prebuilt-read",
            &base,
            json!({"req_format":"native"}),
        );
        request
            .transport
            .extra_headers
            .push(("X-Trace".into(), "initial-only".into()));

        let result = perform_ocr(request).await.unwrap();
        server.await.unwrap();
        assert_eq!(
            result.provider_native_response.as_ref(),
            operation.as_object()
        );
        let requests = seen.lock().unwrap();
        assert_eq!(requests.len(), 3);
        assert!(requests[0].to_ascii_lowercase().contains("x-trace:"));
        for poll in &requests[1..] {
            assert!(!poll.to_ascii_lowercase().contains("x-trace:"));
            assert!(
                poll.to_ascii_lowercase()
                    .contains("ocp-apim-subscription-key: test-key")
            );
        }
    }

    struct SubmissionBoundary {
        request_count: Arc<Mutex<Vec<String>>>,
        post_calls: Arc<Mutex<Vec<(usize, Value)>>>,
    }

    impl crate::ocr::hooks::OcrHooks for SubmissionBoundary {
        fn post_call(
            &self,
            request: crate::ocr::hooks::OcrPostCallRequest,
        ) -> crate::ocr::hooks::OcrHookFuture<'_, crate::ocr::hooks::OcrPostCallRequest> {
            Box::pin(async move {
                self.post_calls.lock().unwrap().push((
                    self.request_count.lock().unwrap().len(),
                    request.original_response.clone(),
                ));
                Ok(request)
            })
        }
    }

    #[tokio::test]
    async fn accepted_response_runs_post_call_for_submission_and_completed_poll() {
        let (base, seen, server) = mock_server(vec![
            MockResponse {
                status: 202,
                headers: vec![("Operation-Location", "{base}/operation".into())],
                body: json!({"submitted": true}),
            },
            MockResponse::json(json!({"status":"succeeded"})),
        ])
        .await;
        let post_calls = Arc::new(Mutex::new(Vec::new()));
        let request = crate::ocr::LiteLLMOcrRequest {
            hooks: Arc::new(SubmissionBoundary {
                request_count: seen.clone(),
                post_calls: post_calls.clone(),
            }),
            ..wire_request("azure_ai/doc-intelligence/prebuilt-read", &base, json!({}))
        };

        perform_ocr(request).await.unwrap();
        server.await.unwrap();
        assert_eq!(seen.lock().unwrap().len(), 2);
        assert_eq!(
            *post_calls.lock().unwrap(),
            [
                (1, json!(r#"{"submitted":true}"#)),
                (2, json!(r#"{"status":"succeeded"}"#)),
            ]
        );
    }

    #[tokio::test]
    async fn polling_forwards_bearer_credentials() {
        let (base, seen, server) = mock_server(vec![
            MockResponse {
                status: 202,
                headers: vec![("Operation-Location", "{base}/operation".into())],
                body: json!({}),
            },
            MockResponse::json(json!({"status":"succeeded"})),
        ])
        .await;
        let mut request = wire_request("azure_ai/doc-intelligence/prebuilt-read", &base, json!({}));
        request.credentials.api_key = None;
        request.transport.extra_headers = vec![("Authorization".into(), "Bearer token".into())];

        perform_ocr(request).await.unwrap();
        server.await.unwrap();
        let requests = seen.lock().unwrap();
        assert!(
            requests[1]
                .to_ascii_lowercase()
                .contains("authorization: bearer token")
        );
    }

    #[tokio::test]
    async fn polling_does_not_follow_redirects() {
        let (base, seen, server) = mock_server(vec![
            MockResponse {
                status: 202,
                headers: vec![("Operation-Location", "{base}/operation".into())],
                body: json!({}),
            },
            MockResponse {
                status: 302,
                headers: vec![("Location", "{base}/redirected".into())],
                body: json!({}),
            },
            MockResponse::json(json!({"status":"succeeded"})),
        ])
        .await;

        let error = perform_ocr(wire_request(
            "azure_ai/doc-intelligence/prebuilt-read",
            &base,
            json!({}),
        ))
        .await
        .unwrap_err();

        assert!(error.to_string().contains("status 302"), "{error}");
        assert_eq!(seen.lock().unwrap().len(), 2);
        server.abort();
    }

    #[tokio::test]
    async fn polling_rejects_terminal_failure() {
        let (base, _, server) = mock_server(vec![
            MockResponse {
                status: 202,
                headers: vec![("Operation-Location", "{base}/operation".into())],
                body: json!({}),
            },
            MockResponse::json(json!({"status":"failed"})),
        ])
        .await;

        let error = perform_ocr(wire_request(
            "azure_ai/doc-intelligence/prebuilt-read",
            &base,
            json!({}),
        ))
        .await
        .unwrap_err();
        server.await.unwrap();
        assert!(error.to_string().contains("status failed"));
    }

    #[tokio::test]
    async fn malformed_provider_pages_report_response_paths() {
        for (analysis, path) in [
            (json!({"pages":null}), "pages"),
            (json!({"pages":[null]}), "pages[0]"),
            (json!({"pages":[{"lines":null}]}), "lines"),
            (json!({"pages":[{"width":"bad"}]}), "width"),
        ] {
            let (base, _, server) = mock_server(vec![MockResponse::json(json!({
                "status":"succeeded",
                "analyzeResult":analysis
            }))])
            .await;
            let error = perform_ocr(wire_request(
                "azure_ai/doc-intelligence/prebuilt-read",
                &base,
                json!({}),
            ))
            .await
            .unwrap_err();
            server.await.unwrap();
            assert!(error.to_string().contains(path), "{error}");
        }
    }

    #[tokio::test]
    async fn rejects_missing_invalid_and_cross_origin_operation_locations() {
        for headers in [
            Vec::new(),
            vec![("Operation-Location", "/relative".into())],
            vec![("Operation-Location", "http://example.com/operation".into())],
            vec![(
                "Operation-Location",
                "http://user:password@127.0.0.1/operation".into(),
            )],
        ] {
            let (base, _, server) = mock_server(vec![MockResponse {
                status: 202,
                headers,
                body: json!({}),
            }])
            .await;
            let error = perform_ocr(wire_request(
                "azure_ai/doc-intelligence/prebuilt-read",
                &base,
                json!({}),
            ))
            .await
            .unwrap_err();
            server.await.unwrap();
            assert!(error.to_string().contains("operation-location"));
        }
    }

    #[tokio::test]
    async fn polling_deadline_bounds_retry_delay() {
        let (base, _, server) = mock_server(vec![
            MockResponse {
                status: 202,
                headers: vec![("Operation-Location", "{base}/operation".into())],
                body: json!({}),
            },
            MockResponse {
                status: 200,
                headers: vec![("Retry-After", "9999".into())],
                body: json!({"status":"notStarted"}),
            },
        ])
        .await;
        let mut request = wire_request("azure_ai/doc-intelligence/prebuilt-read", &base, json!({}));
        request.transport.poll_timeout = std::time::Duration::from_millis(100);

        let error = tokio::time::timeout(std::time::Duration::from_secs(1), perform_ocr(request))
            .await
            .unwrap()
            .unwrap_err();
        server.await.unwrap();
        assert!(error.to_string().contains("timed out"));
    }

    #[tokio::test]
    async fn model_id_is_encoded_and_dot_segments_are_rejected() {
        let (base, seen, server) = mock_server(vec![MockResponse::json(json!({
            "status":"succeeded"
        }))])
        .await;
        perform_ocr(wire_request(
            "azure_ai/doc-intelligence/a ?#é",
            &base,
            json!({}),
        ))
        .await
        .unwrap();
        server.await.unwrap();
        assert!(seen.lock().unwrap()[0].contains("a%20%3F%23%C3%A9:analyze"));

        for model in [
            "azure_ai/doc-intelligence/.",
            "azure_ai/doc-intelligence/..",
        ] {
            let error = perform_ocr(wire_request(model, "http://127.0.0.1:1", json!({})))
                .await
                .unwrap_err();
            assert!(error.to_string().contains("dot segment"));
        }
    }

    #[tokio::test]
    async fn pre_call_guardrail_receives_caller_pages_before_mapping() {
        use std::sync::Arc;

        use crate::ocr::hooks::{OcrHookFuture, OcrHooks, OcrPreCallRequest};

        struct RewritePages;
        impl OcrHooks for RewritePages {
            fn intercepts_requests(&self) -> bool {
                true
            }

            fn pre_call(&self, request: OcrPreCallRequest) -> OcrHookFuture<'_, OcrPreCallRequest> {
                Box::pin(async move {
                    assert_eq!(request.optional_params["pages"], json!([0, 2]));
                    Ok(OcrPreCallRequest {
                        optional_params: json!({"pages": [1]}),
                        ..request
                    })
                })
            }
        }
        let (base, seen, server) =
            mock_server(vec![MockResponse::json(json!({"status": "succeeded"}))]).await;
        let request = wire_request(
            "azure_ai/doc-intelligence/prebuilt-read",
            &base,
            json!({"pages": [0, 2]}),
        )
        .with_host_hooks(Arc::new(RewritePages), None);
        perform_ocr(request).await.unwrap();
        server.await.unwrap();
        let requests = seen.lock().unwrap();
        let target = requests[0].split_whitespace().nth(1).unwrap();
        assert_eq!(
            query_value(&format!("{base}{target}"), "pages").as_deref(),
            Some("2")
        );
        assert_eq!(requests.len(), 1);
    }
}
