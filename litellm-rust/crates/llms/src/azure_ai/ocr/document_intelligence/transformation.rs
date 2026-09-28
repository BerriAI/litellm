use std::{collections::BTreeSet, time::Duration};

use base64::{Engine, engine::general_purpose::STANDARD};
use litellm_auth::{InputSource, Sourced};
use litellm_auth_azure::{AzureAuthInputs, SECRET_NAMES as AZURE_AUTH_SECRET_NAMES};
use litellm_core_utils::{
    call_arguments::CallArguments,
    serde_compat::{FiniteF64, LaxI64},
    url_utils::ApiUrl,
};
use reqwest::Url;
use serde::{Deserialize, Deserializer, Serialize};
use serde_json::{Map, Value};
use serde_with::serde_as;
use tokio::time::Instant;

use crate::base_llm::ocr::{
    document::InlineDocument,
    error::Error,
    handler::{CallHooks, OcrClient, read_json_response},
    settings::OcrSettings,
    transformation::{
        BaseOcrConfig, DecodedOcrResponse, LiteLLMOcrResponse, OCR_INLINE_MAX_BYTES,
        OCR_POLL_RETRY_SECS, OcrConnection, OcrCredentialInputs, OcrDocument, OcrPage,
        OcrPageDimensions, OcrResponseContext, OcrResponseFormat, OcrUsageInfo, PreparedOcrRequest,
        ResolvedOcrCredentials, decode_and_normalize_response, decode_response,
    },
};

const AZURE_DI_SUBSCRIPTION_HEADER: &str = "Ocp-Apim-Subscription-Key";
const AZURE_DI_DEFAULT_WIDTH: f64 = 8.5;
const AZURE_DI_DEFAULT_HEIGHT: f64 = 11.0;

const AZURE_DI_API_KEY_ENV: &str = "AZURE_DOCUMENT_INTELLIGENCE_API_KEY";
const AZURE_DI_ENDPOINT_ENV: &str = "AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT";

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct DocumentIntelligenceParams {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pages: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub features: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(untagged)]
pub enum DocumentIntelligenceRequest {
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
pub struct AzureDocumentIntelligenceOperation {
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
pub struct AzureDocumentIntelligenceOcrConfig;

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

    fn secret_names(&self) -> Vec<&'static str> {
        [
            [AZURE_DI_API_KEY_ENV, AZURE_DI_ENDPOINT_ENV].as_slice(),
            AZURE_AUTH_SECRET_NAMES,
        ]
        .into_iter()
        .flatten()
        .copied()
        .collect()
    }

    fn resolve_connection_params(&self, inputs: OcrCredentialInputs) -> ResolvedOcrCredentials {
        ResolvedOcrCredentials {
            api_key: inputs.api_key.and_then(|key| {
                inputs
                    .dynamic_api_key
                    .filter(|value| !value.value().expose().is_empty())
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
    ) -> Result<DocumentIntelligenceParams, Error> {
        Ok(DocumentIntelligenceParams {
            pages: normalize_pages_param(non_default_params.get("pages"))?,
            features: normalize_features_param(non_default_params.get("features"))?,
        })
    }

    async fn validate_environment(
        &self,
        request: &PreparedOcrRequest,
        client: &OcrClient,
    ) -> Result<Self::Environment, Error> {
        let config = crate::azure_ai::ocr::common_utils::azure_auth_inputs(request)?;
        self.resolve_headers(
            &client.auth().azure,
            &request.connection,
            &config,
            &|name: &str| request.connection.secret(name),
        )
        .await
    }

    fn get_complete_url(
        &self,
        request: &PreparedOcrRequest,
        optional_params: &Self::OcrParams,
        _environment: &Self::Environment,
    ) -> Result<String, Error> {
        let endpoint = nonblank(request.connection.api_base.clone())
            .or_else(|| nonblank(request.connection.secret(AZURE_DI_ENDPOINT_ENV)))
            .ok_or_else(|| Error::Auth(litellm_auth::Error::ProviderAuthentication("Missing Azure Document Intelligence API Base - Set AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT or pass api_base".into())))?;
        self.build_ocr_url(
            &endpoint,
            &request.model,
            optional_params,
            &request
                .connection
                .settings
                .document_intelligence_api_version,
        )
    }

    fn transform_ocr_request(
        &self,
        _model: &str,
        document: OcrDocument,
        _optional_params: &DocumentIntelligenceParams,
        _headers: &[(String, String)],
    ) -> Result<DocumentIntelligenceRequest, Error> {
        build_request(document)
    }

    fn transform_ocr_response(
        &self,
        model: &str,
        raw_response: &[u8],
        request_format: OcrResponseFormat,
    ) -> Result<LiteLLMOcrResponse, Error> {
        decode_and_normalize_response(model, raw_response, request_format, |model, response| {
            transform_completed_response(
                model,
                response,
                OcrSettings::default().document_intelligence_dpi,
            )
        })
    }

    async fn async_transform_ocr_response(
        &self,
        model: &str,
        raw_response: reqwest::Response,
        context: OcrResponseContext<'_>,
    ) -> Result<LiteLLMOcrResponse, Error> {
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
            ..transform_completed_response(
                model,
                decoded.data,
                context.connection.settings.document_intelligence_dpi,
            )?
        })
    }
}

fn normalize_pages_param(pages: Option<&Value>) -> Result<Option<String>, Error> {
    let normalized = match pages {
        None | Some(Value::Null) => return Ok(None),
        Some(Value::Array(pages)) if pages.is_empty() => return Ok(None),
        Some(Value::Array(pages)) if pages.iter().all(Value::is_number) => pages
            .iter()
            .map(|page| {
                let page = page
                    .as_i64()
                    .ok_or_else(|| Error::Pages("page index is out of range".into()))?;
                if page < 0 {
                    return Err(Error::Pages("negative page index".into()));
                }
                page.checked_add(1)
                    .ok_or_else(|| Error::Pages("page index is out of range".into()))
            })
            .collect::<Result<BTreeSet<_>, _>>()?
            .into_iter()
            .map(|page| page.to_string())
            .collect::<Vec<_>>()
            .join(","),
        Some(Value::Array(tokens)) => tokens
            .iter()
            .map(|token| {
                token
                    .as_str()
                    .map(str::trim)
                    .ok_or_else(|| Error::Pages("expected only integers or only strings".into()))
            })
            .collect::<Result<Vec<_>, _>>()?
            .join(","),
        Some(Value::String(range)) => range
            .split(',')
            .map(str::trim)
            .collect::<Vec<_>>()
            .join(","),
        Some(_) => {
            return Err(Error::Pages(
                "expected an array of integers or strings, or a native page range".into(),
            ));
        }
    };
    if !normalized.split(',').all(valid_page_token) {
        return Err(Error::Pages("invalid native page range".into()));
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

fn normalize_features_param(features: Option<&Value>) -> Result<Option<String>, Error> {
    let tokens = match features {
        None | Some(Value::Null) => return Ok(None),
        Some(Value::Array(names)) => names
            .iter()
            .map(|name| name.as_str().ok_or(Error::Features))
            .collect::<Result<Vec<_>, _>>()?,
        Some(Value::String(names)) => names.split(',').collect(),
        Some(_) => return Err(Error::Features),
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
        return Err(Error::Features);
    }
    Ok(Some(normalized.join(",")))
}

fn build_request(document: OcrDocument) -> Result<DocumentIntelligenceRequest, Error> {
    let source = document.source();
    if source.is_empty() {
        return Err(Error::MissingDocumentUrl);
    }
    Ok(if let Some(document) = InlineDocument::parse(source)? {
        DocumentIntelligenceRequest::Base64Source {
            base64_source: STANDARD.encode(document.decode(OCR_INLINE_MAX_BYTES)?),
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
    dpi: i64,
) -> Result<LiteLLMOcrResponse, Error> {
    if response.status != Some(OperationStatus::Succeeded) {
        return Err(Error::OperationStatus(
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
        .map(|page| transform_azure_page(page, dpi))
        .collect::<Result<Vec<_>, _>>()?;
    let pages_processed = i64::try_from(pages.len()).map_err(|_| Error::NumericRange("pages"))?;
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

fn transform_azure_page(page: AzureDocumentIntelligencePage, dpi: i64) -> Result<OcrPage, Error> {
    let index = page
        .page_number
        .unwrap_or(1)
        .checked_sub(1)
        .ok_or(Error::NumericRange("page.pageNumber"))?;
    let dimensions = convert_dimensions(
        page.width.unwrap_or(AZURE_DI_DEFAULT_WIDTH),
        page.height.unwrap_or(AZURE_DI_DEFAULT_HEIGHT),
        page.unit.as_deref().unwrap_or("inch"),
        dpi,
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
    dpi: i64,
) -> Result<OcrPageDimensions, Error> {
    let scale = if unit == "inch" { dpi as f64 } else { 1.0 };
    Ok(OcrPageDimensions {
        width: Some(pixel_dimension(width, scale, "page.width")?),
        height: Some(pixel_dimension(height, scale, "page.height")?),
        dpi: Some(dpi),
    })
}

fn pixel_dimension(value: f64, scale: f64, field: &'static str) -> Result<i64, Error> {
    let value = value * scale;
    if !value.is_finite() || value < i64::MIN as f64 || value >= -(i64::MIN as f64) {
        return Err(Error::NumericRange(field));
    }
    Ok(value.trunc() as i64)
}

async fn read_operation_response(
    http_client: &litellm_http::Client,
    response: reqwest::Response,
    original_url: &str,
    headers: &[(String, String)],
    connection: &OcrConnection,
    native: bool,
    hooks: &dyn CallHooks<Error>,
) -> Result<DecodedOcrResponse<AzureDocumentIntelligenceOperation>, Error> {
    if response.status() != reqwest::StatusCode::ACCEPTED {
        let bytes = crate::base_llm::ocr::handler::read_response_bytes(
            response,
            connection.max_response_bytes,
        )
        .await?;
        hooks.response_received(&bytes).await?;
        return decode_response(&bytes, native);
    }
    let location = response
        .headers()
        .get("operation-location")
        .and_then(|value| value.to_str().ok())
        .ok_or(Error::PollLocation)?
        .to_string();
    let original = Url::parse(original_url).map_err(|_| Error::PollOrigin)?;
    let operation = Url::parse(&location).map_err(|_| Error::PollOrigin)?;
    if original.origin() != operation.origin()
        || !operation.username().is_empty()
        || operation.password().is_some()
    {
        return Err(Error::PollOrigin);
    }
    let bytes =
        crate::base_llm::ocr::handler::read_response_bytes(response, connection.max_response_bytes)
            .await?;
    hooks.response_received(&bytes).await?;
    poll_operation(http_client, operation, headers, connection, native, hooks).await
}

async fn poll_operation(
    http_client: &litellm_http::Client,
    url: Url,
    headers: &[(String, String)],
    connection: &OcrConnection,
    native: bool,
    hooks: &dyn CallHooks<Error>,
) -> Result<DecodedOcrResponse<AzureDocumentIntelligenceOperation>, Error> {
    let deadline = Instant::now()
        .checked_add(connection.settings.poll_timeout)
        .ok_or(Error::PollTimeout)?;

    loop {
        let remaining = deadline
            .checked_duration_since(Instant::now())
            .filter(|remaining| !remaining.is_zero())
            .ok_or(Error::PollTimeout)?;
        let builder = http_client
            .get(url.clone())
            .timeout(remaining.min(connection.timeout));
        let builder = litellm_http::request::with_headers(
            builder,
            headers,
            litellm_http::request::HeaderPolicy::Only(&[
                AZURE_DI_SUBSCRIPTION_HEADER,
                "authorization",
            ]),
        );
        let response =
            tokio::time::timeout_at(deadline, litellm_http::request::http_request(builder))
                .await
                .map_err(|_| Error::PollTimeout)?
                .map_err(litellm_http::transport::Error::from)?;
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
        .map_err(|_| Error::PollTimeout)??;
        match &decoded.data.status {
            Some(OperationStatus::Succeeded) => {
                hooks.response_received(decoded.text.as_bytes()).await?;
                return Ok(decoded);
            }
            Some(OperationStatus::Running | OperationStatus::NotStarted) => {
                tokio::time::timeout_at(deadline, tokio::time::sleep(Duration::from_secs(retry)))
                    .await
                    .map_err(|_| Error::PollTimeout)?;
            }
            status => {
                return Err(Error::OperationStatus(
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
    pub fn analyze_path(model: &str) -> Result<[String; 3], Error> {
        Ok([
            "documentintelligence".into(),
            "documentModels".into(),
            format!("{}:analyze", model_id(model)?),
        ])
    }

    fn build_ocr_url(
        &self,
        endpoint: &str,
        model: &str,
        params: &DocumentIntelligenceParams,
        api_version: &str,
    ) -> Result<String, Error> {
        let path = Self::analyze_path(model)?;
        ApiUrl::parse(endpoint)
            .and_then(|url| url.complete_path(&path.each_ref().map(String::as_str)))
            .map(|url| {
                url.append_query_pairs(
                    [("api-version", api_version)]
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
            .map_err(|_| Error::RequestField {
                path: "api_base".into(),
            })
    }

    async fn resolve_headers(
        &self,
        auth: &litellm_auth_azure::AzureAuthService,
        connection: &OcrConnection,
        config: &AzureAuthInputs,
        env_lookup: &(dyn Fn(&str) -> Option<String> + Sync),
    ) -> Result<Vec<(String, String)>, Error> {
        if litellm_http::request::has_header(&connection.extra_headers, "authorization")
            || litellm_http::request::has_header(
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
        let key = nonblank(
            connection
                .api_key
                .as_ref()
                .map(|key| key.expose().to_string()),
        )
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
        let token = super::super::common_utils::resolve_entra(auth, config, env_lookup)
            .await?
            .ok_or(Error::MissingAzureDocumentIntelligenceCredentials)?;
        super::super::common_utils::validate_destination(connection, token.source())?;
        Ok(
            std::iter::once(("Authorization".into(), format!("Bearer {}", token.value())))
                .chain(connection.extra_headers.clone())
                .collect(),
        )
    }
}

fn model_id(model: &str) -> Result<&str, Error> {
    let model = model.rsplit('/').next().unwrap_or(model);
    if matches!(model, "." | "..") {
        return Err(Error::DotModel);
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

    fn map(value: Value) -> Result<DocumentIntelligenceParams, Error> {
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
            .resolve_headers(
                &Default::default(),
                &connection,
                &Default::default(),
                &|name| (name == AZURE_DI_API_KEY_ENV).then(|| "environment-key".into()),
            )
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
            api_key: Some(litellm_auth::SecretValue::new("request-key")),
            api_key_source: InputSource::Request,
            api_base: Some("https://request.example".into()),
            api_base_source: InputSource::Request,
            ..Default::default()
        };

        let headers = AzureDocumentIntelligenceOcrConfig
            .resolve_headers(
                &Default::default(),
                &connection,
                &Default::default(),
                &|_| None,
            )
            .await
            .unwrap();

        assert_eq!(
            headers[0],
            (AZURE_DI_SUBSCRIPTION_HEADER.into(), "request-key".into())
        );
    }
}
