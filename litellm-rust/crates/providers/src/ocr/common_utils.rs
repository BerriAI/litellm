use std::time::{Duration, Instant};

use base64::engine::general_purpose::STANDARD as BASE64_STANDARD;
use base64::Engine;
use litellm_core::error::CoreError;
use litellm_core::ocr::transformation::OcrProviderConfig;
use litellm_core::{CoreResult, LlmProvider};
use serde_json::{Map, Value};

use crate::azure_ai::ocr::transformation as azure_ai;
use crate::azure_ai::ocr::transformation::{
    AZURE_AI_OCR_CONFIG, AZURE_DOCUMENT_INTELLIGENCE_OCR_CONFIG,
};
use crate::mistral::ocr::transformation as mistral;
use crate::mistral::ocr::transformation::MISTRAL_OCR_CONFIG;
use crate::vertex_ai::ocr::transformation as vertex_ai;
use crate::vertex_ai::ocr::transformation::{VERTEX_AI_DEEPSEEK_OCR_CONFIG, VERTEX_AI_OCR_CONFIG};

use super::http_client;

const ERROR_BODY_MAX_CHARS: usize = 256;
const AZURE_DOCUMENT_INTELLIGENCE_POLL_TIMEOUT_SECS: u64 = 120;

pub(super) fn truncate_error_body(body: &str) -> String {
    if body.chars().count() <= ERROR_BODY_MAX_CHARS {
        return body.to_string();
    }
    let truncated: String = body.chars().take(ERROR_BODY_MAX_CHARS).collect();
    format!("{truncated}... (truncated)")
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum OcrProviderDispatch {
    Mistral,
    AzureAi,
    AzureDocumentIntelligence,
    VertexAiMistral,
    VertexAiDeepSeek,
}

impl OcrProviderDispatch {
    pub(super) fn for_provider_and_model(provider: LlmProvider, model: &str) -> Option<Self> {
        match provider {
            LlmProvider::Mistral => Some(Self::Mistral),
            LlmProvider::AzureAi if is_azure_document_intelligence_model(model) => {
                Some(Self::AzureDocumentIntelligence)
            }
            LlmProvider::AzureAiDocIntelligence => Some(Self::AzureDocumentIntelligence),
            LlmProvider::AzureAi => Some(Self::AzureAi),
            LlmProvider::VertexAi if vertex_ai::is_deepseek_model(model) => {
                Some(Self::VertexAiDeepSeek)
            }
            LlmProvider::VertexAi => Some(Self::VertexAiMistral),
            _ => None,
        }
    }

    pub(super) fn config(self) -> &'static dyn OcrProviderConfig {
        match self {
            Self::Mistral => &MISTRAL_OCR_CONFIG,
            Self::AzureAi => &AZURE_AI_OCR_CONFIG,
            Self::AzureDocumentIntelligence => &AZURE_DOCUMENT_INTELLIGENCE_OCR_CONFIG,
            Self::VertexAiMistral => &VERTEX_AI_OCR_CONFIG,
            Self::VertexAiDeepSeek => &VERTEX_AI_DEEPSEEK_OCR_CONFIG,
        }
    }

    pub(super) fn complete_url(
        self,
        api_base: Option<&str>,
        model: &str,
        optional_params: &Map<String, Value>,
        env_lookup: &dyn Fn(&str) -> Option<String>,
    ) -> CoreResult<String> {
        match self {
            Self::Mistral => Ok(mistral::complete_url(api_base)),
            Self::AzureAi => azure_ai::complete_azure_ai_url(api_base, env_lookup),
            Self::AzureDocumentIntelligence => azure_ai::complete_document_intelligence_url(
                api_base,
                model,
                optional_params,
                env_lookup,
            ),
            Self::VertexAiMistral => {
                vertex_ai::complete_vertex_mistral_url(api_base, model, optional_params, env_lookup)
            }
            Self::VertexAiDeepSeek => {
                vertex_ai::complete_vertex_deepseek_url(api_base, optional_params, env_lookup)
            }
        }
    }

    pub(super) fn resolve_api_key(
        self,
        api_key: Option<&str>,
        env_lookup: &dyn Fn(&str) -> Option<String>,
    ) -> CoreResult<String> {
        match self {
            Self::Mistral => mistral::resolve_api_key(api_key, env_lookup),
            Self::AzureAi => azure_ai::resolve_azure_ai_api_key(api_key, env_lookup),
            Self::AzureDocumentIntelligence => {
                azure_ai::resolve_document_intelligence_api_key(api_key, env_lookup)
            }
            Self::VertexAiMistral | Self::VertexAiDeepSeek => {
                vertex_ai::resolve_vertex_api_key(api_key, env_lookup)
            }
        }
    }

    pub(super) fn needs_data_uri_document(self) -> bool {
        matches!(self, Self::AzureAi | Self::VertexAiMistral)
    }

    pub(super) fn uses_subscription_key(self) -> bool {
        matches!(self, Self::AzureDocumentIntelligence)
    }

    pub(super) fn polls_document_intelligence(self) -> bool {
        matches!(self, Self::AzureDocumentIntelligence)
    }
}

fn is_azure_document_intelligence_model(model: &str) -> bool {
    let model = model.to_ascii_lowercase();
    model.contains("doc-intelligence") || model.contains("documentintelligence")
}

pub(super) fn string_headers(
    extra_headers: Option<Map<String, Value>>,
) -> CoreResult<Vec<(String, String)>> {
    extra_headers
        .unwrap_or_default()
        .into_iter()
        .map(|(key, value)| {
            value
                .as_str()
                .map(|value| (key.clone(), value.to_string()))
                .ok_or_else(|| {
                    CoreError::InvalidRequest(format!(
                        "OCR extra_headers.{key} must be a string, got {}",
                        litellm_core::error::json_type_name(&value)
                    ))
                })
        })
        .collect()
}

pub(super) fn has_authorization_header(headers: &[(String, String)]) -> bool {
    has_header(headers, "authorization")
}

pub(super) fn has_header(headers: &[(String, String)], name: &str) -> bool {
    headers
        .iter()
        .any(|(key, _)| key.eq_ignore_ascii_case(name))
}

fn document_url_field(document: &Value) -> CoreResult<Option<(&str, &str)>> {
    let Some(object) = document.as_object() else {
        return Ok(None);
    };
    let Some(doc_type) = object.get("type").and_then(Value::as_str) else {
        return Ok(None);
    };
    let field = match doc_type {
        "document_url" => "document_url",
        "image_url" => "image_url",
        _ => return Ok(None),
    };
    let Some(url) = object.get(field).and_then(Value::as_str) else {
        return Ok(None);
    };
    Ok(Some((field, url)))
}

fn is_url_requiring_fetch(url: &str) -> bool {
    !url.starts_with("data:") && (url.starts_with("http://") || url.starts_with("https://"))
}

pub(super) async fn convert_document_url_to_data_uri(document: Value) -> CoreResult<Value> {
    let Some((field, url)) = document_url_field(&document)? else {
        return Ok(document);
    };
    if !is_url_requiring_fetch(url) {
        return Ok(document);
    }

    let response = http_client()
        .get(url)
        .send()
        .await
        .map_err(|err| CoreError::Network(err.to_string()))?;
    let status = response.status();
    if !status.is_success() {
        let body = response.text().await.unwrap_or_default();
        return Err(CoreError::Http {
            status: status.as_u16(),
            body: truncate_error_body(&body),
        });
    }
    let content_type = response
        .headers()
        .get(reqwest::header::CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.split(';').next())
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or("application/octet-stream")
        .to_string();
    let bytes = response
        .bytes()
        .await
        .map_err(|err| CoreError::Network(err.to_string()))?;
    let data_uri = format!(
        "data:{content_type};base64,{}",
        BASE64_STANDARD.encode(bytes)
    );

    let mut transformed = document
        .as_object()
        .cloned()
        .ok_or_else(|| CoreError::InvalidRequest("OCR document must be an object".to_string()))?;
    transformed.insert(field.to_string(), Value::String(data_uri));
    Ok(Value::Object(transformed))
}

fn same_origin(left: &str, right: &str) -> bool {
    let Ok(left) = reqwest::Url::parse(left) else {
        return false;
    };
    let Ok(right) = reqwest::Url::parse(right) else {
        return false;
    };
    left.scheme() == right.scheme()
        && left.host_str() == right.host_str()
        && left.port_or_known_default() == right.port_or_known_default()
}

fn retry_after_secs(response: &reqwest::Response) -> u64 {
    response
        .headers()
        .get(reqwest::header::RETRY_AFTER)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or(2)
}

fn operation_status(response_json: &Value) -> CoreResult<&str> {
    let status = response_json
        .get("status")
        .and_then(Value::as_str)
        .ok_or(CoreError::MissingField("status"))?;
    match status {
        "succeeded" => Ok("succeeded"),
        "running" | "notStarted" => Ok("running"),
        "failed" => {
            let message = response_json
                .get("error")
                .and_then(|error| error.get("message"))
                .and_then(Value::as_str)
                .unwrap_or("Unknown error");
            Err(CoreError::InvalidResponse(format!(
                "Azure Document Intelligence analysis failed: {message}"
            )))
        }
        other => Err(CoreError::InvalidResponse(format!(
            "Unknown operation status: {other}"
        ))),
    }
}

pub(super) async fn poll_document_intelligence(
    operation_url: &str,
    original_url: &str,
    headers: &[(String, String)],
    timeout: Option<Duration>,
) -> CoreResult<Value> {
    if !same_origin(operation_url, original_url) {
        return Err(CoreError::InvalidResponse(
            "Azure Document Intelligence: rejected cross-origin polling URL".to_string(),
        ));
    }

    let start = Instant::now();
    let timeout = timeout.unwrap_or(Duration::from_secs(
        AZURE_DOCUMENT_INTELLIGENCE_POLL_TIMEOUT_SECS,
    ));
    loop {
        if start.elapsed() > timeout {
            return Err(CoreError::Network(format!(
                "Azure Document Intelligence operation polling timed out after {} seconds",
                timeout.as_secs()
            )));
        }

        let mut request_builder = http_client().get(operation_url);
        for (key, value) in headers {
            if key.eq_ignore_ascii_case("ocp-apim-subscription-key") {
                request_builder = request_builder.header(key, value);
            }
        }
        let response = request_builder
            .send()
            .await
            .map_err(|err| CoreError::Network(err.to_string()))?;
        let retry_after = retry_after_secs(&response);
        let status = response.status();
        let text = response
            .text()
            .await
            .map_err(|err| CoreError::Network(err.to_string()))?;
        if !status.is_success() {
            return Err(CoreError::Http {
                status: status.as_u16(),
                body: truncate_error_body(&text),
            });
        }
        let response_json: Value = serde_json::from_str(&text).map_err(|err| {
            CoreError::InvalidResponse(format!("invalid Azure DI poll response JSON: {err}"))
        })?;
        if operation_status(&response_json)? == "succeeded" {
            return Ok(response_json);
        }
        tokio::time::sleep(Duration::from_secs(retry_after)).await;
    }
}
