use crate::ocr::error::OcrError;
use crate::ocr::error::OcrRequestError;
use crate::ocr::error::OcrResponseError;
use crate::ocr::error::PagesError;
use std::time::Duration;

use super::hooks::{OcrDuringCallRequest, OcrPreCallRequest};
use super::prepare::{OcrProviderKind, OcrProviderRequest, ocr_provider_config};
use super::types::{OcrConnection, OcrDocument, OcrRequest, OcrRequestFormat, VertexOcrSettings};
use crate::Error;
use crate::auth::azure::AzureAuthInputs;
use crate::routing_utils::provider::{CustomLlmProvider, get_custom_llm_provider};
use percent_encoding::{AsciiSet, NON_ALPHANUMERIC, utf8_percent_encode};
use serde::{
    Deserialize, Deserializer,
    de::{DeserializeOwned, IntoDeserializer},
};
use serde_json::{Map, Value};

#[derive(Debug)]
pub struct DecodedOcrResponse<T> {
    pub data: T,
    pub native: Option<Map<String, Value>>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OcrWireRequest {
    pub model: String,
    pub document: Value,
    pub api_key: Option<String>,
    pub api_base: Option<String>,
    pub custom_llm_provider: Option<String>,
    pub extra_headers: Option<Map<String, Value>>,
    #[serde(default)]
    pub optional_params: Map<String, Value>,
    pub timeout_seconds: Option<f64>,
}

pub fn decode_request(wire: OcrWireRequest) -> Result<OcrRequest, Error> {
    decode_request_with_env(wire, &|name| std::env::var(name).ok())
}

pub fn decode_request_with_env(
    wire: OcrWireRequest,
    env: &dyn Fn(&str) -> Option<String>,
) -> Result<OcrRequest, Error> {
    let provider = get_custom_llm_provider(&wire.model, wire.custom_llm_provider.as_deref())
        .unwrap_or(CustomLlmProvider {
            model: &wire.model,
            custom_llm_provider: "mistral",
        });
    let kind = ocr_provider_config(provider.custom_llm_provider, provider.model)?;
    let params = decode_params(kind, wire.optional_params.clone())?;
    let document = decode_request_value(wire.document, "document")?;
    let key_env = match kind {
        OcrProviderKind::Mistral => "MISTRAL_API_KEY",
        OcrProviderKind::AzureAi => "AZURE_AI_API_KEY",
        OcrProviderKind::AzureDocumentIntelligence => "AZURE_DOCUMENT_INTELLIGENCE_API_KEY",
        OcrProviderKind::VertexAi | OcrProviderKind::VertexAiDeepSeek => "VERTEX_AI_API_KEY",
        OcrProviderKind::ReductoV3 | OcrProviderKind::ReductoLegacy => "REDUCTO_API_KEY",
    };
    let base_env = match kind {
        OcrProviderKind::AzureAi => Some("AZURE_AI_API_BASE"),
        OcrProviderKind::AzureDocumentIntelligence => Some("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT"),
        _ => None,
    };
    let azure_auth = if matches!(
        kind,
        OcrProviderKind::AzureAi | OcrProviderKind::AzureDocumentIntelligence
    ) {
        Some(AzureAuthInputs::from_optional_params(
            &wire.optional_params,
        )?)
    } else {
        None
    };
    let headers = wire
        .extra_headers
        .unwrap_or_default()
        .into_iter()
        .map(|(name, value)| {
            let value = value
                .as_str()
                .ok_or_else(|| OcrRequestError::RequestField {
                    path: format!("extra_headers.{name}"),
                })?;
            Ok((name, value.to_string()))
        })
        .collect::<Result<Vec<_>, OcrRequestError>>()?;
    let timeout = wire
        .timeout_seconds
        .map(|seconds| {
            Duration::try_from_secs_f64(seconds).map_err(|_| OcrRequestError::RequestField {
                path: "timeout_seconds".into(),
            })
        })
        .transpose()?;
    let vertex = VertexOcrSettings {
        project: string_setting(
            &wire.optional_params,
            &["vertex_project", "vertex_ai_project"],
        )?
        .or_else(|| nonblank(env("VERTEXAI_PROJECT"))),
        location: string_setting(
            &wire.optional_params,
            &["vertex_location", "vertex_ai_location"],
        )?
        .or_else(|| nonblank(env("VERTEXAI_LOCATION")))
        .or_else(|| nonblank(env("VERTEX_LOCATION"))),
    };
    let defaults = OcrConnection::default();
    let max_download_bytes = env("MAX_IMAGE_URL_DOWNLOAD_SIZE_MB")
        .and_then(|value| value.parse::<f64>().ok())
        .map(|mb| (mb.max(0.0) * 1024.0 * 1024.0) as u64)
        .unwrap_or(defaults.max_download_bytes);
    let mut request = OcrRequest::new(provider.model.to_string(), document, params);
    request.connection = OcrConnection {
        api_key: nonblank(wire.api_key)
            .or_else(|| nonblank(env(key_env)))
            .or_else(|| {
                matches!(
                    kind,
                    OcrProviderKind::VertexAi | OcrProviderKind::VertexAiDeepSeek
                )
                .then(|| nonblank(env("VERTEXAI_API_KEY")))
                .flatten()
            }),
        api_base: nonblank(wire.api_base).or_else(|| base_env.and_then(|name| nonblank(env(name)))),
        extra_headers: headers,
        azure_auth,
        vertex,
        timeout: timeout.unwrap_or(defaults.timeout),
        poll_timeout: timeout.unwrap_or(defaults.poll_timeout),
        max_download_bytes,
    };
    Ok(request)
}

fn nonblank(value: Option<String>) -> Option<String> {
    value
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
}
fn string_setting(
    params: &Map<String, Value>,
    keys: &[&str],
) -> Result<Option<String>, OcrRequestError> {
    for key in keys {
        if let Some(value) = params.get(*key).filter(|value| !value.is_null()) {
            let value = value
                .as_str()
                .ok_or_else(|| OcrRequestError::RequestField {
                    path: (*key).into(),
                })?;
            if let Some(value) = nonblank(Some(value.into())) {
                return Ok(Some(value));
            }
        }
    }
    Ok(None)
}

pub fn decode_params(
    kind: OcrProviderKind,
    params: Map<String, Value>,
) -> Result<OcrProviderRequest, OcrRequestError> {
    if let Some(format) = params.get("req_format") {
        let format: OcrRequestFormat =
            serde_json::from_value(format.clone()).map_err(|_| OcrRequestError::RequestFormat)?;
        if format == OcrRequestFormat::Native && kind != OcrProviderKind::AzureDocumentIntelligence
        {
            return Err(OcrRequestError::NativeUnsupported(kind.provider_name()));
        }
    }
    if kind == OcrProviderKind::AzureDocumentIntelligence
        && let Some(Value::Array(pages)) = params.get("pages")
    {
        if pages.iter().any(Value::is_boolean) {
            return Err(PagesError::BooleanIndex.into());
        }
        if pages.iter().any(|p| p.is_number() && p.as_i64().is_none()) {
            return Err(PagesError::IndexOutOfRange.into());
        }
        if !pages.iter().all(Value::is_i64) && !pages.iter().all(Value::is_string) {
            return Err(PagesError::MixedElementTypes.into());
        }
    }
    let value = Value::Object(params);
    Ok(match kind {
        OcrProviderKind::Mistral => {
            OcrProviderRequest::Mistral(decode_request_value(value, "optional_params")?)
        }
        OcrProviderKind::AzureAi => {
            OcrProviderRequest::AzureAi(decode_request_value(value, "optional_params")?)
        }
        OcrProviderKind::AzureDocumentIntelligence => {
            OcrProviderRequest::AzureDocumentIntelligence(decode_request_value(
                value,
                "optional_params",
            )?)
        }
        OcrProviderKind::VertexAi => {
            OcrProviderRequest::VertexAi(decode_request_value(value, "optional_params")?)
        }
        OcrProviderKind::VertexAiDeepSeek => {
            OcrProviderRequest::VertexAiDeepSeek(decode_request_value(value, "optional_params")?)
        }
        OcrProviderKind::ReductoV3 => {
            OcrProviderRequest::ReductoV3(decode_request_value(value, "optional_params")?)
        }
        OcrProviderKind::ReductoLegacy => {
            OcrProviderRequest::ReductoLegacy(decode_request_value(value, "optional_params")?)
        }
    })
}

pub fn decode_request_value<T: DeserializeOwned>(
    value: Value,
    prefix: &str,
) -> Result<T, OcrRequestError> {
    serde_path_to_error::deserialize(value.into_deserializer()).map_err(|error| {
        OcrRequestError::RequestField {
            path: format!("{prefix}.{}", error.path()),
        }
    })
}

pub fn decode_response<T: DeserializeOwned>(
    bytes: &[u8],
    native: bool,
) -> Result<DecodedOcrResponse<T>, OcrResponseError> {
    let mut deserializer = serde_json::Deserializer::from_slice(bytes);
    let data = serde_path_to_error::deserialize(&mut deserializer).map_err(|error| {
        OcrResponseError::ResponseField {
            path: error.path().to_string(),
        }
    })?;
    deserializer
        .end()
        .map_err(|_| OcrResponseError::ResponseField {
            path: "response".into(),
        })?;
    let native = if native {
        Some(
            serde_json::from_slice(bytes).map_err(|_| OcrResponseError::ResponseField {
                path: "response".into(),
            })?,
        )
    } else {
        None
    };
    Ok(DecodedOcrResponse { data, native })
}

pub async fn read_json_response<T: DeserializeOwned>(
    response: reqwest::Response,
    native: bool,
) -> Result<DecodedOcrResponse<T>, OcrError> {
    let status = response.status();
    let bytes = response
        .bytes()
        .await
        .map_err(super::client::network_error)?;
    if !status.is_success() {
        return Err(crate::error::TransportError::Http {
            status: status.as_u16(),
            body: super::client::truncate_error_body(&String::from_utf8_lossy(&bytes)),
        }
        .into());
    }
    Ok(decode_response(&bytes, native)?)
}

pub fn decode_pre_call_result(
    original: OcrPreCallRequest,
    value: Value,
) -> Result<OcrPreCallRequest, OcrRequestError> {
    #[derive(Deserialize)]
    struct Changed {
        document: OcrDocument,
        #[serde(default)]
        optional_params: Map<String, Value>,
    }
    let changed: Changed = decode_request_value(value, "guardrail")?;
    Ok(OcrPreCallRequest {
        document: changed.document,
        optional_params: Value::Object(changed.optional_params),
        ..original
    })
}

pub fn decode_during_call_result(
    original: OcrDuringCallRequest,
    value: Value,
) -> Result<OcrDuringCallRequest, OcrRequestError> {
    #[derive(Deserialize)]
    struct Changed {
        body: Value,
    }
    let changed: Changed = decode_request_value(value, "guardrail")?;
    Ok(OcrDuringCallRequest {
        body: changed.body,
        ..original
    })
}

pub(crate) fn encode_model_id(model: &str) -> Result<String, OcrRequestError> {
    const PATH_SEGMENT: &AsciiSet = &NON_ALPHANUMERIC
        .remove(b'-')
        .remove(b'_')
        .remove(b'.')
        .remove(b'~');
    let model = model.rsplit('/').next().unwrap_or(model);
    if matches!(model, "." | "..") {
        return Err(OcrRequestError::DotModel);
    }
    Ok(utf8_percent_encode(model, PATH_SEGMENT).to_string())
}

#[derive(Deserialize)]
#[serde(untagged)]
enum PythonNumber {
    Integer(i64),
    Float(f64),
    String(String),
    Boolean(bool),
}
impl PythonNumber {
    fn integer(self) -> Option<i64> {
        match self {
            Self::Integer(n) => Some(n),
            Self::Boolean(n) => Some(i64::from(n)),
            Self::String(s) => s
                .trim()
                .parse::<i64>()
                .ok()
                .or_else(|| s.trim().parse::<f64>().ok().and_then(exact_integer)),
            Self::Float(n) => exact_integer(n),
        }
    }
    fn float(self) -> Option<f64> {
        match self {
            Self::Integer(n) => Some(n as f64),
            Self::Float(n) => Some(n),
            Self::String(s) => s.trim().parse().ok(),
            Self::Boolean(n) => Some(u8::from(n) as f64),
        }
    }
}
fn exact_integer(n: f64) -> Option<i64> {
    (n.is_finite() && n.fract() == 0.0 && n >= i64::MIN as f64 && n < -(i64::MIN as f64))
        .then_some(n as i64)
}
pub fn optional_i64<'de, D: Deserializer<'de>>(deserializer: D) -> Result<Option<i64>, D::Error> {
    Option::<PythonNumber>::deserialize(deserializer)?
        .map(|n| {
            n.integer()
                .ok_or_else(|| serde::de::Error::custom("expected integer"))
        })
        .transpose()
}
pub fn required_i64<'de, D: Deserializer<'de>>(deserializer: D) -> Result<i64, D::Error> {
    PythonNumber::deserialize(deserializer)?
        .integer()
        .ok_or_else(|| serde::de::Error::custom("expected integer"))
}
pub fn optional_f64<'de, D: Deserializer<'de>>(deserializer: D) -> Result<Option<f64>, D::Error> {
    Option::<PythonNumber>::deserialize(deserializer)?
        .map(|n| {
            n.float()
                .ok_or_else(|| serde::de::Error::custom("expected number"))
        })
        .transpose()
}
pub fn present_nullable<'de, D: Deserializer<'de>, T: Deserialize<'de>>(
    deserializer: D,
) -> Result<Option<Option<T>>, D::Error> {
    Option::<T>::deserialize(deserializer).map(Some)
}
pub fn reducto_page<'de, D: Deserializer<'de>>(deserializer: D) -> Result<Option<i64>, D::Error> {
    let value = Value::deserialize(deserializer)?;
    Ok(match value {
        Value::Number(n) => n
            .as_i64()
            .or_else(|| n.as_f64().and_then(|n| exact_integer(n.trunc()))),
        Value::String(s) => s.trim().parse().ok(),
        Value::Bool(b) => Some(i64::from(b)),
        _ => None,
    })
}

pub(crate) fn decode_deepseek_content(
    text: &str,
) -> Result<
    Option<crate::providers::vertex_ai::ocr::deepseek::types::DeepSeekOcrResult>,
    OcrResponseError,
> {
    match serde_json::from_str::<Value>(text) {
        Err(_) => Ok(None),
        Ok(value) => serde_path_to_error::deserialize(value.into_deserializer())
            .map(Some)
            .map_err(|error| OcrResponseError::ResponseField {
                path: format!("choices[0].message.content.{}", error.path()),
            }),
    }
}
