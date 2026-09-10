use crate::ocr::error::OcrRequestError;
use crate::ocr::error::OcrResponseError;
use std::time::Duration;

use super::hooks::{OcrDuringCallRequest, OcrPreCallRequest};
use super::registry::{decode_integration_request, resolve_wire_integration};
use super::types::{OcrConnection, OcrDocument, OcrRequest};
use crate::Error;
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
    let (model, integration_kind) =
        resolve_wire_integration(&wire.model, wire.custom_llm_provider.as_deref())?;
    let integration = decode_integration_request(integration_kind, wire.optional_params)?;
    let document = decode_request_value(wire.document, "document")?;
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
    let defaults = OcrConnection::default();
    let mut request = OcrRequest::new(model.as_str().to_string(), document, integration);
    request.connection = OcrConnection {
        api_key: nonblank(wire.api_key),
        api_base: nonblank(wire.api_base),
        extra_headers: headers,
        timeout: timeout.unwrap_or(defaults.timeout),
        poll_timeout: timeout.unwrap_or(defaults.poll_timeout),
        max_download_bytes: defaults.max_download_bytes,
    };
    Ok(request)
}

fn nonblank(value: Option<String>) -> Option<String> {
    value
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
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
    if n.fract() != 0.0 {
        return None;
    }
    checked_truncated_i64(n)
}
pub(crate) fn checked_truncated_i64(value: f64) -> Option<i64> {
    let truncated = value.trunc();
    (truncated.is_finite() && truncated >= i64::MIN as f64 && truncated < -(i64::MIN as f64))
        .then_some(truncated as i64)
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
