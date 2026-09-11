use crate::ocr::error::OcrRequestError;
use crate::ocr::error::OcrResponseError;
use std::collections::BTreeMap;
use std::time::Duration;

use super::hooks::{OcrDuringCallRequest, OcrPreCallRequest};
use super::types::{LiteLLMOcrRequest, OcrConnection, OcrDocument};
use crate::Error;
use crate::auth::InputSource;
use serde::{
    Deserialize,
    de::{DeserializeOwned, IntoDeserializer},
};
use serde_json::{Map, Value};

#[derive(Debug)]
pub struct DecodedOcrResponse<T> {
    pub data: T,
    pub native: Option<Value>,
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
    #[serde(default)]
    pub input_sources: BTreeMap<String, InputSource>,
    pub timeout_seconds: Option<f64>,
}

pub fn is_supported_request(model: &str, custom_llm_provider: Option<&str>) -> bool {
    super::registry::resolve_wire_adapter(model, custom_llm_provider).is_ok()
}

pub fn decode_request(wire: OcrWireRequest) -> Result<LiteLLMOcrRequest, Error> {
    let api_key_source = source_for(&wire.input_sources, "api_key");
    let api_base_source = source_for(&wire.input_sources, "api_base");
    let extra_headers_source = source_for(&wire.input_sources, "extra_headers");
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
    let request = LiteLLMOcrRequest::new(
        wire.model,
        document,
        wire.custom_llm_provider.as_deref(),
        wire.optional_params,
    )?;
    let connection = OcrConnection {
        api_key: nonblank(wire.api_key),
        api_key_source,
        api_base: nonblank(wire.api_base),
        api_base_source,
        extra_headers: headers,
        extra_headers_source,
        timeout: timeout.unwrap_or(defaults.timeout),
        max_download_bytes: defaults.max_download_bytes,
    };
    Ok(LiteLLMOcrRequest {
        connection,
        input_sources: wire.input_sources,
        ..request
    })
}

fn source_for(sources: &BTreeMap<String, InputSource>, name: &str) -> InputSource {
    sources.get(name).copied().unwrap_or_default()
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
