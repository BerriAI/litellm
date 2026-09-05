use std::time::Duration;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use serde_json::{Map, Value};

#[derive(FromPyObject)]
pub(crate) struct NativeRequestOptions {
    api_key: Option<String>,
    api_base: Option<String>,
    custom_llm_provider: Option<String>,
    #[pyo3(from_py_with = litellm_python_interop::from_py)]
    extra_headers: Option<Map<String, Value>>,
    #[pyo3(from_py_with = litellm_python_interop::from_py)]
    extra_query: Option<Map<String, Value>>,
    timeout_seconds: Option<f64>,
    #[pyo3(from_py_with = litellm_python_interop::from_py)]
    provider_connection: Option<Map<String, Value>>,
}

impl From<NativeRequestOptions> for litellm_core::request_options::RequestOptions {
    fn from(input: NativeRequestOptions) -> Self {
        Self {
            api_key: input.api_key,
            api_base: input.api_base,
            custom_llm_provider: input.custom_llm_provider,
            extra_headers: input.extra_headers,
            extra_query: input.extra_query,
            timeout: optional_timeout(input.timeout_seconds),
            provider_connection: input.provider_connection.unwrap_or_default(),
        }
    }
}

#[derive(FromPyObject)]
pub(crate) struct NativeRequestAttribution {
    user_api_key_hash: Option<String>,
    user_api_key_user_id: Option<String>,
    user_api_key_team_id: Option<String>,
}

#[derive(FromPyObject)]
pub(crate) struct NativeRequestContext {
    #[pyo3(from_py_with = litellm_python_interop::from_py)]
    metadata: Option<Map<String, Value>>,
    #[pyo3(from_py_with = litellm_python_interop::from_py)]
    litellm_metadata: Option<Map<String, Value>>,
    request_metadata_fields: Vec<String>,
    litellm_call_id: Option<String>,
    request_model: Option<String>,
    attribution: NativeRequestAttribution,
}

impl From<NativeRequestContext> for litellm_core::request_context::LiteLlmRequestContext {
    fn from(input: NativeRequestContext) -> Self {
        Self {
            metadata: input.metadata,
            litellm_metadata: input.litellm_metadata,
            request_metadata_fields: input.request_metadata_fields,
            litellm_call_id: input.litellm_call_id,
            request_model: input.request_model,
            attribution: litellm_core::request_context::RequestAttribution {
                user_api_key_hash: input.attribution.user_api_key_hash,
                user_api_key_user_id: input.attribution.user_api_key_user_id,
                user_api_key_team_id: input.attribution.user_api_key_team_id,
            },
        }
    }
}

pub(crate) fn required_value(
    name: &'static str,
    value: Value,
    expected: fn(&Value) -> bool,
    expected_name: &'static str,
) -> PyResult<Value> {
    if expected(&value) {
        return Ok(value);
    }
    Err(PyValueError::new_err(format!(
        "{name} must be a {expected_name}"
    )))
}

pub(crate) fn optional_timeout(timeout_seconds: Option<f64>) -> Option<Duration> {
    timeout_seconds.and_then(|secs| {
        if secs.is_finite() && secs > 0.0 {
            Some(Duration::from_secs_f64(secs))
        } else {
            None
        }
    })
}

#[cfg(test)]
pub(crate) fn request_fixtures(py: Python<'_>) -> Bound<'_, pyo3::types::PyDict> {
    let locals = pyo3::types::PyDict::new(py);
    py.run(
        c"
from dataclasses import dataclass, replace

@dataclass(frozen=True)
class Options:
    api_key: object = None
    api_base: object = None
    custom_llm_provider: object = None
    extra_headers: object = None
    extra_query: object = None
    timeout_seconds: object = None
    provider_connection: object = None

@dataclass(frozen=True)
class Attribution:
    user_api_key_hash: object = None
    user_api_key_user_id: object = None
    user_api_key_team_id: object = None

@dataclass(frozen=True)
class Context:
    metadata: object = None
    litellm_metadata: object = None
    request_metadata_fields: tuple = ()
    litellm_call_id: object = None
    request_model: object = None
    attribution: Attribution = Attribution()

@dataclass(frozen=True)
class Request:
    model: str = 'model'
    messages: object = None
    body: object = None
    audio: object = None
    document: object = None
    optional_params: object = None
    options: Options = Options()
    value: str = ''
    url: str = ''

context = Context()
",
        Some(&locals),
        Some(&locals),
    )
    .expect("request dataclasses should load");
    locals
}
