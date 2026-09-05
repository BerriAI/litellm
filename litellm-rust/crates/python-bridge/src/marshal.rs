use std::time::Duration;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use serde_json::{Map, Value};

#[derive(FromPyObject)]
struct NativeBedrockOptions {
    aws_access_key_id: Option<String>,
    aws_secret_access_key: Option<String>,
    aws_session_token: Option<String>,
    aws_region_name: Option<String>,
    aws_session_name: Option<String>,
    aws_profile_name: Option<String>,
    aws_role_name: Option<String>,
    aws_web_identity_token: Option<String>,
    aws_sts_endpoint: Option<String>,
    aws_external_id: Option<String>,
    aws_bedrock_runtime_endpoint: Option<String>,
    request_metadata_fields: Vec<String>,
    request_metadata: Option<std::collections::BTreeMap<String, String>>,
}

impl From<NativeBedrockOptions> for litellm_core::request_options::BedrockOptions {
    fn from(input: NativeBedrockOptions) -> Self {
        Self {
            aws_access_key_id: input.aws_access_key_id,
            aws_secret_access_key: input.aws_secret_access_key,
            aws_session_token: input.aws_session_token,
            aws_region_name: input.aws_region_name,
            aws_session_name: input.aws_session_name,
            aws_profile_name: input.aws_profile_name,
            aws_role_name: input.aws_role_name,
            aws_web_identity_token: input.aws_web_identity_token,
            aws_sts_endpoint: input.aws_sts_endpoint,
            aws_external_id: input.aws_external_id,
            aws_bedrock_runtime_endpoint: input.aws_bedrock_runtime_endpoint,
            request_metadata_fields: input.request_metadata_fields,
            request_metadata: input.request_metadata,
        }
    }
}

#[derive(FromPyObject)]
struct NativeAnthropicOptions {
    user_id: Option<String>,
}

impl From<NativeAnthropicOptions> for litellm_core::request_options::AnthropicOptions {
    fn from(input: NativeAnthropicOptions) -> Self {
        Self {
            user_id: input.user_id,
        }
    }
}

#[derive(FromPyObject)]
struct NativeVertexOptions {
    project: Option<String>,
    location: Option<String>,
}

impl From<NativeVertexOptions> for litellm_core::request_options::VertexOptions {
    fn from(input: NativeVertexOptions) -> Self {
        Self {
            project: input.project,
            location: input.location,
        }
    }
}

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
    bedrock: Option<NativeBedrockOptions>,
    anthropic: Option<NativeAnthropicOptions>,
    vertex: Option<NativeVertexOptions>,
}

impl NativeRequestOptions {
    pub(crate) fn provider(&self, default: &'static str) -> &str {
        self.custom_llm_provider.as_deref().unwrap_or(default)
    }
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
            bedrock: input.bedrock.map(Into::into),
            anthropic: input.anthropic.map(Into::into),
            vertex: input.vertex.map(Into::into),
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
    litellm_call_id: Option<String>,
    trace_id: Option<String>,
    request_model: Option<String>,
    attribution: NativeRequestAttribution,
    capabilities: NativeRequestCapabilities,
}

#[derive(FromPyObject)]
struct NativeRequestCapabilities {
    execution_mode: Option<String>,
    stream: bool,
    has_agentic_hook: bool,
    has_custom_client: bool,
    request_format: Option<String>,
    input_source_kind: Option<String>,
    native_response_format: bool,
    websocket_mode: Option<String>,
    requires_connection: bool,
}

impl From<NativeRequestContext> for litellm_core::request_context::LiteLlmRequestContext {
    fn from(input: NativeRequestContext) -> Self {
        Self {
            litellm_call_id: input.litellm_call_id,
            trace_id: input.trace_id,
            request_model: input.request_model,
            attribution: litellm_core::request_context::RequestAttribution {
                user_api_key_hash: input.attribution.user_api_key_hash,
                user_api_key_user_id: input.attribution.user_api_key_user_id,
                user_api_key_team_id: input.attribution.user_api_key_team_id,
            },
            capabilities: litellm_core::request_context::RequestCapabilities {
                execution_mode: input.capabilities.execution_mode,
                stream: input.capabilities.stream,
                has_agentic_hook: input.capabilities.has_agentic_hook,
                has_custom_client: input.capabilities.has_custom_client,
                request_format: input.capabilities.request_format,
                input_source_kind: input.capabilities.input_source_kind,
                native_response_format: input.capabilities.native_response_format,
                websocket_mode: input.capabilities.websocket_mode,
                requires_connection: input.capabilities.requires_connection,
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
    bedrock: object = None
    anthropic: object = None
    vertex: object = None

@dataclass(frozen=True)
class BedrockOptions:
    aws_access_key_id: object = None
    aws_secret_access_key: object = None
    aws_session_token: object = None
    aws_region_name: object = None
    aws_session_name: object = None
    aws_profile_name: object = None
    aws_role_name: object = None
    aws_web_identity_token: object = None
    aws_sts_endpoint: object = None
    aws_external_id: object = None
    aws_bedrock_runtime_endpoint: object = None
    request_metadata_fields: object = ()
    request_metadata: object = None

@dataclass(frozen=True)
class Capabilities:
    execution_mode: object = None
    stream: object = False
    has_agentic_hook: object = False
    has_custom_client: object = False
    request_format: object = None
    input_source_kind: object = None
    native_response_format: object = False
    websocket_mode: object = None
    requires_connection: object = False

@dataclass(frozen=True)
class VertexOptions:
    project: object = None
    location: object = None

@dataclass(frozen=True)
class Attribution:
    user_api_key_hash: object = None
    user_api_key_user_id: object = None
    user_api_key_team_id: object = None

@dataclass(frozen=True)
class Context:
    litellm_call_id: object = None
    trace_id: object = None
    request_model: object = None
    attribution: Attribution = Attribution()
    capabilities: Capabilities = Capabilities()

@dataclass(frozen=True)
class Request:
    model: str = 'model'
    messages: object = None
    body: object = None
    audio: object = None
    document: object = None
    optional_params: object = None
    value: str = ''
    url: str = ''

context = Context()
options = Options()
",
        Some(&locals),
        Some(&locals),
    )
    .expect("request dataclasses should load");
    locals
}
