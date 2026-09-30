mod host;

use pyo3::types::{PyDict, PyTuple};

use crate::logger::{run_async, run_sync};
use litellm_core::chat_completions::{ChatCompletionsRoute, Error, types::ChatCompletionsRequest};
use litellm_llms_types::formats::chat_completions::ChatCompletionsResponse;
use pyo3::prelude::*;
use serde_json::{Map, Value};

use crate::{
    errors::route_error_to_pyerr,
    marshal::{
        RouteOptions, extra_headers_argument, messages_argument, optional_params_argument,
        optional_timeout,
    },
};

async fn execute(
    http: Result<litellm_http::Client, litellm_http::Error>,
    secrets: std::sync::Arc<dyn litellm_secrets::source::SecretSource>,
    messages: Vec<Value>,
    optional_params: Map<String, Value>,
    options: RouteOptions,
) -> Result<ChatCompletionsResponse, Error> {
    let RouteOptions {
        model,
        api_key,
        api_base,
        custom_llm_provider,
        extra_headers,
        timeout,
    } = options;
    ChatCompletionsRoute::new(http?, crate::http::resources().auth.clone(), secrets)
        .execute(
            ChatCompletionsRequest {
                model: &model,
                messages: Value::Array(messages),
                optional_params,
                api_key: api_key.as_deref(),
                api_base: api_base.as_deref(),
                custom_llm_provider: custom_llm_provider.as_deref(),
                extra_headers,
                timeout,
            },
            &(),
            None,
        )
        .await
}

#[pyfunction]
#[pyo3(signature = (model, messages, optional_params=None, api_key=None, api_base=None, custom_llm_provider=None, extra_headers=None, timeout_seconds=None))]
#[expect(
    clippy::too_many_arguments,
    reason = "one parameter per Python keyword"
)]
pub(crate) fn chat_completions(
    py: Python<'_>,
    model: String,
    #[pyo3(from_py_with = messages_argument)] messages: Vec<Value>,
    #[pyo3(from_py_with = optional_params_argument)] optional_params: Option<Map<String, Value>>,
    api_key: Option<String>,
    api_base: Option<String>,
    custom_llm_provider: Option<String>,
    #[pyo3(from_py_with = extra_headers_argument)] extra_headers: Option<Map<String, Value>>,
    timeout_seconds: Option<f64>,
) -> PyResult<Py<PyAny>> {
    let options = RouteOptions {
        model,
        api_key,
        api_base,
        custom_llm_provider,
        extra_headers,
        timeout: optional_timeout(timeout_seconds),
    };
    let http = crate::http::provider_client(py, &PyDict::new(py), false)?;
    let secrets = crate::secrets::source(py)?;
    run_sync(
        py,
        execute(
            http,
            secrets,
            messages,
            optional_params.unwrap_or_default(),
            options,
        ),
        route_error_to_pyerr,
    )
}

#[pyfunction]
#[pyo3(signature = (model, messages, optional_params=None, api_key=None, api_base=None, custom_llm_provider=None, extra_headers=None, timeout_seconds=None))]
#[expect(
    clippy::too_many_arguments,
    reason = "one parameter per Python keyword"
)]
pub(crate) fn achat_completions<'py>(
    py: Python<'py>,
    model: String,
    #[pyo3(from_py_with = messages_argument)] messages: Vec<Value>,
    #[pyo3(from_py_with = optional_params_argument)] optional_params: Option<Map<String, Value>>,
    api_key: Option<String>,
    api_base: Option<String>,
    custom_llm_provider: Option<String>,
    #[pyo3(from_py_with = extra_headers_argument)] extra_headers: Option<Map<String, Value>>,
    timeout_seconds: Option<f64>,
) -> PyResult<Bound<'py, PyAny>> {
    let options = RouteOptions {
        model,
        api_key,
        api_base,
        custom_llm_provider,
        extra_headers,
        timeout: optional_timeout(timeout_seconds),
    };
    let http = crate::http::provider_client(py, &PyDict::new(py), true)?;
    let secrets = crate::secrets::source(py)?;
    run_async(
        py,
        execute(
            http,
            secrets,
            messages,
            optional_params.unwrap_or_default(),
            options,
        ),
        route_error_to_pyerr,
    )
}

fn run_public(
    py: Python<'_>,
    request: Bound<'_, PyAny>,
    args: Bound<'_, PyTuple>,
    kwargs: Bound<'_, PyDict>,
    asynchronous: bool,
) -> PyResult<Py<PyAny>> {
    use super::inference::InferenceHost;
    use litellm_callbacks_legacy_python::LoggingOperation;
    let host = InferenceHost::new(
        request.clone().unbind(),
        "litellm.rust_bridge.chat_completions.route_host",
    );
    let cache_call_type = if asynchronous {
        "acompletion"
    } else {
        "completion"
    };
    crate::cache::admit_native(py, &kwargs, cache_call_type)?;
    let (arguments, hooks) = crate::routes::call_hooks(
        py,
        LoggingOperation::Completion,
        &request,
        &args,
        &kwargs,
        asynchronous,
    )?;
    crate::routes::run_public_call(
        py,
        arguments,
        move |py, arguments, request| {
            let route = ChatCompletionsRoute::new(
                crate::http::provider_client(py, arguments, asynchronous)?
                    .map_err(crate::http::client_error)?,
                crate::http::resources().auth.clone(),
                crate::secrets::source(py)?,
            );
            let (cache, cache_options) =
                crate::cache::configured_native(py, arguments, cache_call_type)?;
            let route = match cache {
                Some(cache) => route.with_cache(litellm_cache_response::ScopedCache::new(
                    cache,
                    litellm_cache_response::CacheScope::Shared,
                )),
                None => route,
            };
            Ok(route.machine(request, cache_options.policy))
        },
        host::ChatCompletionsPythonHost(host),
        hooks,
        asynchronous,
    )
}

#[pyfunction]
pub(crate) fn completion(
    py: Python<'_>,
    request: Bound<'_, PyAny>,
    args: Bound<'_, PyTuple>,
    kwargs: Bound<'_, PyDict>,
) -> PyResult<Py<PyAny>> {
    run_public(py, request, args, kwargs, false)
}

#[pyfunction]
pub(crate) fn acompletion(
    py: Python<'_>,
    request: Bound<'_, PyAny>,
    args: Bound<'_, PyTuple>,
    kwargs: Bound<'_, PyDict>,
) -> PyResult<Py<PyAny>> {
    run_public(py, request, args, kwargs, true)
}
