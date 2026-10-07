mod host;

use pyo3::types::{PyDict, PyTuple};

use crate::execution::{run_async, run_sync};
use litellm_inference_chat::{ChatCompletionsRoute, Error, types::ChatCompletionsRequest};
use litellm_llms_types::formats::chat_completions::ChatCompletionsResponse;
use pyo3::prelude::*;
use serde_json::{Map, Value};

use crate::{
    errors::route_error_to_pyerr,
    marshal::{
        RouteOptions, messages_argument, optional_object_field, required_field, value_route_options,
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
pub(crate) fn chat_completions(py: Python<'_>, call: Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    let call = super::NativeCall::extract(&call)?;
    let messages: Vec<Value> = messages_argument(&required_field(&call.bound, "messages")?)?;
    let optional_params =
        optional_object_field(&call.bound, "optional_params")?.unwrap_or_default();
    let options = value_route_options(&call.bound)?;
    let http = crate::http::provider_client(py, &call.kwargs, false)?;
    let secrets = crate::secrets::source(py)?;
    run_sync(
        py,
        execute(http, secrets, messages, optional_params, options),
        route_error_to_pyerr,
    )
}

#[pyfunction]
pub(crate) fn achat_completions<'py>(
    py: Python<'py>,
    call: Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let call = super::NativeCall::extract(&call)?;
    let messages: Vec<Value> = messages_argument(&required_field(&call.bound, "messages")?)?;
    let optional_params =
        optional_object_field(&call.bound, "optional_params")?.unwrap_or_default();
    let options = value_route_options(&call.bound)?;
    let http = crate::http::provider_client(py, &call.kwargs, true)?;
    let secrets = crate::secrets::source(py)?;
    run_async(
        py,
        execute(http, secrets, messages, optional_params, options),
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
pub(crate) fn completion(py: Python<'_>, call: Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    let call = super::NativeCall::extract(&call)?;
    run_public(py, call.bound.into_any(), call.args, call.kwargs, false)
}

#[pyfunction]
pub(crate) fn acompletion(py: Python<'_>, call: Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    let call = super::NativeCall::extract(&call)?;
    run_public(py, call.bound.into_any(), call.args, call.kwargs, true)
}
