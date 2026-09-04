use std::future::Future;

use litellm_core::Error;
use litellm_core::responses::http_types::{ResponsesApiResponse, ResponsesRequest};
use litellm_core::responses::{responses as run_responses, responses_decline_reason};
use pyo3::prelude::*;
use serde_json::Value;

use crate::errors::chat_completions_error_to_pyerr;
use crate::marshal::{RouteOptions, RouteOptionsInputs, object_or_empty, required_value};

fn prepare_responses(
    inputs: ResponsesInputs,
) -> PyResult<impl Future<Output = Result<ResponsesApiResponse, Error>> + Send + 'static> {
    let input = required_value(
        "input",
        inputs.input,
        |value| value.is_string() || value.is_array(),
        "string or list",
    )?;
    let optional_params = object_or_empty("optional_params", inputs.optional_params)?;
    let options = RouteOptions::from_python(RouteOptionsInputs {
        model: inputs.model,
        api_key: inputs.api_key,
        api_base: inputs.api_base,
        custom_llm_provider: inputs.custom_llm_provider,
        extra_headers: inputs.extra_headers,
        timeout_seconds: inputs.timeout_seconds,
    })?;

    Ok(async move {
        let RouteOptions {
            model,
            api_key,
            api_base,
            custom_llm_provider,
            extra_headers,
            timeout,
        } = options;
        run_responses(ResponsesRequest {
            model: &model,
            input,
            optional_params,
            api_key: api_key.as_deref(),
            api_base: api_base.as_deref(),
            custom_llm_provider: custom_llm_provider.as_deref(),
            extra_headers,
            timeout,
            use_chat_completions_api: inputs.use_chat_completions_api.unwrap_or(false),
        })
        .await
    })
}

#[pyfunction]
#[pyo3(signature = (model, input, optional_params=None, custom_llm_provider=None, use_chat_completions_api=false))]
fn responses_decline(
    model: String,
    #[pyo3(from_py_with = litellm_python_interop::from_py)] input: Value,
    #[pyo3(from_py_with = litellm_python_interop::from_py)] optional_params: Option<Value>,
    custom_llm_provider: Option<String>,
    use_chat_completions_api: bool,
) -> PyResult<Option<String>> {
    let optional_params = object_or_empty("optional_params", optional_params)?;
    Ok(responses_decline_reason(
        &model,
        input,
        &optional_params,
        custom_llm_provider.as_deref(),
        use_chat_completions_api,
    )
    .map(str::to_string))
}

bridge_route! {
    sync = responses,
    asynchronous = aresponses,
    inputs = ResponsesInputs,
    required = {
        model: String,
        #[pyo3(from_py_with = litellm_python_interop::from_py)]
        input: serde_json::Value,
    },
    optional = {
        #[pyo3(from_py_with = litellm_python_interop::from_py)]
        optional_params: Option<serde_json::Value>,
        api_key: Option<String>,
        api_base: Option<String>,
        custom_llm_provider: Option<String>,
        #[pyo3(from_py_with = litellm_python_interop::from_py)]
        extra_headers: Option<serde_json::Value>,
        timeout_seconds: Option<f64>,
        use_chat_completions_api: Option<bool>,
    },
    prepare = prepare_responses,
    errors = chat_completions_error_to_pyerr,
    extra = [responses_decline],
}
