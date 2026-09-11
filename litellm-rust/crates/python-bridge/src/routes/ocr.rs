use std::future::Future;
use std::sync::{Arc, Mutex};

use litellm_ai_gateway::io::ocr::{OcrRequest, ocr as run_ocr};
use litellm_core::ocr::wire::{OcrWireRequest, decode_request, is_supported_request};
use pyo3::prelude::*;
use serde_json::Value;

use crate::errors::{BridgeError, ocr_route_error};
use crate::marshal::{RouteOptions, RouteOptionsInputs, object_or_empty};

#[path = "ocr_callbacks.rs"]
mod callbacks;

fn prepare_ocr(
    inputs: OcrInputs,
) -> PyResult<impl Future<Output = Result<Value, BridgeError>> + Send + 'static> {
    let document: Value =
        Python::attach(|py| litellm_python_interop::from_py(inputs.document.bind(py)))?;
    let hooks = (inputs.logging_obj.is_some() || inputs.token_provider.is_some())
        .then(|| {
            Python::attach(|py| {
                Ok::<_, PyErr>(Arc::new(callbacks::PythonOcrHooks {
                    logger: inputs.logging_obj,
                    token_provider: match inputs.token_provider {
                        Some(provider)
                            if provider.bind(py).is_callable()
                                && provider.bind(py).is_truthy()? =>
                        {
                            Some(provider)
                        }
                        _ => None,
                    },
                    document: inputs.document,
                    document_snapshot: document.clone(),
                    api_key: inputs.api_key.clone(),
                    locals: inputs
                        .callback_loop
                        .map(|event_loop| {
                            pyo3_async_runtimes::TaskLocals::new(event_loop.into_bound(py))
                                .copy_context(py)
                        })
                        .transpose()?,
                    error: Mutex::new(None),
                    execution_body: Mutex::new(None),
                }))
            })
        })
        .transpose()?;
    let options = RouteOptions::from_python(RouteOptionsInputs {
        model: inputs.model,
        api_key: inputs.api_key,
        api_base: inputs.api_base,
        custom_llm_provider: inputs.custom_llm_provider,
        extra_headers: inputs.extra_headers,
        timeout_seconds: inputs.timeout_seconds,
    })?;
    let optional_params = object_or_empty("optional_params", inputs.optional_params)?;
    let input_sources = inputs
        .input_sources
        .map(serde_json::from_value)
        .transpose()
        .map_err(|error| pyo3::exceptions::PyValueError::new_err(error.to_string()))?
        .unwrap_or_default();

    Ok(async move {
        let RouteOptions {
            model,
            api_key,
            api_base,
            custom_llm_provider,
            extra_headers,
            timeout,
        } = options;
        if is_supported_request(&model, custom_llm_provider.as_deref()) {
            let mut request = decode_request(OcrWireRequest {
                model,
                document,
                api_key,
                api_base,
                custom_llm_provider,
                extra_headers,
                optional_params,
                input_sources,
                timeout_seconds: timeout.map(|value| value.as_secs_f64()),
            })
            .map_err(ocr_route_error)?;
            if let Some(call_completion) = &inputs.call_completion {
                callbacks::NativeOcrCompletion::attach(call_completion)
                    .map_err(BridgeError::Host)?;
            }
            if let Some(hooks) = &hooks
                && hooks.token_provider.is_some()
            {
                request.connection.token_provider =
                    Some(litellm_core::auth::TokenProviderHandle::new(hooks.clone()));
            }
            let request = match &hooks {
                Some(hooks) => request.with_host_hooks(hooks.clone(), None),
                None => request,
            };
            return litellm_core::ocr::ocr(request)
                .await
                .map(|response| response.into_json())
                .map_err(|error| {
                    hooks
                        .and_then(|hooks| {
                            hooks
                                .error
                                .lock()
                                .unwrap_or_else(|poisoned| poisoned.into_inner())
                                .take()
                        })
                        .map(BridgeError::Host)
                        .unwrap_or_else(|| ocr_route_error(error))
                });
        }
        run_ocr(OcrRequest {
            model: &model,
            document,
            api_key: api_key.as_deref(),
            api_base: api_base.as_deref(),
            custom_llm_provider: custom_llm_provider.as_deref(),
            extra_headers,
            optional_params,
            timeout,
            callbacks: Vec::new(),
            guardrails: Vec::new(),
            request_metadata: Default::default(),
            litellm_call_id: None,
        })
        .await
        .map_err(ocr_route_error)
    })
}

bridge_route! {
    sync = ocr,
    asynchronous = aocr,
    inputs = OcrInputs,
    required = {
        model: String,
        document: Py<PyAny>,
    },
    optional = {
        api_key: Option<String>,
        api_base: Option<String>,
        custom_llm_provider: Option<String>,
        #[pyo3(from_py_with = litellm_python_interop::from_py)]
        extra_headers: Option<serde_json::Value>,
        #[pyo3(from_py_with = litellm_python_interop::from_py)]
        optional_params: Option<serde_json::Value>,
        #[pyo3(from_py_with = litellm_python_interop::from_py)]
        input_sources: Option<serde_json::Value>,
        timeout_seconds: Option<f64>,
        logging_obj: Option<Py<PyAny>>,
        callback_loop: Option<Py<PyAny>>,
        token_provider: Option<Py<PyAny>>,
        call_completion: Option<Py<PyAny>>,
    },
    prepare = prepare_ocr,
    errors = std::convert::identity,
}

#[cfg(test)]
mod tests {
    use litellm_core::ocr::wire::is_supported_request;

    #[test]
    fn native_activation_includes_migrated_providers() {
        assert!(is_supported_request("model", Some("mistral")));
        assert!(is_supported_request("pixtral-12b", Some("azure_ai")));
        assert!(is_supported_request(
            "documentintelligence/prebuilt-read",
            Some("azure_ai")
        ));
        assert!(is_supported_request("parse-v3", Some("reducto")));
        assert!(is_supported_request("parse-legacy", Some("reducto")));
        assert!(is_supported_request("mistral-ocr", Some("vertex_ai")));
        assert!(is_supported_request("deepseek-ocr", Some("vertex_ai")));
    }
}
