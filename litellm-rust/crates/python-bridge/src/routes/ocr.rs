use crate::callback_bindings::PythonProviderObserver;
use crate::errors::ocr_error_to_pyerr;
use crate::marshal::{NativeRequestContext, NativeRequestOptions};
use litellm_ai_gateway::integrations::types::RequestHooks;
use litellm_ai_gateway::io::ocr::OcrRequest;
use litellm_ai_gateway::io::ocr::ocr_with_observer as run_route;
use litellm_core::Error;
use litellm_core::request_context::LiteLlmRequestContext;
use pyo3::prelude::*;
use serde_json::{Map, Value};
use std::future::Future;

#[derive(FromPyObject)]
struct OcrInputs {
    model: String,
    #[pyo3(from_py_with = litellm_python_interop::from_py)]
    document: Value,
    #[pyo3(from_py_with = litellm_python_interop::from_py)]
    optional_params: Map<String, Value>,
}

fn prepare_ocr(
    input: OcrInputs,
    options: NativeRequestOptions,
    context: NativeRequestContext,
    callback_adapter: Option<Py<PyAny>>,
    python_context: crate::execution::PythonCallContext<'_>,
) -> PyResult<impl Future<Output = Result<Value, Error>> + Send + 'static> {
    let context: LiteLlmRequestContext = context.into();
    let provider_supported = litellm_ai_gateway::io::ocr::ocr_provider_supported(
        &input.model,
        options.provider("mistral"),
        context.capabilities.request_format.as_deref(),
    );
    if let Some(reason) = super::definition::request_decline(provider_supported, &context) {
        return Err(crate::errors::RustBridgeDeclined::new_err(reason));
    }
    let document = input.document;
    let mut observer = PythonProviderObserver::new(callback_adapter, python_context)?;
    Ok(async move {
        run_route(
            OcrRequest {
                model: &input.model,
                document,
                optional_params: input.optional_params,
            },
            &options.into(),
            &context,
            RequestHooks {
                callbacks: Vec::new(),
                guardrails: Vec::new(),
            },
            &mut observer,
        )
        .await
    })
}

#[pyfunction]
#[pyo3(signature = (model, custom_llm_provider, *, context))]
fn ocr_decline(
    model: &str,
    custom_llm_provider: &str,
    context: NativeRequestContext,
) -> Option<String> {
    let context: LiteLlmRequestContext = context.into();
    let provider_supported = litellm_ai_gateway::io::ocr::ocr_provider_supported(
        model,
        custom_llm_provider,
        context.capabilities.request_format.as_deref(),
    );
    super::definition::request_decline(provider_supported, &context)
}

bridge_route! {
    sync = ocr,
    asynchronous = aocr,
    request = OcrInputs,
    prepare = prepare_ocr,
    errors = ocr_error_to_pyerr,
    extra = [ocr_decline],
}
