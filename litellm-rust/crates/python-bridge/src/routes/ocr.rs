use crate::callback_bindings::PythonProviderObserver;
use crate::errors::{RustBridgeDeclined, ocr_error_to_pyerr};
use crate::marshal::{NativeRequestContext, NativeRequestOptions};
use litellm_core::Error;
use litellm_core::auth::AuthPreflight;
use litellm_core::ocr::{
    OcrRequest, decode_document, ocr as run_route, parameter_names, preflight,
};
use litellm_core::request_context::LiteLlmRequestContext;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use serde_json::{Map, Value};
use std::future::Future;

#[derive(FromPyObject)]
struct OcrInputs {
    model: String,
    #[pyo3(from_py_with = litellm_python_interop::from_py)]
    document: Value,
    optional_params: Py<PyDict>,
    options: NativeRequestOptions,
}

fn admitted<T>(result: AuthPreflight<T>) -> PyResult<T> {
    match result {
        AuthPreflight::Ready(value) => Ok(value),
        AuthPreflight::Declined(reason) => Err(RustBridgeDeclined::new_err(reason)),
    }
}

fn prepare_ocr(
    input: OcrInputs,
    context: NativeRequestContext,
    callback_adapter: Option<Py<PyAny>>,
    python_context: crate::execution::PythonCallContext<'_>,
) -> PyResult<impl Future<Output = Result<Value, Error>> + Send + 'static> {
    if let Some(reason) = ocr_decline(
        &input.model,
        input.options.provider(""),
        false,
        false,
        false,
        None,
    ) {
        return Err(RustBridgeDeclined::new_err(reason));
    }
    let document = admitted(decode_document(input.document))?;
    let mut optional_params = Map::new();
    for name in parameter_names() {
        if let Some(value) = input
            .optional_params
            .bind(python_context.module.py())
            .get_item(name)?
        {
            optional_params.insert(name.to_owned(), litellm_python_interop::from_py(&value)?);
        }
    }
    let plan = admitted(
        preflight(
            OcrRequest {
                model: &input.model,
                document,
                optional_params,
                options: input.options.into(),
            },
            &|name| std::env::var(name).ok(),
        )
        .map_err(ocr_error_to_pyerr)?,
    )?;
    let runtime = crate::auth::runtime(&python_context.module)?;
    let context: LiteLlmRequestContext = context.into();
    let mut observer = PythonProviderObserver::new(callback_adapter, python_context)?;
    Ok(async move {
        run_route(plan, &context, runtime, &mut observer)
            .await
            .map(|response| response.into_json())
    })
}

#[pyfunction]
#[pyo3(signature = (model, custom_llm_provider, *, stream=false, has_agentic_hook=false, has_custom_client=false, request_format=None))]
fn ocr_decline(
    model: &str,
    custom_llm_provider: &str,
    stream: bool,
    has_agentic_hook: bool,
    has_custom_client: bool,
    request_format: Option<&str>,
) -> Option<String> {
    super::definition::request_decline(
        litellm_core::ocr::ocr_provider_supported(model, custom_llm_provider),
        stream,
        has_agentic_hook,
        has_custom_client,
        request_format,
    )
}

bridge_route! {
    sync = ocr,
    asynchronous = aocr,
    request = OcrInputs,
    prepare = prepare_ocr,
    errors = ocr_error_to_pyerr,
    extra = [ocr_decline],
}
