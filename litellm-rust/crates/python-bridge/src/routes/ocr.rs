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
    document: Py<PyAny>,
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
    let provider_admitted = litellm_ai_gateway::io::ocr::ocr_admitted(
        &input.model,
        options.provider("mistral"),
        context.capabilities.request_format.as_deref(),
    );
    if let litellm_core::native_outcome::NativeOutcome::Declined(decline) =
        super::definition::admission(provider_admitted, &context)
    {
        return Err(crate::errors::RustBridgeDeclined::new_err(decline.reason()));
    }
    let py = python_context.py;
    let document = if input
        .document
        .bind(py)
        .get_item("type")
        .and_then(|value| value.extract::<String>())
        .is_ok_and(|kind| kind == "file")
    {
        py.import("litellm.ocr.main")?
            .getattr("convert_file_document_to_url_document")?
            .call1((input.document.bind(py),))?
            .unbind()
    } else {
        input.document
    };
    let document: Value = litellm_python_interop::from_py(document.bind(py))?;
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

bridge_route! {
    sync = ocr,
    asynchronous = aocr,
    request = OcrInputs,
    prepare = prepare_ocr,
    errors = ocr_error_to_pyerr,
}
