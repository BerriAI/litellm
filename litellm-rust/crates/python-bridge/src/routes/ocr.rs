use crate::errors::ocr_error_to_pyerr;
use crate::marshal::{NativeRequestContext, NativeRequestOptions, required_value};
use crate::ocr_callbacks::PythonOcrObserver;
use litellm_core::Error;
use litellm_core::ocr::{OcrRequest, ocr_with_observer};
use litellm_core::request_context::LiteLlmRequestContext;
use litellm_core::request_options::RequestOptions;
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
    let provider_admitted = litellm_core::ocr::ocr_admitted(
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
    let document = required_value("document", document, Value::is_object, "dict")?;
    let call_id = context.litellm_call_id.clone();
    let options: RequestOptions = options.into();
    let mut observer = PythonOcrObserver::new(callback_adapter, python_context)?;
    Ok(async move {
        ocr_with_observer(
            OcrRequest {
                model: &input.model,
                document,
                api_key: options.api_key.as_deref(),
                api_base: options.api_base.as_deref(),
                custom_llm_provider: options.custom_llm_provider.as_deref(),
                extra_headers: options.extra_headers,
                optional_params: input.optional_params,
                timeout: options.timeout,
                litellm_call_id: call_id.as_deref(),
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
