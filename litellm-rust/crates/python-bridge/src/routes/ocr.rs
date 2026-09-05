use crate::errors::ocr_error_to_pyerr;
use crate::marshal::{NativeRequestContext, NativeRequestOptions};
use litellm_ai_gateway::integrations::types::RequestHooks;
use litellm_ai_gateway::io::ocr::OcrRequest;
use litellm_ai_gateway::io::ocr::ocr as run_route;
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
) -> PyResult<impl Future<Output = Result<Value, Error>> + Send + 'static> {
    let context: LiteLlmRequestContext = context.into();
    let document = input.document;
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
