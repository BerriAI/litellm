use std::future::Future;

use crate::errors::ocr_error_to_pyerr;
use litellm_core::Error;
use litellm_core::ocr::wire::{OcrWireRequest, decode_request};
use pyo3::prelude::*;
use serde_json::Value;

fn prepare_ocr(
    inputs: OcrInputs,
) -> PyResult<impl Future<Output = Result<Value, Error>> + Send + 'static> {
    let client = crate::transport::ocr_client().map_err(ocr_error_to_pyerr)?;
    let wire: OcrWireRequest =
        litellm_core::ocr::wire::decode_request_value(inputs.request, "request")
            .map_err(|error| ocr_error_to_pyerr(error.into()))?;
    let mut request = decode_request(wire).map_err(ocr_error_to_pyerr)?;
    request.connection.max_download_bytes = litellm_core::ocr::types::download_limit_bytes(
        std::env::var("MAX_IMAGE_URL_DOWNLOAD_SIZE_MB")
            .ok()
            .as_deref(),
    );
    Ok(async move {
        client
            .perform(request)
            .await
            .map(|response| response.into_json())
    })
}

bridge_route! {
    sync = ocr,
    asynchronous = aocr,
    inputs = OcrInputs,
    required = {
        #[pyo3(from_py_with = litellm_python_interop::from_py)]
        request: serde_json::Value,
    },
    optional = {},
    prepare = prepare_ocr,
    errors = ocr_error_to_pyerr,
}
