mod callbacks;
mod document;
mod errors;
mod host;
mod project;

use litellm_core::ocr::{NativeOutcome, OcrAdmission, OcrCall, OcrClient};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};

use self::errors::to_pyerr as ocr_error_to_pyerr;
use self::host::PythonOcrHost;
use crate::errors::RustBridgeDeclined;
use crate::lifecycle::{PythonCallState, Signature, run_call};

const SIGNATURE: Signature = Signature {
    name: "ocr",
    parameters: &[
        "model",
        "document",
        "api_key",
        "api_base",
        "timeout",
        "custom_llm_provider",
        "extra_headers",
    ],
    required: 2,
};

const ASYNC_SIGNATURE: Signature = Signature {
    name: "aocr",
    ..SIGNATURE
};

fn admitted_call(outcome: NativeOutcome<OcrCall>) -> PyResult<OcrCall> {
    match outcome {
        NativeOutcome::Completed(call) => Ok(call),
        NativeOutcome::Declined(reason) => Err(RustBridgeDeclined::new_err(format!(
            "native OCR admission declined: {reason:?}"
        ))),
    }
}

fn call(
    py: Python<'_>,
    args: Bound<'_, PyTuple>,
    kwargs: Option<Bound<'_, PyDict>>,
    asynchronous: bool,
) -> PyResult<Py<PyAny>> {
    let kwargs = kwargs.unwrap_or_else(|| PyDict::new(py));
    let signature = if asynchronous {
        &ASYNC_SIGNATURE
    } else {
        &SIGNATURE
    };
    signature.bind(&args, &kwargs)?;
    let client = OcrClient::shared().map_err(ocr_error_to_pyerr)?;
    let call = admitted_call(OcrCall::admit(
        client,
        OcrAdmission {
            asynchronous,
            ..OcrAdmission::all()
        },
    ))?;
    let host = PythonOcrHost::new(PythonCallState::new(
        py,
        args.unbind(),
        kwargs.copy()?.unbind(),
        asynchronous,
        signature.name,
    )?);
    run_call(py, call, host)
}

#[pyfunction]
#[pyo3(signature = (*args, **kwargs))]
fn ocr(
    py: Python<'_>,
    args: Bound<'_, PyTuple>,
    kwargs: Option<Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    call(py, args, kwargs, false)
}

#[pyfunction]
#[pyo3(signature = (*args, **kwargs))]
fn aocr(
    py: Python<'_>,
    args: Bound<'_, PyTuple>,
    kwargs: Option<Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    call(py, args, kwargs, true)
}

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    super::add_function(module, wrap_pyfunction!(ocr, module)?)?;
    super::add_function(module, wrap_pyfunction!(aocr, module)?)
}

#[cfg(test)]
mod tests {
    use litellm_core::ocr::{Error, OcrDecline};

    use super::*;

    #[test]
    fn typed_initial_decline_uses_bridge_decline_contract() {
        Python::initialize();
        Python::attach(|py| {
            let Err(error) = admitted_call(NativeOutcome::Declined(OcrDecline::HostOperations))
            else {
                panic!("unsupported host operations should decline admission");
            };
            assert!(error.is_instance_of::<RustBridgeDeclined>(py));
        });
    }

    #[test]
    fn post_admission_error_does_not_use_bridge_decline_contract() {
        Python::initialize();
        Python::attach(|py| {
            let error = ocr_error_to_pyerr(Error::InvalidRequest("callback result".into()));
            assert!(!error.is_instance_of::<RustBridgeDeclined>(py));
        });
    }
}
