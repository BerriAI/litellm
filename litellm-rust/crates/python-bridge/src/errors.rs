use litellm_core::error::Error;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

pyo3::create_exception!(
    _native,
    RustBridgeDeclined,
    pyo3::exceptions::PyException,
    "Core admission declined without effects, so the host may select its legacy path once."
);

pyo3::create_exception!(
    _native,
    RustHostCallbackError,
    pyo3::exceptions::PyException
);

pyo3::create_exception!(
    _native,
    RustBridgeUnavailable,
    pyo3::exceptions::PyException
);

pyo3::create_exception!(
    _native,
    RustUpstreamError,
    pyo3::exceptions::PyException,
    "The provider call was already issued and failed. Args are (status, message); status is 0 when there was no HTTP response."
);

pub(crate) fn core_error_to_pyerr(err: Error) -> PyErr {
    match err {
        Error::Auth(message) => PyValueError::new_err(message),
        Error::InvalidProvider(_)
        | Error::InvalidRequest(_)
        | Error::InvalidType { .. }
        | Error::MissingField(_)
        | Error::MissingDocumentUrl => PyValueError::new_err(err.to_string()),
        other => PyRuntimeError::new_err(other.to_string()),
    }
}

pub(crate) fn execution_error_to_pyerr(error: Error) -> PyErr {
    match error {
        Error::Http { status, body } => RustUpstreamError::new_err((status, body)),
        Error::Connect(message) | Error::Network(message) | Error::InvalidResponse(message) => {
            RustUpstreamError::new_err((0u16, message))
        }
        other => core_error_to_pyerr(other),
    }
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = module.py();
    module.add(
        "RustBridgeUnavailable",
        py.get_type::<RustBridgeUnavailable>(),
    )?;
    module.add("RustBridgeDeclined", py.get_type::<RustBridgeDeclined>())?;
    module.add(
        "RustHostCallbackError",
        py.get_type::<RustHostCallbackError>(),
    )?;
    module.add("RustUpstreamError", py.get_type::<RustUpstreamError>())
}

pub(crate) fn host_callback_error(py: Python<'_>, error: PyErr) -> PyErr {
    let wrapped = RustHostCallbackError::new_err(error.to_string());
    wrapped.set_context(py, Some(error.clone_ref(py)));
    wrapped.set_cause(py, Some(error));
    wrapped
}

pub(crate) fn terminal_pyerr(error: PyErr) -> PyErr {
    Python::attach(|py| {
        if error.is_instance_of::<RustBridgeDeclined>(py)
            || error.is_instance_of::<RustBridgeUnavailable>(py)
        {
            return host_callback_error(py, error);
        }
        error
    })
}

pub(crate) fn admit(
    result: Result<(), litellm_core::call_lifecycle::admission::AdmissionDecline>,
) -> PyResult<()> {
    result.map_err(|reason| RustBridgeDeclined::new_err(reason.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn execution_failures_never_authorize_fallback() {
        Python::initialize();
        Python::attach(|py| {
            for error in [
                Error::Unsupported("unsupported option"),
                Error::Auth("credential resolution failed".into()),
                Error::InvalidProvider("unknown".into()),
                Error::InvalidRequest("invalid request".into()),
                Error::InvalidType {
                    expected: "object",
                    actual: "string",
                },
                Error::MissingField("model"),
                Error::MissingDocumentUrl,
                Error::MissingApiKey {
                    provider: "anthropic",
                },
                Error::MissingAzureAiCredentials,
                Error::MissingAzureDocumentIntelligenceCredentials,
                Error::MissingReductoApiKey,
                Error::Routing("routing failed".into()),
            ] {
                let expected = core_error_to_pyerr(error.clone());
                let actual = execution_error_to_pyerr(error);
                assert!(!actual.is_instance_of::<RustBridgeDeclined>(py));
                assert!(actual.get_type(py).is(expected.get_type(py)));
                assert_eq!(actual.to_string(), expected.to_string());
            }
        });
    }

    #[test]
    fn upstream_failures_retain_the_status_and_message_contract() {
        Python::initialize();
        Python::attach(|py| {
            for (error, status, message) in [
                (
                    Error::Connect("connection refused".into()),
                    0,
                    "connection refused",
                ),
                (
                    Error::Http {
                        status: 429,
                        body: "rate limited".into(),
                    },
                    429u16,
                    "rate limited",
                ),
                (
                    Error::Network("connection lost".into()),
                    0,
                    "connection lost",
                ),
                (
                    Error::InvalidResponse("invalid JSON".into()),
                    0,
                    "invalid JSON",
                ),
            ] {
                let actual = execution_error_to_pyerr(error);
                assert!(actual.is_instance_of::<RustUpstreamError>(py));
                let args: (u16, String) =
                    actual.value(py).getattr("args").unwrap().extract().unwrap();
                assert_eq!(args, (status, message.to_string()));
            }
        });
    }

    #[test]
    fn terminal_reserved_errors_are_wrapped_with_the_original_cause() {
        Python::initialize();
        Python::attach(|py| {
            for error in [
                RustBridgeDeclined::new_err("callback decline"),
                RustBridgeUnavailable::new_err("callback unavailable"),
            ] {
                let original = error.value(py).clone().unbind();
                let wrapped = terminal_pyerr(error);
                assert!(wrapped.is_instance_of::<RustHostCallbackError>(py));
                let cause = wrapped.value(py).getattr("__cause__").unwrap();
                assert!(cause.is(original.bind(py)));
            }
        });
    }
}
