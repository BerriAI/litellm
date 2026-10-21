use super::operations::{PythonMutationError, PythonMutationResponse};
use litellm_host_python::{json_loads, to_py};
use litellm_secrets::cyberark;
use pyo3::{exceptions::PyValueError, prelude::*, types::PyDict};

pub(super) fn mutation_value(
    result: Result<PythonMutationResponse, PythonMutationError>,
    context: &super::vault::ErrorContext,
) -> PyResult<Py<PyAny>> {
    Python::attach(|py| match result {
        Ok(PythonMutationResponse::Value(value)) => to_py(py, &value),
        Ok(PythonMutationResponse::Json(body)) => match json_value(py, &body) {
            Ok(value) => Ok(value),
            Err(error) => error_value(py, error.value(py).str()?.extract()?),
        },
        Err(PythonMutationError::Vault(failure)) => {
            super::vault::failure_value(py, *failure, context)
        }
        Err(PythonMutationError::CyberarkWrite { name, failure }) => {
            let message = cyberark_failure(py, &name, *failure)?;
            to_py(
                py,
                &serde_json::json!({"status": "error", "message": message}),
            )
        }
        Err(PythonMutationError::CurrentMissing(name)) => Err(PyValueError::new_err(format!(
            "Current secret {name} not found"
        ))),
        Err(PythonMutationError::ReplacementMissing(name)) => Err(PyValueError::new_err(format!(
            "Failed to verify new secret {name}"
        ))),
        Err(PythonMutationError::ReplacementMismatch) => {
            Err(PyValueError::new_err("New secret value mismatch"))
        }
        Err(PythonMutationError::Unsupported) => Err(PyValueError::new_err(
            "native secret manager mutation is unavailable",
        )),
    })
}

fn cyberark_failure(
    py: Python<'_>,
    name: &str,
    failure: cyberark::WriteFailure,
) -> PyResult<String> {
    let message = match failure.source {
        cyberark::Error::Status(status) | cyberark::Error::AuthStatus(status) => {
            let url = failure
                .request_url
                .as_ref()
                .map_or("", reqwest::Url::as_str);
            http_message(py, "POST", url, status)?
        }
        cyberark::Error::Operation(litellm_secrets_types::Error::UnsafeSecretName) => {
            format!("Invalid secret_name {}", name.into_pyobject(py)?.repr()?)
        }
        cyberark::Error::Http(source) if failure.authentication => match os_error_code(&source) {
            Some(code) => {
                let reason = py.import("os")?.getattr("strerror")?.call1((code,))?;
                py.import("builtins")?
                    .getattr("OSError")?
                    .call1((code, reason))?
                    .str()?
                    .extract()?
            }
            None => cyberark::Error::Http(source).to_string(),
        },
        cyberark::Error::Http(source) if source.is_connect() => {
            "All connection attempts failed".to_owned()
        }
        source => source.to_string(),
    };
    Ok(if failure.authentication {
        format!("Could not authenticate to CyberArk Conjur: {message}")
    } else {
        message
    })
}

fn os_error_code(error: &(dyn std::error::Error + 'static)) -> Option<i32> {
    error
        .downcast_ref::<std::io::Error>()
        .and_then(std::io::Error::raw_os_error)
        .or_else(|| error.source().and_then(os_error_code))
}

pub(super) fn json_value(py: Python<'_>, body: &[u8]) -> PyResult<Py<PyAny>> {
    json_loads(py, body)
}

pub(super) fn error_value(py: Python<'_>, message: String) -> PyResult<Py<PyAny>> {
    to_py(
        py,
        &serde_json::json!({"status": "error", "message": message}),
    )
}

pub(super) fn http_message(
    py: Python<'_>,
    method: &str,
    url: &str,
    status: u16,
) -> PyResult<String> {
    let httpx = py.import("httpx")?;
    let request = httpx.getattr("Request")?.call1((method, url))?;
    let kwargs = PyDict::new(py);
    kwargs.set_item("request", request)?;
    let response = httpx.getattr("Response")?.call((status,), Some(&kwargs))?;
    match response.call_method0("raise_for_status") {
        Err(error) => error.value(py).str()?.extract(),
        Ok(_) => Ok(format!("HTTP {status}")),
    }
}
