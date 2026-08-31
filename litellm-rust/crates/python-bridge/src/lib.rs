mod diagnostics;
mod errors;
mod marshal;
mod routes;

use std::panic::{AssertUnwindSafe, catch_unwind};

use litellm_ai_gateway::io::responses_ws::ResponsesWebSocketConnection as RustResponsesWebSocketConnection;
use litellm_python_interop::panic_to_pyerr;
use pyo3::prelude::*;
use pyo3::types::PyAny;

use crate::errors::core_error_to_pyerr;
use crate::marshal::{marshal_headers, optional_timeout};

#[pyclass]
struct ResponsesWebSocketConnection {
    inner: RustResponsesWebSocketConnection,
}

struct NewResponsesWebSocketConnection(ResponsesWebSocketConnection);

impl<'py> IntoPyObject<'py> for NewResponsesWebSocketConnection {
    type Target = ResponsesWebSocketConnection;
    type Output = Bound<'py, ResponsesWebSocketConnection>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> PyResult<Self::Output> {
        catch_unwind(AssertUnwindSafe(|| Py::new(py, self.0)))
            .map_err(panic_to_pyerr)?
            .map(|value| value.into_bound(py))
    }
}

#[pymethods]
impl ResponsesWebSocketConnection {
    #[classmethod]
    #[pyo3(signature = (url, headers=None, timeout_seconds=None))]
    fn connect<'py>(
        _cls: &Bound<'py, pyo3::types::PyType>,
        py: Python<'py>,
        url: String,
        headers: Option<Py<PyAny>>,
        timeout_seconds: Option<f64>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let headers = marshal_headers(py, headers)?;
        let timeout = optional_timeout(timeout_seconds);
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let inner = RustResponsesWebSocketConnection::connect_url(&url, &headers, timeout)
                .await
                .map_err(core_error_to_pyerr)?;
            Ok(NewResponsesWebSocketConnection(
                ResponsesWebSocketConnection { inner },
            ))
        })
    }

    fn send_text<'py>(&self, py: Python<'py>, text: String) -> PyResult<Bound<'py, PyAny>> {
        let inner = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            inner.send_text(text).await.map_err(core_error_to_pyerr)
        })
    }

    fn recv_text<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let inner = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            inner.recv_text().await.map_err(core_error_to_pyerr)
        })
    }

    fn close<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let inner = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            inner.close().await.map_err(core_error_to_pyerr)
        })
    }
}

#[pymodule]
fn _native(module: &Bound<'_, PyModule>) -> PyResult<()> {
    errors::register(module)?;
    routes::register(module)?;
    module.add_class::<ResponsesWebSocketConnection>()?;
    diagnostics::register(module)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn module_registration_preserves_the_public_surface() {
        Python::initialize();
        Python::attach(|py| {
            let module = PyModule::new(py, "_native").expect("module should be created");
            _native(&module).expect("module should register");

            let expected = [
                "RustBridgeDeclined",
                "RustUpstreamError",
                "ocr",
                "aocr",
                "transcription",
                "atranscription",
                "messages",
                "amessages",
                "chat_completions_decline",
                "chat_completions",
                "achat_completions",
                "ResponsesWebSocketConnection",
                "gil_stats",
            ];

            let public_names: Vec<String> = module
                .dict()
                .keys()
                .extract::<Vec<String>>()
                .expect("module names should be strings")
                .into_iter()
                .filter(|name| !name.starts_with("__"))
                .collect();
            assert_eq!(public_names, expected);
        });
    }
}
