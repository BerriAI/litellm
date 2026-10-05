use pyo3::{prelude::*, types::PyBytes};
use serde::Serialize;

pub(super) fn payload(
    py: Python<'_>,
    response: litellm_http::response::Response<impl Serialize>,
) -> PyResult<Py<PyAny>> {
    let body = litellm_host_python::to_py(py, &response.body)?;
    let hidden = py
        .import("litellm.rust_bridge.transport")?
        .getattr("hidden_params")?
        .call1((
            headers(py, &response.head.headers)?,
            response.head.status.as_u16(),
        ))?;
    body.bind(py)
        .cast::<pyo3::types::PyDict>()?
        .set_item("_hidden_params", hidden)?;
    Ok(body)
}

pub(super) fn headers(py: Python<'_>, headers: &reqwest::header::HeaderMap) -> PyResult<Py<PyAny>> {
    py.import("httpx")?
        .getattr("Headers")?
        .call1((header_pairs(py, headers),))
        .map(Bound::unbind)
}

fn header_pairs<'py>(
    py: Python<'py>,
    headers: &reqwest::header::HeaderMap,
) -> Vec<(Bound<'py, PyBytes>, Bound<'py, PyBytes>)> {
    headers
        .iter()
        .map(|(name, value)| {
            (
                PyBytes::new(py, name.as_str().as_bytes()),
                PyBytes::new(py, value.as_bytes()),
            )
        })
        .collect()
}

pub(crate) fn upstream_error(
    py: Python<'_>,
    status: u16,
    body: String,
    provider_headers: &reqwest::header::HeaderMap,
) -> PyResult<PyErr> {
    let error = crate::errors::RustUpstreamError::new_err((status, body));
    error
        .value(py)
        .setattr("headers", header_pairs(py, provider_headers))?;
    Ok(error)
}
