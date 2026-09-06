use std::future::Future;

use litellm_core::Error;
use litellm_core::messages::types::AnthropicMessagesResponse;
use litellm_python_interop::{Pythonized, from_py, release_gil, to_py};
use pyo3::prelude::*;
use serde_json::Value;

use crate::errors::core_error_to_pyerr;

fn prepare_echo(inputs: EchoInputs) -> PyResult<impl Future<Output = Result<Value, Error>> + Send> {
    Ok(async move { Ok(inputs.body) })
}

bridge_route! {
    sync = echo,
    asynchronous = aecho,
    inputs = EchoInputs,
    required = {
        #[pyo3(from_py_with = litellm_python_interop::from_py)]
        body: serde_json::Value,
    },
    optional = {},
    prepare = prepare_echo,
    errors = core_error_to_pyerr,
    trace = disabled,
}

#[pyclass(frozen)]
struct Payload {
    value: Value,
}

#[pymethods]
impl Payload {
    #[new]
    fn new(#[pyo3(from_py_with = from_py)] value: Value) -> Self {
        Self { value }
    }

    fn encode(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        to_py(py, &self.value)
    }
}

#[pyclass(frozen)]
struct TypedResponse {
    value: AnthropicMessagesResponse,
}

#[pymethods]
impl TypedResponse {
    #[new]
    fn new(#[pyo3(from_py_with = from_py)] value: AnthropicMessagesResponse) -> Self {
        Self { value }
    }

    fn encode<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        Pythonized(&self.value).into_pyobject(py)
    }
}

#[pyfunction]
fn decode(#[pyo3(from_py_with = from_py)] value: Value) -> bool {
    !std::hint::black_box(value).is_null()
}

#[pyfunction]
fn roundtrip<'py>(
    py: Python<'py>,
    #[pyo3(from_py_with = from_py)] value: Value,
) -> PyResult<Bound<'py, PyAny>> {
    Pythonized(value).into_pyobject(py)
}

#[pyfunction]
fn gil_roundtrip(py: Python<'_>) -> bool {
    release_gil(py, || true)
}

pub(super) fn register_namespace(module: &Bound<'_, PyModule>) -> PyResult<()> {
    let benchmarks = PyModule::new(module.py(), "_bench")?;
    register(&benchmarks)?;
    benchmarks.add_class::<Payload>()?;
    benchmarks.add_class::<TypedResponse>()?;
    benchmarks.add_function(wrap_pyfunction!(decode, &benchmarks)?)?;
    benchmarks.add_function(wrap_pyfunction!(roundtrip, &benchmarks)?)?;
    benchmarks.add_function(wrap_pyfunction!(gil_roundtrip, &benchmarks)?)?;
    module.add_submodule(&benchmarks)
}
