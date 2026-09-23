use pyo3::{
    prelude::*,
    types::{PyDict, PyTuple},
};

use crate::errors::RustBridgeDeclined;

#[pyfunction]
pub(crate) fn embedding(
    _request: Bound<'_, PyAny>,
    _args: Bound<'_, PyTuple>,
    _kwargs: Bound<'_, PyDict>,
) -> PyResult<Py<PyAny>> {
    Err(RustBridgeDeclined::new_err(
        "native embeddings route is not implemented",
    ))
}

#[pyfunction]
pub(crate) fn aembedding(
    _request: Bound<'_, PyAny>,
    _args: Bound<'_, PyTuple>,
    _kwargs: Bound<'_, PyDict>,
) -> PyResult<Py<PyAny>> {
    Err(RustBridgeDeclined::new_err(
        "native embeddings route is not implemented",
    ))
}

#[cfg(test)]
mod tests {
    use pyo3::{
        prelude::*,
        types::{PyDict, PyTuple},
    };

    use crate::errors::RustBridgeDeclined;

    #[test]
    fn both_entrypoints_decline_before_provider_execution() {
        Python::initialize();
        Python::attach(|py| {
            let request = PyDict::new(py);
            let args = PyTuple::empty(py);
            let kwargs = PyDict::new(py);

            for entrypoint in [super::embedding, super::aembedding] {
                let error = entrypoint(request.clone().into_any(), args.clone(), kwargs.clone())
                    .expect_err("native embeddings must decline until a route machine exists");
                assert!(error.is_instance_of::<RustBridgeDeclined>(py));
            }
        });
    }
}
