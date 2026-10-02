use pyo3::{
    prelude::*,
    types::{PyDict, PyTuple},
};

use crate::errors::RustBridgeDeclined;

#[pyfunction]
#[pyo3(signature = (request, args, kwargs))]
pub(crate) fn embedding(
    request: Bound<'_, PyAny>,
    args: Bound<'_, PyTuple>,
    kwargs: Bound<'_, PyDict>,
) -> PyResult<Py<PyAny>> {
    drop((request, args, kwargs));
    Err(RustBridgeDeclined::new_err(
        "native embeddings route is not implemented",
    ))
}

#[pyfunction]
#[pyo3(signature = (request, args, kwargs))]
pub(crate) fn aembedding(
    request: Bound<'_, PyAny>,
    args: Bound<'_, PyTuple>,
    kwargs: Bound<'_, PyDict>,
) -> PyResult<Py<PyAny>> {
    drop((request, args, kwargs));
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
