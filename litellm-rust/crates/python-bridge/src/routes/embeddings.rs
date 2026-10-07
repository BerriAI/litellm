use pyo3::prelude::*;

use crate::errors::RustBridgeDeclined;

#[pyfunction]
pub(crate) fn embedding(call: Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    drop(super::NativeCall::extract(&call)?);
    Err(RustBridgeDeclined::new_err(
        "native embeddings route is not implemented",
    ))
}

#[pyfunction]
pub(crate) fn aembedding(call: Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    embedding(call)
}

#[cfg(test)]
mod tests {
    use pyo3::{prelude::*, types::PyDict};
    use rstest::rstest;

    use crate::errors::RustBridgeDeclined;

    #[rstest]
    #[case::sync(false)]
    #[case::asynchronous(true)]
    fn both_entrypoints_decline_before_provider_execution(#[case] asynchronous: bool) {
        Python::initialize();
        Python::attach(|py| {
            let locals = PyDict::new(py);
            py.run(
                c"from types import SimpleNamespace
call = SimpleNamespace(args=(), kwargs={}, bound={'model':'test-model','input':'hello'})",
                Some(&locals),
                Some(&locals),
            )
            .unwrap();
            let call = locals.get_item("call").unwrap().unwrap();
            let error = if asynchronous {
                super::aembedding(call)
            } else {
                super::embedding(call)
            }
            .expect_err("native embeddings must decline until a route machine exists");
            assert!(error.is_instance_of::<RustBridgeDeclined>(py));
        });
    }
}
