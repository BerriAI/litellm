use pyo3::prelude::*;

use super::NativeCall;
use crate::errors::RustBridgeDeclined;

#[pyfunction]
pub(crate) fn embedding(call: NativeCall<'_>) -> PyResult<Py<PyAny>> {
    drop(call);
    Err(RustBridgeDeclined::new_err(
        "native embeddings route is not implemented",
    ))
}

#[pyfunction]
pub(crate) fn aembedding(call: NativeCall<'_>) -> PyResult<Py<PyAny>> {
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
call = SimpleNamespace(args=(), kwargs={'model':'test-model','input':'hello'}, base={})",
                Some(&locals),
                Some(&locals),
            )
            .unwrap();
            let call: super::NativeCall<'_> =
                locals.get_item("call").unwrap().unwrap().extract().unwrap();
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
