use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use pyo3::types::PyBytes;

#[pyfunction]
pub fn bundled_model_cost_map(py: Python<'_>) -> Bound<'_, PyBytes> {
    static BYTES: PyOnceLock<Py<PyBytes>> = PyOnceLock::new();
    BYTES
        .get_or_init(py, || {
            PyBytes::new(py, litellm_model_catalog::BUNDLED_REGISTRY).unbind()
        })
        .bind(py)
        .clone()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[rstest::rstest]
    fn repeated_calls_reuse_the_bundled_registry_bytes() {
        Python::initialize();
        let first = Python::attach(|py| bundled_model_cost_map(py).unbind());
        Python::attach(|py| {
            let second = bundled_model_cost_map(py);
            assert!(first.bind(py).is(&second));
            assert_eq!(second.as_bytes(), litellm_model_catalog::BUNDLED_REGISTRY);
        });
    }
}
