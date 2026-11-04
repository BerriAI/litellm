mod aliases;

use litellm_model_catalog::{SizeProblem, is_model_key, size_problem};
use pyo3::prelude::*;
use pyo3::pybacked::PyBackedBytes;
use pyo3::sync::PyOnceLock;
use pyo3::types::{PyBytes, PyString};

pub(crate) use aliases::{AliasWarning, expand_model_aliases};

/// Why a fetched map has too few models, as main's `_check_model_count_not_reduced` decides
#[pyclass(frozen, module = "litellm.rust_bridge._native")]
pub(crate) enum ModelCountProblem {
    BelowMinimum { actual: usize, minimum: usize },
    Shrunk { actual: usize, reference: usize },
}

/// Top-level keys of `model_cost` that are models. A non-string key is counted, as Python counts it
#[pyfunction]
pub(crate) fn count_model_entries(model_cost: &Bound<'_, PyAny>) -> PyResult<usize> {
    model_cost.try_iter()?.try_fold(0, |count, key| {
        let key = key?;
        let is_model = match key.cast::<PyString>() {
            Ok(key) => is_model_key(&key.to_string_lossy()),
            Err(_) => true,
        };
        Ok(count + usize::from(is_model))
    })
}

/// A negative minimum or reference passes the check that uses it, as it does in Python
#[pyfunction]
pub(crate) fn model_count_problem(
    fetched_map: &Bound<'_, PyAny>,
    backup_model_count: i64,
    min_model_count: i64,
    max_shrink_ratio: f64,
) -> PyResult<Option<ModelCountProblem>> {
    let problem = size_problem(
        count_model_entries(fetched_map)?,
        usize::try_from(backup_model_count).unwrap_or(0),
        usize::try_from(min_model_count).unwrap_or(0),
        max_shrink_ratio,
    );
    Ok(problem.map(|problem| match problem {
        SizeProblem::BelowMinimum { actual, minimum } => {
            ModelCountProblem::BelowMinimum { actual, minimum }
        }
        SizeProblem::Shrunk {
            actual, reference, ..
        } => ModelCountProblem::Shrunk { actual, reference },
    }))
}

#[pyfunction]
pub(crate) fn git_blob_id(body: PyBackedBytes) -> String {
    litellm_model_catalog::revision(&body)
}

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
