use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyCFunction;

pub(super) fn add_function(
    module: &Bound<'_, PyModule>,
    function: Bound<'_, PyCFunction>,
) -> PyResult<()> {
    let name: String = function.getattr("__name__")?.extract()?;
    if module.hasattr(&name)? {
        return Err(PyRuntimeError::new_err(format!(
            "duplicate native route: {name}"
        )));
    }
    module.add_function(function)
}

#[cfg(test)]
#[path = "../../tests/unit/routes/definition.rs"]
mod tests;
