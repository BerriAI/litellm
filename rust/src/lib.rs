use pyo3::{
    Bound, PyResult,
    types::{PyModule, PyModuleMethods},
    wrap_pyfunction,
};

mod pattern_prefilter;
use pattern_prefilter::build_pattern_prefilter;

// Function name must match Cargo.toml's [lib] name and pyproject.toml's
// module-name: pyo3 derives the Python module name from it.
#[pyo3::pymodule]
fn litellm_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(build_pattern_prefilter, m)?)?;
    Ok(())
}
