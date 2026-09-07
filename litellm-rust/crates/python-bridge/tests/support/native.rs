use pyo3::prelude::*;
use pyo3::types::PyDict;

pub fn native_globals(py: Python<'_>) -> PyResult<Bound<'_, PyDict>> {
    let module = pyo3::wrap_pymodule!(_native::_native)(py).into_bound(py);
    let globals = PyDict::new(py);
    globals.set_item("native", &module)?;
    Ok(globals)
}

pub fn run_fixture(
    py: Python<'_>,
    globals: &Bound<'_, PyDict>,
    source: &str,
    filename: &str,
) -> PyResult<()> {
    let builtins = py.import("builtins")?;
    let code = builtins.call_method1("compile", (source, filename, "exec"))?;
    builtins.call_method1("exec", (code, globals))?;
    Ok(())
}
