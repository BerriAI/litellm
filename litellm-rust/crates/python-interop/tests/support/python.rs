#![allow(dead_code)]

use std::ffi::CStr;

use pyo3::prelude::*;
use pyo3::types::PyDict;
use rstest::fixture;

pub struct InitializedPython;

impl InitializedPython {
    pub fn attach<F, R>(&self, f: F) -> R
    where
        F: for<'py> FnOnce(Python<'py>) -> R,
    {
        Python::attach(f)
    }
}

#[fixture]
#[once]
pub fn initialized_python() -> InitializedPython {
    Python::initialize();
    InitializedPython
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

pub fn scope<'py>(py: Python<'py>, source: &CStr) -> PyResult<Bound<'py, PyDict>> {
    let globals = PyDict::new(py);
    py.run(source, Some(&globals), None)?;
    Ok(globals)
}

pub fn item<'py>(globals: &Bound<'py, PyDict>, name: &str) -> Bound<'py, PyAny> {
    globals.get_item(name).unwrap().unwrap()
}
