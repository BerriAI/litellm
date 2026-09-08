use pyo3::prelude::*;

#[cfg(feature = "panic-test")]
#[pyfunction]
fn _panic_for_test() {
    panic!("intentional PyO3 panic smoke test");
}

pub(crate) fn register(_module: &Bound<'_, PyModule>) -> PyResult<()> {
    #[cfg(feature = "panic-test")]
    _module.add_function(wrap_pyfunction!(_panic_for_test, _module)?)?;
    Ok(())
}
