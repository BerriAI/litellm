use pyo3::prelude::*;
use pyo3::types::PyDict;
use rstest::fixture;

use super::python::{InitializedPython, initialized_python, run_fixture};
use super::{Backend, callback_owner};

#[fixture]
pub fn scenario_scope(initialized_python: &InitializedPython) -> Py<PyDict> {
    initialized_python.attach(|py| {
        let globals = PyDict::new(py);
        globals
            .set_item(
                "factory",
                Py::new(py, callback_owner::OwnerFactory::default()).unwrap(),
            )
            .unwrap();
        globals
            .set_item(
                "AWAIT_ADAPTER_FILENAME",
                litellm_python_interop::AWAIT_ADAPTER_FILENAME
                    .to_str()
                    .unwrap(),
            )
            .unwrap();
        run_fixture(
            py,
            &globals,
            include_str!("../fixtures/callback_lifecycle.py"),
            concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/tests/fixtures/callback_lifecycle.py"
            ),
        )
        .unwrap();
        globals.unbind()
    })
}

pub fn run_scenario_fixture(
    scenario_scope: Py<PyDict>,
    scenario: &str,
    backend: Backend,
) -> PyResult<()> {
    Python::attach(|py| {
        let globals = scenario_scope.bind(py);
        globals.get_item("run_scenario")?.unwrap().call1((
            scenario,
            matches!(backend, Backend::PreparedCall),
            globals.get_item("factory")?.unwrap(),
        ))?;
        Ok(())
    })
}
