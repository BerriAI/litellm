use pyo3::prelude::*;

#[path = "support/mod.rs"]
mod support;

use support::native::{native_globals, run_fixture};

#[test]
fn retained_routes_preserve_callbacks_context_wire_and_ownership() -> PyResult<()> {
    Python::initialize();
    Python::attach(|py| {
        let globals = native_globals(py)?;
        run_fixture(
            py,
            &globals,
            include_str!("fixtures/retained_http_contract.py"),
            concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/tests/fixtures/retained_http_contract.py"
            ),
        )?;
        globals
            .get_item("run_ownership_contract")?
            .unwrap()
            .call0()?;
        Ok(())
    })
}

#[test]
fn retained_routes_surface_transport_failures_as_runtime_error() -> PyResult<()> {
    Python::initialize();
    Python::attach(|py| {
        let globals = native_globals(py)?;
        run_fixture(
            py,
            &globals,
            include_str!("fixtures/retained_http_contract.py"),
            concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/tests/fixtures/retained_http_contract.py"
            ),
        )?;
        globals.get_item("run_error_contract")?.unwrap().call0()?;
        Ok(())
    })
}
