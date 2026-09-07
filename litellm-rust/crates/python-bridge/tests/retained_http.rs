use pyo3::prelude::*;
use rstest::rstest;
use serial_test::serial;

#[path = "support/mod.rs"]
mod support;

use support::native::{native_globals, run_fixture};

#[test]
#[serial(python_interpreter)]
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
#[serial(python_interpreter)]
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

#[rstest]
#[case::original_retain_mutate("retain_mutate", "original")]
#[case::original_caller_closure("caller_closure", "original")]
#[case::original_delayed_after_prepare_before_encode_read(
    "delayed_after_prepare_before_encode_read",
    "original"
)]
#[case::original_field_replace("field_replace", "original")]
#[case::original_mutate_then_caught_error_observe("mutate_then_caught_error_observe", "original")]
#[case::copy_caller_inputs_retain_mutate("retain_mutate", "copy_caller_inputs_before_prepare")]
#[case::copy_caller_inputs_caller_closure("caller_closure", "copy_caller_inputs_before_prepare")]
#[case::copy_roots_retain_mutate("retain_mutate", "copy_roots_after_prepare")]
#[case::copy_roots_delayed_after_prepare_before_encode_read(
    "delayed_after_prepare_before_encode_read",
    "copy_roots_after_prepare"
)]
#[case::copy_roots_mutate_then_caught_error_observe(
    "mutate_then_caught_error_observe",
    "copy_roots_after_prepare"
)]
#[case::encode_logging_replacements_field_replace("field_replace", "encode_logging_replacements")]
#[case::reconstruct_logging_envelope_retain_mutate("retain_mutate", "reconstruct_logging_envelope")]
#[case::reconstruct_logging_envelope_field_replace("field_replace", "reconstruct_logging_envelope")]
#[case::reconstruct_root_tuple_retain_mutate("retain_mutate", "reconstruct_root_tuple")]
#[case::reconstruct_root_tuple_delayed_after_prepare_before_encode_read(
    "delayed_after_prepare_before_encode_read",
    "reconstruct_root_tuple"
)]
#[case::reconstruct_root_tuple_field_replace("field_replace", "reconstruct_root_tuple")]
#[serial(python_interpreter)]
fn retained_production_transport_comparison_matrix(
    #[case] scenario: &str,
    #[case] variant: &str,
) -> PyResult<()> {
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
            .get_item("run_comparison_contract")?
            .unwrap()
            .call1((scenario, variant))?;
        Ok(())
    })
}
