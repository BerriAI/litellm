use pyo3::prelude::*;
use rstest::rstest;
use serial_test::serial;

#[path = "support/mod.rs"]
mod support;

use support::native::{native_globals, run_fixture};

#[rstest]
#[case::differential_callbacks_wire("differential_callbacks_wire")]
#[case::negative_control_copied_caller_document("negative_control_copied_caller_document")]
#[case::negative_control_rebound_logging_body("negative_control_rebound_logging_body")]
#[case::negative_control_rebound_logging_headers("negative_control_rebound_logging_headers")]
#[case::differential_callback_retained_mutation_after_post_received(
    "differential_callback_retained_mutation_after_post_received"
)]
#[case::public_rust_dispatch_wire_fallback_and_escaping_base_exception(
    "public_rust_dispatch_wire_fallback_and_escaping_base_exception"
)]
#[case::collection_after_success_and_failures("collection_after_success_and_failures")]
#[case::callback_retained_graph_remains_usable_then_collects(
    "callback_retained_graph_remains_usable_then_collects"
)]
#[case::collection_after_cancellation_during_blocked_transport(
    "collection_after_cancellation_during_blocked_transport"
)]
#[ignore = "requires repo Python"]
#[serial(python_interpreter)]
fn retained_real_production_boundary_differential_and_lifecycle(
    #[case] scenario: &str,
) -> PyResult<()> {
    Python::initialize();
    Python::attach(|py| {
        let globals = native_globals(py)?;
        run_fixture(
            py,
            &globals,
            include_str!("fixtures/ocr_retained.py"),
            concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/tests/fixtures/ocr_retained.py"
            ),
        )?;
        let case = globals
            .get_item("RealBoundaryTests")?
            .unwrap()
            .call1((format!("test_{scenario}"),))?;
        let outcome = case.call_method0("debug");
        let cleanup = case.call_method0("doCleanups");
        outcome?;
        cleanup?;
        Ok(())
    })
}
