use pyo3::prelude::*;
use pyo3::types::PyDict;
use rstest::{fixture, rstest};
use serial_test::serial;

use crate::support::Backend;
use crate::support::python::run_fixture;
use crate::support::scenarios::{run_scenario_fixture, scenario_scope};

#[fixture]
fn component_scope(scenario_scope: Py<PyDict>) -> Py<PyDict> {
    Python::attach(|py| {
        run_fixture(
            py,
            scenario_scope.bind(py),
            include_str!("../fixtures/callback_components.py"),
            concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/tests/fixtures/callback_components.py"
            ),
        )
        .unwrap();
    });
    scenario_scope
}

#[fixture]
fn integration_scope(scenario_scope: Py<PyDict>) -> Py<PyDict> {
    Python::attach(|py| {
        run_fixture(
            py,
            scenario_scope.bind(py),
            include_str!("../fixtures/callback_integrations.py"),
            concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/tests/fixtures/callback_integrations.py"
            ),
        )
        .unwrap();
    });
    scenario_scope
}

#[rstest]
#[case::identity_and_ignored_returns("pre_call_identity_and_ignored_returns")]
#[case::mutations_visible_to_later_callbacks("pre_call_mutations_visible_to_later_callbacks")]
#[case::mutation_survives_failure("pre_call_mutation_survives_failure")]
#[ignore = "requires the repository Python environment and LiteLLM on PYTHONPATH"]
#[serial(python_interpreter)]
fn pre_call_contract(
    component_scope: Py<PyDict>,
    #[case] scenario: &str,
    #[values(Backend::Python, Backend::PreparedCall)] backend: Backend,
) -> PyResult<()> {
    run_scenario_fixture(component_scope, scenario, backend)
}

#[rstest]
#[case::real_post_call_logging("real_post_call_logging")]
#[case::real_post_call_dict_response("real_post_call_dict_response")]
#[case::real_sync_logging("real_sync_logging")]
#[case::real_sync_logging_hook_failure("real_sync_logging_hook_failure")]
#[case::real_sync_failure_chain("real_sync_failure_chain")]
#[case::real_async_failure_chain("real_async_failure_chain")]
#[case::real_async_logging("real_async_logging")]
#[case::real_copy_boundaries("real_copy_boundaries")]
#[case::real_logging_worker("real_logging_worker")]
#[case::real_sync_stream_copies("real_sync_stream_copies")]
#[case::real_stream_completion("real_stream_completion")]
#[case::real_stream_close("real_stream_close")]
#[case::real_stream_cancellation("real_stream_cancellation")]
#[ignore = "requires the repository Python environment and LiteLLM on PYTHONPATH"]
#[serial(python_interpreter)]
fn component_contract(
    component_scope: Py<PyDict>,
    #[case] scenario: &str,
    #[values(Backend::Python, Backend::PreparedCall)] backend: Backend,
) -> PyResult<()> {
    run_scenario_fixture(component_scope, scenario, backend)
}

#[rstest]
#[case::real_logging_queue_copy_control("real_logging_queue_copy_control")]
#[case::real_crowdstrike_translator_identity("real_crowdstrike_translator_identity")]
#[case::real_rubrik_block_lifecycle("real_rubrik_block_lifecycle")]
#[case::real_parallel_guardrail_sharing_and_exception_order("real_parallel_guardrail_snapshots")]
#[case::real_purview_sync_background_and_active_loop("real_purview_sync_background")]
#[ignore = "requires the repository Python environment and LiteLLM on PYTHONPATH"]
#[serial(python_interpreter)]
fn integration_contract(
    integration_scope: Py<PyDict>,
    #[case] scenario: &str,
    #[values(Backend::Python, Backend::PreparedCall)] backend: Backend,
) -> PyResult<()> {
    run_scenario_fixture(integration_scope, scenario, backend)
}
