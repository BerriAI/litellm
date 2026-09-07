use pyo3::prelude::*;
use pyo3::types::PyDict;
use rstest::{fixture, rstest};
use serial_test::{parallel, serial};

#[path = "support/callback_owner.rs"]
mod callback_owner;

#[path = "support/mod.rs"]
mod support;

use support::python::{InitializedPython, initialized_python, run_fixture};

#[fixture]
fn scenario_scope(initialized_python: &InitializedPython) -> Py<PyDict> {
    let _ = initialized_python;
    Python::attach(|py| {
        let globals = PyDict::new(py);
        globals
            .set_item(
                "factory",
                Py::new(py, callback_owner::OwnerFactory::default()).unwrap(),
            )
            .unwrap();
        run_fixture(
            py,
            &globals,
            include_str!("fixtures/callback_lifecycle.py"),
            concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/tests/fixtures/callback_lifecycle.py"
            ),
        )
        .unwrap();
        globals.unbind()
    })
}

#[rstest]
#[case::awaitable_kinds("awaitable_kinds")]
#[case::identity_and_context("identity_and_context")]
#[case::exceptions("exceptions")]
#[case::exception_ownership("exception_ownership")]
#[case::cancellation_before_start("cancellation_before_start")]
#[case::cancellation_unwinds("cancellation_unwinds")]
#[case::cancellation_during_cleanup("cancellation_during_cleanup")]
#[case::cancellation_suppressed("cancellation_suppressed")]
#[case::registration_and_gc("registration_and_gc")]
#[case::background_and_session("background_and_session")]
#[case::stream_lifecycle("stream_lifecycle")]
#[case::sync_stream_lifecycle("sync_stream_lifecycle")]
#[case::repeated_ownership("repeated_ownership")]
#[case::retained_field_replacement("retained_field_replacement")]
#[case::queued_graph_ownership("queued_graph_ownership")]
#[case::detached_work_after_error("detached_work_after_error")]
#[serial(python_interpreter)]
fn lifecycle_contract(
    scenario_scope: Py<PyDict>,
    #[case] scenario: &str,
    #[values(false, true)] retained: bool,
) -> PyResult<()> {
    Python::attach(|py| {
        scenario_scope
            .bind(py)
            .get_item("run_scenario")?
            .unwrap()
            .call1((
                scenario,
                retained,
                scenario_scope.bind(py).get_item("factory")?.unwrap(),
            ))?;
        Ok(())
    })
}

#[rstest]
#[case::real_async_logging("real_async_logging")]
#[case::real_pre_call_logging("real_pre_call_logging")]
#[case::real_copy_boundaries("real_copy_boundaries")]
#[case::real_logging_worker("real_logging_worker")]
#[case::real_sync_stream_copies("real_sync_stream_copies")]
#[case::real_stream_completion("real_stream_completion")]
#[case::real_stream_close("real_stream_close")]
#[case::real_stream_cancellation("real_stream_cancellation")]
#[ignore = "requires the repository Python environment and LiteLLM on PYTHONPATH"]
#[serial(python_interpreter)]
fn component_contract(
    scenario_scope: Py<PyDict>,
    #[case] scenario: &str,
    #[values(false, true)] retained: bool,
) -> PyResult<()> {
    Python::attach(|py| {
        let globals = scenario_scope.bind(py);
        run_fixture(
            py,
            globals,
            include_str!("fixtures/callback_components.py"),
            concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/tests/fixtures/callback_components.py"
            ),
        )?;
        globals.get_item("run_scenario")?.unwrap().call1((
            scenario,
            retained,
            scenario_scope.bind(py).get_item("factory")?.unwrap(),
        ))?;
        Ok(())
    })
}

#[rstest]
#[case::real_logging_queue_chain("real_logging_queue_chain")]
#[case::real_crowdstrike_translator_identity("real_crowdstrike_translator_identity")]
#[case::real_rubrik_block_lifecycle("real_rubrik_block_lifecycle")]
#[case::real_parallel_guardrail_snapshots("real_parallel_guardrail_snapshots")]
#[case::real_purview_sync_background("real_purview_sync_background")]
#[ignore = "requires the repository Python environment and LiteLLM on PYTHONPATH"]
#[serial(python_interpreter)]
fn integration_contract(
    scenario_scope: Py<PyDict>,
    #[case] scenario: &str,
    #[values(false, true)] retained: bool,
) -> PyResult<()> {
    Python::attach(|py| {
        let globals = scenario_scope.bind(py);
        run_fixture(
            py,
            globals,
            include_str!("fixtures/callback_integrations.py"),
            concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/tests/fixtures/callback_integrations.py"
            ),
        )?;
        globals.get_item("run_scenario")?.unwrap().call1((
            scenario,
            retained,
            globals.get_item("factory")?.unwrap(),
        ))?;
        Ok(())
    })
}

#[rstest]
#[parallel(python_interpreter)]
fn detached_release(initialized_python: &InitializedPython) -> PyResult<()> {
    use litellm_python_interop::{InvocationMode, PreparedCall};
    use pyo3::types::PyTuple;
    let _ = initialized_python;
    let (call, reference) = Python::attach(|py| -> PyResult<_> {
        let globals = PyDict::new(py);
        py.run(
            c"import weakref\nclass Value: pass\nvalue = Value()\nreference = weakref.ref(value)",
            Some(&globals),
            None,
        )?;
        let value = globals.get_item("value")?.unwrap();
        let call = PreparedCall::new(
            InvocationMode::Direct,
            py.None(),
            PyTuple::new(py, [value])?.unbind(),
            None,
        );
        let reference = globals.get_item("reference")?.unwrap().unbind();
        globals.del_item("value")?;
        Ok((call, reference))
    })?;
    drop(call);
    Python::attach(|py| {
        assert!(reference.call0(py)?.is_none(py));
        Ok(())
    })
}

#[rstest]
#[case::direct(false)]
#[case::awaited(true)]
#[parallel(python_interpreter)]
fn outcome_identifies_binding(scenario_scope: Py<PyDict>, #[case] awaited: bool) -> PyResult<()> {
    use litellm_python_interop::{InvocationMode, InvocationOutcome, PreparedCall};
    use pyo3::types::PyTuple;
    Python::attach(|py| {
        let callback = py.eval(c"lambda: None", Some(scenario_scope.bind(py)), None)?;
        let call = PreparedCall::new(
            if awaited {
                InvocationMode::Await
            } else {
                InvocationMode::Direct
            },
            callback.unbind(),
            PyTuple::empty(py).unbind(),
            None,
        );
        match call.invoke(py)? {
            InvocationOutcome::Returned(value) => {
                assert!(!awaited);
                assert!(value.is_none(py));
            }
            InvocationOutcome::Awaitable(value) => {
                assert!(awaited);
                value.call_method0(py, "close")?;
            }
        }
        Ok(())
    })
}

#[rstest]
#[parallel(python_interpreter)]
fn awaited_raise_surfaces_when_driven(scenario_scope: Py<PyDict>) -> PyResult<()> {
    use litellm_python_interop::{InvocationMode, InvocationOutcome, PreparedCall};
    use pyo3::types::PyTuple;
    Python::attach(|py| {
        let globals = scenario_scope.bind(py);
        py.run(
            c"events = []\nerror = ValueError('await failure')\nasync def callback():\n    events.append('started')\n    raise error\ndef drive(coroutine):\n    try:\n        coroutine.send(None)\n        return False\n    except BaseException as caught:\n        return caught is error\n",
            Some(globals),
            None,
        )?;
        let started = || -> PyResult<usize> { globals.get_item("events")?.unwrap().len() };
        let call = PreparedCall::new(
            InvocationMode::Await,
            globals.get_item("callback")?.unwrap().unbind(),
            PyTuple::empty(py).unbind(),
            None,
        );
        let pending = match call.invoke(py)? {
            InvocationOutcome::Awaitable(value) => value,
            InvocationOutcome::Returned(_) => panic!("await binding produced a settled outcome"),
        };
        assert_eq!(started()?, 0);
        assert!(
            globals
                .get_item("drive")?
                .unwrap()
                .call1((pending,))?
                .extract::<bool>()?
        );
        assert_eq!(started()?, 1);
        Ok(())
    })
}
