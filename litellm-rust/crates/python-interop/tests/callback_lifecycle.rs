use std::process::Command;
use std::time::{Duration, Instant};

use litellm_python_interop::{InvocationMode, InvocationOutcome, PreparedCall};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};
use rstest::{fixture, rstest};
use serial_test::{parallel, serial};

#[path = "support/callback_owner.rs"]
mod callback_owner;

#[path = "support/mod.rs"]
mod support;

use support::python::{InitializedPython, initialized_python, item, run_fixture};

#[test]
fn cold_awaited_adapter_initialization_allows_reentry() -> PyResult<()> {
    let test = "cold_awaited_adapter_initialization_allows_reentry";
    let child_env = "LITELLM_INTEROP_COLD_REENTRY_CHILD";
    if std::env::var(child_env).as_deref() != Ok(test) {
        let mut child = Command::new(std::env::current_exe().unwrap())
            .args(["--exact", test, "--nocapture"])
            .env(child_env, test)
            .spawn()
            .unwrap();
        let deadline = Instant::now() + Duration::from_secs(15);
        loop {
            if let Some(status) = child.try_wait().unwrap() {
                assert!(
                    status.success(),
                    "awaited adapter reentry child failed: {status}"
                );
                return Ok(());
            }
            if Instant::now() >= deadline {
                child.kill().unwrap();
                child.wait().unwrap();
                panic!("awaited adapter reentry did not complete within 15 seconds");
            }
            std::thread::sleep(Duration::from_millis(10));
        }
    }

    let globals = scenario_scope(&initialized_python());
    Python::attach(|py| {
        globals
            .bind(py)
            .get_item("cold_awaited_adapter_reentry")?
            .unwrap()
            .call1((globals.bind(py).get_item("factory")?.unwrap(),))?;
        Ok(())
    })
}

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
#[case::detached_work_after_error("detached_work_after_error")]
#[serial(python_interpreter)]
fn lifecycle_contract(
    scenario_scope: Py<PyDict>,
    #[case] scenario: &str,
    #[values(false, true)] retained: bool,
) -> PyResult<()> {
    run_scenario_fixture(scenario_scope, scenario, retained, None)
}

#[rstest]
#[case::identity_and_ignored_returns("pre_call_identity_and_ignored_returns")]
#[case::mutations_visible_to_later_callbacks("pre_call_mutations_visible_to_later_callbacks")]
#[case::mutation_survives_failure("pre_call_mutation_survives_failure")]
#[ignore = "requires the repository Python environment and LiteLLM on PYTHONPATH"]
#[serial(python_interpreter)]
fn pre_call_contract(
    scenario_scope: Py<PyDict>,
    #[case] scenario: &str,
    #[values(false, true)] retained: bool,
) -> PyResult<()> {
    run_scenario_fixture(
        scenario_scope,
        scenario,
        retained,
        Some((
            include_str!("fixtures/callback_components.py"),
            concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/tests/fixtures/callback_components.py"
            ),
        )),
    )
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
    scenario_scope: Py<PyDict>,
    #[case] scenario: &str,
    #[values(false, true)] retained: bool,
) -> PyResult<()> {
    run_scenario_fixture(
        scenario_scope,
        scenario,
        retained,
        Some((
            include_str!("fixtures/callback_components.py"),
            concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/tests/fixtures/callback_components.py"
            ),
        )),
    )
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
    scenario_scope: Py<PyDict>,
    #[case] scenario: &str,
    #[values(false, true)] retained: bool,
) -> PyResult<()> {
    run_scenario_fixture(
        scenario_scope,
        scenario,
        retained,
        Some((
            include_str!("fixtures/callback_integrations.py"),
            concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/tests/fixtures/callback_integrations.py"
            ),
        )),
    )
}

fn run_scenario_fixture(
    scenario_scope: Py<PyDict>,
    scenario: &str,
    retained: bool,
    fixture: Option<(&str, &str)>,
) -> PyResult<()> {
    Python::attach(|py| {
        let globals = scenario_scope.bind(py);
        if let Some((source, filename)) = fixture {
            run_fixture(py, globals, source, filename)?;
        }
        globals.get_item("run_scenario")?.unwrap().call1((
            scenario,
            retained,
            globals.get_item("factory")?.unwrap(),
        ))?;
        Ok(())
    })
}

#[rstest]
#[case::retained_lifetime("deferred_lifetime", "identity")]
#[case::prepared_ownership("deferred_lifetime", "missing_handoff")]
#[case::externally_owned_retained("borrowed_lifetime", "identity")]
#[case::original_coroutine("direct_coroutine", "identity")]
#[case::passthrough_coroutine("direct_coroutine", "result_passthrough")]
#[serial(python_interpreter)]
fn control_contract(
    scenario_scope: Py<PyDict>,
    #[case] witness: &str,
    #[case] control: &str,
    #[values(false, true)] retained: bool,
    #[values(false, true)] awaited: bool,
) -> PyResult<()> {
    run_control_fixture(scenario_scope, witness, control, retained, awaited)
}

// The `weak` control wraps nothing: it holds only weak references and never
// calls the factory, so the `retained` axis has no effect on it.
#[rstest]
#[case::expired_borrow("deferred_lifetime")]
#[case::externally_owned_borrow("borrowed_lifetime")]
#[serial(python_interpreter)]
fn weak_control(
    scenario_scope: Py<PyDict>,
    #[case] witness: &str,
    #[values(false, true)] awaited: bool,
) -> PyResult<()> {
    run_control_fixture(scenario_scope, witness, "weak", false, awaited)
}

#[rstest]
#[case::retained("identity")]
#[case::missing_handoff("missing_handoff")]
#[serial(python_interpreter)]
fn pending_handoff_control(
    scenario_scope: Py<PyDict>,
    #[case] control: &str,
    #[values(false, true)] retained: bool,
) -> PyResult<()> {
    run_control_fixture(scenario_scope, "pending_handoff", control, retained, true)
}

fn run_control_fixture(
    scenario_scope: Py<PyDict>,
    witness: &str,
    control: &str,
    retained: bool,
    awaited: bool,
) -> PyResult<()> {
    Python::attach(|py| {
        let globals = scenario_scope.bind(py);
        run_fixture(
            py,
            globals,
            include_str!("fixtures/callback_controls.py"),
            concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/tests/fixtures/callback_controls.py"
            ),
        )?;
        globals.get_item("run_control")?.unwrap().call1((
            witness,
            control,
            retained,
            awaited,
            globals.get_item("factory")?.unwrap(),
        ))?;
        Ok(())
    })
}

fn invoke_direct_callback(
    py: Python<'_>,
    globals: &Bound<'_, PyDict>,
    callback: &str,
    argument: &str,
    retained: bool,
) -> PyResult<Py<PyAny>> {
    let args = PyTuple::new(py, [item(globals, argument)])?;
    if !retained {
        let factory = item(globals, "ReferenceFactory").call0()?;
        let owner = factory.call_method1("prepare", (item(globals, callback), args))?;
        let result = owner.call_method0("invoke");
        owner.call_method0("close")?;
        assert_eq!(factory.getattr("live")?.extract::<usize>()?, 0);
        return result.map(Bound::unbind);
    }
    let call = PreparedCall::new(
        InvocationMode::Direct,
        item(globals, callback).unbind(),
        args.unbind(),
        None,
    );
    match call.invoke(py)? {
        InvocationOutcome::Returned(value) => Ok(value),
        InvocationOutcome::Awaitable(_) => panic!("direct callback produced an awaitable outcome"),
    }
}

#[rstest]
#[serial(python_interpreter)]
fn retained_field_survives_replacement_and_observes_original_mutations(
    scenario_scope: Py<PyDict>,
    #[values(false, true)] retained: bool,
) -> PyResult<()> {
    Python::attach(|py| {
        let globals = scenario_scope.bind(py);
        py.run(
            c"
original = {'messages': [{'content': 'original'}]}
replacement = {'messages': [{'content': 'replacement'}]}
event = {'payload': original, 'alias': original}
saved = []

def retain(value):
    saved.append(value['payload'])

def replace(value):
    value['payload'] = replacement
    value['alias']['messages'][0]['content'] = 'mutated original'
",
            Some(globals),
            None,
        )?;
        for callback in ["retain", "replace"] {
            assert!(invoke_direct_callback(py, globals, callback, "event", retained)?.is_none(py));
        }
        let original = item(globals, "original");
        let replacement = item(globals, "replacement");
        let event = item(globals, "event");
        let saved = item(globals, "saved").get_item(0)?;
        assert!(saved.is(&original));
        assert!(event.get_item("alias")?.is(&original));
        assert!(event.get_item("payload")?.is(&replacement));
        assert_eq!(
            saved
                .get_item("messages")?
                .get_item(0)?
                .get_item("content")?
                .extract::<String>()?,
            "mutated original"
        );
        replacement
            .get_item("messages")?
            .get_item(0)?
            .set_item("content", "mutated replacement")?;
        assert_eq!(
            event
                .get_item("payload")?
                .get_item("messages")?
                .get_item(0)?
                .get_item("content")?
                .extract::<String>()?,
            "mutated replacement"
        );
        assert_eq!(
            original
                .get_item("messages")?
                .get_item(0)?
                .get_item("content")?
                .extract::<String>()?,
            "mutated original"
        );
        Ok(())
    })
}

#[rstest]
#[serial(python_interpreter)]
fn queued_graph_outlives_invocation_and_stays_live_until_serialized(
    scenario_scope: Py<PyDict>,
    #[values(false, true)] retained: bool,
) -> PyResult<()> {
    Python::attach(|py| {
        let globals = scenario_scope.bind(py);
        py.run(
            c"
queue = asyncio.Queue()
sentinel = Value()
reference = weakref.ref(sentinel)
payload = {'sentinel': sentinel, 'nested': {'status': 'queued'}}
snapshot = json.dumps(payload['nested'])

def enqueue(value):
    queue.put_nowait(value)
",
            Some(globals),
            None,
        )?;
        assert!(invoke_direct_callback(py, globals, "enqueue", "payload", retained)?.is_none(py));
        py.run(c"del sentinel, payload\ngc.collect()", Some(globals), None)?;
        let reference = item(globals, "reference");
        assert!(!reference.call0()?.is_none());
        let queue = item(globals, "queue");
        let queued = queue.call_method0("get_nowait")?;
        queued
            .get_item("nested")?
            .set_item("status", "changed before flush")?;
        let json = item(globals, "json");
        let flushed = json.call_method1(
            "loads",
            (json.call_method1("dumps", (queued.get_item("nested")?,))?,),
        )?;
        assert_eq!(
            flushed.get_item("status")?.extract::<String>()?,
            "changed before flush"
        );
        let snapshot = json.call_method1("loads", (item(globals, "snapshot"),))?;
        assert_eq!(snapshot.get_item("status")?.extract::<String>()?, "queued");
        assert!(queued.get_item("sentinel")?.is(reference.call0()?));
        queue.call_method0("task_done")?;
        drop(queued);
        py.run(c"gc.collect()", Some(globals), None)?;
        assert!(reference.call0()?.is_none());
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
