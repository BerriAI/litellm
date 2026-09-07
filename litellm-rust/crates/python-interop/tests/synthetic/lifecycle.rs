use std::io::Read;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use litellm_python_interop::{InvocationMode, InvocationOutcome, PreparedCall};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};
use rstest::rstest;
use serial_test::{parallel, serial};

use crate::support::Backend;
use crate::support::python::{InitializedPython, initialized_python, item, run_fixture};
use crate::support::scenarios::{run_scenario_fixture, scenario_scope};

#[test]
fn cold_awaited_adapter_initialization_allows_reentry() -> PyResult<()> {
    let test = concat!(
        module_path!(),
        "::cold_awaited_adapter_initialization_allows_reentry"
    )
    .split_once("::")
    .unwrap()
    .1;
    let child_env = "LITELLM_INTEROP_COLD_REENTRY_CHILD";
    if std::env::var(child_env).as_deref() != Ok(test) {
        let mut child = Command::new(std::env::current_exe().unwrap())
            .args(["--exact", test, "--nocapture"])
            .env(child_env, test)
            .stdout(Stdio::piped())
            .spawn()
            .unwrap();
        let deadline = Instant::now() + Duration::from_secs(15);
        loop {
            if let Some(status) = child.try_wait().unwrap() {
                let mut output = String::new();
                child
                    .stdout
                    .take()
                    .unwrap()
                    .read_to_string(&mut output)
                    .unwrap();
                assert!(
                    status.success(),
                    "awaited adapter reentry child failed: {status}\n{output}"
                );
                assert!(
                    output.contains("test result: ok. 1 passed; 0 failed; 0 ignored;"),
                    "awaited adapter reentry child did not run exactly one test:\n{output}"
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
    #[values(Backend::Python, Backend::PreparedCall)] backend: Backend,
) -> PyResult<()> {
    run_scenario_fixture(scenario_scope, scenario, backend)
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
    #[values(Backend::Python, Backend::PreparedCall)] backend: Backend,
    #[values(false, true)] awaited: bool,
) -> PyResult<()> {
    run_control_fixture(scenario_scope, witness, control, backend, awaited)
}

#[rstest]
#[case::expired_borrow("deferred_lifetime")]
#[case::externally_owned_borrow("borrowed_lifetime")]
#[serial(python_interpreter)]
fn weak_control(
    scenario_scope: Py<PyDict>,
    #[case] witness: &str,
    #[values(false, true)] awaited: bool,
) -> PyResult<()> {
    run_control_fixture(scenario_scope, witness, "weak", Backend::Python, awaited)
}

#[rstest]
#[case::retained("identity")]
#[case::missing_handoff("missing_handoff")]
#[serial(python_interpreter)]
fn pending_handoff_control(
    scenario_scope: Py<PyDict>,
    #[case] control: &str,
    #[values(Backend::Python, Backend::PreparedCall)] backend: Backend,
) -> PyResult<()> {
    run_control_fixture(scenario_scope, "pending_handoff", control, backend, true)
}

fn run_control_fixture(
    scenario_scope: Py<PyDict>,
    witness: &str,
    control: &str,
    backend: Backend,
    awaited: bool,
) -> PyResult<()> {
    Python::attach(|py| {
        let globals = scenario_scope.bind(py);
        run_fixture(
            py,
            globals,
            include_str!("../fixtures/callback_controls.py"),
            concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/tests/fixtures/callback_controls.py"
            ),
        )?;
        globals.get_item("run_control")?.unwrap().call1((
            witness,
            control,
            matches!(backend, Backend::PreparedCall),
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
    backend: Backend,
) -> PyResult<Py<PyAny>> {
    let args = PyTuple::new(py, [item(globals, argument)])?;
    if matches!(backend, Backend::Python) {
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
    #[values(Backend::Python, Backend::PreparedCall)] backend: Backend,
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
            assert!(invoke_direct_callback(py, globals, callback, "event", backend)?.is_none(py));
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
    #[values(Backend::Python, Backend::PreparedCall)] backend: Backend,
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
        assert!(invoke_direct_callback(py, globals, "enqueue", "payload", backend)?.is_none(py));
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
