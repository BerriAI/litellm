use litellm_python_interop::{InvocationMode, InvocationOutcome, PreparedCall};
use pyo3::exceptions::{PyAssertionError, PyKeyboardInterrupt, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};
use rstest::rstest;

#[path = "support/mod.rs"]
mod support;

use support::python::{InitializedPython, initialized_python, item, run_fixture, scope};

#[rstest]
fn retains_aliases_mutations_and_original_result(
    initialized_python: &InitializedPython,
) -> PyResult<()> {
    let _ = initialized_python;
    Python::attach(|py| {
        let globals = scope(
            py,
            c"
def callback(data, *, alias):
    observed.append(data['nested'] is alias)
    saved.append(data)
    alias['value'] = 'during'
    return data
",
        )?;
        let shared = PyDict::new(py);
        shared.set_item("value", "before")?;
        let payload = PyDict::new(py);
        payload.set_item("nested", &shared)?;
        let saved = PyList::empty(py);
        let observed = PyList::empty(py);
        globals.set_item("saved", &saved)?;
        globals.set_item("observed", &observed)?;
        let keywords = PyDict::new(py);
        keywords.set_item("alias", &shared)?;
        let invocation = PreparedCall::new(
            InvocationMode::Direct,
            item(&globals, "callback").unbind(),
            PyTuple::new(py, [&payload])?.unbind(),
            Some(keywords.unbind()),
        );
        let result = invoke_direct(&invocation, py)?;
        assert!(result.bind(py).is(&payload));
        drop(invocation);
        assert_eq!(observed.extract::<Vec<bool>>()?, [true]);
        assert!(saved.get_item(0)?.is(&payload));
        assert_eq!(item(&shared, "value").extract::<String>()?, "during");
        shared.set_item("value", "after")?;
        assert_eq!(
            saved
                .get_item(0)?
                .get_item("nested")?
                .get_item("value")?
                .extract::<String>()?,
            "after"
        );
        Ok(())
    })
}

#[rstest]
fn preserves_exception_identity_cause_traceback_and_prior_mutation(
    initialized_python: &InitializedPython,
) -> PyResult<()> {
    let _ = initialized_python;
    Python::attach(|py| {
        let globals = scope(
            py,
            c"
def callback(data):
    data['changed'] = True
    raise error from cause
",
        )?;
        let payload = PyDict::new(py);
        let original = PyKeyboardInterrupt::new_err("original");
        let cause = PyValueError::new_err("cause");
        globals.set_item("error", original.value(py))?;
        globals.set_item("cause", cause.value(py))?;
        let invocation = PreparedCall::new(
            InvocationMode::Direct,
            item(&globals, "callback").unbind(),
            PyTuple::new(py, [&payload])?.unbind(),
            None,
        );
        let error = invoke_direct(&invocation, py).unwrap_err();
        assert!(error.value(py).is(original.value(py)));
        drop(invocation);
        assert!(item(&payload, "changed").extract::<bool>()?);
        assert!(error.value(py).getattr("__cause__")?.is(cause.value(py)));
        let frames = py
            .import("traceback")?
            .call_method1("extract_tb", (error.traceback(py),))?;
        assert_eq!(
            frames
                .get_item(frames.len()? - 1)?
                .getattr("name")?
                .extract::<String>()?,
            "callback"
        );
        Ok(())
    })
}

#[rstest]
fn returns_coroutine_without_executing_it(initialized_python: &InitializedPython) -> PyResult<()> {
    let _ = initialized_python;
    Python::attach(|py| {
        let globals = scope(
            py,
            c"
async def work():
    started.append(True)
def callback():
    return coroutine
",
        )?;
        let started = PyList::empty(py);
        globals.set_item("started", &started)?;
        let coroutine = item(&globals, "work").call0()?;
        globals.set_item("coroutine", &coroutine)?;
        let invocation = PreparedCall::new(
            InvocationMode::Direct,
            item(&globals, "callback").unbind(),
            PyTuple::empty(py).unbind(),
            None,
        );
        let result = invoke_direct(&invocation, py)?;
        let inspect = py.import("inspect")?;
        let state = inspect.call_method1("getcoroutinestate", (&coroutine,));
        coroutine.call_method0("close")?;
        assert!(result.bind(py).is(&coroutine));
        assert!(started.is_empty());
        assert!(state?.eq(inspect.getattr("CORO_CREATED")?)?);
        Ok(())
    })
}

#[pyfunction]
fn reenter(py: Python<'_>, callback: Py<PyAny>, payload: Py<PyAny>) -> PyResult<Py<PyAny>> {
    invoke_direct(
        &PreparedCall::new(
            InvocationMode::Direct,
            callback,
            PyTuple::new(py, [payload])?.unbind(),
            None,
        ),
        py,
    )
}

#[rstest]
fn preserves_current_context_thread_and_reentry(
    initialized_python: &InitializedPython,
) -> PyResult<()> {
    let _ = initialized_python;
    Python::attach(|py| {
        let globals = scope(
            py,
            c"
import threading
def inner(data):
    observed.append((context.get(), threading.get_ident()))
    data['inner'] = True
    context.set('inner')
    return data
def outer():
    observed.append((context.get(), threading.get_ident()))
    context.set('outer')
    return reenter(inner, payload)
",
        )?;
        let context = py
            .import("contextvars")?
            .getattr("ContextVar")?
            .call1(("prepared_call_context",))?;
        let thread = py
            .import("threading")?
            .call_method0("get_ident")?
            .extract::<u64>()?;
        let payload = PyDict::new(py);
        let observed = PyList::empty(py);
        globals.set_item("context", &context)?;
        globals.set_item("payload", &payload)?;
        globals.set_item("observed", &observed)?;
        globals.set_item("reenter", wrap_pyfunction!(reenter, py)?)?;
        let invocation = PreparedCall::new(
            InvocationMode::Direct,
            item(&globals, "outer").unbind(),
            PyTuple::empty(py).unbind(),
            None,
        );
        let token = context.call_method1("set", ("caller",))?;
        let result = invoke_direct(&invocation, py);
        let final_context = context.call_method0("get");
        context.call_method1("reset", (token,))?;
        assert!(result?.bind(py).is(&payload));
        assert!(item(&payload, "inner").extract::<bool>()?);
        assert_eq!(final_context?.extract::<String>()?, "inner");
        assert_eq!(
            observed.extract::<Vec<(String, u64)>>()?,
            [("caller".to_owned(), thread), ("outer".to_owned(), thread)]
        );
        Ok(())
    })
}

#[rstest]
fn owns_arguments_until_release_and_preserves_callback_retention(
    initialized_python: &InitializedPython,
) -> PyResult<()> {
    let _ = initialized_python;
    let (invocation, globals) = Python::attach(|py| {
        let globals = scope(
            py,
            c"
class Value:
    pass
class Callback:
    def __call__(self, value, *, other):
        saved.append(value)
        observed.append(other is other_ref())
",
        )?;
        globals.set_item("saved", PyList::empty(py))?;
        globals.set_item("observed", PyList::empty(py))?;
        let value = item(&globals, "Value").call0()?;
        let other = item(&globals, "Value").call0()?;
        let callback = item(&globals, "Callback").call0()?;
        let weakref = py.import("weakref")?;
        for (name, object) in [
            ("value_ref", &value),
            ("other_ref", &other),
            ("callback_ref", &callback),
        ] {
            globals.set_item(name, weakref.call_method1("ref", (object,))?)?;
        }
        let keywords = PyDict::new(py);
        keywords.set_item("other", other)?;
        let invocation = PreparedCall::new(
            InvocationMode::Direct,
            callback.unbind(),
            PyTuple::new(py, [value])?.unbind(),
            Some(keywords.unbind()),
        );
        Ok::<_, PyErr>((invocation, globals.unbind()))
    })?;
    Python::attach(|py| {
        let globals = globals.bind(py);
        for name in ["value_ref", "other_ref", "callback_ref"] {
            assert!(!item(globals, name).call0()?.is_none(), "{name}");
        }
        assert!(invoke_direct(&invocation, py)?.is_none(py));
        drop(invocation);
        let gc = py.import("gc")?;
        gc.call_method0("collect")?;
        assert_eq!(item(globals, "observed").extract::<Vec<bool>>()?, [true]);
        assert!(item(globals, "callback_ref").call0()?.is_none());
        assert!(item(globals, "other_ref").call0()?.is_none());
        let saved = item(globals, "saved");
        assert!(item(globals, "value_ref").call0()?.is(saved.get_item(0)?));
        saved.get_item(0)?.setattr("still_usable", true)?;
        saved.call_method0("clear")?;
        gc.call_method0("collect")?;
        assert!(item(globals, "value_ref").call0()?.is_none());
        Ok(())
    })
}

fn prepare_pre_call(
    py: Python<'_>,
    logger: &Bound<'_, PyAny>,
    view: &Bound<'_, PyDict>,
) -> PyResult<PreparedCall> {
    let keywords = PyDict::new(py);
    keywords.set_item("input", "OCR document processing")?;
    keywords.set_item("api_key", py.None())?;
    keywords.set_item("additional_args", view)?;
    Ok(PreparedCall::new(
        InvocationMode::Direct,
        logger.getattr("pre_call")?.unbind(),
        PyTuple::empty(py).unbind(),
        Some(keywords.unbind()),
    ))
}

#[rstest]
fn checked_runner_rejects_unhandled_background_failures(
    initialized_python: &InitializedPython,
) -> PyResult<()> {
    let _ = initialized_python;
    Python::attach(|py| {
        let globals = PyDict::new(py);
        run_fixture(
            py,
            &globals,
            include_str!("fixtures/callback_lifecycle.py"),
            concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/tests/fixtures/callback_lifecycle.py"
            ),
        )?;
        py.run(
            c"
async def fail():
    raise RuntimeError('background task regression')

async def scenario(cyclic, handled, observed):
    task = asyncio.create_task(fail())
    if cyclic:
        task.cycle = task
    await checkpoint()
    observed['done'] = task.done()
    if handled:
        try:
            task.result()
        except RuntimeError as error:
            observed['error'] = str(error)
    del task
",
            Some(&globals),
            None,
        )?;
        for cyclic in [false, true] {
            for handled in [false, true] {
                let observed = PyDict::new(py);
                let owners = item(&globals, "ReferenceFactory").call0()?;
                let scenario = item(&globals, "scenario").call1((cyclic, handled, &observed))?;
                let result = item(&globals, "run_checked").call1((owners, scenario));
                assert!(
                    item(&observed, "done").extract::<bool>()?,
                    "cyclic={cyclic}, handled={handled}"
                );
                if handled {
                    result?;
                    assert_eq!(
                        item(&observed, "error").extract::<String>()?,
                        "background task regression"
                    );
                } else {
                    let error = result.unwrap_err();
                    assert!(error.is_instance_of::<PyAssertionError>(py), "{error}");
                    let message = error.value(py).str()?.to_str()?.to_owned();
                    assert!(
                        message.starts_with("unhandled background failures: "),
                        "{message}"
                    );
                    assert!(message.contains("background task regression"), "{message}");
                }
            }
        }
        Ok(())
    })
}

#[rstest]
#[ignore = "requires the repository Python environment and LiteLLM on PYTHONPATH"]
fn real_ocr_logging_preserves_execution_roots_and_continues_after_error(
    initialized_python: &InitializedPython,
) -> PyResult<()> {
    let _ = initialized_python;
    Python::attach(|py| {
        let globals = scope(
            py,
            c"
from litellm.integrations.custom_logger import CustomLogger

class Retain(CustomLogger):
    def log_pre_api_call(self, model, messages, kwargs):
        order.append('retain')
        self.view = kwargs['additional_args']
        self.headers = self.view['headers']
        self.body = self.view['complete_input_dict']
        self.snapshot = (self.headers['X-Trace'], self.body['document']['value'])
        return {'ignored_replacement': True}

class MutateThenFail(CustomLogger):
    def log_pre_api_call(self, model, messages, kwargs):
        order.append('mutate_then_fail')
        view = kwargs['additional_args']
        view['headers']['X-Trace'] = 'mutated'
        view['complete_input_dict']['document']['value'] = 'mutated'
        view['headers'] = {'X-Trace': 'replacement'}
        view['complete_input_dict'] = {'replacement': True}
        raise RuntimeError('expected callback failure')

class Observe(CustomLogger):
    def log_pre_api_call(self, model, messages, kwargs):
        order.append('observe')
        self.view = kwargs['additional_args']
        self.snapshot = (
            tuple(sorted(self.view['headers'].items())),
            self.view['complete_input_dict'].get('replacement'),
            'document' in self.view['complete_input_dict'],
        )

",
        )?;
        let order = PyList::empty(py);
        globals.set_item("order", &order)?;
        let first = item(&globals, "Retain").call0()?;
        let last = item(&globals, "Observe").call0()?;
        let document = PyDict::new(py);
        document.set_item("value", "original")?;
        let headers = PyDict::new(py);
        headers.set_item("X-Trace", "original")?;
        let body = PyDict::new(py);
        body.set_item("document", &document)?;
        body.set_item("alias", &document)?;
        let view = PyDict::new(py);
        view.set_item("headers", &headers)?;
        view.set_item("complete_input_dict", &body)?;
        view.set_item("api_base", "https://example.invalid/ocr")?;
        let keywords = PyDict::new(py);
        keywords.set_item("model", "test")?;
        keywords.set_item("messages", PyList::empty(py))?;
        keywords.set_item("stream", false)?;
        keywords.set_item("call_type", "ocr")?;
        keywords.set_item(
            "start_time",
            py.import("datetime")?
                .getattr("datetime")?
                .call_method0("now")?,
        )?;
        keywords.set_item("litellm_call_id", "retained-test")?;
        keywords.set_item("function_id", "retained-test")?;
        keywords.set_item(
            "dynamic_input_callbacks",
            PyList::new(
                py,
                [&first, &item(&globals, "MutateThenFail").call0()?, &last],
            )?,
        )?;
        let logger = py
            .import("litellm.litellm_core_utils.litellm_logging")?
            .getattr("Logging")?
            .call((), Some(&keywords))?;
        let invocation = prepare_pre_call(py, &logger, &view)?;
        assert!(invoke_direct(&invocation, py)?.is_none(py));
        drop(invocation);
        assert!(headers.is(first.getattr("headers")?));
        assert!(body.is(first.getattr("body")?));
        assert!(view.is(last.getattr("view")?));
        assert_eq!(item(&headers, "X-Trace").extract::<String>()?, "mutated");
        assert_eq!(
            order.extract::<Vec<String>>()?,
            ["retain", "mutate_then_fail", "observe"]
        );
        assert_eq!(
            first.getattr("snapshot")?.extract::<(String, String)>()?,
            ("original".to_owned(), "original".to_owned())
        );
        assert_eq!(
            last.getattr("snapshot")?
                .extract::<(Vec<(String, String)>, bool, bool)>()?,
            (
                vec![("X-Trace".to_owned(), "replacement".to_owned())],
                true,
                false
            )
        );
        assert!(first.getattr("view")?.is(last.getattr("view")?));
        assert!(first.getattr("body")?.get_item("document")?.is(&document));
        assert!(first.getattr("body")?.get_item("alias")?.is(&document));
        assert_eq!(item(&document, "value").extract::<String>()?, "mutated");
        assert_eq!(
            last.getattr("view")?
                .get_item("headers")?
                .get_item("X-Trace")?
                .extract::<String>()?,
            "replacement"
        );
        let replacement = PyDict::new(py);
        replacement.set_item("replacement", true)?;
        assert!(
            last.getattr("view")?
                .get_item("complete_input_dict")?
                .eq(replacement)?
        );
        document.set_item("value", "after invocation")?;
        assert_eq!(
            first
                .getattr("body")?
                .get_item("document")?
                .get_item("value")?
                .extract::<String>()?,
            "after invocation"
        );
        drop((headers, body, view));
        assert_eq!(
            first
                .getattr("headers")?
                .get_item("X-Trace")?
                .extract::<String>()?,
            "mutated"
        );
        Ok(())
    })
}

fn invoke_direct(call: &PreparedCall, py: Python<'_>) -> PyResult<Py<PyAny>> {
    match call.invoke(py)? {
        InvocationOutcome::Returned(value) => Ok(value),
        InvocationOutcome::Awaitable(_) => panic!("direct binding produced an awaitable outcome"),
    }
}
