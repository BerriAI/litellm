use litellm_python_interop::{InvocationMode, InvocationOutcome, PreparedCall};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};
use rstest::rstest;

#[path = "support/mod.rs"]
mod support;

use support::python::{InitializedPython, initialized_python, item, scope};

#[rstest]
fn retains_aliases_mutations_and_original_result(
    initialized_python: &InitializedPython,
) -> PyResult<()> {
    let _ = initialized_python;
    Python::attach(|py| {
        let globals = scope(
            py,
            c"
shared = {'value': 'before'}
payload = {'nested': shared}
saved = []
def callback(data, *, alias):
    assert data['nested'] is alias
    saved.append(data)
    alias['value'] = 'during'
    return data
",
        )?;
        let payload = item(&globals, "payload");
        let keywords = PyDict::new(py);
        keywords.set_item("alias", item(&globals, "shared"))?;
        let invocation = PreparedCall::new(
            InvocationMode::Direct,
            item(&globals, "callback").unbind(),
            PyTuple::new(py, [&payload])?.unbind(),
            Some(keywords.unbind()),
        );
        let result = invoke_direct(&invocation, py)?;
        assert!(result.bind(py).is(&payload));
        drop(invocation);
        py.run(
            c"
assert saved[0] is payload
assert shared['value'] == 'during'
shared['value'] = 'after'
assert saved[0]['nested']['value'] == 'after'
",
            Some(&globals),
            None,
        )
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
payload = {}
error = KeyboardInterrupt('original')
cause = ValueError('cause')
def callback(data):
    data['changed'] = True
    raise error from cause
",
        )?;
        let invocation = PreparedCall::new(
            InvocationMode::Direct,
            item(&globals, "callback").unbind(),
            PyTuple::new(py, [item(&globals, "payload")])?.unbind(),
            None,
        );
        let error = invoke_direct(&invocation, py).unwrap_err();
        assert!(error.value(py).is(item(&globals, "error")));
        drop(invocation);
        py.run(
            c"
import traceback
assert payload['changed'] is True
assert error.__cause__ is cause
assert traceback.extract_tb(error.__traceback__)[-1].name == 'callback'
",
            Some(&globals),
            None,
        )
    })
}

#[rstest]
fn returns_coroutine_without_executing_it(initialized_python: &InitializedPython) -> PyResult<()> {
    let _ = initialized_python;
    Python::attach(|py| {
        let globals = scope(
            py,
            c"
import inspect
started = []
async def work():
    started.append(True)
coroutine = work()
def callback():
    return coroutine
",
        )?;
        let invocation = PreparedCall::new(
            InvocationMode::Direct,
            item(&globals, "callback").unbind(),
            PyTuple::empty(py).unbind(),
            None,
        );
        let result = invoke_direct(&invocation, py)?;
        assert!(result.bind(py).is(item(&globals, "coroutine")));
        py.run(
            c"
assert started == []
assert inspect.getcoroutinestate(coroutine) == inspect.CORO_CREATED
coroutine.close()
",
            Some(&globals),
            None,
        )
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
import contextvars
import threading
context = contextvars.ContextVar('prepared_call_context')
token = context.set('caller')
thread = threading.get_ident()
payload = {}
def inner(data):
    assert context.get() == 'outer'
    assert threading.get_ident() == thread
    data['inner'] = True
    context.set('inner')
    return data
def outer():
    assert context.get() == 'caller'
    assert threading.get_ident() == thread
    context.set('outer')
    return reenter(inner, payload)
",
        )?;
        globals.set_item("reenter", wrap_pyfunction!(reenter, py)?)?;
        let invocation = PreparedCall::new(
            InvocationMode::Direct,
            item(&globals, "outer").unbind(),
            PyTuple::empty(py).unbind(),
            None,
        );
        let result = invoke_direct(&invocation, py)?;
        assert!(result.bind(py).is(item(&globals, "payload")));
        py.run(
            c"
try:
    assert payload['inner'] is True
    assert context.get() == 'inner'
finally:
    context.reset(token)
",
            Some(&globals),
            None,
        )
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
import gc
import weakref
saved = []
class Value:
    pass
class Callback:
    def __call__(self, value, *, other):
        saved.append(value)
        assert other is other_ref()
value = Value()
other = Value()
callback = Callback()
value_ref = weakref.ref(value)
other_ref = weakref.ref(other)
callback_ref = weakref.ref(callback)
",
        )?;
        let keywords = PyDict::new(py);
        keywords.set_item("other", item(&globals, "other"))?;
        let invocation = PreparedCall::new(
            InvocationMode::Direct,
            item(&globals, "callback").unbind(),
            PyTuple::new(py, [item(&globals, "value")])?.unbind(),
            Some(keywords.unbind()),
        );
        py.run(c"del value, other, callback", Some(&globals), None)?;
        Ok::<_, PyErr>((invocation, globals.unbind()))
    })?;
    Python::attach(|py| {
        let globals = globals.bind(py);
        py.run(
            c"assert all(ref() is not None for ref in (value_ref, other_ref, callback_ref))",
            Some(globals),
            None,
        )?;
        assert!(invoke_direct(&invocation, py)?.is_none(py));
        drop(invocation);
        py.run(
            c"
gc.collect()
assert callback_ref() is None
assert other_ref() is None
assert value_ref() is saved[0]
saved[0].still_usable = True
saved.clear()
gc.collect()
assert value_ref() is None
",
            Some(globals),
            None,
        )
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
#[ignore = "requires the repository Python environment and LiteLLM on PYTHONPATH"]
fn real_ocr_logging_preserves_execution_roots_and_continues_after_error(
    initialized_python: &InitializedPython,
) -> PyResult<()> {
    let _ = initialized_python;
    Python::attach(|py| {
        let globals = scope(
            py,
            c"
from datetime import datetime
from litellm.integrations.custom_logger import CustomLogger
from litellm.litellm_core_utils.litellm_logging import Logging

class Retain(CustomLogger):
    def log_pre_api_call(self, model, messages, kwargs):
        self.view = kwargs['additional_args']
        self.headers = self.view['headers']
        self.body = self.view['complete_input_dict']
        return {'ignored_replacement': True}

class MutateThenFail(CustomLogger):
    def log_pre_api_call(self, model, messages, kwargs):
        view = kwargs['additional_args']
        view['headers']['X-Trace'] = 'mutated'
        view['complete_input_dict']['document']['value'] = 'mutated'
        view['headers'] = {'X-Trace': 'replacement'}
        view['complete_input_dict'] = {'replacement': True}
        raise RuntimeError('expected callback failure')

class Observe(CustomLogger):
    def log_pre_api_call(self, model, messages, kwargs):
        self.view = kwargs['additional_args']

first = Retain()
last = Observe()
document = {'value': 'original'}
headers = {'X-Trace': 'original'}
body = {'document': document, 'alias': document}
view = {'headers': headers, 'complete_input_dict': body, 'api_base': 'https://example.invalid/ocr'}
logger = Logging(
    model='test', messages=[], stream=False, call_type='ocr',
    start_time=datetime.now(), litellm_call_id='retained-test', function_id='retained-test',
    dynamic_input_callbacks=[first, MutateThenFail(), last],
)
",
        )?;
        let headers = item(&globals, "headers").unbind();
        let body = item(&globals, "body").unbind();
        let view = item(&globals, "view").cast_into::<PyDict>()?.unbind();
        let invocation = prepare_pre_call(py, &item(&globals, "logger"), view.bind(py))?;
        assert!(invoke_direct(&invocation, py)?.is_none(py));
        drop(invocation);
        py.run(c"del headers, body, view", Some(&globals), None)?;
        assert!(
            headers
                .bind(py)
                .is(item(&globals, "first").getattr("headers")?)
        );
        assert!(body.bind(py).is(item(&globals, "first").getattr("body")?));
        assert!(view.bind(py).is(item(&globals, "last").getattr("view")?));
        assert_eq!(
            headers.bind(py).get_item("X-Trace")?.extract::<String>()?,
            "mutated"
        );
        py.run(
            c"
assert first.view is last.view
assert first.body['document'] is document
assert first.body['alias'] is document
assert document['value'] == 'mutated'
assert last.view['headers']['X-Trace'] == 'replacement'
assert last.view['complete_input_dict'] == {'replacement': True}
document['value'] = 'after invocation'
assert first.body['document']['value'] == 'after invocation'
",
            Some(&globals),
            None,
        )?;
        drop((headers, body, view));
        py.run(
            c"assert first.headers['X-Trace'] == 'mutated'",
            Some(&globals),
            None,
        )
    })
}

fn invoke_direct(call: &PreparedCall, py: Python<'_>) -> PyResult<Py<PyAny>> {
    match call.invoke(py)? {
        InvocationOutcome::Returned(value) => Ok(value),
        InvocationOutcome::Awaitable(_) => panic!("direct binding produced an awaitable outcome"),
    }
}
