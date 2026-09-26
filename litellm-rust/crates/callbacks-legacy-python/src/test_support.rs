use std::ffi::CStr;

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};

use crate::{LegacyLogging, LegacySurface, PublicCall};

/// The parameters of every `callbacks_legacy_python` function, as the real module declares them.
/// `tests/unit/rust_bridge/test_callbacks_legacy_python.py` pins this file to the Python
/// signatures, and [`namespace`] binds every fake call against it.
pub(crate) const PYTHON_CONTRACT: &str = include_str!("../python_contract.json");

/// Stand-ins for `callbacks_legacy_python`, the only Python module the crate calls. Tests
/// share one interpreter and run concurrently, so each fake is installed idempotently and
/// forwards to the per-test `StubLogger` it is handed (directly, or as `kwargs['logger']`).
/// Every fake is bound against the contract first, so a call the real module would reject
/// fails here too.
const STUBS: &CStr = c"
import contextvars
import inspect
import json
import sys
import traceback
import types

for name in ('litellm', 'litellm.rust_bridge', 'litellm.rust_bridge.callbacks_legacy_python'):
    sys.modules.setdefault(name, types.ModuleType(name))

legacy = sys.modules['litellm.rust_bridge.callbacks_legacy_python']
CONTRACT = json.loads(python_contract)


def contracted(name, fake):
    signature = inspect.Signature(
        [inspect.Parameter(parameter, inspect.Parameter.POSITIONAL_OR_KEYWORD) for parameter in CONTRACT[name]]
    )

    def checked(*args, **kwargs):
        signature.bind(*args, **kwargs)
        return fake(*args, **kwargs)

    return checked


if not hasattr(legacy, 'is_internal'):
    legacy.is_internal = contextvars.ContextVar('is_internal_call', default=False)

FAKES = {
    'setup': lambda call_type, args, kwargs, start, asynchronous: types.SimpleNamespace(
        logger=kwargs['logger_factory'](kwargs) if 'logger_factory' in kwargs else kwargs['logger'],
        kwargs=kwargs,
    ),
    'finalize': lambda response, logger, kwargs, start, end: logger.record('finalize', response),
    'update_logging': lambda logger, kwargs, model, optional_params, litellm_params, provider: logger.update_from_kwargs(
        kwargs=kwargs,
        model=model,
        optional_params=optional_params,
        litellm_params=litellm_params,
        custom_llm_provider=provider,
    ),
    'pre_call': lambda logger, input, api_key, additional_args: logger.pre_call(input, api_key, additional_args),
    'post_call': lambda logger, original_response, api_key, additional_args: logger.post_call(
        original_response, api_key, additional_args
    ),
    'defers_async_logging': lambda logger: bool(getattr(logger, '_defer_async_logging', False)),
    'defer_success': lambda logger, pending: setattr(logger, '_native_pending_logging', pending),
    'sync_success_for_async_call': lambda logger, response, start, end: logger.handle_sync_success_callbacks_for_async_calls(
        response, start, end
    ),
    'failure_handler': lambda logger, error, start, end, asynchronous: (
        logger.async_failure_handler if asynchronous else logger.failure_handler
    )(error, ''.join(traceback.format_exception(error)), start, end),
    'submit_success': lambda logger, response, start, end: logger.record('submit', (response, start, end)),
    'async_success_handler': lambda logger, response, start, end: logger.async_success_handler(response, start, end),
    'enqueue_logging': lambda coroutine: coroutine.enqueue(),
    'restore_context': lambda logger: logger.record('restore', None),
    'custom_pricing_fields': lambda: ('ocr_cost_per_page',),
    'is_internal_call': lambda: legacy.is_internal.get(),
    'execute_pre_request_hooks': lambda model, messages, kwargs: kwargs['logger'].pre_request(model, messages, kwargs),
    'async_pre_call_deployment_hook': lambda logger, kwargs, call_type: kwargs['logger'].hook('pre', kwargs, call_type),
    'async_post_call_success_deployment_hook': lambda kwargs, response, call_type: kwargs['logger'].hook(
        'success', response, call_type
    ),
    'async_post_call_failure_deployment_hook': lambda kwargs, error, call_type: kwargs['logger'].hook('failure', error, call_type),
    'stream_opened': lambda logger: logger.record('stream_opened', None),
    'stream_success': lambda logger, request_body, chunks, start, end, first_chunk: logger.record(
        'stream_success', list(chunks)
    ),
    'stream_failure': lambda logger, request_body, chunks, error: logger.record('stream_failure', error),
}
assert FAKES.keys() == CONTRACT.keys(), sorted(FAKES.keys() ^ CONTRACT.keys())
for name, fake in FAKES.items():
    setattr(legacy, name, contracted(name, fake))


unraisable = sys.modules.setdefault(
    'litellm_test_unraisable', types.ModuleType('litellm_test_unraisable')
)
if not hasattr(unraisable, 'events'):
    unraisable.events = []
    sys.unraisablehook = lambda event: unraisable.events.append((event.object, event.exc_value))


def unraisable_from(owner):
    return [error for source, error in unraisable.events if source is owner]


class StubCoroutine:
    def __init__(self, logger):
        self.logger = logger

    def enqueue(self):
        self.logger.record('enqueued', None)
        self.logger.on_enqueue(self)

    def close(self):
        self.logger.record('closed', None)


class StubLogger:
    def __init__(self):
        self.calls = []
        self.hooks = {}
        self.on_enqueue = lambda coroutine: None

    def record(self, name, value):
        self.calls.append((name, value))

    def names(self):
        return [name for name, _ in self.calls]

    def hook(self, phase, value, call_type):
        self.record(phase + '_hook', call_type)
        return self.hooks.get(phase, lambda value: 'awaitable')(value)

    def pre_request(self, model, messages, kwargs):
        self.record('pre_request', (model, messages, kwargs))
        return self.hooks.get('pre_request', lambda model, messages, kwargs: 'awaitable')(model, messages, kwargs)

    def failure_handler(self, error, trace, start, end):
        self.record('failure_handler', error)

    def async_failure_handler(self, error, trace, start, end):
        self.record('async_failure_handler', error)
        return 'awaitable'

    def success_handler(self, response, start, end):
        self.record('success_handler', response)

    def async_success_handler(self, response, start, end):
        self.record('async_success_handler', response)
        return StubCoroutine(self)

    def handle_sync_success_callbacks_for_async_calls(self, response, start, end):
        self.record('sync_success_for_async_call', response)


logger = StubLogger()
";

/// A namespace with the stubs, `StubLogger` and a fresh `logger`, after `script` ran in it.
pub(crate) fn namespace<'py>(py: Python<'py>, script: &CStr) -> Bound<'py, PyDict> {
    let locals = PyDict::new(py);
    locals.set_item("python_contract", PYTHON_CONTRACT).unwrap();
    py.run(STUBS, Some(&locals), Some(&locals)).unwrap();
    py.run(script, Some(&locals), Some(&locals)).unwrap();
    locals
}

pub(crate) fn run(py: Python<'_>, locals: &Bound<'_, PyDict>, code: &CStr) {
    py.run(code, Some(locals), Some(locals)).unwrap();
}

pub(crate) fn local<'py>(locals: &Bound<'py, PyDict>, name: &str) -> Bound<'py, PyAny> {
    locals.get_item(name).unwrap().unwrap()
}

/// A legacy call over the namespace's `kwargs` (or none) and `request` (or `None`).
pub(crate) fn legacy_call(
    py: Python<'_>,
    locals: &Bound<'_, PyDict>,
    asynchronous: bool,
) -> LegacyLogging {
    let request = locals
        .get_item("request")
        .unwrap()
        .unwrap_or_else(|| py.None().into_bound(py));
    let kwargs = locals
        .get_item("kwargs")
        .unwrap()
        .map(|kwargs| kwargs.cast_into::<PyDict>().unwrap())
        .unwrap_or_else(|| PyDict::new(py));
    let call = PublicCall::capture(&request, &PyTuple::empty(py), &kwargs).unwrap();
    LegacyLogging::new(
        py,
        LegacySurface {
            call_type: "test",
            input_description: "test input",
            stream: None,
        },
        call,
        asynchronous,
    )
}
