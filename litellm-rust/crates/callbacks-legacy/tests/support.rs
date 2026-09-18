use std::ffi::CStr;

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};

use crate::{LegacyLogging, LegacySurface, PublicCall};

/// Stand-ins for every litellm function the legacy contract calls. Tests share one
/// interpreter and run concurrently, so each stub is installed idempotently and forwards to
/// the per-test `StubLogger` it is handed (directly, or as `kwargs['logger']`).
const STUBS: &CStr = c"
import contextvars
import sys
import types

for name in (
    'litellm',
    'litellm.utils',
    'litellm.types',
    'litellm.types.utils',
    'litellm._internal_context',
    'litellm.litellm_core_utils',
    'litellm.litellm_core_utils.logging_worker',
    'litellm.litellm_core_utils.litellm_logging',
    'litellm.rust_bridge',
    'litellm.rust_bridge.legacy_callbacks',
):
    sys.modules.setdefault(name, types.ModuleType(name))

legacy = sys.modules['litellm.rust_bridge.legacy_callbacks']
legacy.setup = lambda call_type, args, kwargs, start, asynchronous: types.SimpleNamespace(
    logger=kwargs['logger_factory'](kwargs) if 'logger_factory' in kwargs else kwargs['logger'],
    kwargs=kwargs,
    bridge_owned=True,
)
legacy.deployment_callbacks_needed = lambda: True
legacy.check_limits = lambda arguments: arguments['logger'].check_limits(arguments)
legacy.callbacks_needed = lambda logger, phase: logger.needed.get(phase, True)
legacy.success_bookkeeping = lambda logger, response, start, end, asynchronous: logger.record(
    'success_bookkeeping', asynchronous
)
legacy.failure_bookkeeping = lambda logger, error, start, end, asynchronous: logger.record(
    'failure_bookkeeping', asynchronous
)
legacy.finalize = lambda response, logger, kwargs, start, end: logger.record('finalize', response)

utils = sys.modules['litellm.utils']
utils.async_pre_call_deployment_hook = lambda kwargs, call_type: kwargs['logger'].hook(
    'pre', kwargs, call_type
)
utils.async_post_call_success_deployment_hook = lambda kwargs, response, call_type: kwargs[
    'logger'
].hook('success', response, call_type)
utils.async_post_call_failure_deployment_hook = lambda kwargs, error, call_type: kwargs[
    'logger'
].hook('failure', error, call_type)
utils._restore_correlation_context_if_supported = lambda logger: logger.record('restore', None)

internal = sys.modules['litellm._internal_context']
if not hasattr(internal, 'is_internal_call'):
    internal.is_internal_call = contextvars.ContextVar('is_internal_call', default=False)

sys.modules['litellm.types.utils'].CustomPricingLiteLLMParams = type(
    'CustomPricingLiteLLMParams', (), {'model_fields': {'ocr_cost_per_page': None}}
)


unraisable = sys.modules.setdefault(
    'litellm_test_unraisable', types.ModuleType('litellm_test_unraisable')
)
if not hasattr(unraisable, 'events'):
    unraisable.events = []
    sys.unraisablehook = lambda event: unraisable.events.append((event.object, event.exc_value))


def unraisable_from(owner):
    return [error for source, error in unraisable.events if source is owner]


class Worker:
    def ensure_initialized_and_enqueue(self, coroutine):
        return coroutine.enqueue()


class Executor:
    def submit(self, run, handler, *args):
        handler.__self__.record('submit', args)


sys.modules['litellm.litellm_core_utils.logging_worker'].GLOBAL_LOGGING_WORKER = Worker()
sys.modules['litellm.litellm_core_utils.litellm_logging'].executor = Executor()


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
        self.needed = {}
        self.hooks = {}
        self.on_enqueue = lambda coroutine: None

    def record(self, name, value):
        self.calls.append((name, value))

    def names(self):
        return [name for name, _ in self.calls]

    def hook(self, phase, value, call_type):
        self.record(phase + '_hook', call_type)
        return self.hooks.get(phase, lambda value: 'awaitable')(value)

    def check_limits(self, arguments):
        self.record('check_limits', arguments)

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
        },
        call,
        asynchronous,
    )
}
