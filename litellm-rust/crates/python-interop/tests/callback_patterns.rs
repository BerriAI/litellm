use std::ffi::CStr;

use litellm_python_interop::{InvocationMode, InvocationOutcome, PreparedCall};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};
use rstest::rstest;
use serial_test::serial;

#[path = "support/mod.rs"]
mod support;

use support::python::{InitializedPython, initialized_python, item, run_fixture, scope};

#[derive(Clone, Copy, Debug)]
enum Backend {
    Python,
    PreparedCall,
}

#[pyfunction]
fn invoke_prepared<'py>(
    py: Python<'py>,
    callback: Py<PyAny>,
    args: Py<PyTuple>,
    awaited: bool,
) -> PyResult<Bound<'py, PyAny>> {
    let mode = if awaited {
        InvocationMode::Await
    } else {
        InvocationMode::Direct
    };
    let call = PreparedCall::new(mode, callback, args, None);
    let outcome = call.invoke(py);
    drop(call);
    match outcome? {
        InvocationOutcome::Returned(value) => {
            assert_eq!(mode, InvocationMode::Direct);
            Ok(value.into_bound(py))
        }
        InvocationOutcome::Awaitable(value) => {
            assert_eq!(mode, InvocationMode::Await);
            Ok(value.into_bound(py))
        }
    }
}

fn pattern_scope<'py>(
    py: Python<'py>,
    backend: Backend,
    mode: InvocationMode,
) -> PyResult<Bound<'py, PyDict>> {
    let globals = scope(
        py,
        c"
import asyncio
import copy
import gc
import json
import threading
import weakref
from unittest import TestCase
from callback_lifecycle import ReferenceFactory, run_checked

class Value:
    pass

def invoke(callback, *args):
    if retained:
        return invoke_prepared(callback, args, awaited)
    if not awaited:
        return callback(*args)

    async def await_callback():
        return await callback(*args)

    return await_callback()

async def async_call(callback, *args):
    async def async_callback(*values):
        return callback(*values)

    result = invoke(async_callback if awaited else callback, *args)
    return await result if awaited else result

def call(callback, *args):
    if not awaited:
        return invoke(callback, *args)
    return run_async(async_call(callback, *args))

def run_async(scenario):
    return run_checked(ReferenceFactory(), scenario)
",
    )?;
    globals.set_item("invoke_prepared", wrap_pyfunction!(invoke_prepared, py)?)?;
    globals.set_item("retained", matches!(backend, Backend::PreparedCall))?;
    globals.set_item("awaited", mode == InvocationMode::Await)?;
    Ok(globals)
}

#[track_caller]
fn run_pattern(
    python: &InitializedPython,
    backend: Backend,
    mode: InvocationMode,
    parameters: impl for<'py> FnOnce(&Bound<'py, PyDict>) -> PyResult<()>,
    scenario: &CStr,
) -> PyResult<()> {
    let location = std::panic::Location::caller();
    python.attach(|py| {
        let globals = pattern_scope(py, backend, mode)?;
        parameters(&globals)?;
        run_fixture(
            py,
            &globals,
            scenario.to_str().unwrap(),
            &format!("{}:{}", location.file(), location.line()),
        )?;
        let result = py.run(
            c"
if asyncio.iscoroutinefunction(test_pattern):
    run_async(test_pattern())
else:
    test_pattern()
",
            Some(&globals),
            None,
        );
        if let Err(error) = &result {
            error.print(py);
        }
        result
    })
}

#[rstest]
#[case::flush_after_edit_sees_mutation(false, false, false)]
#[case::flush_before_edit_keeps_original(true, false, true)]
#[case::copied_editor_leaves_queue_unchanged(false, true, true)]
#[serial(python_interpreter)]
fn reference_queue_observes_edits_until_serialization_and_owns_arguments(
    initialized_python: &InitializedPython,
    #[case] flush_before_edit: bool,
    #[case] copy_before_edit: bool,
    #[case] flushed_has_tools: bool,
    #[values(Backend::Python, Backend::PreparedCall)] backend: Backend,
    #[values(InvocationMode::Direct, InvocationMode::Await)] mode: InvocationMode,
) -> PyResult<()> {
    run_pattern(
        initialized_python,
        backend,
        mode,
        |globals| {
            globals.set_item("flush_before_edit", flush_before_edit)?;
            globals.set_item("copy_before_edit", copy_before_edit)?;
            globals.set_item("flushed_has_tools", flushed_has_tools)
        },
        c"
def test_pattern():
    queue, snapshots, observations = [], [], []
    payload = {'model_parameters': {'tools': ['lookup'], 'stream': True}}
    response = Value()
    response_ref = weakref.ref(response)
    event = {'payload': payload}

    def enqueue_live(kwargs, result):
        queue.append((kwargs['payload'], kwargs, result))

    def enqueue_snapshot(kwargs, result):
        snapshots.append(json.dumps(kwargs['payload']))

    def pop_tools(kwargs, result):
        kwargs['payload']['model_parameters'].pop('tools')

    def observe(kwargs, result):
        observations.append('tools' in kwargs['payload']['model_parameters'])

    call(enqueue_live, event, response)
    call(enqueue_snapshot, event, response)
    early_flush = json.dumps(queue[0][0]) if flush_before_edit else None
    edited = copy.deepcopy(event) if copy_before_edit else event
    call(pop_tools, edited, response)
    call(observe, event, response)

    assert queue[0][0] is payload
    assert queue[0][1] is event
    assert queue[0][2] is response
    assert observations == [copy_before_edit]
    assert json.loads(snapshots[0]) == {
        'model_parameters': {'tools': ['lookup'], 'stream': True}
    }

    del response, event, payload, edited
    gc.collect()
    assert response_ref() is not None, 'queued response must outlive invocation'
    flushed = json.loads(early_flush if early_flush is not None else json.dumps(queue[0][0]))
    assert ('tools' in flushed['model_parameters']) == flushed_has_tools
    assert flushed['model_parameters']['stream'] is True
    queue.clear()
    gc.collect()
    assert response_ref() is None, 'clearing the queue must release its response'
",
    )
}

#[rstest]
#[serial(python_interpreter)]
fn shallow_queue_keeps_nested_aliases_but_not_replaced_fields(
    initialized_python: &InitializedPython,
    #[values(Backend::Python, Backend::PreparedCall)] backend: Backend,
    #[values(InvocationMode::Direct, InvocationMode::Await)] mode: InvocationMode,
) -> PyResult<()> {
    initialized_python.attach(|py| {
        let globals = pattern_scope(py, backend, mode)?;
        py.run(
            c"
queue = []
messages = [{'content': 'original'}]
response = Value()
response.messages = messages
event = {'messages': messages, 'status': 'queued'}
replacement = [{'content': 'replacement'}]

def enqueue(data, result):
    queue.append({**data})
    queue.append(result.__dict__)
",
            Some(&globals),
            None,
        )?;
        let event = item(&globals, "event");
        let response = item(&globals, "response");
        let messages = item(&globals, "messages");
        item(&globals, "call").call1((item(&globals, "enqueue"), &event, &response))?;

        messages.get_item(0)?.set_item("content", "edited")?;
        event.set_item("messages", item(&globals, "replacement"))?;
        event.set_item("status", "sent")?;

        let queue = item(&globals, "queue");
        let shallow_entry = queue.get_item(0)?;
        let attributes_entry = queue.get_item(1)?;
        assert!(!shallow_entry.is(&event));
        assert!(attributes_entry.is(response.getattr("__dict__")?));
        assert!(shallow_entry.get_item("messages")?.is(&messages));
        assert!(attributes_entry.get_item("messages")?.is(&messages));
        let serialized = item(&globals, "json").call_method1("dumps", (&shallow_entry,))?;
        assert_eq!(
            serde_json::from_str::<serde_json::Value>(serialized.extract::<&str>()?).unwrap(),
            serde_json::json!({"messages": [{"content": "edited"}], "status": "queued"})
        );
        assert!(
            event
                .get_item("messages")?
                .is(item(&globals, "replacement"))
        );
        Ok(())
    })
}

#[rstest]
#[case::ignored_return(false)]
#[case::caught_error_does_not_roll_back(true)]
#[serial(python_interpreter)]
fn sequential_loggers_observe_prior_mutations_and_opaque_state(
    initialized_python: &InitializedPython,
    #[case] raises: bool,
    #[values(Backend::Python, Backend::PreparedCall)] backend: Backend,
    #[values(InvocationMode::Direct, InvocationMode::Await)] mode: InvocationMode,
) -> PyResult<()> {
    run_pattern(
        initialized_python,
        backend,
        mode,
        |globals| globals.set_item("raises", raises),
        c"
def test_pattern():
    saved, seen, errors, order = [], [], [], []
    state = {'optional_params': {'tools': ['lookup']}, 'cache_hit': None}
    lock = threading.Lock()
    failure = RuntimeError('after mutation')

    def retain(kwargs):
        order.append('retain')
        saved.append(kwargs)

    def mutate(kwargs):
        order.append('mutate')
        kwargs['optional_params'].pop('tools')
        kwargs['cache_hit'] = False
        kwargs['flush_lock'] = lock
        if raises:
            raise failure
        return {'replacement': True}

    def observe(kwargs):
        order.append('observe')
        seen.append((kwargs, kwargs['cache_hit'], kwargs['flush_lock']))

    for callback in (retain, mutate, observe):
        try:
            call(callback, state)
        except RuntimeError as error:
            errors.append(error)

    assert order == ['retain', 'mutate', 'observe']
    assert errors == ([failure] if raises else [])
    if raises:
        assert errors[0] is failure
    assert saved[0] is seen[0][0] is state
    assert seen[0][1] is False
    assert seen[0][2] is lock
    assert state['optional_params'] == {}
    assert 'replacement' not in state
    with TestCase().assertRaises(TypeError):
        json.dumps(state)
",
    )
}

#[rstest]
#[case::replacement_list(false, false)]
#[case::in_place_redaction(true, false)]
#[case::equal_but_independently_copied_subset_does_not_match(false, true)]
#[serial(python_interpreter)]
fn redaction_matches_message_identity_and_preserves_unscanned_messages(
    initialized_python: &InitializedPython,
    #[case] redact_in_place: bool,
    #[case] copy_subset: bool,
    #[values(Backend::Python, Backend::PreparedCall)] backend: Backend,
    #[values(InvocationMode::Direct, InvocationMode::Await)] mode: InvocationMode,
) -> PyResult<()> {
    run_pattern(
        initialized_python,
        backend,
        mode,
        |globals| {
            globals.set_item("redact_in_place", redact_in_place)?;
            globals.set_item("copy_subset", copy_subset)
        },
        c"
def test_pattern():
    first, second = {'content': 'private'}, {'content': 'keep'}
    messages = [first, second]
    subset = [first]
    state = {'messages': messages}

    def redact(full, selected):
        if not {id(message) for message in selected} <= {id(message) for message in full}:
            return None
        if redact_in_place:
            selected[0]['content'] = 'masked'
            return full
        replacements = {id(selected[0]): {'content': 'masked'}}
        return [replacements.get(id(message), message) for message in full]

    scanned = copy.deepcopy(subset) if copy_subset else subset
    assert scanned == subset
    result = call(redact, messages, scanned)
    if result is not None and result is not messages:
        state['messages'] = result

    assert messages[0] is subset[0] is first
    assert state['messages'][1] is second
    if copy_subset:
        assert scanned[0] is not first
        assert result is None
        assert state['messages'] is messages
        assert first['content'] == 'private'
    elif redact_in_place:
        assert result is state['messages'] is messages
        assert first['content'] == 'masked'
    else:
        assert state['messages'] is result
        assert result is not messages
        assert result[0] is not first
        assert result[0] == {'content': 'masked'}
        assert first['content'] == 'private'
",
    )
}

#[rstest]
#[case::adopt_replacement(false)]
#[case::merge_into_live_request(true)]
#[serial(python_interpreter)]
fn dispatcher_applies_returned_state_only_at_its_replacement_or_merge_boundary(
    initialized_python: &InitializedPython,
    #[case] merge: bool,
    #[values(Backend::Python, Backend::PreparedCall)] backend: Backend,
    #[values(InvocationMode::Direct, InvocationMode::Await)] mode: InvocationMode,
) -> PyResult<()> {
    initialized_python.attach(|py| {
        let globals = pattern_scope(py, backend, mode)?;
        py.run(
            c"
original = {'messages': [{'content': 'original'}], 'request_id': 'retained'}
replacement = {'messages': [{'content': 'redacted'}], 'verdict': 'allow'}
seen = []

def rewrite(data):
    data['checkpoint'] = True
    return replacement

def observe(data):
    seen.append(data)
",
            Some(&globals),
            None,
        )?;
        let original = item(&globals, "original");
        let old_messages = original.get_item("messages")?;
        let replacement = item(&globals, "replacement");
        let call = item(&globals, "call");
        let returned = call.call1((item(&globals, "rewrite"), &original))?;
        assert!(returned.is(&replacement));
        let current = if merge {
            original.call_method1("update", (&returned,))?;
            original.clone()
        } else {
            returned
        };
        call.call1((item(&globals, "observe"), &current))?;

        assert!(item(&globals, "seen").get_item(0)?.is(&current));
        assert!(
            current
                .get_item("messages")?
                .is(replacement.get_item("messages")?)
        );
        assert_eq!(
            old_messages
                .get_item(0)?
                .get_item("content")?
                .extract::<String>()?,
            "original"
        );
        assert!(original.get_item("checkpoint")?.extract::<bool>()?);
        assert_eq!(current.is(&original), merge);
        if merge {
            assert_eq!(
                current.get_item("request_id")?.extract::<String>()?,
                "retained"
            );
            assert!(current.get_item("checkpoint")?.extract::<bool>()?);
        } else {
            assert!(original.get_item("messages")?.is(&old_messages));
            assert!(!current.contains("request_id")?);
            assert!(!current.contains("checkpoint")?);
        }
        Ok(())
    })
}

#[rstest]
#[case::failure_hook_only(false)]
#[case::success_hook_checks_block_flag(true)]
#[serial(python_interpreter)]
fn block_exception_preserves_stashed_context_for_later_hooks(
    initialized_python: &InitializedPython,
    #[case] dispatch_success: bool,
    #[values(Backend::Python, Backend::PreparedCall)] backend: Backend,
    #[values(InvocationMode::Direct, InvocationMode::Await)] mode: InvocationMode,
) -> PyResult<()> {
    run_pattern(
        initialized_python,
        backend,
        mode,
        |globals| globals.set_item("dispatch_success", dispatch_success),
        c"
def test_pattern():
    logger = Value()
    logger.details = {}
    request, failures, successes, success_calls = {}, [], [], []
    block = RuntimeError('blocked')

    def pre_call(data):
        logger.details['blocked'] = True
        data['logging_object'] = logger
        raise block

    def success(details):
        success_calls.append(details)
        if not details.get('blocked'):
            successes.append(details)

    def failure(data, error):
        failures.append((data.pop('logging_object'), error))

    with TestCase().assertRaises(RuntimeError) as caught:
        call(pre_call, request)
    assert caught.exception is block
    assert request['logging_object'] is logger
    assert logger.details == {'blocked': True}

    if dispatch_success:
        call(success, logger.details)
        assert successes == []
        assert success_calls[0] is logger.details
        unblocked_details = {}
        call(success, unblocked_details)
        assert successes == [unblocked_details]
        assert successes[0] is unblocked_details
    assert len(success_calls) == (2 if dispatch_success else 0)
    call(failure, request, caught.exception)

    assert len(failures) == 1
    assert failures[0][0] is logger
    assert failures[0][1] is block
    assert request == {}
    assert logger.details == {'blocked': True}
",
    )
}

#[rstest]
#[case::shared_inputs_lose_an_increment(false, false)]
#[case::reverse_completion_changes_writer_order(false, true)]
#[case::independent_snapshots_discard_edits(true, false)]
#[serial(python_interpreter)]
fn parallel_callbacks_share_state_without_isolation_or_automatic_return_merging(
    initialized_python: &InitializedPython,
    #[case] snapshots: bool,
    #[case] reverse: bool,
    #[values(Backend::Python, Backend::PreparedCall)] backend: Backend,
    #[values(InvocationMode::Direct, InvocationMode::Await)] mode: InvocationMode,
) -> PyResult<()> {
    run_pattern(
        initialized_python,
        backend,
        mode,
        |globals| {
            globals.set_item("snapshots", snapshots)?;
            globals.set_item("reverse", reverse)
        },
        c"
async def test_pattern():
    state = {'count': 0, 'metadata': {'writers': []}}
    entered = [asyncio.Event(), asyncio.Event()]
    release = [asyncio.Event(), asyncio.Event()]
    completed = [asyncio.Event(), asyncio.Event()]

    async def increment(data, index):
        previous = data['count']
        entered[index].set()
        await release[index].wait()
        data['count'] = previous + 1
        data['metadata']['writers'].append(index)
        completed[index].set()
        return {'ignored_replacement': True}

    inputs = [copy.deepcopy(state) for _ in range(2)] if snapshots else [state, state]
    tasks = [asyncio.create_task(invoke(increment, data, index)) for index, data in enumerate(inputs)]
    await asyncio.gather(*(event.wait() for event in entered))
    assert state == {'count': 0, 'metadata': {'writers': []}}
    assert all(not task.done() for task in tasks)

    order = [1, 0] if reverse else [0, 1]
    for position, index in enumerate(order):
        release[index].set()
        await completed[index].wait()
        assert inputs[index]['count'] == 1
        if position == 0:
            assert not tasks[1 - index].done()
            assert inputs[index]['metadata']['writers'] == [index]

    assert await asyncio.gather(*tasks) == [{'ignored_replacement': True}] * 2
    if snapshots:
        assert state == {'count': 0, 'metadata': {'writers': []}}
        assert inputs[0] is not inputs[1]
        for index, data in enumerate(inputs):
            assert data is not state
            assert data == {'count': 1, 'metadata': {'writers': [index]}}
    else:
        assert inputs[0] is inputs[1] is state
        assert state == {'count': 1, 'metadata': {'writers': order}}
",
    )
}

#[rstest]
#[case::after_return(false)]
#[case::after_block_exception(true)]
#[serial(python_interpreter)]
fn background_task_keeps_arguments_and_updates_live_state_after_callback_finishes(
    initialized_python: &InitializedPython,
    #[case] raises: bool,
    #[values(Backend::Python, Backend::PreparedCall)] backend: Backend,
    #[values(InvocationMode::Direct, InvocationMode::Await)] mode: InvocationMode,
) -> PyResult<()> {
    run_pattern(
        initialized_python,
        backend,
        mode,
        |globals| globals.set_item("raises", raises),
        c"
async def test_pattern():
    entered, release = asyncio.Event(), asyncio.Event()
    tasks, observed_loops = [], []
    state = Value()
    state.metadata = {'audit': 'pending'}
    reference = weakref.ref(state)
    event_loop = asyncio.get_running_loop()

    async def audit(data):
        observed_loops.append(asyncio.get_running_loop())
        entered.set()
        await release.wait()
        data.metadata['audit'] = 'complete'

    def schedule(data):
        observed_loops.append(asyncio.get_running_loop())
        tasks.append(asyncio.create_task(audit(data)))
        if raises:
            raise RuntimeError('blocked after scheduling')

    if raises:
        with TestCase().assertRaisesRegex(RuntimeError, 'blocked after scheduling'):
            await async_call(schedule, state)
    else:
        assert await async_call(schedule, state) is None

    metadata = state.metadata
    snapshot = copy.deepcopy(metadata)
    del state
    await entered.wait()
    gc.collect()
    assert reference() is not None
    assert metadata == {'audit': 'pending'}
    assert len(tasks) == 1
    assert not tasks[0].done(), 'callback completion must not wait for background work'
    assert tasks[0].get_loop() is event_loop
    assert len(observed_loops) == 2
    assert all(loop is event_loop for loop in observed_loops)

    release.set()
    assert await tasks[0] is None
    assert metadata == {'audit': 'complete'}
    assert snapshot == {'audit': 'pending'}
    tasks.clear()
    gc.collect()
    assert reference() is None
",
    )
}

#[rstest]
#[case::no_redaction_returns_original("passthrough")]
#[case::shallow_redaction_copies_only_response("shallow_redaction")]
#[case::per_key_copy_failure_shares_only_uncopyable_value("per_key_fallback")]
#[case::whole_copy_success_isolates_nested_values("deep_copy")]
#[case::whole_copy_failure_returns_original("whole_object_fallback")]
#[serial(python_interpreter)]
fn explicit_copy_boundaries_preserve_copy_depth_and_failure_fallback(
    initialized_python: &InitializedPython,
    #[case] copy_policy: &str,
    #[values(Backend::Python, Backend::PreparedCall)] backend: Backend,
    #[values(InvocationMode::Direct, InvocationMode::Await)] mode: InvocationMode,
) -> PyResult<()> {
    run_pattern(
        initialized_python,
        backend,
        mode,
        |globals| globals.set_item("copy_policy", copy_policy),
        c"
def test_pattern():
    lock = threading.Lock()
    state = {'messages': [{'content': 'private'}], 'response': {'content': 'private'}}
    if copy_policy in ('per_key_fallback', 'whole_object_fallback'):
        state['opaque'] = {'lock': lock, 'edits': []}

    def passthrough(data):
        return data

    def shallow_redaction(data):
        result = {**data, 'response': copy.deepcopy(data['response'])}
        result['response']['content'] = 'masked'
        return result

    def per_key_fallback(data):
        result = {}
        for key, value in data.items():
            try:
                result[key] = copy.deepcopy(value)
            except TypeError:
                result[key] = value
        return result

    def whole_object_fallback(data):
        try:
            return copy.deepcopy(data)
        except TypeError:
            return data

    callbacks = {
        'passthrough': passthrough,
        'shallow_redaction': shallow_redaction,
        'per_key_fallback': per_key_fallback,
        'deep_copy': whole_object_fallback,
        'whole_object_fallback': whole_object_fallback,
    }
    result = call(callbacks[copy_policy], state)
    returns_original = copy_policy in ('passthrough', 'whole_object_fallback')
    shares_messages = returns_original or copy_policy == 'shallow_redaction'

    assert (result is state) == returns_original
    assert (result['response'] is state['response']) == returns_original
    assert (result['messages'] is state['messages']) == shares_messages
    result['messages'][0]['content'] = 'later edit'
    assert state['messages'][0]['content'] == ('later edit' if shares_messages else 'private')
    assert state['response']['content'] == 'private'
    assert result['response']['content'] == ('masked' if copy_policy == 'shallow_redaction' else 'private')

    if 'opaque' in state:
        assert result['opaque'] is state['opaque']
        assert result['opaque']['lock'] is lock
        result['opaque']['edits'].append('shared despite copy')
        assert state['opaque']['edits'] == ['shared despite copy']
",
    )
}
