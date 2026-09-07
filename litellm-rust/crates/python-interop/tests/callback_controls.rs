use std::ffi::CStr;

use litellm_python_interop::{InvocationMode, InvocationOutcome, PreparedCall};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};
use rstest::rstest;
use serial_test::serial;

#[path = "support/mod.rs"]
mod support;

use support::python::{InitializedPython, initialized_python, item, scope};

enum ControlCall {
    Reference(Py<PyAny>),
    Prepared(PreparedCall),
}

impl ControlCall {
    fn new(
        py: Python<'_>,
        callback: Bound<'_, PyAny>,
        args: Bound<'_, PyTuple>,
        kwargs: Option<Bound<'_, PyDict>>,
        retained: bool,
        mode: InvocationMode,
    ) -> PyResult<Self> {
        if retained {
            return Ok(Self::Prepared(PreparedCall::new(
                mode,
                callback.unbind(),
                args.unbind(),
                kwargs.map(Bound::unbind),
            )));
        }
        let factory = py
            .import("callback_lifecycle")?
            .getattr("ReferenceFactory")?
            .call0()?;
        Ok(Self::Reference(
            factory
                .call_method1(
                    "prepare",
                    (callback, args, kwargs, mode == InvocationMode::Await),
                )?
                .unbind(),
        ))
    }

    fn invoke(&self, py: Python<'_>, mode: InvocationMode) -> PyResult<Py<PyAny>> {
        match self {
            Self::Reference(owner) => owner.call_method0(py, "invoke"),
            Self::Prepared(call) => match call.invoke(py)? {
                InvocationOutcome::Returned(value) => {
                    assert_eq!(mode, InvocationMode::Direct);
                    Ok(value)
                }
                InvocationOutcome::Awaitable(value) => {
                    assert_eq!(mode, InvocationMode::Await);
                    Ok(value)
                }
            },
        }
    }
}

#[rstest]
#[case::original_graph(c"(positional, keywords)", (true, true, true), (1, 1, 1))]
#[case::rebuilt_envelopes(c"(tuple(value for value in positional), dict(keywords))", (true, true, true), (1, 1, 1))]
#[case::shallow_payload(c"((copy.copy(positional[0]),), keywords)", (false, true, true), (0, 1, 1))]
#[case::copied_graph_keeps_cross_argument_alias(c"copy.deepcopy((positional, keywords))", (false, false, true), (0, 0, 0))]
#[case::separate_copies_break_cross_argument_alias(c"(copy.deepcopy(positional), copy.deepcopy(keywords))", (false, false, false), (0, 0, 0))]
#[serial(python_interpreter)]
fn argument_copy_boundaries_determine_identity_and_mutation_visibility(
    initialized_python: &InitializedPython,
    #[case] transform: &CStr,
    #[case] expected_identity: (bool, bool, bool),
    #[case] live_stages: (u8, u8, u8),
    #[values(false, true)] retained: bool,
    #[values(InvocationMode::Direct, InvocationMode::Await)] mode: InvocationMode,
) -> PyResult<()> {
    initialized_python.attach(|py| {
        let globals = scope(
            py,
            c"
import copy
from types import SimpleNamespace

nested = SimpleNamespace(stage=0)
original = SimpleNamespace(nested=nested, stage=0)
positional, keywords = (original,), {'alias': nested}

def observe(value, *, alias):
    return value, alias, (value.stage, value.nested.stage, alias.stage)

async def observe_async(value, *, alias):
    return observe(value, alias=alias)
",
        )?;
        let transformed = py.eval(transform, Some(&globals), None)?;
        let call = ControlCall::new(
            py,
            item(
                &globals,
                if mode == InvocationMode::Await {
                    "observe_async"
                } else {
                    "observe"
                },
            ),
            transformed.get_item(0)?.cast_into::<PyTuple>()?,
            Some(transformed.get_item(1)?.cast_into::<PyDict>()?),
            retained,
            mode,
        )?;
        let original = item(&globals, "original");
        let nested = item(&globals, "nested");
        original.setattr("stage", 1)?;
        nested.setattr("stage", 1)?;
        let pending = call.invoke(py, mode)?;
        original.setattr("stage", 2)?;
        nested.setattr("stage", 2)?;
        let result = if mode == InvocationMode::Await {
            let lifecycle = py.import("callback_lifecycle")?;
            lifecycle.call_method1(
                "run_checked",
                (lifecycle.call_method0("ReferenceFactory")?, pending),
            )?
        } else {
            pending.into_bound(py)
        };
        drop(call);
        let value = result.get_item(0)?;
        let alias = result.get_item(1)?;
        assert_eq!(
            (
                value.is(&original),
                value.getattr("nested")?.is(&nested),
                value.getattr("nested")?.is(&alias)
            ),
            expected_identity
        );
        let stage = if mode == InvocationMode::Await { 2 } else { 1 };
        assert_eq!(
            result.get_item(2)?.extract::<(u8, u8, u8)>()?,
            (
                live_stages.0 * stage,
                live_stages.1 * stage,
                live_stages.2 * stage
            )
        );
        Ok(())
    })
}

#[rstest]
#[case::original_result(None, (true, true))]
#[case::passthrough_result(Some(c"lambda value: value"), (true, true))]
#[case::shallow_result(Some(c"copy.copy"), (false, true))]
#[case::deep_result(Some(c"copy.deepcopy"), (false, false))]
#[serial(python_interpreter)]
fn result_copy_boundaries_determine_root_and_nested_identity(
    initialized_python: &InitializedPython,
    #[case] transform: Option<&CStr>,
    #[case] expected_identity: (bool, bool),
    #[values(false, true)] retained: bool,
    #[values(InvocationMode::Direct, InvocationMode::Await)] mode: InvocationMode,
) -> PyResult<()> {
    initialized_python.attach(|py| {
        let globals = scope(
            py,
            c"
import copy
from types import SimpleNamespace

original = SimpleNamespace(nested=SimpleNamespace(stage=0), stage=0)

def callback():
    return original

async def callback_async():
    return original
",
        )?;
        let call = ControlCall::new(
            py,
            item(
                &globals,
                if mode == InvocationMode::Await {
                    "callback_async"
                } else {
                    "callback"
                },
            ),
            PyTuple::empty(py),
            None,
            retained,
            mode,
        )?;
        let pending = call.invoke(py, mode)?;
        let settled = if mode == InvocationMode::Await {
            let lifecycle = py.import("callback_lifecycle")?;
            lifecycle.call_method1(
                "run_checked",
                (lifecycle.call_method0("ReferenceFactory")?, pending),
            )?
        } else {
            pending.into_bound(py)
        };
        let result = if let Some(transform) = transform {
            py.eval(transform, Some(&globals), None)?
                .call1((settled,))?
        } else {
            settled
        };
        drop(call);
        let original = item(&globals, "original");
        assert_eq!(
            (
                result.is(&original),
                result.getattr("nested")?.is(original.getattr("nested")?)
            ),
            expected_identity
        );
        Ok(())
    })
}
