//! Conversions between the router's typed values and the plain Python values the call's
//! driver reads and writes.

use std::sync::Arc;

use litellm_host_python::from_py;
use litellm_router::{
    engine::{Outcome, Override, RouterCall},
    failure::{Classified, Raised, Rejection},
    host::{Attempt, Invoked, MockFailure, Op, Target, TypedFallback},
    operation::Operation,
    pyrepr::PyNumber,
    settings::Fallbacks,
};
use pyo3::{
    exceptions::PyValueError,
    prelude::*,
    types::{PyDict, PyList, PyTuple},
};

use super::PyObj;

pub(super) fn router_call(arguments: &Bound<'_, PyDict>) -> PyResult<RouterCall> {
    let operation: String = required(arguments, "operation")?;
    let model: String = required(arguments, "model")?;
    let mut call = RouterCall::new(Operation::from_name(&operation), model);
    call.num_retries = optional(arguments, "num_retries")?;
    call.disable_fallbacks = optional(arguments, "disable_fallbacks")?.unwrap_or(false);
    call.fallbacks = fallbacks_override(arguments, "fallbacks")?;
    call.context_window_fallbacks = fallbacks_override(arguments, "context_window_fallbacks")?;
    call.content_policy_fallbacks = fallbacks_override(arguments, "content_policy_fallbacks")?;
    call.shared_logging = optional(arguments, "shared_logging")?.unwrap_or(false);
    call.web_search = optional(arguments, "web_search")?.unwrap_or(false);
    call.mock = optional::<String>(arguments, "mock")?
        .map(|name| match name.as_str() {
            "fallbacks" => Ok(MockFailure::Fallbacks),
            "context_window_fallbacks" => Ok(MockFailure::ContextWindowFallbacks),
            "content_policy_fallbacks" => Ok(MockFailure::ContentPolicyFallbacks),
            _ => Err(PyValueError::new_err(format!(
                "unknown mock failure {name:?}"
            ))),
        })
        .transpose()?;
    Ok(call)
}

fn required<T: for<'de> serde::Deserialize<'de>>(
    arguments: &Bound<'_, PyDict>,
    key: &str,
) -> PyResult<T> {
    optional(arguments, key)?
        .ok_or_else(|| PyValueError::new_err(format!("router call is missing {key:?}")))
}

fn optional<T: for<'de> serde::Deserialize<'de>>(
    arguments: &Bound<'_, PyDict>,
    key: &str,
) -> PyResult<Option<T>> {
    match arguments.get_item(key)? {
        Some(value) if !value.is_none() => from_py(&value).map(Some),
        _ => Ok(None),
    }
}

fn fallbacks_override(arguments: &Bound<'_, PyDict>, key: &str) -> PyResult<Override<Fallbacks>> {
    match arguments.get_item(key)? {
        None => Ok(Override::Inherit),
        Some(value) if value.is_none() => Ok(Override::Set(None)),
        Some(value) => from_py(&value).map(|fallbacks| Override::Set(Some(fallbacks))),
    }
}

/// `("ok", response)` or `("error", exception, classified)`.
pub(super) fn invoked(
    _py: Python<'_>,
    result: &Bound<'_, PyAny>,
) -> PyResult<Invoked<PyObj, PyObj>> {
    let result = result.cast::<PyTuple>()?;
    let tag: String = result.get_item(0)?.extract()?;
    match tag.as_str() {
        "ok" => Ok(Invoked::Success(Arc::new(result.get_item(1)?.unbind()))),
        "error" => Ok(Invoked::Failure {
            error: Arc::new(result.get_item(1)?.unbind()),
            classified: from_py::<Classified>(&result.get_item(2)?)?,
        }),
        _ => Err(PyValueError::new_err(format!(
            "unknown invoke result {tag:?}"
        ))),
    }
}

pub(super) fn attempt<'py>(py: Python<'py>, attempt: &Attempt<PyObj>) -> PyResult<Py<PyAny>> {
    let dict = PyDict::new(py);
    match &attempt.target {
        Target::Deployment(id) => dict.set_item("deployment_id", id)?,
        Target::Mock(mock) => dict.set_item(
            "mock",
            match mock {
                MockFailure::Fallbacks => "fallbacks",
                MockFailure::ContextWindowFallbacks => "context_window_fallbacks",
                MockFailure::ContentPolicyFallbacks => "content_policy_fallbacks",
            },
        )?,
    }
    dict.set_item("model_group", &attempt.model_group)?;
    dict.set_item("bucket", attempt.bucket)?;
    dict.set_item("fallback_depth", attempt.fallback_depth)?;
    dict.set_item("model_group_size", attempt.retry.model_group_size)?;
    dict.set_item("attempted_retries", attempt.retry.attempted_retries)?;
    dict.set_item("max_retries", attempt.retry.max_retries)?;
    dict.set_item("ops", ops(py, &attempt.ops)?)?;
    Ok(dict.into_any().unbind())
}

pub(super) fn outcome(py: Python<'_>, outcome: &Outcome) -> PyResult<Py<PyAny>> {
    let dict = PyDict::new(py);
    dict.set_item("model_group", &outcome.model_group)?;
    dict.set_item("deployment_id", &outcome.deployment_id)?;
    dict.set_item("attempted_retries", outcome.attempted_retries)?;
    dict.set_item("max_retries", outcome.max_retries)?;
    dict.set_item("attempted_fallbacks", outcome.attempted_fallbacks)?;
    Ok(dict.into_any().unbind())
}

pub(super) fn ops(py: Python<'_>, ops: &[Op<PyObj>]) -> PyResult<Py<PyAny>> {
    let list = PyList::empty(py);
    for op in ops {
        list.append(self::op(py, op)?)?;
    }
    Ok(list.into_any().unbind())
}

fn op<'py>(py: Python<'py>, op: &Op<PyObj>) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    match op {
        Op::LogRetry {
            bucket,
            model,
            error,
        } => {
            dict.set_item("op", "log_retry")?;
            dict.set_item("bucket", bucket)?;
            dict.set_item("model", model)?;
            dict.set_item("error", raised(py, error)?)?;
        }
        Op::OpenBucket { id, copy_of, stamp } => {
            dict.set_item("op", "open_bucket")?;
            dict.set_item("id", id)?;
            dict.set_item("copy_of", copy_of)?;
            dict.set_item("original_model_group", &stamp.original_model_group)?;
            dict.set_item("model_group", &stamp.model_group)?;
            dict.set_item("attempted_fallbacks", stamp.attempted_fallbacks)?;
            dict.set_item("max_fallbacks", stamp.max_fallbacks)?;
        }
        Op::StampRetries {
            error,
            max_retries,
            num_retries,
        } => {
            dict.set_item("op", "stamp_retries")?;
            dict.set_item("error", raised(py, error)?)?;
            dict.set_item("max_retries", max_retries)?;
            dict.set_item("num_retries", num_retries)?;
        }
        Op::MissingTypedFallbacks {
            error,
            kind,
            model_group,
        } => {
            dict.set_item("op", "missing_typed_fallbacks")?;
            dict.set_item("error", raised(py, error)?)?;
            dict.set_item(
                "kind",
                match kind {
                    TypedFallback::ContextWindow => "context_window",
                    TypedFallback::ContentPolicy => "content_policy",
                },
            )?;
            dict.set_item("model_group", model_group)?;
        }
        Op::NoFallbackGroup {
            error,
            lookup_groups,
        } => {
            dict.set_item("op", "no_fallback_group")?;
            dict.set_item("error", raised(py, error)?)?;
            dict.set_item("lookup_groups", lookup_groups)?;
        }
        Op::FallbackOutcome {
            error,
            model_group,
            attempted,
            last,
        } => {
            dict.set_item("op", "fallback_outcome")?;
            dict.set_item("error", raised(py, error)?)?;
            dict.set_item("model_group", model_group)?;
            dict.set_item("attempted", attempted)?;
            dict.set_item(
                "last",
                last.as_ref().map(|last| raised(py, last)).transpose()?,
            )?;
        }
    }
    Ok(dict)
}

/// The exception object itself, or a dict describing a rejection the router raised.
pub(super) fn raised(py: Python<'_>, raised: &Raised<PyObj>) -> PyResult<Py<PyAny>> {
    match raised {
        Raised::Host(error) => Ok(error.clone_ref(py)),
        Raised::Router { id, rejection } => {
            let dict = PyDict::new(py);
            dict.set_item("rejection_id", id)?;
            match rejection {
                Rejection::NoDeploymentsAvailable {
                    model,
                    cooldown_time,
                    cooldown_list,
                    model_ids,
                    enable_pre_call_checks,
                } => {
                    dict.set_item("kind", "no_deployments_available")?;
                    dict.set_item("model", model)?;
                    match cooldown_time {
                        PyNumber::Int(value) => dict.set_item("cooldown_time", value)?,
                        PyNumber::Float(value) => dict.set_item("cooldown_time", value)?,
                    }
                    dict.set_item("cooldown_list", cooldown_list)?;
                    dict.set_item("model_ids", model_ids)?;
                    dict.set_item("enable_pre_call_checks", enable_pre_call_checks)?;
                }
                Rejection::NoHealthyDeployments { model } => {
                    dict.set_item("kind", "no_healthy_deployments")?;
                    dict.set_item("model", model)?;
                }
            }
            Ok(dict.into_any().unbind())
        }
    }
}
