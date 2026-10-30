//! `RouterCoreHandle`: router-core behind a step-driven handle. Python drives each call through
//! [`RouterCall::next`], running every attempt, sleep, fallback check, selection and cache op the
//! call loop asks for and handing the outcome back; Rust never awaits Python. `select` is driven
//! the same way, since reading cooldowns is a cache step.

mod config;
mod host;

use std::{
    sync::{Arc, Mutex, PoisonError},
    time::Duration,
};

use litellm_error::Report;
use litellm_host_python::{from_py_argument, json_loads, to_py};
use litellm_router_core::{AttemptError, Error, FailureKind, Op, Router, RouterEvent};
use pyo3::{
    exceptions::{PyRuntimeError, PyValueError},
    prelude::*,
    types::{PyBool, PyDict, PyList, PyString},
};
use serde_json::Value;

use self::host::{
    Answer, Ask, CacheOp, CacheOutcome, DriveError, Driver, Finish, Stage, Step, StepHost, StepKind,
};
use crate::errors::RustBridgeDeclined;

/// The `error_kind` of a `failed` step.
#[derive(Clone, Copy, Debug, strum::IntoStaticStr)]
enum ErrorKind {
    #[strum(serialize = "config")]
    Config,
    #[strum(serialize = "unknown_model")]
    UnknownModel,
    #[strum(serialize = "no_healthy_deployments")]
    NoHealthyDeployments,
    #[strum(serialize = "attempt")]
    Attempt,
    #[strum(serialize = "cache")]
    Cache,
    #[strum(serialize = "retired")]
    Retired,
}

impl From<&Error> for ErrorKind {
    fn from(error: &Error) -> Self {
        match error {
            Error::Config(_) => Self::Config,
            Error::UnknownModel { .. } => Self::UnknownModel,
            Error::NoHealthyDeployments { .. } => Self::NoHealthyDeployments,
            Error::Attempt(_) => Self::Attempt,
            Error::Cache(_) => Self::Cache,
            Error::Retired => Self::Retired,
        }
    }
}

fn config_error_to_pyerr(error: config::Error) -> PyErr {
    let message = Report::new(&error).to_string();
    if error.declined() {
        RustBridgeDeclined::new_err(message)
    } else {
        PyValueError::new_err(message)
    }
}

fn drive_error_to_pyerr(error: DriveError) -> PyErr {
    let message = Report::new(&error).to_string();
    if error.is_request() {
        PyValueError::new_err(message)
    } else {
        PyRuntimeError::new_err(message)
    }
}

/// A document Python passes as a mapping or as its JSON text.
fn document(value: &Bound<'_, PyAny>) -> PyResult<Value> {
    match value.cast::<PyString>() {
        Ok(text) => {
            let parsed = json_loads(value.py(), text.to_str()?.as_bytes())?;
            from_py_argument(parsed.bind(value.py()))
        }
        Err(_) => from_py_argument(value),
    }
}

/// A router-core router whose host operations Python runs.
#[pyclass(frozen)]
pub(crate) struct RouterCoreHandle {
    router: Router,
}

#[pymethods]
impl RouterCoreHandle {
    #[new]
    fn new(config: &Bound<'_, PyAny>, dynamic: &Bound<'_, PyAny>) -> PyResult<Self> {
        let (config, dynamic) = (document(config)?, document(dynamic)?);
        let build = || {
            let config = config::router_config(config)?;
            let dynamic = config::dynamic_config(dynamic)?;
            Ok(Router::new(config, dynamic, Arc::new(StepHost))?)
        };
        build()
            .map(|router| Self { router })
            .map_err(config_error_to_pyerr)
    }

    /// Start a call; nothing runs until the first `next(None)`.
    fn start(&self, op: &str, request: &Bound<'_, PyAny>) -> PyResult<RouterCall> {
        let op: Op = op.parse().unwrap_or_else(|_| Op::Other(op.to_owned()));
        let request = config::request(document(request)?, None).map_err(config_error_to_pyerr)?;
        let router = self.router.clone();
        Ok(RouterCall::new(async move {
            router.call(op, request).await.map(Finish::Call)
        }))
    }

    /// Pick the deployment `model` routes to now, without calling it. Driven like `start`: its
    /// steps are the cache reads and selections the pick needs, and it ends with `done`
    /// (`deployment_id`, `model_group`) or `failed`.
    fn select(&self, model: &str, request: &Bound<'_, PyAny>) -> PyResult<RouterCall> {
        let request =
            config::request(document(request)?, Some(model)).map_err(config_error_to_pyerr)?;
        let router = self.router.clone();
        Ok(RouterCall::new(async move {
            let deployment = router.select(&Op::Completion, &request).await?;
            Ok(Finish::Selected {
                deployment_id: deployment.id.clone(),
                model_group: deployment.model_name.clone(),
            })
        }))
    }

    /// Apply `{"upsert": [deployments], "delete": [ids]}` and return the new registry version.
    fn apply(&self, changes: &Bound<'_, PyAny>) -> PyResult<u64> {
        let changes = config::changes(document(changes)?).map_err(config_error_to_pyerr)?;
        self.router
            .apply(changes)
            .map_err(|error| config_error_to_pyerr(error.into()))
    }

    fn registry_version(&self) -> u64 {
        self.router.registry().version()
    }

    fn retire(&self) {
        self.router.retire();
    }
}

/// One router call, driven a step at a time by Python.
#[pyclass(frozen)]
pub(crate) struct RouterCall {
    driver: Mutex<Driver>,
}

impl RouterCall {
    fn new(
        future: impl std::future::Future<Output = Result<Finish, Error>> + Send + 'static,
    ) -> Self {
        Self {
            driver: Mutex::new(Driver::new(future)),
        }
    }
}

#[pymethods]
impl RouterCall {
    /// Hand back the outcome of the last step (`None` on the first call) and return the next
    /// step, or the call's end.
    #[pyo3(signature = (outcome))]
    fn next<'py>(
        &self,
        py: Python<'py>,
        outcome: Option<Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let mut driver = self.driver.lock().unwrap_or_else(PoisonError::into_inner);
        let answer = match driver.waiting() {
            None => match outcome {
                None => None,
                Some(_) => return Err(drive_error_to_pyerr(DriveError::Unasked)),
            },
            Some(ask) => Some(answer(ask, outcome)?),
        };
        let step = driver.resume(answer).map_err(drive_error_to_pyerr)?;
        let dict = step_dict(py, step)?;
        dict.set_item("events", events_list(py, driver.take_events())?)?;
        Ok(dict)
    }
}

/// Read Python's outcome for `ask`.
fn answer(ask: &Ask, outcome: Option<Bound<'_, PyAny>>) -> PyResult<Answer> {
    let missing = || {
        drive_error_to_pyerr(DriveError::MissingAnswer {
            expected: ask.kind(),
        })
    };
    match ask {
        Ask::Sleep { .. } => Ok(Answer::Slept),
        Ask::AllowFallback { .. } => Ok(Answer::AllowFallback(
            outcome.ok_or_else(missing)?.extract()?,
        )),
        Ask::Select { .. } => Ok(Answer::Select(outcome.ok_or_else(missing)?.extract()?)),
        Ask::Attempt { .. } => {
            let outcome = outcome.ok_or_else(missing)?;
            attempt_outcome(outcome.cast::<PyDict>()?).map(Answer::Attempt)
        }
        Ask::Cache { op, .. } => cache_outcome(*op, outcome).map(Answer::Cache),
    }
}

/// `get`: the JSON value or None; `batch_get`: a list of JSON values or None, one per key;
/// `increment`: the new int or float; `set` and `delete`: None.
fn cache_outcome(op: CacheOp, outcome: Option<Bound<'_, PyAny>>) -> PyResult<CacheOutcome> {
    let outcome = outcome.filter(|outcome| !outcome.is_none());
    let json = |value: Option<Value>| value.filter(|value| !value.is_null());
    match op {
        CacheOp::Get => outcome
            .map(|outcome| from_py_argument::<Value>(&outcome))
            .transpose()
            .map(|value| CacheOutcome::Value(json(value))),
        CacheOp::BatchGet => {
            let outcome = outcome.ok_or_else(|| {
                PyValueError::new_err("a `batch_get` cache outcome must be a list, not None")
            })?;
            if !outcome.is_instance_of::<PyList>() {
                return Err(PyValueError::new_err(
                    "a `batch_get` cache outcome must be a list with one entry per key",
                ));
            }
            let values: Vec<Option<Value>> = from_py_argument(&outcome)?;
            Ok(CacheOutcome::Values(values.into_iter().map(json).collect()))
        }
        CacheOp::Increment => outcome
            .filter(|outcome| !outcome.is_instance_of::<PyBool>())
            .and_then(|outcome| outcome.extract::<f64>().ok())
            .map(CacheOutcome::Number)
            .ok_or_else(|| {
                PyValueError::new_err("an `increment` cache outcome must be the new int or float")
            }),
        CacheOp::Set | CacheOp::Delete => match outcome {
            None => Ok(CacheOutcome::Stored),
            Some(_) => Err(PyValueError::new_err(format!(
                "a `{op}` cache step has no outcome, so pass None"
            ))),
        },
    }
}

/// `{"ok": True, "total_tokens": int}` or `{"ok": False, "kind": str, "status": int | None,
/// "retry_after": float | None, "num_retries": int | None, "detail": str | None}`.
fn attempt_outcome(outcome: &Bound<'_, PyDict>) -> PyResult<Result<u64, AttemptError>> {
    fn item<'py, T: FromPyObjectOwned<'py>>(
        outcome: &Bound<'py, PyDict>,
        key: &str,
    ) -> PyResult<Option<T>> {
        match outcome.get_item(key)? {
            Some(value) if !value.is_none() => Ok(Some(value.extract().map_err(Into::into)?)),
            Some(_) | None => Ok(None),
        }
    }
    let ok: bool = item(outcome, "ok")?
        .ok_or_else(|| PyValueError::new_err("an attempt outcome needs `ok`"))?;
    if ok {
        return Ok(Ok(item(outcome, "total_tokens")?.unwrap_or(0)));
    }
    let kind: String = item(outcome, "kind")?
        .ok_or_else(|| PyValueError::new_err("a failed attempt outcome needs `kind`"))?;
    let kind: FailureKind = kind
        .parse()
        .map_err(|_| PyValueError::new_err(format!("`{kind}` is not a router failure kind")))?;
    let retry_after = item::<f64>(outcome, "retry_after")?
        .map(|seconds| {
            Duration::try_from_secs_f64(seconds)
                .map_err(|_| PyValueError::new_err("`retry_after` must be a non-negative number"))
        })
        .transpose()?;
    let mut failure = AttemptError::new(kind, item(outcome, "status")?);
    failure.retry_after = retry_after;
    failure.num_retries = item(outcome, "num_retries")?;
    failure.detail = item::<String>(outcome, "detail")?.map(Into::into);
    Ok(Err(failure))
}

/// `[{"kind": "retries_exhausted", "model_group": str, "attempted": int, "max": int}
///   | {"kind": "fallback", "to": str, "order": int | None, "level": int}]`.
fn events_list(py: Python<'_>, events: Vec<RouterEvent>) -> PyResult<Bound<'_, PyList>> {
    let list = PyList::empty(py);
    for event in events {
        let dict = PyDict::new(py);
        match event {
            RouterEvent::RetriesExhausted {
                model_group,
                attempted,
                max,
            } => {
                dict.set_item("kind", "retries_exhausted")?;
                dict.set_item("model_group", model_group)?;
                dict.set_item("attempted", attempted)?;
                dict.set_item("max", max)?;
            }
            RouterEvent::Fallback {
                to, order, level, ..
            } => {
                dict.set_item("kind", "fallback")?;
                dict.set_item("to", to)?;
                dict.set_item("order", order)?;
                dict.set_item("level", level)?;
            }
            _ => continue,
        }
        list.append(dict)?;
    }
    Ok(list)
}

fn step_dict(py: Python<'_>, step: Step) -> PyResult<Bound<'_, PyDict>> {
    let dict = PyDict::new(py);
    let kind = match &step {
        Step::Ask(ask) => ask.kind(),
        Step::Done(_) => StepKind::Done,
        Step::Failed(_) => StepKind::Failed,
    };
    dict.set_item("kind", <&'static str>::from(kind))?;
    match step {
        Step::Ask(ask) => ask_items(py, &dict, ask)?,
        Step::Done(finish) => done_items(&dict, finish)?,
        Step::Failed(error) => {
            let attempt = error.attempt();
            dict.set_item("error", Report::new(&error).to_string())?;
            dict.set_item("error_kind", <&'static str>::from(ErrorKind::from(&error)))?;
            dict.set_item("attempt_failed", attempt.is_some())?;
            dict.set_item("status", attempt.and_then(|failure| failure.status))?;
            dict.set_item(
                "failure_kind",
                attempt.map(|failure| <&'static str>::from(failure.kind)),
            )?;
        }
    }
    Ok(dict)
}

fn ask_items(py: Python<'_>, dict: &Bound<'_, PyDict>, ask: Ask) -> PyResult<()> {
    match ask {
        Ask::Attempt {
            deployment_id,
            model,
            model_group,
            target_order,
            candidate_ids,
            retry,
            fallback_depth,
            fallback_level,
            timeout,
        } => {
            dict.set_item("deployment_id", deployment_id)?;
            dict.set_item("model", model)?;
            dict.set_item("model_group", model_group)?;
            dict.set_item("target_order", target_order)?;
            dict.set_item("candidate_ids", PyList::new(py, candidate_ids)?)?;
            dict.set_item("retry", retry)?;
            dict.set_item("fallback_depth", fallback_depth)?;
            dict.set_item("fallback_level", fallback_level)?;
            dict.set_item("timeout", timeout)
        }
        Ask::Sleep { seconds } => dict.set_item("seconds", seconds),
        Ask::AllowFallback { target } => dict.set_item("target", target),
        Ask::Select {
            stage,
            model_group,
            candidates,
        } => {
            let (stage, strategy) = match stage {
                Stage::Plugins => ("plugins", None),
                Stage::Strategy(strategy) => ("strategy", Some(strategy)),
            };
            dict.set_item("stage", stage)?;
            dict.set_item("strategy", strategy)?;
            dict.set_item("model_group", model_group)?;
            dict.set_item("candidates", PyList::new(py, candidates)?)
        }
        Ask::Cache {
            op,
            key,
            keys,
            value,
            ttl,
        } => {
            dict.set_item("op", <&'static str>::from(op))?;
            dict.set_item("key", key)?;
            dict.set_item("keys", keys.map(|keys| PyList::new(py, keys)).transpose()?)?;
            dict.set_item("value", value.map(|value| to_py(py, &value)).transpose()?)?;
            dict.set_item("ttl", ttl)
        }
    }
}

fn done_items(dict: &Bound<'_, PyDict>, finish: Finish) -> PyResult<()> {
    match finish {
        Finish::Call(response) => {
            dict.set_item("deployment_id", response.deployment_id)?;
            dict.set_item("model_group", response.model_group)?;
            dict.set_item("attempted_retries", response.attempted_retries)?;
            dict.set_item("max_retries", response.max_retries)?;
            dict.set_item("attempted_fallbacks", response.attempted_fallbacks)?;
            dict.set_item("total_tokens", response.output.total_tokens)
        }
        Finish::Selected {
            deployment_id,
            model_group,
        } => {
            dict.set_item("deployment_id", deployment_id)?;
            dict.set_item("model_group", model_group)
        }
    }
}

#[cfg(test)]
mod tests {
    use pyo3::{ffi::c_str, prelude::*, types::PyDict};
    use rstest::rstest;

    const CONFIG: &str = r#"{"model_list": [
        {"model_name": "primary", "litellm_params": {"model": "openai/a"}, "model_info": {"id": "a1"}}
    ]}"#;

    const PAIR: &str = r#"{"model_list": [
        {"model_name": "primary", "litellm_params": {"model": "openai/a"}, "model_info": {"id": "a1"}},
        {"model_name": "primary", "litellm_params": {"model": "openai/b"}, "model_info": {"id": "a2"}}
    ]}"#;

    #[rstest]
    fn python_drives_calls_and_selections_through_the_handle() {
        Python::initialize();
        Python::attach(|py| {
            let locals = PyDict::new(py);
            locals.set_item("native", crate::native_module(py)).unwrap();
            locals.set_item("config", CONFIG).unwrap();
            locals.set_item("pair", PAIR).unwrap();
            py.run(
                c_str!(
                    r#"
import time
store = {}
asked = []

def answer_cache(step):
    asked.append(step)
    op, key = step["op"], step["key"]
    if op == "get":
        return store.get(key)
    if op == "batch_get":
        return [store.get(k) for k in step["keys"]]
    if op == "set":
        store[key] = step["value"]
        return None
    if op == "increment":
        store[key] = store.get(key, 0) + step["value"]
        return store[key]
    if op == "delete":
        store.pop(key, None)
        return None
    raise AssertionError(op)

def drive(call, outcome):
    step = call.next(outcome)
    while step["kind"] == "cache":
        step = call.next(answer_cache(step))
    return step

handle = native.RouterCoreHandle(config, {"num_retries": 1})
picked = drive(handle.select("primary", {}), None)
assert (picked["kind"], picked["deployment_id"], picked["model_group"]) == ("done", "a1", "primary"), picked
assert [step["op"] for step in asked] == ["batch_get"]
assert list(asked[0]) == ["kind", "op", "key", "keys", "value", "ttl", "events"], list(asked[0])
call = handle.start("acompletion", '{"model": "primary"}')
version = handle.registry_version()
assert handle.apply({"delete": ["gone"]}) == version + 1
steps = [drive(call, None)]
steps.append(drive(call, {"ok": False, "kind": "RateLimitError", "status": 429}))
steps.append(drive(call, None))
steps.append(drive(call, {"ok": True, "total_tokens": 5}))
kinds = [step["kind"] for step in steps]

cooled = native.RouterCoreHandle(pair, {"num_retries": 0})
call = cooled.start("acompletion", {"model": "primary"})
tried = drive(call, None)["deployment_id"]
assert drive(call, {"ok": False, "kind": "RateLimitError", "status": 429})["kind"] == "failed"
assert f"deployment:{tried}:cooldown" not in store, store
store[f"deployment:{tried}:cooldown"] = {"exception_received": "x", "status_code": "429", "timestamp": time.time(), "cooldown_time": 60.0}
other = drive(cooled.select("primary", {}), None)
assert other["kind"] == "done" and other["deployment_id"] != tried, other

call = handle.select("primary", {})
step = call.next(None)
assert step["op"] == "batch_get"
for bad in ([], [None, None], "nope", None):
    try:
        call.next(bad)
    except ValueError:
        pass
    else:
        raise AssertionError(f"{bad!r} should not answer a batch_get of one key")
assert drive(call, [None])["kind"] == "done"
"#
                ),
                Some(&locals),
                Some(&locals),
            )
            .unwrap();
            let kinds: Vec<String> = locals
                .get_item("kinds")
                .unwrap()
                .unwrap()
                .extract()
                .unwrap();
            assert_eq!(kinds, ["attempt", "sleep", "attempt", "done"]);
        });
    }

    #[rstest]
    #[case::unknown_key(r#"{"routing_strategy_args": {}}"#, true)]
    #[case::bad_json("{", false)]
    #[case::not_an_object("[]", false)]
    fn construction_failures_raise_the_class_python_reads(
        #[case] config: &str,
        #[case] declined: bool,
    ) {
        Python::initialize();
        Python::attach(|py| {
            let error = crate::native_module(py)
                .getattr("RouterCoreHandle")
                .and_then(|handle| handle.call1((config, "{}")))
                .unwrap_err();
            assert_eq!(
                error.is_instance_of::<crate::errors::RustBridgeDeclined>(py),
                declined,
                "{error}"
            );
            assert_eq!(
                error.is_instance_of::<pyo3::exceptions::PyValueError>(py),
                !declined,
                "{error}"
            );
        });
    }
}
