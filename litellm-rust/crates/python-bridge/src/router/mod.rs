//! `RouterCoreHandle`: router-core behind a step-driven handle. Python drives each call through
//! [`RouterCall::next`], running every attempt, sleep, fallback check, selection and cache op the
//! call loop asks for and handing the outcome back; Rust never awaits Python. `select` is driven
//! the same way, since reading cooldowns is a cache step.

mod config;
mod host;
mod retries;
mod stream;

use std::{
    sync::{Arc, Mutex, PoisonError},
    time::{Duration, SystemTime},
};

use litellm_error::Report;
use litellm_host_python::{from_py_argument, json_loads, to_py};
use litellm_router_core::{
    AttemptError, Blocked, Deployment, DeploymentSpec, Error, FailureKind, FailureReport, Op,
    PreRouted, RetryCause, Router, RouterEvent,
    health::{UsageMetric, current_minute, usage_key},
    registry::{DeploymentFilter, ResolveKind, ResolveScope},
    strategy::routing_context,
};
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
    #[strum(serialize = "paused")]
    Paused,
    #[strum(serialize = "not_pass_through")]
    NotPassThrough,
    #[strum(serialize = "no_pass_through_deployments")]
    NoPassThroughDeployments,
    #[strum(serialize = "only_markers")]
    OnlyMarkers,
    #[strum(serialize = "ptu_reserved")]
    PtuReserved,
    #[strum(serialize = "context_window_exceeded")]
    ContextWindowExceeded,
    #[strum(serialize = "unsupported_params")]
    UnsupportedParams,
    #[strum(serialize = "routing_plugins_left_nothing")]
    RoutingPluginsLeftNothing,
    #[strum(serialize = "routing_plugins_need_async")]
    RoutingPluginsNeedAsync,
    #[strum(serialize = "multiple_teams")]
    MultipleTeams,
    #[strum(serialize = "reserved")]
    Reserved,
    #[strum(serialize = "invalid_router_weights")]
    InvalidRouterWeights,
    #[strum(serialize = "non_numeric_weight")]
    NonNumericWeight,
    #[strum(serialize = "missing_routing_group")]
    MissingRoutingGroup,
    #[strum(serialize = "default_deployment_raises")]
    DefaultDeploymentRaises,
    #[strum(serialize = "index_out_of_range")]
    IndexOutOfRange,
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
            Error::Paused { .. } => Self::Paused,
            Error::NotPassThrough { .. } => Self::NotPassThrough,
            Error::NoPassThroughDeployments { .. } => Self::NoPassThroughDeployments,
            Error::OnlyMarkers { .. } => Self::OnlyMarkers,
            Error::PtuReserved { .. } => Self::PtuReserved,
            Error::ContextWindowExceeded { .. } => Self::ContextWindowExceeded,
            Error::UnsupportedParams { .. } => Self::UnsupportedParams,
            Error::RoutingPluginsLeftNothing { .. } => Self::RoutingPluginsLeftNothing,
            Error::RoutingPluginsNeedAsync => Self::RoutingPluginsNeedAsync,
            Error::MultipleTeams { .. } => Self::MultipleTeams,
            Error::Reserved { .. } => Self::Reserved,
            Error::InvalidRouterWeights => Self::InvalidRouterWeights,
            Error::NonNumericWeight { .. } => Self::NonNumericWeight,
            Error::MissingRoutingGroup { .. } => Self::MissingRoutingGroup,
            Error::DefaultDeploymentRaises { .. } => Self::DefaultDeploymentRaises,
            Error::IndexOutOfRange { .. } => Self::IndexOutOfRange,
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
                deployment: host::deployment_record(&deployment),
            })
        }))
    }

    /// List the deployments a pick for `model` would choose among, driven like `select`.
    fn healthy(&self, model: &str, request: &Bound<'_, PyAny>) -> PyResult<RouterCall> {
        let request =
            config::request(document(request)?, Some(model)).map_err(config_error_to_pyerr)?;
        let router = self.router.clone();
        Ok(RouterCall::new(async move {
            let candidates = router.healthy(&Op::Completion, &request).await?;
            // A deployment named by its id is listed alone, as a dict (`_core_async_get_healthy_deployments`).
            let single = candidates.kind.is_single() && candidates.healthy.len() == 1;
            let deployments = candidates.healthy;
            Ok(Finish::Healthy {
                single,
                deployment_ids: deployments.iter().map(|d| d.id.clone()).collect(),
                deployments: deployments
                    .iter()
                    .map(|d| host::deployment_record(d))
                    .collect(),
            })
        }))
    }

    /// Apply `{"upsert": [deployments], "delete": [ids], "settings": {dynamic document}}` and
    /// return the new registry version.
    fn apply(&self, changes: &Bound<'_, PyAny>) -> PyResult<u64> {
        let changes = config::changes(document(changes)?).map_err(config_error_to_pyerr)?;
        self.router
            .apply(changes)
            .map_err(|error| config_error_to_pyerr(error.into()))
    }

    fn registry_version(&self) -> u64 {
        self.router.registry().version()
    }

    /// Record a failure the router did not see itself (a health check, an interceptor, a
    /// logging callback), driven like `start`: its steps are cache ops, and it ends with `done`
    /// (`cooled`).
    #[pyo3(signature = (id, failure, deployment_cooldown=false, count_only=false, requested_group=None, cooldown_time=None, uncounted=false))]
    #[allow(clippy::too_many_arguments)]
    fn report_failure(
        &self,
        id: &str,
        failure: &Bound<'_, PyDict>,
        deployment_cooldown: bool,
        count_only: bool,
        requested_group: Option<String>,
        cooldown_time: Option<f64>,
        uncounted: bool,
    ) -> PyResult<RouterCall> {
        let failure = match attempt_outcome(failure)? {
            Err(failure) => failure,
            Ok(_) => {
                return Err(PyValueError::new_err(
                    "a reported failure needs `ok: False`",
                ));
            }
        };
        let router = self.router.clone();
        let id = id.to_owned();
        let report = FailureReport {
            deployment_cooldown,
            cooldown_time,
            count_only,
            requested_group,
            uncounted,
        };
        Ok(RouterCall::new(async move {
            router
                .report_failure(&id, &failure, report)
                .await
                .map(|cooled| Finish::Reported { cooled })
        }))
    }

    /// Record a success a logging callback saw, driven like `report_failure`; it ends with
    /// `done` (`key`: the success counter's key, or None for an unknown deployment).
    fn report_success(&self, id: &str, total_tokens: f64, requests: f64) -> RouterCall {
        let router = self.router.clone();
        let id = id.to_owned();
        RouterCall::new(async move {
            router
                .report_success(&id, total_tokens, requests)
                .await
                .map(|key| Finish::Counted { key })
        })
    }

    /// Count a success against `id`, configured or not; ends with `done` (`key`).
    fn count_success(&self, id: &str) -> RouterCall {
        let router = self.router.clone();
        let id = id.to_owned();
        RouterCall::new(async move {
            router
                .count_success(&id)
                .await
                .map(|key| Finish::Counted { key: Some(key) })
        })
    }

    /// The usage keys of this minute for every deployment of `model_group`:
    /// `[(id, tpm_key, rpm_key)]`, or None when the group has no deployment.
    fn usage_keys(&self, model_group: &str) -> Option<Vec<(String, String, String)>> {
        let minute = current_minute(SystemTime::now());
        let group = self.router.registry().group(model_group)?;
        Some(
            group
                .iter()
                .map(|deployment| {
                    let model = &deployment.params.model;
                    (
                        deployment.id.clone(),
                        usage_key(&deployment.id, model, UsageMetric::Tpm, &minute),
                        usage_key(&deployment.id, model, UsageMetric::Rpm, &minute),
                    )
                })
                .collect(),
        )
    }

    /// Every configured deployment, as `model_list` entries with their ids, in config order.
    fn deployments<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let registry = self.router.registry();
        let records = registry
            .deployments(&DeploymentFilter::default())
            .iter()
            .map(|deployment| to_py(py, &host::deployment_record(deployment)))
            .collect::<PyResult<Vec<_>>>()?;
        PyList::new(py, records)
    }

    /// The deployment with this id, or None.
    fn deployment(&self, py: Python<'_>, id: &str) -> PyResult<Option<Py<PyAny>>> {
        self.router
            .registry()
            .deployment(id)
            .map(|deployment| to_py(py, &host::deployment_record(&deployment)))
            .transpose()
    }

    /// What `name` resolves to: `{"model_group", "kind", "ids"}`, or None for an unknown name.
    #[pyo3(signature = (name, specific_deployment=false))]
    fn resolve<'py>(
        &self,
        py: Python<'py>,
        name: &str,
        specific_deployment: bool,
    ) -> PyResult<Option<Bound<'py, PyDict>>> {
        let registry = self.router.registry();
        let Some(resolution) = registry.resolve(
            name,
            ResolveScope {
                specific_deployment,
            },
        ) else {
            return Ok(None);
        };
        let dict = PyDict::new(py);
        dict.set_item("model_group", resolution.group)?;
        dict.set_item("kind", format!("{:?}", resolution.kind))?;
        dict.set_item(
            "ids",
            PyList::new(py, resolution.deployments.iter().map(|d| d.id.clone()))?,
        )?;
        Ok(Some(dict))
    }

    /// The deployments a pick for `model` would resolve to for `request`, before any team,
    /// cooldown or strategy filter (`_core_get_routable_deployments`); empty when it refuses.
    fn routable<'py>(
        &self,
        py: Python<'py>,
        model: &str,
        request: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyList>> {
        let request =
            config::request(document(request)?, Some(model)).map_err(config_error_to_pyerr)?;
        let records = litellm_router_core::resolve::routable(
            &self.router.registry(),
            &self.router.settings().dynamic,
            &request,
        )
        .iter()
        .map(|d| to_py(py, &host::deployment_record(d)))
        .collect::<PyResult<Vec<_>>>()?;
        PyList::new(py, records)
    }

    /// `_get_routing_context` for a pick of `model`: `(strategy, scope, group)`, spelled as a
    /// strategy select step spells them. A name that resolves to nothing reads its routing group
    /// membership as is.
    #[pyo3(signature = (model, routing_strategy=None))]
    fn routing_context(
        &self,
        model: &str,
        routing_strategy: Option<&str>,
    ) -> (String, &'static str, Option<String>) {
        let registry = self.router.registry();
        let settings = self.router.settings();
        let (kind, group) = registry
            .resolve(model, ResolveScope::default())
            .map_or((ResolveKind::Group, model.to_owned()), |resolution| {
                (resolution.kind, resolution.group)
            });
        let (strategy, scope) = routing_context(
            &registry,
            &settings.routing_strategy,
            routing_strategy,
            &kind,
            &group,
        );
        let (scope, group) = scope.wire();
        (strategy.to_string(), scope, group.map(str::to_owned))
    }

    fn retire(&self) {
        self.router.retire();
    }

    /// What may still take over a stream from `origin` that fails before content, and the
    /// retry budget for a failure of `kind` (see `stream::policy`).
    #[pyo3(signature = (request, origin, kind=None))]
    fn stream_policy<'py>(
        &self,
        py: Python<'py>,
        request: &Bound<'py, PyAny>,
        origin: &Bound<'py, PyAny>,
        kind: Option<&str>,
    ) -> PyResult<Bound<'py, PyDict>> {
        stream::policy(py, &self.router, request, origin, kind)
    }

    /// Retry a stream from `origin` that failed before content within its group, driven like
    /// `start`: `done` when a retry returned a response, `failed` with the newest failure.
    fn retry_stream(
        &self,
        op: &str,
        request: &Bound<'_, PyAny>,
        origin: &Bound<'_, PyAny>,
        failure: &Bound<'_, PyDict>,
    ) -> PyResult<RouterCall> {
        stream::recover(
            &self.router,
            stream::Recovery::Retry,
            op,
            request,
            origin,
            failure,
        )
    }

    /// The fallback walk for a stream from `origin` that failed before content, driven like
    /// `start`.
    fn fallback_stream(
        &self,
        op: &str,
        request: &Bound<'_, PyAny>,
        origin: &Bound<'_, PyAny>,
        failure: &Bound<'_, PyDict>,
    ) -> PyResult<RouterCall> {
        stream::recover(
            &self.router,
            stream::Recovery::Fallback,
            op,
            request,
            origin,
            failure,
        )
    }

    /// `get_num_retries_from_retry_policy` for a failure shaped like an attempt outcome.
    #[pyo3(signature = (failure, model_group=None))]
    fn retry_policy_retries(
        &self,
        failure: &Bound<'_, PyDict>,
        model_group: Option<&str>,
    ) -> PyResult<Option<u32>> {
        retries::policy_retries(&self.router, failure, model_group)
    }

    /// `_core_get_allowed_fails_from_policy` for a failure shaped like an attempt outcome.
    fn allowed_fails_for(&self, failure: &Bound<'_, PyDict>) -> PyResult<Option<u32>> {
        retries::allowed_fails(&self.router, failure)
    }

    /// `should_retry_this_error` as a predicate, for a failure shaped like an attempt outcome.
    fn should_retry(
        &self,
        failure: &Bound<'_, PyDict>,
        counts: (usize, usize),
        fallbacks: (bool, bool, bool),
    ) -> PyResult<bool> {
        retries::should_retry(failure, counts, fallbacks)
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
        Ask::Select {
            stage: Stage::PreCall,
            ..
        } => {
            let facts = from_py_argument(&outcome.ok_or_else(missing)?)?;
            serde_json::from_value(facts)
                .map(Answer::PreCall)
                .map_err(|error| PyValueError::new_err(format!("bad pre-call facts: {error}")))
        }
        Ask::Select {
            stage: Stage::RoutingPlugins { .. },
            ..
        } => {
            // `{"models": [...] | None}`: a bare None would read as no outcome.
            let outcome = outcome.ok_or_else(missing)?;
            let models = outcome.cast::<PyDict>()?.get_item("models")?;
            Ok(Answer::Models(match models {
                Some(models) if !models.is_none() => Some(models.extract()?),
                _ => None,
            }))
        }
        Ask::Select {
            stage: Stage::CallbackFilters,
            ..
        } => {
            // The remaining ids, or a failure mapping as an attempt reports one.
            let outcome = outcome.ok_or_else(missing)?;
            if let Ok(failure) = outcome.cast::<PyDict>() {
                return match attempt_outcome(failure)? {
                    Err(error) => Ok(Answer::Filtered(Err(error))),
                    Ok(_) => Err(PyValueError::new_err(
                        "a `callback_filters` outcome is the remaining ids or a failure",
                    )),
                };
            }
            Ok(Answer::Filtered(Ok(outcome.extract()?)))
        }
        Ask::Select { .. } => Ok(Answer::Select(outcome.ok_or_else(missing)?.extract()?)),
        Ask::PreRoute { .. } => pre_route_outcome(outcome).map(Answer::PreRoute),
        Ask::Attempt { .. } => {
            let outcome = outcome.ok_or_else(missing)?;
            let outcome = outcome.cast::<PyDict>()?;
            let blocked = match outcome.get_item("blocked")? {
                Some(value) if !value.is_none() => {
                    let name: String = value.extract()?;
                    Some(name.parse::<Blocked>().map_err(|_| {
                        PyValueError::new_err(format!(
                            "`{name}` is not a way a response is blocked"
                        ))
                    })?)
                }
                Some(_) | None => None,
            };
            attempt_outcome(outcome).map(|result| match (result, blocked) {
                (Ok(total_tokens), Some(blocked)) => Answer::Blocked(total_tokens, blocked),
                (result, _) => Answer::Attempt(result),
            })
        }
        Ask::Cache { op, .. } => cache_outcome(*op, outcome).map(Answer::Cache),
    }
}

/// A pre-route answer: `None`, a model group name, or a custom strategy's deployment document.
fn pre_route_outcome(outcome: Option<Bound<'_, PyAny>>) -> PyResult<Option<PreRouted>> {
    let Some(outcome) = outcome.filter(|outcome| !outcome.is_none()) else {
        return Ok(None);
    };
    if let Ok(model) = outcome.cast::<PyString>() {
        return Ok(Some(PreRouted::Model(model.to_str()?.to_owned())));
    }
    let spec: DeploymentSpec = serde_json::from_value(document(&outcome)?)
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    let deployment = Deployment::from_spec(spec)
        .map_err(|error| PyValueError::new_err(Report::new(&error).to_string()))?;
    Ok(Some(PreRouted::Deployment(Arc::new(deployment))))
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
pub(crate) fn attempt_outcome(outcome: &Bound<'_, PyDict>) -> PyResult<Result<u64, AttemptError>> {
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
    failure.recorded = item::<bool>(outcome, "recorded")?.unwrap_or(false);
    failure.names_provider = item::<bool>(outcome, "llm_provider")?.unwrap_or(false);
    failure.classes = item::<Vec<String>>(outcome, "classes")?
        .unwrap_or_default()
        .into_iter()
        .map(Into::into)
        .collect();
    Ok(Err(failure))
}

/// `[{"kind": "retries_exhausted", "model_group": str, "attempted": int, "max": int}
///   | {"kind": "fallback", "to": str, "order": int | None, "level": int}]`.
/// The `cause` of a `retry_logged` or `retry_policy_raised` event, with what names it.
fn cause_items(dict: &Bound<'_, PyDict>, cause: RetryCause) -> PyResult<()> {
    match cause {
        RetryCause::Attempt => dict.set_item("cause", "attempt"),
        RetryCause::Level => dict.set_item("cause", "level"),
        RetryCause::Other => dict.set_item("cause", "other"),
        RetryCause::Pick(picked) => {
            dict.set_item("cause", "pick")?;
            pick_items(dict, &picked.0)
        }
    }
}

/// A pick's error, as the `failed` step names it (`error_kind` and what names it), with the
/// model its message names (`failed_model`) when the error carries one.
fn pick_items(dict: &Bound<'_, PyDict>, error: &Error) -> PyResult<()> {
    error_items(dict, error)?;
    match error {
        Error::NoHealthyDeployments { model } | Error::UnknownModel { model } => {
            dict.set_item("failed_model", model)
        }
        _ => Ok(()),
    }
}

/// `error_kind` and the fields that name the error.
fn error_items(dict: &Bound<'_, PyDict>, error: &Error) -> PyResult<()> {
    dict.set_item("error", Report::new(error).to_string())?;
    dict.set_item("error_kind", <&'static str>::from(ErrorKind::from(error)))?;
    match error {
        Error::OnlyMarkers { model }
        | Error::PtuReserved { model }
        | Error::MultipleTeams { model }
        | Error::DefaultDeploymentRaises { model }
        | Error::IndexOutOfRange { model } => dict.set_item("error_model", model),
        Error::ContextWindowExceeded { model, detail } => {
            dict.set_item("error_model", model)?;
            dict.set_item("error_detail", detail)
        }
        Error::UnsupportedParams { model, params } => {
            dict.set_item("error_model", model)?;
            dict.set_item("error_params", PyList::new(dict.py(), params)?)
        }
        Error::RoutingPluginsLeftNothing { models } => {
            dict.set_item("error_params", PyList::new(dict.py(), models)?)
        }
        Error::Reserved {
            model,
            until,
            timezone,
        } => {
            dict.set_item("error_model", model)?;
            dict.set_item("error_detail", format!("{until} {timezone}"))
        }
        Error::NonNumericWeight { metric } => dict.set_item("error_detail", metric),
        Error::MissingRoutingGroup { group } => dict.set_item("error_model", group),
        _ => Ok(()),
    }
}

fn events_list(py: Python<'_>, events: Vec<RouterEvent>) -> PyResult<Bound<'_, PyList>> {
    let list = PyList::empty(py);
    for event in events {
        let dict = PyDict::new(py);
        match event {
            RouterEvent::RetriesExhausted {
                model_group,
                attempted,
                max,
                level,
                picked,
            } => {
                dict.set_item("kind", "retries_exhausted")?;
                dict.set_item("model_group", model_group)?;
                dict.set_item("attempted", attempted)?;
                dict.set_item("max", max)?;
                dict.set_item("level", level)?;
                dict.set_item("picked", picked)?;
            }
            RouterEvent::Fallback {
                from,
                to,
                order,
                depth,
                level,
                scope,
                parent,
                max_fallbacks,
            } => {
                dict.set_item("kind", "fallback")?;
                dict.set_item("from", from)?;
                dict.set_item("to", to)?;
                dict.set_item("order", order)?;
                dict.set_item("depth", depth)?;
                dict.set_item("level", level)?;
                dict.set_item("scope", scope)?;
                dict.set_item("parent", parent)?;
                dict.set_item("max_fallbacks", max_fallbacks)?;
            }
            RouterEvent::Blocked {
                level,
                model,
                blocked,
            } => {
                dict.set_item("kind", "blocked")?;
                dict.set_item("level", level)?;
                dict.set_item("model", model)?;
                dict.set_item("blocked", <&'static str>::from(blocked))?;
            }
            RouterEvent::FallbackDone {
                from,
                to,
                level,
                scope,
                parent,
                succeeded,
            } => {
                dict.set_item("kind", "fallback_done")?;
                dict.set_item("from", from)?;
                dict.set_item("to", to)?;
                dict.set_item("level", level)?;
                dict.set_item("scope", scope)?;
                dict.set_item("parent", parent)?;
                dict.set_item("succeeded", succeeded)?;
            }
            RouterEvent::RetryLogged {
                scope,
                level,
                model,
                attempted_retries,
                cause,
            } => {
                dict.set_item("kind", "retry_logged")?;
                dict.set_item("scope", scope)?;
                dict.set_item("level", level)?;
                dict.set_item("model", model)?;
                dict.set_item("attempted_retries", attempted_retries)?;
                cause_items(&dict, cause)?;
            }
            RouterEvent::RetryPolicyRaised {
                level,
                model,
                cause,
            } => {
                dict.set_item("kind", "retry_policy_raised")?;
                dict.set_item("level", level)?;
                dict.set_item("model", model)?;
                cause_items(&dict, cause)?;
            }
            RouterEvent::LevelFailed {
                level,
                model,
                error,
                handling,
            } => {
                dict.set_item("kind", "level_failed")?;
                dict.set_item("level", level)?;
                dict.set_item("model", model)?;
                dict.set_item("handling", handling)?;
                pick_items(&dict, &error.0)?;
            }
            RouterEvent::MockFailed {
                mock,
                level,
                model,
                num_retries,
            } => {
                dict.set_item("kind", "mock_failed")?;
                dict.set_item("mock", <&'static str>::from(mock))?;
                dict.set_item("level", level)?;
                dict.set_item("model", model)?;
                dict.set_item("num_retries", num_retries)?;
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
            error_items(&dict, &error)?;
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
            deployment,
            model,
            model_group,
            target_order,
            candidate_ids,
            retry,
            fallback_depth,
            fallback_level,
            timeout,
            params,
            scope,
            max_retries,
            model_group_size,
        } => {
            dict.set_item("scope", scope)?;
            dict.set_item("max_retries", max_retries)?;
            dict.set_item("model_group_size", model_group_size)?;
            dict.set_item("deployment_id", deployment_id)?;
            dict.set_item("deployment", to_py(py, &deployment)?)?;
            dict.set_item("model", model)?;
            dict.set_item("model_group", model_group)?;
            dict.set_item("target_order", target_order)?;
            dict.set_item("candidate_ids", PyList::new(py, candidate_ids)?)?;
            dict.set_item("retry", retry)?;
            dict.set_item("fallback_depth", fallback_depth)?;
            dict.set_item("fallback_level", fallback_level)?;
            dict.set_item("timeout", timeout)?;
            dict.set_item("params", params.map(|p| to_py(py, &p)).transpose()?)
        }
        Ask::Sleep { seconds } => dict.set_item("seconds", seconds),
        Ask::AllowFallback { target } => dict.set_item("target", target),
        Ask::Select {
            stage,
            model_group,
            name,
            scope,
            group,
            pool,
            candidates,
        } => {
            let (stage, strategy) = match stage {
                Stage::Plugins => ("plugins", None),
                Stage::Strategy(strategy) => ("strategy", Some(strategy)),
                Stage::Tags => ("tags", None),
                Stage::PreCall => ("pre_call", None),
                Stage::TextOnly => ("text_only", None),
                Stage::CallbackFilters => ("callback_filters", None),
                Stage::RoutingPlugins { run } => {
                    dict.set_item("run", run)?;
                    ("routing_plugins", None)
                }
            };
            dict.set_item("pool", PyList::new(py, pool)?)?;
            dict.set_item("stage", stage)?;
            dict.set_item("strategy", strategy)?;
            dict.set_item("model_group", model_group)?;
            dict.set_item("model", name)?;
            dict.set_item("scope", scope)?;
            dict.set_item("group", group)?;
            dict.set_item("candidates", PyList::new(py, candidates)?)
        }
        Ask::PreRoute {
            model,
            fallback_depth,
            retry,
        } => {
            dict.set_item("model", model)?;
            dict.set_item("fallback_depth", fallback_depth)?;
            dict.set_item("retry", retry)
        }
        Ask::Cache {
            op,
            key,
            keys,
            value,
            ttl,
            local,
        } => {
            dict.set_item("op", <&'static str>::from(op))?;
            dict.set_item("key", key)?;
            dict.set_item("keys", keys.map(|keys| PyList::new(py, keys)).transpose()?)?;
            dict.set_item("value", value.map(|value| to_py(py, &value)).transpose()?)?;
            dict.set_item("ttl", ttl)?;
            dict.set_item("local", local)
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
            deployment,
        } => {
            dict.set_item("deployment_id", deployment_id)?;
            dict.set_item("model_group", model_group)?;
            dict.set_item("deployment", to_py(dict.py(), &deployment)?)
        }
        Finish::Reported { cooled } => dict.set_item("cooled", cooled),
        Finish::Counted { key } => dict.set_item("key", key),
        Finish::Healthy {
            single,
            deployment_ids,
            deployments,
        } => {
            dict.set_item("single", single)?;
            dict.set_item("deployment_ids", deployment_ids)?;
            let records = deployments
                .iter()
                .map(|d| to_py(dict.py(), d))
                .collect::<PyResult<Vec<_>>>()?;
            dict.set_item("deployments", PyList::new(dict.py(), records)?)
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
assert list(asked[0]) == ["kind", "op", "key", "keys", "value", "ttl", "local", "events"], list(asked[0])
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
attempt = drive(call, None)
tried = attempt["deployment_id"]
assert attempt["deployment"]["model_info"]["id"] == tried, attempt
assert attempt["deployment"]["litellm_params"]["model"].startswith("openai/"), attempt
assert [d["model_info"]["id"] for d in cooled.deployments()] == ["a1", "a2"]
assert cooled.deployment("a2")["model_name"] == "primary"
assert cooled.resolve("primary")["ids"] == ["a1", "a2"]
assert cooled.resolve("nope") is None
assert drive(call, {"ok": False, "kind": "RateLimitError", "status": 429})["kind"] == "failed"
# The core records the failure itself: the 429 cools the deployment down in the host's store.
assert f"deployment:{tried}:cooldown" in store, store
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
