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
use litellm_router_core::lookup::{ListScope, Lookup, Row};
use litellm_router_core::{
    AttemptError, Blocked, BuildPath, Built, Deployment, DeploymentSpec, Error, FailureKind,
    FailureReport, Op, OutcomeCallbacks, PreRouted, Provided, RetryCause, Router, RouterEvent,
    SecretAt, SuccessReport,
    build::Install,
    health::{UsageMetric, current_minute, usage_key},
    registry::{DeploymentFilter, ResolveKind, ResolveScope},
    strategy::routing_context,
};
use pyo3::{
    exceptions::{PyRuntimeError, PyValueError},
    prelude::*,
    types::{PyBool, PyDict, PyList, PyString, PyTuple},
};
use serde_json::Value;

use self::host::{
    CacheOp, CacheOutcome, DriveError, Driver, Finish, HostCall, HostCallResponse,
    PythonRandomness, Stage, Step, StepHost, StepKind,
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
    #[strum(serialize = "endpoint_not_supported")]
    EndpointNotSupported,
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
    #[strum(serialize = "model_info")]
    ModelInfo,
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
            Error::EndpointNotSupported { .. } => Self::EndpointNotSupported,
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
            Error::ModelInfo { .. } => Self::ModelInfo,
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

/// The `OutcomeCallbacks` a reported outcome came through; none is a report no callback makes.
fn outcome_callbacks(value: Option<&Bound<'_, PyAny>>) -> PyResult<OutcomeCallbacks> {
    let Some(value) = value else {
        return Ok(OutcomeCallbacks::default());
    };
    serde_json::from_value(document(value)?)
        .map_err(|error| PyValueError::new_err(error.to_string()))
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
            let config = config::parse_router_config(config)?;
            let dynamic = config::parse_dynamic_config(dynamic)?;
            Ok(Router::builder(config, dynamic, Arc::new(StepHost))
                .randomness(Arc::new(PythonRandomness))
                .build()?)
        };
        build()
            .map(|router| Self { router })
            .map_err(config_error_to_pyerr)
    }

    /// Start a call; nothing runs until the first `next(None)`.
    fn start(&self, op: &str, request: &Bound<'_, PyAny>) -> PyResult<RouterCall> {
        let op = Op::new(op);
        let request =
            config::parse_request(document(request)?, None).map_err(config_error_to_pyerr)?;
        let router = self.router.clone();
        Ok(RouterCall::new(async move {
            router.call(op, request).await.map(Finish::Call)
        }))
    }

    /// Pick the deployment `model` routes to now, without calling it. Driven like `start`: its
    /// steps are the cache reads and selections the pick needs, and it ends with `done`
    /// (`deployment_id`, `model_group`) or `failed`.
    fn select(&self, model: &str, request: &Bound<'_, PyAny>) -> PyResult<RouterCall> {
        let request = config::parse_request(document(request)?, Some(model))
            .map_err(config_error_to_pyerr)?;
        let router = self.router.clone();
        Ok(RouterCall::new(async move {
            let deployment = router.select(&Op::from("acompletion"), &request).await?;
            Ok(Finish::Selected {
                deployment_id: deployment.id.clone(),
                model_group: deployment.model_name.clone(),
                deployment: host::deployment_record(&deployment),
            })
        }))
    }

    /// List the deployments a pick for `model` would choose among, driven like `select`.
    fn healthy(&self, model: &str, request: &Bound<'_, PyAny>) -> PyResult<RouterCall> {
        let request = config::parse_request(document(request)?, Some(model))
            .map_err(config_error_to_pyerr)?;
        let router = self.router.clone();
        Ok(RouterCall::new(async move {
            let candidates = router.healthy(&Op::from("acompletion"), &request).await?;
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
        let changes = config::parse_changes(document(changes)?).map_err(config_error_to_pyerr)?;
        self.router
            .apply(changes)
            .map_err(|error| config_error_to_pyerr(error.into()))
    }

    /// Build `entries` (model_list entries) through the host, driven like `start`: its steps
    /// are `secrets`, `validate`, `provider` and `added`, and it ends with `done` (`built`: one
    /// `{"index", "id", "provider"}` per deployment installed). `path` is `create`
    /// (`set_model_list`) or `add` (`add_deployment`); `install` is `replace`, `append` or
    /// `upsert`.
    #[pyo3(signature = (entries, path, resolve_secrets, install))]
    fn build(
        &self,
        entries: &Bound<'_, PyAny>,
        path: &str,
        resolve_secrets: bool,
        install: &str,
    ) -> PyResult<RouterCall> {
        let entries: Vec<serde_json::Map<String, Value>> =
            serde_json::from_value(document(entries)?)
                .map_err(|error| PyValueError::new_err(format!("bad model_list: {error}")))?;
        let path: BuildPath = path
            .parse()
            .map_err(|_| PyValueError::new_err(format!("`{path}` is not a build path")))?;
        let install: Install = install
            .parse()
            .map_err(|_| PyValueError::new_err(format!("`{install}` is not a build install")))?;
        let router = self.router.clone();
        Ok(RouterCall::new(async move {
            router
                .build(path, entries, resolve_secrets, install)
                .await
                .map(Finish::Built)
        }))
    }

    /// `get_router_model_info` for the deployment `query` describes (a
    /// `RouterModelInfoQuery` document), driven like `start`: its steps are `model_info`
    /// lookups, and it ends with `done` (`info`: the merged map or None, `user_keys`: the keys
    /// whose values are the deployment's own).
    fn router_model_info(&self, query: &Bound<'_, PyAny>) -> PyResult<RouterCall> {
        let query: litellm_router_core::model_info::RouterModelInfoQuery =
            serde_json::from_value(document(query)?)
                .map_err(|error| PyValueError::new_err(format!("bad model-info query: {error}")))?;
        let router = self.router.clone();
        Ok(RouterCall::new(async move {
            router
                .router_model_info(&query)
                .await
                .map(Finish::RouterModelInfo)
        }))
    }

    /// `get_deployment_model_info` for deployment `model_id` serving `model_name`, driven like
    /// `router_model_info`; it ends with `done` (`info`: the merged map or None).
    fn deployment_model_info(
        &self,
        model_id: &str,
        model_name: &Bound<'_, PyAny>,
    ) -> PyResult<RouterCall> {
        let model_id = model_id.to_owned();
        let model_name = from_py_argument::<Value>(model_name)?;
        let router = self.router.clone();
        Ok(RouterCall::new(async move {
            router
                .deployment_model_info(&model_id, &model_name)
                .await
                .map(Finish::DeploymentModelInfo)
        }))
    }

    /// `_set_model_group_info` for the deployments `query` names (a `GroupInfoQuery` document),
    /// driven like `router_model_info`; it ends with `done` (`value`: None when no deployment
    /// matched, else `{"assignments": {field: value}}`, what the merge set on the group the host
    /// built from its first deployment).
    fn model_group_info(&self, query: &Bound<'_, PyAny>) -> PyResult<RouterCall> {
        let query: litellm_router_core::group_info::GroupInfoQuery =
            serde_json::from_value(document(query)?)
                .map_err(|error| PyValueError::new_err(format!("bad group-info query: {error}")))?;
        let router = self.router.clone();
        Ok(RouterCall::new(async move {
            let info = router.model_group_info(&query).await?;
            Ok(Finish::Value(info.map_or(
                Value::Null,
                |info| serde_json::json!({"assignments": info.assignments}),
            )))
        }))
    }

    /// `get_configured_token_limits` for the deployment with `id` and its `configured` model_info,
    /// driven like `router_model_info`; it ends with `done` (`value`: `[max_input, max_output]`).
    fn configured_token_limits(
        &self,
        id: &Bound<'_, PyAny>,
        configured: &Bound<'_, PyAny>,
    ) -> PyResult<RouterCall> {
        let id = from_py_argument::<Value>(id)?;
        let Value::Object(configured) = document(configured)? else {
            return Err(PyValueError::new_err(
                "the configured model_info must be a dict",
            ));
        };
        let router = self.router.clone();
        Ok(RouterCall::new(async move {
            let (input, output) = router.configured_token_limits(id, &configured).await?;
            Ok(Finish::Value(serde_json::json!([input, output])))
        }))
    }

    /// `get_model_listing_info` over the `deployments` a listed name names, driven like
    /// `router_model_info`; it ends with `done` (`value`: `{"cost_map_keys", "max_input_tokens",
    /// "max_output_tokens"}`).
    fn model_listing_info(&self, deployments: &Bound<'_, PyAny>) -> PyResult<RouterCall> {
        let deployments: Vec<Value> = from_py_argument(deployments)?;
        let router = self.router.clone();
        Ok(RouterCall::new(async move {
            let listing = router.model_listing_info(&deployments).await?;
            Ok(Finish::Value(serde_json::json!({
                "cost_map_keys": listing.cost_map_keys,
                "max_input_tokens": listing.max_input_tokens,
                "max_output_tokens": listing.max_output_tokens,
            })))
        }))
    }

    /// `get_configured_mode` / `get_configured_display_name`: `value` when it is a string that is
    /// not blank.
    #[staticmethod]
    fn configured_text(value: &Bound<'_, PyAny>) -> PyResult<Option<String>> {
        let value = from_py_argument::<Value>(value).ok();
        Ok(litellm_router_core::group_info::configured_text(
            value.as_ref(),
        ))
    }

    /// `get_configured_service_tiers`: each of `model_infos`' `service_tiers`, unvalidated.
    #[staticmethod]
    fn configured_service_tiers(
        py: Python<'_>,
        model_infos: &Bound<'_, PyAny>,
    ) -> PyResult<Vec<Py<PyAny>>> {
        let model_infos: Vec<serde_json::Map<String, Value>> = from_py_argument(model_infos)?;
        litellm_router_core::group_info::configured_service_tiers(&model_infos)
            .iter()
            .map(|tiers| to_py(py, tiers))
            .collect()
    }

    /// The routing-cache key deployment `id` serving litellm `model` counts `metric` (`tpm`,
    /// `rpm`, `itpm` or `otpm`) under in `minute` (`HH-MM`).
    fn usage_key(&self, id: &str, model: &str, metric: &str, minute: &str) -> PyResult<String> {
        let metric: UsageMetric = metric
            .parse()
            .map_err(|_| PyValueError::new_err(format!("`{metric}` is not a usage metric")))?;
        Ok(usage_key(id, model, metric, minute))
    }

    /// Keep the providers `other`'s build resolved for the deployments this router serves too.
    fn adopt_providers(&self, other: &Self) {
        self.router.adopt_providers(&other.router);
    }

    fn registry_version(&self) -> u64 {
        self.router.registry().version()
    }

    /// The version the deployments before the newest appends last changed at: a list read at
    /// or after it is current once it reads `deployments_from(len(list))`.
    fn registry_rewritten(&self) -> u64 {
        self.router.registry().rewritten()
    }

    /// Record a failure the router did not see itself (a health check, an interceptor, a
    /// logging callback), driven like `start`: its steps are cache ops, and it ends with `done`
    /// (`cooled`, `cooldown_time`). `counts_request` also counts the failed request against the deployment's
    /// per-minute requests, as an async call's failure callback does. `callbacks` is the logged
    /// request's `OutcomeCallbacks`.
    #[pyo3(signature = (id, failure, deployment_cooldown=false, count_only=false, requested_group=None, cooldown_time=None, uncounted=false, counts_request=false, usage_model=None, callbacks=None))]
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
        counts_request: bool,
        usage_model: Option<String>,
        callbacks: Option<&Bound<'_, PyAny>>,
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
            counts_request,
            usage_model,
            callbacks: outcome_callbacks(callbacks)?,
        };
        Ok(RouterCall::new(async move {
            router
                .report_failure(&id, &failure, report)
                .await
                .map(|cooldown| Finish::Reported { cooldown })
        }))
    }

    /// Record a success a logging callback saw, driven like `report_failure`; it ends with
    /// `done` (`key`: the tokens counter's key when usage was added, else None).
    /// `usage_model` is the model its usage is keyed by, and `tracks_usage` the host's answer to
    /// whether it counts any (see `Output::tracks_usage`). `callbacks`: see `report_failure`.
    #[pyo3(signature = (id, total_tokens, requests, usage_model=None, tracks_usage=None, callbacks=None))]
    fn report_success(
        &self,
        id: &str,
        total_tokens: f64,
        requests: f64,
        usage_model: Option<String>,
        tracks_usage: Option<bool>,
        callbacks: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<RouterCall> {
        let router = self.router.clone();
        let id = id.to_owned();
        let report = SuccessReport {
            tracks_usage,
            usage_model,
            callbacks: outcome_callbacks(callbacks)?,
        };
        Ok(RouterCall::new(async move {
            router
                .report_success(&id, total_tokens, requests, report)
                .await
                .map(|key| Finish::Counted { key })
        }))
    }

    /// Count a success against `id`, configured or not; ends with `done` (`key`). `callbacks`:
    /// see `report_failure`.
    #[pyo3(signature = (id, callbacks=None))]
    fn count_success(
        &self,
        id: &str,
        callbacks: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<RouterCall> {
        let router = self.router.clone();
        let id = id.to_owned();
        let callbacks = outcome_callbacks(callbacks)?;
        Ok(RouterCall::new(async move {
            router
                .count_success(&id, callbacks)
                .await
                .map(|key| Finish::Counted { key })
        }))
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

    /// The deployments from position `start` on, as `deployments()` lists them.
    fn deployments_from<'py>(&self, py: Python<'py>, start: usize) -> PyResult<Bound<'py, PyList>> {
        let registry = self.router.registry();
        let records = registry
            .deployments_from(start)
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

    // -- registry lookups (router-core `lookup.rs`): rows the host reads its stored entries at --

    /// `_core_get_all_deployments`, as rows.
    #[pyo3(signature = (model_name, team_id=None, rename=None))]
    fn lookup_all<'py>(
        &self,
        py: Python<'py>,
        model_name: &str,
        team_id: Option<&str>,
        rename: Option<&str>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let registry = self.router.registry();
        rows_to_py(py, &Lookup::new(&registry).all(model_name, team_id, rename))
    }

    /// `_team_deployments_across_teams`: the rows, and how many teams own them.
    fn lookup_across_teams<'py>(
        &self,
        py: Python<'py>,
        model: &str,
    ) -> PyResult<(Bound<'py, PyAny>, usize)> {
        let registry = self.router.registry();
        let (rows, teams) = Lookup::new(&registry).across_teams(model);
        Ok((rows_to_py(py, &rows)?, teams))
    }

    /// `_core_config_deployments`, as rows.
    fn lookup_config<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let registry = self.router.registry();
        rows_to_py(py, &Lookup::new(&registry).config())
    }

    /// `_core_get_model_ids`: the ids, or the row the read raises at.
    #[pyo3(signature = (model_name=None, exclude_team_models=false))]
    fn lookup_model_ids<'py>(
        &self,
        py: Python<'py>,
        model_name: Option<&str>,
        exclude_team_models: bool,
    ) -> PyResult<(Vec<String>, Option<Bound<'py, PyAny>>)> {
        let registry = self.router.registry();
        match Lookup::new(&registry).model_ids(model_name, exclude_team_models) {
            Ok(ids) => Ok((ids, None)),
            Err(raw) => Ok((Vec::new(), Some(row_to_py(py, &raw)?))),
        }
    }

    /// `_core_get_model_list`, as rows.
    #[pyo3(signature = (model_name, team_id, scope))]
    fn lookup_model_list<'py>(
        &self,
        py: Python<'py>,
        model_name: Option<&str>,
        team_id: Option<&str>,
        scope: &Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let scope = list_scope(scope)?;
        let registry = self.router.registry();
        rows_to_py(
            py,
            &Lookup::new(&registry).model_list(model_name, team_id, &scope),
        )
    }

    /// `_core_get_model_names`, or the row the read raises at.
    fn lookup_model_names<'py>(
        &self,
        py: Python<'py>,
        team_id: Option<&str>,
        scope: &Bound<'_, PyAny>,
    ) -> PyResult<(Vec<String>, Option<Bound<'py, PyAny>>)> {
        let scope = list_scope(scope)?;
        let registry = self.router.registry();
        match Lookup::new(&registry).model_names(team_id, &scope) {
            Ok(names) => Ok((names, None)),
            Err(raw) => Ok((Vec::new(), Some(row_to_py(py, &raw)?))),
        }
    }

    /// `_core_get_model_access_groups`: `(group, names)` pairs in the order the groups appear,
    /// or the row the read raises at.
    fn lookup_access_groups<'py>(
        &self,
        py: Python<'py>,
        model_name: Option<&str>,
        access_group: Option<&str>,
        team_id: Option<&str>,
        scope: &Bound<'_, PyAny>,
    ) -> PyResult<(AccessGroups, Option<Bound<'py, PyAny>>)> {
        let scope = list_scope(scope)?;
        let registry = self.router.registry();
        match Lookup::new(&registry).access_groups(model_name, access_group, team_id, &scope) {
            Ok(groups) => Ok((groups, None)),
            Err(raw) => Ok((Vec::new(), Some(row_to_py(py, &raw)?))),
        }
    }

    /// Whether the whole list lists any row, and the name of its first row whose litellm model
    /// is `model_id` (or ends in `/model_id` or `:model_id`); or the row the read raises at.
    fn lookup_name_serving<'py>(
        &self,
        py: Python<'py>,
        model_id: &str,
        scope: &Bound<'_, PyAny>,
    ) -> PyResult<(bool, Option<String>, Option<Bound<'py, PyAny>>)> {
        let scope = list_scope(scope)?;
        let registry = self.router.registry();
        match Lookup::new(&registry).name_serving(model_id, &scope) {
            Ok((listed, name)) => Ok((listed, name, None)),
            Err(raw) => Ok((false, None, Some(row_to_py(py, &raw)?))),
        }
    }

    /// Whether `model_names` (the group names) holds `name`.
    fn has_model_name(&self, name: &str) -> bool {
        self.router
            .registry()
            .host_indexes()
            .model_names
            .contains(&Value::String(name.to_owned()))
    }

    /// `_core_team_model_has_alternatives`, or the row the read raises at.
    fn lookup_team_model_has_alternatives<'py>(
        &self,
        py: Python<'py>,
        id: &str,
    ) -> PyResult<(bool, Option<Bound<'py, PyAny>>)> {
        let registry = self.router.registry();
        match Lookup::new(&registry).team_model_has_alternatives(id) {
            Ok(found) => Ok((found, None)),
            Err(raw) => Ok((false, Some(row_to_py(py, &raw)?))),
        }
    }

    /// The position of the group's first deployment the team may use, or the row the read
    /// raises at.
    fn lookup_group_usable_by_team<'py>(
        &self,
        py: Python<'py>,
        model_group: &str,
        team_id: Option<&str>,
    ) -> PyResult<(Option<usize>, Option<Bound<'py, PyAny>>)> {
        let registry = self.router.registry();
        match Lookup::new(&registry).group_deployment_usable_by_team(model_group, team_id) {
            Ok(position) => Ok((position, None)),
            Err(raw) => Ok((None, Some(row_to_py(py, &raw)?))),
        }
    }

    /// The position of the team public name's first deployment, or the row the read raises at.
    fn lookup_team_public_name<'py>(
        &self,
        py: Python<'py>,
        model_id: &str,
        team_id: &str,
    ) -> PyResult<(Option<usize>, Option<Bound<'py, PyAny>>)> {
        let registry = self.router.registry();
        match Lookup::new(&registry).team_public_name_deployment(model_id, team_id) {
            Ok(position) => Ok((position, None)),
            Err(raw) => Ok((None, Some(row_to_py(py, &raw)?))),
        }
    }

    /// `_routable_deployments`, as rows.
    fn lookup_routable<'py>(
        &self,
        py: Python<'py>,
        model_group: &str,
        team_id: Option<&str>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let registry = self.router.registry();
        rows_to_py(py, &Lookup::new(&registry).routable(model_group, team_id))
    }

    /// `_core_wildcard_route_deployments`: the team router's routes, else the global router's.
    fn lookup_wildcard_route<'py>(
        &self,
        py: Python<'py>,
        model: &str,
        team_id: Option<&str>,
        requests: Vec<String>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let registry = self.router.registry();
        rows_to_py(
            py,
            &Lookup::new(&registry).wildcard_route(model, team_id, &requests),
        )
    }

    /// `_core_get_candidate_model_ids_for_route` for the resolved name: the members' ids when it
    /// names a callable routing group (`members`), else for a name that is no model group the
    /// team's or the wildcard route's, else the group's; or the row the read raises at.
    fn lookup_candidate_ids<'py>(
        &self,
        py: Python<'py>,
        resolved: &str,
        team_id: Option<&str>,
        members: Option<Vec<String>>,
        requests: Vec<String>,
    ) -> PyResult<(Vec<String>, Option<Bound<'py, PyAny>>)> {
        let registry = self.router.registry();
        let lookup = Lookup::new(&registry);
        match lookup.candidate_ids(resolved, team_id, members.as_deref(), &requests) {
            Ok(ids) => Ok((ids, None)),
            Err(raw) => Ok((Vec::new(), Some(row_to_py(py, &raw)?))),
        }
    }

    /// Whether the id map holds `id` (`_has_model_id`).
    fn has_model_id(&self, id: &str) -> bool {
        self.router.registry().host_indexes().has_id(id)
    }

    /// What the id map holds for `id`, as written (a position, or what a caller wrote), or None.
    fn id_position<'py>(&self, py: Python<'py>, id: &str) -> PyResult<Option<Bound<'py, PyAny>>> {
        let registry = self.router.registry();
        registry
            .host_indexes()
            .id_entry(id)
            .map(|held| to_py(py, held).map(|value| value.into_bound(py)))
            .transpose()
    }

    /// The id map's entries, in order: `(id, what it holds)`.
    fn id_positions<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let registry = self.router.registry();
        let pairs: Vec<Value> = registry
            .host_indexes()
            .id_map
            .iter()
            .map(|(id, held)| Value::Array(vec![Value::String(id.clone()), held.clone()]))
            .collect();
        to_py(py, &Value::Array(pairs)).map(|value| value.into_bound(py))
    }

    /// What the name index holds for `name`, as written, or None when it does not index it.
    fn named_positions<'py>(
        &self,
        py: Python<'py>,
        name: &str,
    ) -> PyResult<Option<Bound<'py, PyAny>>> {
        let registry = self.router.registry();
        registry
            .host_indexes()
            .named(name)
            .map(|held| to_py(py, held).map(|value| value.into_bound(py)))
            .transpose()
    }

    /// What the team index holds for `(team, public)`, as written, or None.
    fn team_positions<'py>(
        &self,
        py: Python<'py>,
        team: &str,
        public: &str,
    ) -> PyResult<Option<Bound<'py, PyAny>>> {
        let registry = self.router.registry();
        registry
            .host_indexes()
            .team_named(team, public)
            .map(|held| to_py(py, held).map(|value| value.into_bound(py)))
            .transpose()
    }

    /// `team_public_model_names_of`: the public names the team index holds for `team`.
    fn team_public_names_of(&self, team: &str) -> Vec<String> {
        self.router
            .registry()
            .host_indexes()
            .team_public_names_of(team)
    }

    /// The positions of the deployments whose `litellm_params.model` is `model`.
    fn litellm_model_positions(&self, model: &str) -> Vec<usize> {
        self.router
            .registry()
            .deployments(&DeploymentFilter::default())
            .iter()
            .enumerate()
            .filter(|(_, deployment)| {
                deployment.raw_params.get("model").and_then(Value::as_str) == Some(model)
            })
            .map(|(position, _)| position)
            .collect()
    }

    /// `PatternMatchRouter.route` over the wildcard router (or `team`'s): the first pattern's
    /// deployments, each with the model the wildcards matched, or None.
    #[pyo3(signature = (team, request, filtered=None))]
    fn pattern_route<'py>(
        &self,
        py: Python<'py>,
        team: Option<&str>,
        request: &str,
        filtered: Option<Vec<String>>,
    ) -> PyResult<Option<Bound<'py, PyAny>>> {
        let registry = self.router.registry();
        let indexes = registry.host_indexes();
        let router = match team {
            None => Some(&indexes.patterns),
            Some(team) => indexes
                .team_patterns
                .iter()
                .find(|(held, _)| held == team)
                .map(|(_, router)| router),
        };
        router
            .and_then(|router| router.route(request, filtered.as_deref()))
            .map(|routed| to_py(py, &Value::Array(routed)).map(|value| value.into_bound(py)))
            .transpose()
    }

    /// `get_models_by_pattern`: the models the first matching (request, pattern) pair routes to.
    fn models_by_pattern(&self, team: Option<&str>, requests: Vec<String>) -> Option<Vec<String>> {
        let registry = self.router.registry();
        let indexes = registry.host_indexes();
        let router = match team {
            None => Some(&indexes.patterns),
            Some(team) => indexes
                .team_patterns
                .iter()
                .find(|(held, _)| held == team)
                .map(|(_, router)| router),
        };
        router.and_then(|router| router.models_by_pattern(&requests))
    }

    /// The indexes the host hands out (`HostIndexes`): id map, name and team indexes, group and
    /// deployment names, team public names, provider default ids and the pattern routers.
    fn host_indexes<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let indexes = serde_json::to_value(self.router.registry().host_indexes())
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        to_py(py, &indexes).map(|value| value.into_bound(py))
    }

    /// `provider_default_deployment_ids` as the indexes hold it now: the ids of the wildcard
    /// deployments the cooldown check skips.
    fn provider_default_ids(&self) -> Vec<String> {
        self.router
            .registry()
            .host_indexes()
            .provider_default_ids
            .clone()
    }

    /// The provider the host resolved when it built the deployment with this id (its
    /// `provider` build step), or None when there is no such deployment or it never went
    /// through a host build.
    fn provider(&self, id: &str) -> Option<String> {
        self.router
            .registry()
            .deployment(id)
            .and_then(|deployment| deployment.resolved_provider.clone())
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
        let request = config::parse_request(document(request)?, Some(model))
            .map_err(config_error_to_pyerr)?;
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
        let response = match driver.pending_call() {
            None => match outcome {
                None => None,
                Some(_) => return Err(drive_error_to_pyerr(DriveError::UnexpectedResponse)),
            },
            Some(call) => Some(host_call_response(call, outcome)?),
        };
        let step = driver.resume(response).map_err(drive_error_to_pyerr)?;
        let dict = step_dict(py, step)?;
        dict.set_item("events", events_list(py, driver.take_events())?)?;
        Ok(dict)
    }
}

/// `(group, names)` pairs, in the order the groups appear.
type AccessGroups = Vec<(String, Vec<String>)>;

/// Lookup rows as the host reads them.
fn rows_to_py<'py>(py: Python<'py>, rows: &[Row]) -> PyResult<Bound<'py, PyAny>> {
    let rows = Value::Array(rows.iter().map(Row::wire).collect());
    to_py(py, &rows).map(|value| value.into_bound(py))
}

fn row_to_py<'py>(py: Python<'py>, row: &Row) -> PyResult<Bound<'py, PyAny>> {
    to_py(py, &row.wire()).map(|value| value.into_bound(py))
}

/// `{"aliases": [{alias, target}], "groups": [{name, members}], "wildcard": [requests]}`.
fn list_scope(scope: &Bound<'_, PyAny>) -> PyResult<ListScope> {
    serde_json::from_value(document(scope)?)
        .map_err(|error| PyValueError::new_err(format!("bad lookup scope: {error}")))
}

/// Read Python's outcome for `call`.
fn host_call_response(
    call: &HostCall,
    outcome: Option<Bound<'_, PyAny>>,
) -> PyResult<HostCallResponse> {
    let missing = || {
        drive_error_to_pyerr(DriveError::MissingResponse {
            expected: call.kind(),
        })
    };
    // `{"raised": True, ...}`: the step's work raised, reported as an attempt's failure.
    if let HostCall::Select { .. } | HostCall::PreRoute { .. } | HostCall::AllowFallback { .. } =
        call
        && let Some(failure) = outcome.as_ref().and_then(|o| o.cast::<PyDict>().ok())
        && failure.contains("raised")?
    {
        return match attempt_outcome(failure)? {
            Err(error) => Ok(HostCallResponse::Raised(error)),
            Ok(_) => Err(PyValueError::new_err("a `raised` outcome is a failure")),
        };
    }
    match call {
        HostCall::Sleep { .. } => Ok(HostCallResponse::Slept),
        HostCall::AllowFallback { .. } => Ok(HostCallResponse::AllowFallback(
            outcome.ok_or_else(missing)?.extract()?,
        )),
        HostCall::RefuseFallbacks { .. } => Ok(HostCallResponse::RefuseFallbacks(
            outcome.ok_or_else(missing)?.extract()?,
        )),
        HostCall::Select {
            stage: Stage::PreCall,
            ..
        } => {
            let facts = from_py_argument(&outcome.ok_or_else(missing)?)?;
            serde_json::from_value(facts)
                .map(HostCallResponse::PreCall)
                .map_err(|error| PyValueError::new_err(format!("bad pre-call facts: {error}")))
        }
        HostCall::Select {
            stage: Stage::RoutingPlugins { .. } | Stage::ProviderRequest,
            ..
        } => {
            // `{"models": [...] | None}`: a bare None would read as no outcome.
            let outcome = outcome.ok_or_else(missing)?;
            let models = outcome.cast::<PyDict>()?.get_item("models")?;
            Ok(HostCallResponse::Models(match models {
                Some(models) if !models.is_none() => Some(models.extract()?),
                _ => None,
            }))
        }
        HostCall::Select {
            stage: Stage::CallbackFilters { .. },
            ..
        } => {
            // The remaining ids, alone or paired with the id a filter prefers, or a failure
            // mapping as an attempt reports one.
            let outcome = outcome.ok_or_else(missing)?;
            if let Ok(failure) = outcome.cast::<PyDict>() {
                return match attempt_outcome(failure)? {
                    Err(error) => Ok(HostCallResponse::Filtered(Err(error))),
                    Ok(_) => Err(PyValueError::new_err(
                        "a `callback_filters` outcome is the remaining ids or a failure",
                    )),
                };
            }
            if let Ok(answer) = outcome.cast::<PyTuple>() {
                return Ok(HostCallResponse::Filtered(Ok(answer.extract()?)));
            }
            Ok(HostCallResponse::Filtered(Ok((outcome.extract()?, None))))
        }
        HostCall::Select {
            stage: Stage::Strategy { records, .. },
            ..
        } if !records.is_empty() => Ok(HostCallResponse::Positions(
            outcome.ok_or_else(missing)?.extract()?,
        )),
        HostCall::Select { .. } => Ok(HostCallResponse::Select(
            outcome.ok_or_else(missing)?.extract()?,
        )),
        HostCall::PreRoute { .. } => pre_route_outcome(outcome).map(HostCallResponse::PreRoute),
        HostCall::Attempt { .. } => {
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
            let deferred = outcome
                .get_item("deferred")?
                .is_some_and(|flag| flag.is_truthy().unwrap_or(false));
            let tracks_usage = outcome
                .get_item("tracks_usage")?
                .map(|flag| flag.extract::<Option<bool>>())
                .transpose()?
                .flatten();
            attempt_outcome(outcome).map(|result| match result {
                Ok(total_tokens) if blocked.is_some() || deferred || tracks_usage.is_some() => {
                    HostCallResponse::Served {
                        total_tokens,
                        blocked,
                        deferred,
                        tracks_usage,
                    }
                }
                result => HostCallResponse::Attempt(result),
            })
        }
        HostCall::Cache { op, .. } => cache_outcome(*op, outcome).map(HostCallResponse::Cache),
        HostCall::Secrets { .. } => Ok(HostCallResponse::Secrets(from_py_argument(
            &outcome.ok_or_else(missing)?,
        )?)),
        // `None` is a response here (no row, no entry), so a missing outcome reads as one.
        HostCall::ModelInfo { .. } => Ok(HostCallResponse::ModelInfo(match outcome {
            Some(outcome) if !outcome.is_none() => from_py_argument(&outcome)?,
            _ => Value::Null,
        })),
        HostCall::Build { kind, .. } => {
            let outcome = outcome.ok_or_else(missing)?;
            let outcome = outcome.cast::<PyDict>()?;
            let invalid = outcome
                .get_item("invalid")?
                .is_some_and(|flag| flag.is_truthy().unwrap_or(false));
            match kind {
                StepKind::Provider => Ok(HostCallResponse::Provided(if invalid {
                    Provided::Invalid
                } else {
                    Provided::Provider(match outcome.get_item("provider")? {
                        Some(provider) if !provider.is_none() => Some(provider.extract()?),
                        _ => None,
                    })
                })),
                _ => {
                    let built = if invalid {
                        Built::Invalid
                    } else {
                        match outcome.get_item("entry")? {
                            Some(entry) if !entry.is_none() => {
                                Built::Entry(from_py_argument(&entry)?)
                            }
                            _ => Built::Skip,
                        }
                    };
                    Ok(if *kind == StepKind::Validate {
                        HostCallResponse::Validated(built)
                    } else {
                        HostCallResponse::Added(built)
                    })
                }
            }
        }
    }
}

/// A pre-route response: `None`, a model group name, or a custom strategy's deployment document.
fn pre_route_outcome(outcome: Option<Bound<'_, PyAny>>) -> PyResult<Option<PreRouted>> {
    let Some(outcome) = outcome.filter(|outcome| !outcome.is_none()) else {
        return Ok(None);
    };
    if let Ok(model) = outcome.cast::<PyString>() {
        return Ok(Some(PreRouted::Model(model.to_str()?.to_owned())));
    }
    // `(model, group)`: the model to resolve, served as the group a session binding stamped.
    if let Ok(pair) = outcome.cast::<PyTuple>()
        && pair.len() == 2
    {
        return Ok(Some(PreRouted::Bound {
            model: pair.get_item(0)?.extract()?,
            group: pair.get_item(1)?.extract()?,
        }));
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
        CacheOp::Set | CacheOp::Delete | CacheOp::IncrementUsage => match outcome {
            None => Ok(CacheOutcome::Stored),
            Some(_) => Err(PyValueError::new_err(format!(
                "a `{op}` cache step has no outcome, so pass None"
            ))),
        },
    }
}

/// `{"ok": True, "total_tokens": int}` (an attempt's may add `"blocked": str`, `"deferred": bool`
/// and `"tracks_usage": bool`) or `{"ok": False, "kind": str, "status": int | None,
/// "retry_after": float | None, "num_retries": int | None, "detail": str | None,
/// "outside_deployment": bool, "caller_scoped": bool, "status_absent": bool,
/// "uncounted_request": bool}`.
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
    failure.failed_deployment_id = item(outcome, "failed_deployment_id")?;
    failure.detail = item::<String>(outcome, "detail")?.map(Into::into);
    failure.recorded = item::<bool>(outcome, "recorded")?.unwrap_or(false);
    failure.names_provider = item::<bool>(outcome, "llm_provider")?.unwrap_or(false);
    failure.agentic_followup = item::<bool>(outcome, "agentic_followup")?.unwrap_or(false);
    failure.outside_deployment = item::<bool>(outcome, "outside_deployment")?.unwrap_or(false);
    failure.status_absent = item::<bool>(outcome, "status_absent")?.unwrap_or(false);
    failure.uncounted_request = item::<bool>(outcome, "uncounted_request")?.unwrap_or(false);
    failure.caller_scoped = item::<bool>(outcome, "caller_scoped")?.unwrap_or(false);
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
        Error::NoHealthyDeployments { model, .. } | Error::UnknownModel { model } => {
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
        Error::EndpointNotSupported { model, message } => {
            dict.set_item("error_model", model)?;
            dict.set_item("error_detail", message)
        }
        Error::OnlyMarkers { model }
        | Error::UnknownModel { model }
        | Error::NoPassThroughDeployments { model }
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
        Error::NoHealthyDeployments {
            request_ids: Some(ids),
            ..
        } => dict.set_item("error_request_ids", PyList::new(dict.py(), ids)?),
        Error::NonNumericWeight { metric } => dict.set_item("error_detail", metric),
        Error::MissingRoutingGroup { group } => dict.set_item("error_model", group),
        Error::ModelInfo { exception, detail } => {
            dict.set_item("error_exception", *exception)?;
            dict.set_item("error_detail", detail)
        }
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
            RouterEvent::Retry { retry, .. } => {
                dict.set_item("kind", "retry")?;
                dict.set_item("retry", retry)?;
            }
            RouterEvent::FallingBack { level, model } => {
                dict.set_item("kind", "falling_back")?;
                dict.set_item("level", level)?;
                dict.set_item("model", model)?;
            }
            RouterEvent::OrderChainFailed { level, last_hop } => {
                dict.set_item("kind", "order_chain_failed")?;
                dict.set_item("level", level)?;
                dict.set_item("last_hop", last_hop)?;
            }
            RouterEvent::FallbacksFailed {
                level,
                chain,
                last_hop,
            } => {
                dict.set_item("kind", "fallbacks_failed")?;
                dict.set_item("level", level)?;
                dict.set_item("chain", chain)?;
                dict.set_item("last_hop", last_hop)?;
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
                unroutable,
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
                if let Some(model) = unroutable {
                    dict.set_item("unroutable", to_py(py, &model)?)?;
                }
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
            RouterEvent::RetryCountRaised {
                level,
                model,
                cause,
            } => {
                dict.set_item("kind", "retry_count_raised")?;
                dict.set_item("level", level)?;
                dict.set_item("model", model)?;
                cause_items(&dict, cause)?;
            }
            RouterEvent::RetriesOpened {
                scope,
                max_retries,
                model_group_size,
            } => {
                dict.set_item("kind", "retries_opened")?;
                dict.set_item("scope", scope)?;
                dict.set_item("max_retries", max_retries)?;
                dict.set_item("model_group_size", model_group_size)?;
            }
            RouterEvent::RetryBudget { scope, max_retries } => {
                dict.set_item("kind", "retry_budget")?;
                dict.set_item("scope", scope)?;
                dict.set_item("max_retries", max_retries)?;
            }
            RouterEvent::PickFailed {
                level,
                model,
                error,
            } => {
                dict.set_item("kind", "pick_failed")?;
                dict.set_item("level", level)?;
                dict.set_item("model", model)?;
                pick_items(&dict, &error.0)?;
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
            RouterEvent::CooldownSet {
                deployment_id,
                cooldown,
            } => {
                dict.set_item("kind", "cooldown_set")?;
                dict.set_item("deployment_id", deployment_id)?;
                dict.set_item("cooldown_time", cooldown.as_secs_f64())?;
            }
            RouterEvent::PickStarted { model } => {
                dict.set_item("kind", "pick_started")?;
                dict.set_item("model", model)?;
            }
            RouterEvent::Selected {
                deployment_id,
                pick,
                ..
            } => {
                dict.set_item("kind", "selected")?;
                dict.set_item("deployment_id", deployment_id)?;
                dict.set_item("pick_seconds", pick.as_secs_f64())?;
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
        Step::HostCall(call) => call.kind(),
        Step::Done(_) => StepKind::Done,
        Step::Failed(_) => StepKind::Failed,
    };
    dict.set_item("kind", <&'static str>::from(kind))?;
    match step {
        Step::HostCall(call) => host_call_items(py, &dict, call)?,
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

fn host_call_items(py: Python<'_>, dict: &Bound<'_, PyDict>, call: HostCall) -> PyResult<()> {
    match call {
        HostCall::Attempt {
            deployment_id,
            deployment,
            model,
            model_group,
            routed,
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
            dict.set_item("routed", routed)?;
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
        HostCall::Sleep { seconds } => dict.set_item("seconds", seconds),
        HostCall::AllowFallback { target } => dict.set_item("target", target),
        HostCall::RefuseFallbacks { level, targets } => {
            dict.set_item("level", level)?;
            let targets = targets.0.iter().cloned().map(Value::Object).collect();
            dict.set_item("targets", to_py(py, &Value::Array(targets))?)
        }
        HostCall::Select {
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
                Stage::Strategy { strategy, records } => {
                    if !records.is_empty() {
                        dict.set_item("records", to_py(py, &Value::Array(records))?)?;
                    }
                    ("strategy", Some(strategy))
                }
                Stage::Tags => ("tags", None),
                Stage::PreCall => ("pre_call", None),
                Stage::TextOnly => ("text_only", None),
                Stage::RouteIncapable { handler } => {
                    dict.set_item("handler", handler)?;
                    ("route_incapable", None)
                }
                Stage::MediaRefusing => ("media_refusing", None),
                Stage::EmbeddingBatch => ("embedding_batch", None),
                Stage::OverrideCheck(strategy) => ("override_check", Some(strategy)),
                Stage::ProviderRequest => ("provider_request", None),
                Stage::CallbackFilters {
                    excluded_ids,
                    skipped_ids,
                    target_order,
                } => {
                    dict.set_item("excluded_ids", excluded_ids)?;
                    dict.set_item("skipped_ids", skipped_ids)?;
                    dict.set_item("target_order", target_order)?;
                    ("callback_filters", None)
                }
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
        HostCall::PreRoute {
            model,
            fallback_depth,
            retry,
        } => {
            dict.set_item("model", model)?;
            dict.set_item("fallback_depth", fallback_depth)?;
            dict.set_item("retry", retry)
        }
        HostCall::Cache {
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
        HostCall::Secrets { index, refs } => {
            dict.set_item("index", index)?;
            let refs = refs
                .into_iter()
                .map(|secret| {
                    let item = PyDict::new(py);
                    item.set_item("name", secret.name)?;
                    match secret.at {
                        SecretAt::Param(key) => {
                            item.set_item("key", key)?;
                            item.set_item("env_only", false)?;
                        }
                        SecretAt::DataSource { source, key } => {
                            item.set_item("source", source)?;
                            item.set_item("key", key)?;
                            item.set_item("env_only", true)?;
                        }
                    }
                    Ok(item)
                })
                .collect::<PyResult<Vec<_>>>()?;
            dict.set_item("refs", PyList::new(py, refs)?)
        }
        HostCall::Build {
            kind: _,
            path,
            index,
            entry,
            declared_id,
            duplicate_id,
            provider,
        } => {
            dict.set_item("path", <&'static str>::from(path))?;
            dict.set_item("index", index)?;
            dict.set_item("entry", to_py(py, &entry)?)?;
            dict.set_item("declared_id", declared_id)?;
            dict.set_item("duplicate_id", duplicate_id)?;
            dict.set_item("provider", provider)
        }
        HostCall::ModelInfo { lookup } => {
            if let Value::Object(lookup) = lookup {
                for (key, value) in lookup {
                    dict.set_item(key, to_py(py, &value)?)?;
                }
            }
            Ok(())
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
        Finish::Reported { cooldown } => {
            dict.set_item("cooled", cooldown.is_some())?;
            dict.set_item("cooldown_time", cooldown.map(|c| c.as_secs_f64()))
        }
        Finish::Built(built) => {
            let py = dict.py();
            let records = built
                .into_iter()
                .map(|deployment| {
                    let item = PyDict::new(py);
                    item.set_item("index", deployment.index)?;
                    item.set_item(
                        "id",
                        deployment.spec.model_info.get("id").and_then(Value::as_str),
                    )?;
                    item.set_item("provider", deployment.spec.provider)?;
                    Ok(item)
                })
                .collect::<PyResult<Vec<_>>>()?;
            dict.set_item("built", PyList::new(py, records)?)
        }
        Finish::Counted { key } => dict.set_item("key", key),
        Finish::RouterModelInfo(merged) => {
            let py = dict.py();
            match merged {
                Some(merged) => {
                    dict.set_item("info", to_py(py, &Value::Object(merged.info))?)?;
                    dict.set_item("user_keys", merged.user_keys)
                }
                None => dict.set_item("info", py.None()),
            }
        }
        Finish::DeploymentModelInfo(info) => {
            let py = dict.py();
            match info {
                Some(info) => dict.set_item("info", to_py(py, &Value::Object(info))?),
                None => dict.set_item("info", py.None()),
            }
        }
        Finish::Value(value) => dict.set_item("value", to_py(dict.py(), &value)?),
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
