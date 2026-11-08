use std::sync::Mutex;

use litellm_host_python::{from_py, to_py};
use litellm_request_auth_core::{self as core, credential, protocol::*};
use pyo3::{
    exceptions::{PyRuntimeError, PyTypeError},
    prelude::*,
};

type ProjectedStep = (Py<PyAny>, Option<NativeAuthContinuation>);

#[pyfunction]
pub(crate) fn request_auth_managed(
    py: Python<'_>,
    facts: &Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    let facts: core::managed::ManagedFacts = from_py(facts)?;
    to_py(py, &py.detach(move || core::managed::decide(facts)))
}

#[pyfunction]
pub(crate) fn request_auth_admission(
    py: Python<'_>,
    facts: &Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    let facts: core::admission::AdmissionFacts = from_py(facts)?;
    to_py(py, &py.detach(move || core::admission::plan(facts)))
}

#[pyfunction]
pub(crate) fn request_auth_access(py: Python<'_>, facts: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    let facts: core::access::AccessFacts = from_py(facts)?;
    to_py(py, &py.detach(move || core::access::decide(facts)))
}

#[pyfunction]
pub(crate) fn request_auth_budget_window_start(
    py: Python<'_>,
    facts: &Bound<'_, PyAny>,
) -> PyResult<Option<f64>> {
    let facts: core::common_budgets::WindowFacts = from_py(facts)?;
    Ok(py.detach(move || core::common_budgets::window_start(&facts)))
}

#[pyfunction]
pub(crate) fn request_auth_model_budget_resolve(
    py: Python<'_>,
    facts: &Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    let facts: core::model_budget_data::ResolveFacts = from_py(facts)?;
    to_py(
        py,
        &py.detach(move || core::model_budget_data::resolve(&facts)),
    )
}

#[pyfunction]
pub(crate) fn request_auth_model_budget_query(
    py: Python<'_>,
    facts: &Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    let facts: core::model_budget_data::QueryFacts = from_py(facts)?;
    to_py(
        py,
        &py.detach(move || core::model_budget_data::query(&facts)),
    )
}

#[pyfunction]
pub(crate) fn request_auth_model_budget_check(
    py: Python<'_>,
    facts: &Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    let facts: core::model_budget_data::Snapshot = from_py(facts)?;
    to_py(
        py,
        &py.detach(move || core::model_budget_data::check(&facts)),
    )
}

#[pyfunction]
pub(crate) fn request_auth_budget_price(
    py: Python<'_>,
    facts: &Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    let facts: core::budget_pricing::ModelPriceFacts = from_py(facts)?;
    to_py(py, &py.detach(move || core::budget_pricing::price(&facts)))
}

#[pyfunction]
pub(crate) fn request_auth_budget_route(
    py: Python<'_>,
    facts: &Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    let facts: core::budget_pricing::RouteBudgetFacts = from_py(facts)?;
    to_py(py, &py.detach(move || core::budget_pricing::route(&facts)))
}

#[pyfunction]
pub(crate) fn request_auth_common_budget(
    py: Python<'_>,
    facts: &Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    let facts: core::common_budgets::BudgetFacts = from_py(facts)?;
    to_py(py, &py.detach(move || core::common_budgets::check(&facts)))
}

#[pyfunction]
pub(crate) fn request_auth_budget_alert(
    py: Python<'_>,
    facts: &Bound<'_, PyAny>,
) -> PyResult<bool> {
    let facts: core::common_budgets::AlertFacts = from_py(facts)?;
    Ok(py.detach(move || core::common_budgets::alert(&facts)))
}

#[pyfunction]
pub(crate) fn request_auth_member_limit(
    py: Python<'_>,
    facts: &Bound<'_, PyAny>,
) -> PyResult<Option<f64>> {
    let facts: core::common_budgets::MemberLimitFacts = from_py(facts)?;
    py.detach(move || core::common_budgets::member_limit(&facts))
        .map_err(map_error)
}

fn map_error(error: core::Error) -> PyErr {
    let message = litellm_error::Report::new(&error).to_string();
    match error {
        core::Error::IncompatibleDateTimes => PyTypeError::new_err(message),
        core::Error::InvalidAnswer => PyRuntimeError::new_err(message),
        core::Error::Pattern(_) => PyRuntimeError::new_err(message),
    }
}

fn project(py: Python<'_>, step: Result<Step, core::Error>) -> PyResult<ProjectedStep> {
    let step = step.map_err(map_error)?;
    Ok((
        to_py(py, &step.node)?,
        step.resume.map(|resume| NativeAuthContinuation {
            resume: Mutex::new(Some(resume)),
        }),
    ))
}

#[pyclass(frozen, module = "litellm.rust_bridge._native")]
pub(crate) struct NativeAuthContinuation {
    resume: Mutex<Option<Continuation>>,
}

#[pymethods]
impl NativeAuthContinuation {
    fn resume(&self, py: Python<'_>, answer: &Bound<'_, PyAny>) -> PyResult<ProjectedStep> {
        let resume = self
            .resume
            .lock()
            .map_err(|_| PyRuntimeError::new_err("auth continuation lock poisoned"))?
            .take()
            .ok_or_else(|| PyRuntimeError::new_err("auth continuation already consumed"))?;
        let answer: Answer = from_py(answer)?;
        project(py, py.detach(move || resume(answer)))
    }
}

#[pyfunction]
pub(crate) fn request_auth_start(
    py: Python<'_>,
    facts: &Bound<'_, PyAny>,
    settings: &Bound<'_, PyAny>,
    now: &Bound<'_, PyAny>,
    resolved: bool,
) -> PyResult<ProjectedStep> {
    let facts: RequestFacts = from_py(facts)?;
    let settings: Settings = from_py(settings)?;
    let now: DateTime = from_py(now)?;
    project(
        py,
        py.detach(move || {
            if resolved {
                core::start_resolved_key(facts, settings, now)
            } else {
                core::start(facts, settings, now)
            }
        }),
    )
}

#[derive(serde::Deserialize, strum::VariantArray)]
enum BudgetEntry {
    Key,
    User,
    TeamMember,
}

#[pyfunction]
pub(crate) fn request_auth_budgets(
    py: Python<'_>,
    entry: &Bound<'_, PyAny>,
) -> PyResult<ProjectedStep> {
    let entry: BudgetEntry = from_py(entry)?;
    project(
        py,
        py.detach(move || match entry {
            BudgetEntry::Key => core::check_model_budgets(),
            BudgetEntry::User => core::check_user_model_budgets(),
            BudgetEntry::TeamMember => core::check_team_member_model_budgets(),
        }),
    )
}

#[pyfunction]
pub(crate) fn request_auth_select_api_key(
    py: Python<'_>,
    keys: &Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    let keys: credential::PresentedKeys = from_py(keys)?;
    to_py(py, &py.detach(move || credential::select_api_key(keys)))
}

#[pyfunction]
pub(crate) fn request_auth_select_websocket_api_key(
    py: Python<'_>,
    keys: &Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    let keys: credential::WebSocketKeys = from_py(keys)?;
    to_py(
        py,
        &py.detach(move || credential::select_websocket_api_key(keys)),
    )
}

#[pyfunction]
pub(crate) fn request_auth_scheme_credential(value: &str) -> Option<String> {
    credential::scheme_credential(value)
}

#[pyfunction]
pub(crate) fn request_auth_litellm_header_token(value: &str) -> String {
    credential::litellm_header_token(value)
}

#[pyfunction]
pub(crate) fn request_auth_route_access(py: Python<'_>, request: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    let request: core::route_access::Request = from_py(request)?;
    let decision = py.detach(move || core::route_access::decide(request)).map_err(map_error)?;
    to_py(py, &decision)
}

#[pyfunction]
pub(crate) fn request_auth_org_route(py: Python<'_>, facts: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    let facts: core::org_routes::Facts = from_py(facts)?;
    to_py(py, &py.detach(move || core::org_routes::decide(facts)))
}

#[pyfunction]
pub(crate) fn request_auth_org_admin(py: Python<'_>, memberships: &Bound<'_, PyAny>, organizations: Vec<String>) -> PyResult<bool> {
    let memberships: Vec<core::org_routes::Membership> = from_py(memberships)?;
    Ok(py.detach(move || core::org_routes::user_is_admin(&memberships, &organizations)))
}

#[pyfunction]
pub(crate) fn request_auth_model_access(py: Python<'_>, request: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    let request: core::model_access::Request = from_py(request)?;
    let decision = py.detach(move || core::model_access::decide(request)).map_err(map_error)?;
    to_py(py, &decision)
}

#[pyfunction]
pub(crate) fn request_auth_model_groups(py: Python<'_>, facts: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    let facts: core::model_groups::Facts = from_py(facts)?;
    to_py(py, &py.detach(move || core::model_groups::decide(facts)))
}

#[pyfunction]
pub(crate) fn request_auth_fallback_access(py: Python<'_>, facts: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    let facts: core::fallback_access::Facts = from_py(facts)?;
    let decision = py.detach(move || core::fallback_access::decide(facts)).map_err(map_error)?;
    to_py(py, &decision)
}

#[pyfunction]
pub(crate) fn request_auth_budget_exemption(
    py: Python<'_>,
    facts: &Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    let facts: core::budget_pricing::ExemptionFacts = from_py(facts)?;
    to_py(
        py,
        &py.detach(move || core::budget_pricing::exemption(&facts)),
    )
}
