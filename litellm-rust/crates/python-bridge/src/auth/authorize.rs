use litellm_gateway_auth_core::{
    authorize::{AuthorizeRequest, AuthorizeTeam},
    budgets::{
        BudgetFacts, BudgetWindow, EndUserBudgetFacts, KeyBudgetFacts, TeamBudgetFacts, UserBudgetFacts,
    },
    route_access::OrgMembership,
};
use pyo3::prelude::*;

use super::route_access::RouteFacts;

/// `core.AuthorizeFacts`: what the Python host read about the request before `common_checks` runs.
#[derive(FromPyObject)]
pub(crate) struct AuthorizeFacts {
    #[pyo3(attribute)]
    route: String,
    #[pyo3(attribute)]
    method: Option<String>,
    #[pyo3(attribute)]
    is_llm_api_route: bool,
    #[pyo3(attribute)]
    dispatched_to_pass_through: bool,
    #[pyo3(attribute)]
    has_model: bool,
    #[pyo3(attribute)]
    skip_budget_checks: bool,
    #[pyo3(attribute)]
    user_id: Option<String>,
    #[pyo3(attribute)]
    agent_id: Option<String>,
    #[pyo3(attribute)]
    team_id: Option<String>,
    #[pyo3(attribute)]
    team_blocked: bool,
    #[pyo3(attribute)]
    team_requires_trace_id: bool,
    #[pyo3(attribute)]
    enforce_user_param: bool,
    #[pyo3(attribute)]
    has_user_param: bool,
    #[pyo3(attribute)]
    reject_clientside_metadata_tags: bool,
    #[pyo3(attribute)]
    has_client_tags: bool,
    #[pyo3(attribute)]
    block_unpriced_models: bool,
    #[pyo3(attribute)]
    max_budget: f64,
    #[pyo3(attribute)]
    global_proxy_spend: Option<f64>,
    #[pyo3(attribute)]
    route_access: RouteFacts,
    #[pyo3(attribute)]
    budgets: BudgetFactsPy,
}

/// `authorization.BudgetFacts`
#[derive(FromPyObject)]
pub(crate) struct BudgetFactsPy {
    #[pyo3(attribute)]
    key: Option<KeyBudgetPy>,
    #[pyo3(attribute)]
    team: Option<TeamBudgetPy>,
    #[pyo3(attribute)]
    user: Option<UserBudgetPy>,
    #[pyo3(attribute)]
    end_user: Option<EndUserBudgetPy>,
    #[pyo3(attribute)]
    apply_user_budget_to_team_keys: bool,
    #[pyo3(attribute)]
    is_info_route: bool,
    #[pyo3(attribute)]
    fail_closed_budget_enforcement: bool,
    #[pyo3(attribute)]
    now: f64,
}

#[derive(FromPyObject)]
struct KeyBudgetPy {
    #[pyo3(attribute)]
    token: String,
    #[pyo3(attribute)]
    org_id: Option<String>,
    #[pyo3(attribute, from_py_with = budget_windows)]
    windows: Vec<BudgetWindow>,
}

#[derive(FromPyObject)]
struct TeamBudgetPy {
    #[pyo3(attribute)]
    team_id: String,
    #[pyo3(attribute)]
    max_budget: Option<f64>,
    #[pyo3(attribute)]
    spend: Option<f64>,
    #[pyo3(attribute)]
    soft_budget: Option<f64>,
    #[pyo3(attribute)]
    organization_id: Option<String>,
    #[pyo3(attribute, from_py_with = budget_windows)]
    windows: Vec<BudgetWindow>,
}

#[derive(FromPyObject)]
struct UserBudgetPy {
    #[pyo3(attribute)]
    user_id: String,
    #[pyo3(attribute)]
    max_budget: Option<f64>,
    #[pyo3(attribute)]
    spend: Option<f64>,
}

#[derive(FromPyObject)]
struct EndUserBudgetPy {
    #[pyo3(attribute)]
    end_user_id: String,
    #[pyo3(attribute)]
    max_budget: Option<f64>,
    #[pyo3(attribute)]
    spend: Option<f64>,
}

/// `budget_limits` as `(budget_duration, max_budget)` pairs.
fn budget_windows(value: &Bound<'_, PyAny>) -> PyResult<Vec<BudgetWindow>> {
    Ok(value
        .extract::<Vec<(String, f64)>>()?
        .into_iter()
        .map(|(duration, max_budget)| BudgetWindow { duration, max_budget })
        .collect())
}

impl BudgetFactsPy {
    fn facts(&self) -> BudgetFacts<'_> {
        BudgetFacts {
            key: self.key.as_ref().map(|key| KeyBudgetFacts {
                token: &key.token,
                org_id: key.org_id.as_deref(),
                windows: &key.windows,
            }),
            team: self.team.as_ref().map(|team| TeamBudgetFacts {
                team_id: &team.team_id,
                max_budget: team.max_budget,
                spend: team.spend,
                soft_budget: team.soft_budget,
                organization_id: team.organization_id.as_deref(),
                windows: &team.windows,
            }),
            user: self.user.as_ref().map(|user| UserBudgetFacts {
                user_id: &user.user_id,
                max_budget: user.max_budget,
                spend: user.spend,
            }),
            end_user: self.end_user.as_ref().map(|end_user| EndUserBudgetFacts {
                end_user_id: &end_user.end_user_id,
                max_budget: end_user.max_budget,
                spend: end_user.spend,
            }),
            apply_user_budget_to_team_keys: self.apply_user_budget_to_team_keys,
            is_info_route: self.is_info_route,
            fail_closed_budget_enforcement: self.fail_closed_budget_enforcement,
            now: self.now,
        }
    }
}

impl AuthorizeFacts {
    pub(crate) fn memberships(&self) -> Option<Vec<OrgMembership<'_>>> {
        self.route_access.memberships()
    }

    pub(crate) fn request<'a>(&'a self, memberships: Option<&'a [OrgMembership<'a>]>) -> AuthorizeRequest<'a> {
        AuthorizeRequest {
            route: &self.route,
            method: self.method.as_deref(),
            is_llm_api_route: self.is_llm_api_route,
            dispatched_to_pass_through: self.dispatched_to_pass_through,
            has_model: self.has_model,
            skip_budget_checks: self.skip_budget_checks,
            user_id: self.user_id.as_deref(),
            agent_id: self.agent_id.as_deref(),
            team: self.team_id.as_deref().map(|team_id| AuthorizeTeam {
                team_id,
                blocked: self.team_blocked,
                requires_trace_id: self.team_requires_trace_id,
            }),
            enforce_user_param: self.enforce_user_param,
            has_user_param: self.has_user_param,
            reject_clientside_metadata_tags: self.reject_clientside_metadata_tags,
            has_client_tags: self.has_client_tags,
            block_unpriced_models: self.block_unpriced_models,
            max_budget: self.max_budget,
            global_proxy_spend: self.global_proxy_spend,
            route_access: self.route_access.access(memberships),
            budgets: self.budgets.facts(),
        }
    }
}
