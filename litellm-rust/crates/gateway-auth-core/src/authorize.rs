use crate::{
    AuthHost, AuthorizeDenial, BudgetExceeded, Error,
    budgets::{self, BudgetFacts},
    host::MembershipRow,
    route_access::{RouteAccess, route_access},
    routes::is_route_allowed,
};

const MODEL_LIST_ROUTES: &[&str] = &["/models", "/v1/models", "/claude_code_gateway/v1/models"];

/// The team the authenticated identity acts for, as the checks read it.
#[derive(Clone, Copy, Debug)]
pub struct AuthorizeTeam<'a> {
    pub team_id: &'a str,
    pub blocked: bool,
    /// `metadata.require_trace_id is True`
    pub requires_trace_id: bool,
}

/// What `common_checks` reads about the request and the identity it already authenticated. The
/// host reads the request's model, body and Python globals once, before the stage runs.
#[derive(Clone, Copy, Debug, Default)]
pub struct AuthorizeRequest<'a> {
    pub route: &'a str,
    pub method: Option<&'a str>,
    /// `RouteChecks.is_llm_api_route(route)`
    pub is_llm_api_route: bool,
    /// `request_dispatched_to_pass_through_endpoint(request)`
    pub dispatched_to_pass_through: bool,
    /// The request names a model.
    pub has_model: bool,
    /// The caller's `skip_budget_checks`, else `request_skips_budget_checks`.
    pub skip_budget_checks: bool,
    pub user_id: Option<&'a str>,
    pub agent_id: Option<&'a str>,
    pub team: Option<AuthorizeTeam<'a>>,
    /// `general_settings.enforce_user_param`
    pub enforce_user_param: bool,
    /// `"user" in request_body`
    pub has_user_param: bool,
    /// `general_settings.reject_clientside_metadata_tags`
    pub reject_clientside_metadata_tags: bool,
    /// The request set `metadata.tags` itself.
    pub has_client_tags: bool,
    /// `litellm.block_requests_for_models_without_pricing`
    pub block_unpriced_models: bool,
    /// `litellm.max_budget`
    pub max_budget: f64,
    pub global_proxy_spend: Option<f64>,
    pub route_access: RouteAccess<'a>,
    pub budgets: BudgetFacts<'a>,
}

/// The team membership the checks read: unread when no check needs it, else the row or none.
#[derive(Debug, PartialEq)]
pub enum Membership<'a, M> {
    Unread,
    Read(Option<&'a M>),
}

impl<M> Clone for Membership<'_, M> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<M> Copy for Membership<'_, M> {}

/// `common_checks`, in its order. The checks the core cannot decide yet are host steps that raise
/// their own error.
pub async fn authorize<H: AuthHost>(host: &H, request: &AuthorizeRequest<'_>) -> Result<(), Error> {
    let membership_user_id = request
        .user_id
        .filter(|_| request.has_model || !request.skip_budget_checks);
    let membership_row: Option<Option<MembershipRow<H>>> = match (request.team, membership_user_id) {
        (Some(team), Some(user_id)) => Some(host.load_team_membership(user_id, team.team_id).await?),
        _ => None,
    };
    let membership = match &membership_row {
        Some(row) => Membership::Read(row.as_ref()),
        None => Membership::Unread,
    };

    if request.block_unpriced_models && request.is_llm_api_route {
        let models = host.unpriced_models().await?;
        if !models.is_empty() {
            return Err(AuthorizeDenial::UnpricedModels { models }.into());
        }
    }
    if let Some(team) = request.team.filter(|team| team.blocked) {
        return Err(AuthorizeDenial::TeamBlocked {
            team_id: team.team_id.to_owned(),
        }
        .into());
    }
    host.check_team_model_access(membership).await?;
    if let Some(agent_id) = request.agent_id.filter(|id| !id.is_empty())
        && host.agent_requires_trace_id(agent_id).await?
        && !host.has_trace_id(true).await?
    {
        return Err(AuthorizeDenial::AgentTraceIdRequired.into());
    }
    team_trace_id(host, request).await?;
    host.check_model_grants().await?;
    host.check_project().await?;
    let matched_access_groups = host.stamp_access_groups(membership).await?;
    if request.reject_clientside_metadata_tags && request.is_llm_api_route && request.has_client_tags {
        return Err(AuthorizeDenial::ClientsideTags.into());
    }
    if !request.skip_budget_checks {
        host.merge_budget_tags().await?;
        budgets::check(host, request, membership, &matched_access_groups).await?;
    }
    if request.enforce_user_param
        && request.method.is_some_and(|method| method.eq_ignore_ascii_case("POST"))
        && request.is_llm_api_route
        && !is_route_allowed(request.route, &["mcp_routes"])
        && !request.has_user_param
    {
        return Err(AuthorizeDenial::UserParamMissing.into());
    }
    global_proxy_budget(request)?;
    host.check_guardrail_modification().await?;
    route_access(host, &request.route_access).await?;
    host.check_object_access().await
}

/// The team's `require_trace_id` gate on LLM, pass-through, MCP and agent calls that change state.
async fn team_trace_id<H: AuthHost>(host: &H, request: &AuthorizeRequest<'_>) -> Result<(), Error> {
    let Some(team) = request.team.filter(|team| team.requires_trace_id) else {
        return Ok(());
    };
    let route = request.route;
    let pass_through =
        is_route_allowed(route, &["passthrough_routes_wildcard"]) || request.dispatched_to_pass_through;
    let applies = !matches!(request.method, Some("GET" | "HEAD" | "OPTIONS"))
        && (request.is_llm_api_route || pass_through)
        && !is_route_allowed(route, &["trace_telemetry_routes"]);
    if !applies {
        return Ok(());
    }
    let headers_only = is_route_allowed(route, &["mcp_inference_routes", "agent_inference_routes"]) || pass_through;
    if host.has_trace_id(headers_only).await? {
        return Ok(());
    }
    Err(AuthorizeDenial::TeamTraceIdRequired {
        team_id: team.team_id.to_owned(),
    }
    .into())
}

/// `_global_proxy_budget_check`
fn global_proxy_budget(request: &AuthorizeRequest<'_>) -> Result<(), Error> {
    let Some(spend) = request.global_proxy_spend else {
        return Ok(());
    };
    let max_budget = request.max_budget;
    let applies = max_budget > 0.0
        && !request.skip_budget_checks
        && request.is_llm_api_route
        && !MODEL_LIST_ROUTES.contains(&request.route);
    if !applies || !max_budget.is_finite() || spend <= max_budget {
        return Ok(());
    }
    Err(BudgetExceeded {
        current_cost: spend,
        max_budget,
        message: format!("Budget has been exceeded! Current cost: {spend}, Max budget: {max_budget}"),
        entity_type: Some("proxy".to_owned()),
        entity_id: None,
    }
    .into())
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;
    use crate::{
        route_access::{RouteRequest, RouteToken},
        testing::ScriptedHost,
    };

    const CHAT: &str = "/chat/completions";

    fn request() -> AuthorizeRequest<'static> {
        AuthorizeRequest {
            route: CHAT,
            method: Some("POST"),
            is_llm_api_route: true,
            has_model: true,
            route_access: RouteAccess {
                request: RouteRequest {
                    route: CHAT,
                    method: Some("POST"),
                    is_llm_api_route: true,
                    ..RouteRequest::default()
                },
                user: None,
                token: Some(RouteToken::default()),
            },
            ..AuthorizeRequest::default()
        }
    }

    fn team(blocked: bool, requires_trace_id: bool) -> Option<AuthorizeTeam<'static>> {
        Some(AuthorizeTeam {
            team_id: "t1",
            blocked,
            requires_trace_id,
        })
    }

    const CHECKS: &[&str] = &[
        "check_team_model_access -",
        "check_model_grants",
        "check_project",
        "stamp_access_groups -",
        "merge_budget_tags",
        "tag_budgets",
        "check_guardrail_modification",
        "check_object_access",
    ];

    #[tokio::test]
    async fn every_check_runs_in_common_checks_order() {
        let host = ScriptedHost::default();
        assert_eq!(authorize(&host, &request()).await, Ok(()));
        assert_eq!(host.calls(), CHECKS);
    }

    #[rstest]
    #[case::with_a_model(true, false, true)]
    #[case::budget_checks_only(false, false, true)]
    #[case::free_request_without_a_model(false, true, false)]
    #[tokio::test]
    async fn the_membership_is_read_only_when_a_model_or_budget_check_needs_it(
        #[case] has_model: bool,
        #[case] skip_budget_checks: bool,
        #[case] read: bool,
    ) {
        let host = ScriptedHost::default();
        let request = AuthorizeRequest {
            has_model,
            skip_budget_checks,
            user_id: Some("u1"),
            team: team(false, false),
            ..request()
        };
        assert_eq!(authorize(&host, &request).await, Ok(()));
        assert_eq!(host.calls().first().map(String::as_str) == Some("membership u1 t1"), read);
        assert_eq!(
            host.calls().contains(&"check_team_model_access membership u1 t1".to_owned()),
            read
        );
    }

    #[tokio::test]
    async fn a_blocked_team_is_denied_before_any_model_check() {
        let host = ScriptedHost::default();
        let request = AuthorizeRequest {
            team: team(true, false),
            ..request()
        };
        assert_eq!(
            authorize(&host, &request).await.unwrap_err().public(),
            Some(crate::PublicDenial::Plain {
                message: "Team=t1 is blocked. Update via `/team/unblock` if you're an admin.".to_owned()
            })
        );
        assert!(host.calls().is_empty());
    }

    #[rstest]
    #[case::one(&["gpt-x"], "Model 'gpt-x' has no pricing")]
    #[case::several(&["a", "b"], "Models 'a', 'b' have no pricing")]
    #[tokio::test]
    async fn unpriced_models_are_blocked_when_the_flag_is_on(#[case] models: &[&str], #[case] opening: &str) {
        let host = ScriptedHost {
            unpriced: models.iter().map(|model| (*model).to_owned()).collect(),
            ..ScriptedHost::default()
        };
        let request = AuthorizeRequest {
            block_unpriced_models: true,
            ..request()
        };
        let denial = authorize(&host, &request).await.unwrap_err().public();
        let Some(crate::PublicDenial::Proxy {
            status,
            message,
            param,
            ..
        }) = denial
        else {
            panic!("expected a proxy denial");
        };
        assert_eq!((status, param.as_deref()), (403, Some("model")));
        assert!(message.starts_with(opening), "{message}");
    }

    #[rstest]
    #[case::missing_header(false, Some("400"))]
    #[case::header_sent(true, None)]
    #[tokio::test]
    async fn an_agent_that_requires_a_trace_id_needs_the_header(
        #[case] has_trace_id: bool,
        #[case] status: Option<&str>,
    ) {
        let host = ScriptedHost {
            agent_requires_trace_id: true,
            has_trace_id,
            ..ScriptedHost::default()
        };
        let request = AuthorizeRequest {
            agent_id: Some("a1"),
            ..request()
        };
        let outcome = authorize(&host, &request).await;
        assert_eq!(
            outcome.err().and_then(|error| error.public()).map(|denial| match denial {
                crate::PublicDenial::Proxy { status, .. } => status.to_string(),
                other => format!("{other:?}"),
            }),
            status.map(str::to_owned)
        );
    }

    #[rstest]
    #[case::llm_post(CHAT, Some("POST"), true, Some("has_trace_id false"))]
    #[case::read_only(CHAT, Some("GET"), false, None)]
    #[case::not_an_llm_or_pass_through_route("/mcp/tools/call", Some("POST"), false, None)]
    #[tokio::test]
    async fn a_team_that_requires_a_trace_id_gates_state_changing_calls(
        #[case] route: &'static str,
        #[case] method: Option<&'static str>,
        #[case] is_llm_api_route: bool,
        #[case] asked: Option<&str>,
    ) {
        let host = ScriptedHost::default();
        let request = AuthorizeRequest {
            route,
            method,
            is_llm_api_route,
            team: team(false, true),
            ..request()
        };
        let denied = authorize(&host, &request).await.is_err();
        assert_eq!(denied, asked.is_some());
        assert_eq!(
            host.calls().iter().find(|call| call.starts_with("has_trace_id")).map(String::as_str),
            asked
        );
    }

    #[rstest]
    #[case::over(2.0, 1.0, true)]
    #[case::at(1.0, 1.0, false)]
    #[case::unlimited(5.0, 0.0, false)]
    fn the_proxy_budget_denies_spend_past_it(#[case] spend: f64, #[case] max_budget: f64, #[case] denied: bool) {
        let request = AuthorizeRequest {
            max_budget,
            global_proxy_spend: Some(spend),
            ..request()
        };
        assert_eq!(global_proxy_budget(&request).is_err(), denied);
    }

    #[rstest]
    #[case::post_without_user("POST", false, CHAT, true)]
    #[case::user_sent("POST", true, CHAT, false)]
    #[case::mcp_route("POST", false, "/mcp/tools/call", false)]
    #[case::get("GET", false, CHAT, false)]
    #[tokio::test]
    async fn enforce_user_param_needs_a_user_on_llm_posts(
        #[case] method: &'static str,
        #[case] has_user_param: bool,
        #[case] route: &'static str,
        #[case] denied: bool,
    ) {
        let host = ScriptedHost::default();
        let request = AuthorizeRequest {
            route,
            method: Some(method),
            enforce_user_param: true,
            has_user_param,
            ..request()
        };
        assert_eq!(
            authorize(&host, &request).await.err().and_then(|error| error.public()),
            denied.then(|| crate::PublicDenial::Plain {
                message: "'user' param not passed in. 'enforce_user_param'=True".to_owned()
            })
        );
    }
}
