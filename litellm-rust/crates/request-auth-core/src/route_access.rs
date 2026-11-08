use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::{Error, patterns};

#[derive(Clone, Default, Deserialize)]
pub struct RouteCatalog {
    pub groups: BTreeMap<String, Vec<String>>,
    pub viewer_blocked: Vec<String>,
    pub viewer_write_blocked: Vec<String>,
    pub self_credentials: Vec<(String, String)>,
    pub managed_models: Vec<String>,
    pub managed_paths: Vec<String>,
    pub managed_realtime: Vec<String>,
    pub managed_live: Vec<String>,
    pub managed_mcp: Vec<String>,
    pub unbilled: Vec<String>,
}

impl RouteCatalog {
    fn group(&self, name: &str) -> &[String] {
        self.groups.get(name).map_or(&[], Vec::as_slice)
    }

    fn access(&self, route: &str, name: &str) -> Result<bool, Error> {
        patterns::route_access(route, self.group(name))
    }

    fn contains(&self, route: &str, name: &str) -> bool {
        self.group(name).iter().any(|entry| entry == route)
    }
}

#[derive(Clone, Default, Deserialize)]
pub struct RouteContext {
    pub route: String,
    pub method: Option<String>,
    pub catalog: RouteCatalog,
    pub registered: bool,
    pub registered_auth: Option<bool>,
    pub configured_eu_jev: bool,
}

#[derive(Clone, Default, Deserialize)]
pub struct PassThroughFacts {
    pub key_allowed: Vec<String>,
    pub team_allowed: Vec<String>,
    pub denied: Vec<String>,
    pub forwarded: Vec<String>,
    pub jwt_team_routes: Vec<String>,
}

#[derive(Deserialize)]
pub struct VirtualRouteFacts {
    pub context: RouteContext,
    pub allowed_routes: Option<Vec<String>>,
    pub passthrough: PassThroughFacts,
    pub password_reset_required: bool,
}

#[derive(Deserialize)]
pub struct ViewerFacts {
    pub context: RouteContext,
    pub request_keys: Vec<String>,
}

#[derive(Deserialize)]
pub struct NonAdminFacts {
    pub viewer: ViewerFacts,
    pub role: Option<String>,
    pub custom_admin_routes: Vec<String>,
    pub premium: bool,
    pub query_user_id: Option<String>,
    pub token_user_id: Option<String>,
    pub permissions: Vec<String>,
    pub organization_ids: Vec<String>,
    pub admin_organization_ids: Vec<String>,
    pub team_service_account: bool,
    pub allowed_routes: Option<Vec<String>>,
    pub passthrough: PassThroughFacts,
}

#[derive(Deserialize)]
pub struct MatchFacts {
    pub route: String,
    pub patterns: Vec<String>,
    pub mode: MatchMode,
}

#[derive(Deserialize, strum::VariantArray)]
pub enum MatchMode {
    Access,
    Template,
    Prefix,
    Wildcard,
    Denied,
    Jwt,
}

#[derive(Deserialize, strum::VariantArray)]
pub enum Classification {
    Llm,
    Unbilled,
    Discovery,
    SelfCredential,
    Azure,
    Managed,
}

#[derive(Deserialize)]
pub enum Request {
    Match(MatchFacts),
    Classify {
        context: RouteContext,
        kind: Classification,
    },
    Virtual(VirtualRouteFacts),
    Viewer(ViewerFacts),
    NonAdmin(NonAdminFacts),
    PassThrough {
        context: RouteContext,
        facts: PassThroughFacts,
        required: bool,
    },
    DeniedPassThrough {
        route: String,
        facts: PassThroughFacts,
    },
}

#[derive(Debug, PartialEq, Eq, Serialize)]
pub struct Decision {
    pub allowed: bool,
    pub denial: Option<Denial>,
    pub detail: Option<String>,
}

#[derive(Debug, PartialEq, Eq, Serialize, strum::VariantArray)]
pub enum Denial {
    VirtualRoute,
    PasswordReset,
    PassThrough,
    DeniedPassThrough,
    CustomAdmin,
    UserInfo,
    AdminOnly,
    ViewerInference,
    ViewerWrite,
    ViewerParameter,
}

impl Decision {
    fn allow() -> Self {
        Self {
            allowed: true,
            denial: None,
            detail: None,
        }
    }
    fn predicate(value: bool) -> Self {
        Self {
            allowed: value,
            denial: None,
            detail: None,
        }
    }
    fn deny(denial: Denial) -> Self {
        Self {
            allowed: false,
            denial: Some(denial),
            detail: None,
        }
    }
    fn detail(denial: Denial, detail: String) -> Self {
        Self {
            allowed: false,
            denial: Some(denial),
            detail: Some(detail),
        }
    }
}

fn denied_pattern(route: &str, pattern: &str) -> bool {
    let normalized = pattern.trim_end_matches('/');
    normalized.is_empty()
        || patterns::prefix(route, normalized)
        || patterns::wildcard(route, pattern)
}

fn jwt_grant(route: &str, entries: &[String]) -> bool {
    entries.iter().any(|entry| {
        !entry.trim_end_matches('*').trim_matches('/').is_empty()
            && patterns::wildcard(route, entry)
    })
}

fn passthrough_grant(route: &str, facts: &PassThroughFacts) -> bool {
    let entries = if facts.key_allowed.is_empty() {
        &facts.team_allowed
    } else {
        &facts.key_allowed
    };
    entries.iter().any(|entry| patterns::prefix(route, entry))
}

fn denied_passthrough(facts: &PassThroughFacts) -> Option<&str> {
    facts
        .denied
        .iter()
        .find(|denied| {
            facts
                .forwarded
                .iter()
                .any(|route| denied_pattern(route, denied))
        })
        .map(String::as_str)
}

fn require_passthrough(route: &str, facts: &PassThroughFacts) -> Decision {
    if let Some(denied) = denied_passthrough(facts) {
        return Decision::detail(Denial::DeniedPassThrough, denied.to_owned());
    }
    if passthrough_grant(route, facts) || jwt_grant(route, &facts.jwt_team_routes) {
        return Decision::allow();
    }
    Decision::deny(Denial::PassThrough)
}

fn azure(route: &str) -> Result<bool, Error> {
    Ok(patterns::regex_match(
        "^/openai/deployments/[^/]+/[^/]+/chat/completions(?=\\n?\\z)",
        route,
    )? || patterns::regex_match("^/engines/[^/]+/chat/completions(?=\\n?\\z)", route)?)
}

fn llm(context: &RouteContext) -> Result<bool, Error> {
    let route = &context.route;
    let catalog = &context.catalog;
    for group in ["openai_routes", "anthropic_routes", "google_routes"] {
        if catalog.contains(route, group) {
            return Ok(true);
        }
    }
    for group in [
        "mcp_inference_routes",
        "agent_inference_routes",
        "litellm_native_routes",
    ] {
        if catalog.access(route, group)? {
            return Ok(true);
        }
    }
    for group in ["openai_routes", "google_routes", "anthropic_routes"] {
        for pattern in catalog.group(group) {
            if pattern.contains('{') && patterns::route_pattern(route, pattern)? {
                return Ok(true);
            }
            if group == "openai_routes"
                && pattern.ends_with('*')
                && patterns::wildcard(route, pattern)
            {
                return Ok(true);
            }
        }
    }
    Ok(azure(route)?
        || catalog
            .group("mapped_pass_through_routes")
            .iter()
            .any(|pattern| patterns::prefix(route, pattern))
        || (route.trim_end_matches('/') == "/eu_jev/v1/systemone" && !context.configured_eu_jev))
}

fn discovery(context: &RouteContext) -> bool {
    context.method.as_deref() == Some("GET")
        && (context.route == "/v1/mcp/server"
            || context
                .route
                .strip_prefix("/v1/mcp/server/")
                .is_some_and(|tail| !tail.is_empty() && !tail.contains('/')))
}

fn self_credential(context: &RouteContext) -> Result<bool, Error> {
    for (method, pattern) in &context.catalog.self_credentials {
        if context.method.as_ref() == Some(method)
            && patterns::route_access(&context.route, std::slice::from_ref(pattern))?
        {
            return Ok(true);
        }
    }
    Ok(false)
}

fn managed(context: &RouteContext) -> Result<bool, Error> {
    let route = &context.route;
    let method = context.method.as_deref();
    let catalog = &context.catalog;
    if route == "/agents" || route == "/v1/agents" {
        return Ok(matches!(method, None | Some("GET" | "HEAD")));
    }
    if catalog.managed_realtime.contains(route) || catalog.managed_live.contains(route) {
        return Ok(matches!(method, None | Some("GET")));
    }
    if catalog.managed_models.contains(route)
        || patterns::route_access(route, &catalog.managed_paths)?
    {
        return Ok(matches!(method, None | Some("POST")));
    }
    Ok(patterns::route_access(route, &catalog.managed_mcp)?
        || catalog.access(route, "agent_inference_routes")?)
}

fn virtual_route(facts: &VirtualRouteFacts) -> Result<Decision, Error> {
    let Some(allowed) = &facts.allowed_routes else {
        return Ok(Decision::allow());
    };
    if allowed.is_empty() {
        return Ok(Decision::allow());
    }
    let context = &facts.context;
    let route = &context.route;
    if allowed
        .iter()
        .any(|pattern| patterns::prefix(route, pattern))
    {
        return Ok(Decision::allow());
    }
    let mut denied_passthrough = false;
    for group in allowed {
        if !context.catalog.groups.contains_key(group) {
            continue;
        }
        let is_llm = matches!(
            group.as_str(),
            "llm_api_routes" | "llm_api_routes_without_mcp"
        );
        if context.catalog.access(route, group)? {
            if (is_llm || group == "openai_routes") && context.registered_auth == Some(true) {
                if passthrough_grant(route, &facts.passthrough) {
                    return Ok(Decision::allow());
                }
                denied_passthrough = true;
            } else {
                return Ok(Decision::allow());
            }
        }
        if !is_llm {
            continue;
        }
        if route == "/auto_router/session" && context.method.as_deref() == Some("GET") {
            return Ok(Decision::allow());
        }
        if context.registered {
            if context.registered_auth == Some(true) {
                if passthrough_grant(route, &facts.passthrough) {
                    return Ok(Decision::allow());
                }
                denied_passthrough = true;
            } else {
                return Ok(Decision::allow());
            }
        }
        if group == "llm_api_routes"
            && (discovery(context) || context.catalog.access(route, "agent_management_routes")?)
        {
            return Ok(Decision::allow());
        }
    }
    if allowed
        .iter()
        .any(|pattern| patterns::wildcard(route, pattern))
    {
        return Ok(Decision::allow());
    }
    Ok(Decision::deny(if denied_passthrough {
        Denial::PassThrough
    } else if facts.password_reset_required {
        Denial::PasswordReset
    } else {
        Denial::VirtualRoute
    }))
}

fn viewer(facts: &ViewerFacts) -> Result<Decision, Error> {
    let context = &facts.context;
    let route = &context.route;
    let catalog = &context.catalog;
    if llm(context)? {
        return Ok(Decision::deny(Denial::ViewerInference));
    }
    let invalid = facts
        .request_keys
        .iter()
        .find(|key| key.as_str() != "user_email");
    if catalog.access(route, "management_routes")? {
        if route == "/user/update" {
            if let Some(key) = invalid {
                return Ok(Decision::detail(Denial::ViewerParameter, key.clone()));
            }
        } else if patterns::route_access(route, &catalog.viewer_blocked)?
            || (route.starts_with("/key/")
                && (route.ends_with("/regenerate") || route.ends_with("/reset_spend")))
        {
            return Ok(Decision::deny(Denial::ViewerWrite));
        }
    }
    if matches!(
        context.method.as_deref().unwrap_or("GET"),
        "GET" | "HEAD" | "OPTIONS"
    ) {
        return Ok(Decision::allow());
    }
    if route == "/user/update" {
        return Ok(invalid.map_or_else(Decision::allow, |key| {
            Decision::detail(Denial::ViewerParameter, key.clone())
        }));
    }
    if route == "/user/password/change" || route == "/session/logout" || self_credential(context)? {
        return Ok(Decision::allow());
    }
    if patterns::route_access(route, &catalog.viewer_write_blocked)?
        || (route.starts_with("/key/") && route.ends_with("/regenerate"))
    {
        return Ok(Decision::deny(Denial::ViewerWrite));
    }
    if catalog.access(route, "admin_viewer_routes")?
        || catalog.access(route, "global_spend_tracking_routes")?
    {
        return Ok(Decision::allow());
    }
    Ok(Decision::deny(Denial::ViewerWrite))
}

fn non_admin(facts: &NonAdminFacts) -> Result<Decision, Error> {
    let context = &facts.viewer.context;
    let route = &context.route;
    let catalog = &context.catalog;
    let role = facts.role.as_deref();
    if facts.premium && facts.custom_admin_routes.contains(route) {
        return Ok(Decision::deny(Denial::CustomAdmin));
    }
    if context.registered_auth == Some(true) {
        return Ok(require_passthrough(route, &facts.passthrough));
    }
    if llm(context)? {
        return Ok(Decision::allow());
    }
    if catalog.contains(route, "info_routes") {
        if route == "/user/info"
            && facts
                .query_user_id
                .as_ref()
                .is_some_and(|id| !id.is_empty())
            && facts.query_user_id != facts.token_user_id
            && role != Some("proxy_admin_viewer")
        {
            return Ok(Decision::deny(Denial::UserInfo));
        }
        return Ok(Decision::allow());
    }
    if catalog.contains(route, "global_spend_tracking_routes")
        && facts
            .permissions
            .iter()
            .any(|entry| entry == "get_spend_routes")
    {
        return Ok(Decision::allow());
    }
    if patterns::prefix(route, "/lens") {
        return Ok(Decision::allow());
    }
    if role == Some("proxy_admin_viewer") {
        return viewer(&facts.viewer);
    }
    let org_admin = !facts.organization_ids.is_empty()
        && facts
            .organization_ids
            .iter()
            .all(|id| facts.admin_organization_ids.contains(id));
    if (role == Some("internal_user")
        && (catalog.access(route, "internal_user_routes")? || self_credential(context)?))
        || (org_admin && catalog.access(route, "org_admin_allowed_routes")?)
        || (role == Some("internal_user_viewer")
            && catalog.access(route, "internal_user_view_only_routes")?)
        || catalog.access(route, "self_managed_routes")?
        || route.starts_with("/v1/mcp/")
        || route.starts_with("/mcp-rest/")
        || passthrough_grant(route, &facts.passthrough)
        || (facts.team_service_account
            && catalog.access(route, "team_service_account_key_routes")?)
    {
        return Ok(Decision::allow());
    }
    if facts.allowed_routes.as_ref().is_some_and(|entries| {
        entries
            .iter()
            .any(|entry| patterns::prefix(route, entry) || patterns::wildcard(route, entry))
    }) {
        return Ok(Decision::allow());
    }
    Ok(Decision::deny(Denial::AdminOnly))
}

pub fn decide(request: Request) -> Result<Decision, Error> {
    match request {
        Request::Virtual(facts) => virtual_route(&facts),
        Request::Viewer(facts) => viewer(&facts),
        Request::NonAdmin(facts) => non_admin(&facts),
        Request::PassThrough {
            context,
            facts,
            required,
        } => Ok(if required {
            require_passthrough(&context.route, &facts)
        } else {
            Decision::predicate(passthrough_grant(&context.route, &facts))
        }),
        Request::DeniedPassThrough { route: _, facts } => Ok(denied_passthrough(&facts)
            .map_or_else(Decision::allow, |entry| {
                Decision::detail(Denial::DeniedPassThrough, entry.to_owned())
            })),
        Request::Match(facts) => {
            let matched = match facts.mode {
                MatchMode::Access => patterns::route_access(&facts.route, &facts.patterns)?,
                MatchMode::Jwt => jwt_grant(&facts.route, &facts.patterns),
                MatchMode::Template => {
                    let mut found = false;
                    for pattern in &facts.patterns {
                        if patterns::route_pattern(&facts.route, pattern)? {
                            found = true;
                            break;
                        }
                    }
                    found
                }
                MatchMode::Prefix => facts
                    .patterns
                    .iter()
                    .any(|pattern| patterns::prefix(&facts.route, pattern)),
                MatchMode::Wildcard => facts
                    .patterns
                    .iter()
                    .any(|pattern| patterns::wildcard(&facts.route, pattern)),
                MatchMode::Denied => facts
                    .patterns
                    .iter()
                    .any(|pattern| denied_pattern(&facts.route, pattern)),
            };
            Ok(Decision::predicate(matched))
        }
        Request::Classify { context, kind } => Ok(Decision::predicate(match kind {
            Classification::Llm => llm(&context)?,
            Classification::Azure => azure(&context.route)?,
            Classification::Discovery => discovery(&context),
            Classification::SelfCredential => self_credential(&context)?,
            Classification::Managed => managed(&context)?,
            Classification::Unbilled => {
                let tail = context.route.rsplit('/').next().unwrap_or("");
                let (resource, action) = tail.split_once(':').unwrap_or((tail, ""));
                context.catalog.unbilled.contains(&context.route)
                    || matches!(resource, "count_tokens" | "count-tokens")
                    || action == "countTokens"
            }
        })),
    }
}
