use http::Method;

use super::{
    groups::RouteGroups,
    table::{
        MANAGED_DISCOVERY_ROUTES, MANAGED_LIVE_ROUTES, MANAGED_MCP_EXCLUDED,
        MANAGED_MODEL_EXTRA_ROUTES, MANAGED_MODEL_OPERATIONS, MANAGED_MODEL_PATHS,
        MANAGED_MODEL_PREFIXES, MANAGED_REALTIME_ROUTES, RouteGroup,
    },
};

impl RouteGroups {
    fn is_managed_tool_route(&self, route: &str) -> bool {
        let mcp = self
            .entries(RouteGroup::McpInference)
            .into_iter()
            .flat_map(|entries| {
                entries
                    .routes
                    .iter()
                    .zip(&entries.templates)
                    .filter(|(entry, _)| !MANAGED_MCP_EXCLUDED.contains(&entry.as_str()))
            });
        { mcp }.any(|(_, template)| template.matches(route))
            || self.matches(route, RouteGroup::AgentInference)
    }

    pub fn managed_route(&self, route: &str) -> ManagedRoute {
        if MANAGED_DISCOVERY_ROUTES.contains(&route) {
            return ManagedRoute::Discovery;
        }
        if MANAGED_REALTIME_ROUTES.contains(&route) {
            return ManagedRoute::Realtime;
        }
        if MANAGED_LIVE_ROUTES.contains(&route) {
            return ManagedRoute::Live;
        }
        if is_managed_model_route(route) {
            return ManagedRoute::Model;
        }
        if self.is_managed_tool_route(route) {
            return ManagedRoute::Tool;
        }
        ManagedRoute::Other
    }

    /// `managed_agent_route_allowed`: agent discovery is read-only, realtime and live sessions open
    /// with a GET, inference is a POST, and MCP and agent-to-agent calls take any method.
    pub fn managed_agent_route_allowed(&self, route: &str, method: Option<&Method>) -> bool {
        match self.managed_route(route) {
            ManagedRoute::Discovery => {
                matches!(method, None | Some(&Method::GET | &Method::HEAD))
            }
            ManagedRoute::Realtime | ManagedRoute::Live => {
                matches!(method, None | Some(&Method::GET))
            }
            ManagedRoute::Model => matches!(method, None | Some(&Method::POST)),
            ManagedRoute::Tool => true,
            ManagedRoute::Other => false,
        }
    }
}

fn is_managed_model_route(route: &str) -> bool {
    MANAGED_MODEL_EXTRA_ROUTES.contains(&route)
        || MANAGED_MODEL_PREFIXES.iter().any(|prefix| {
            route
                .strip_prefix(prefix)
                .and_then(|rest| rest.strip_prefix('/'))
                .is_some_and(|operation| MANAGED_MODEL_OPERATIONS.contains(&operation))
        })
        || MANAGED_MODEL_PATHS.matches(route)
}

/// The kind of route a managed agent identity calls, which decides the methods it may use and how
/// its request names a model.
#[derive(Clone, Copy, Debug, PartialEq, Eq, strum::IntoStaticStr)]
pub enum ManagedRoute {
    #[strum(serialize = "discovery")]
    Discovery,
    #[strum(serialize = "realtime")]
    Realtime,
    #[strum(serialize = "live")]
    Live,
    #[strum(serialize = "model")]
    Model,
    /// MCP and agent-to-agent calls.
    #[strum(serialize = "tool")]
    Tool,
    #[strum(serialize = "other")]
    Other,
}

#[cfg(test)]
mod tests {
    use http::Method;
    use rstest::rstest;

    use crate::test_support::route_groups;

    #[rstest]
    #[case::discovery_get("/agents", Some(Method::GET), true)]
    #[case::discovery_post("/v1/agents", Some(Method::POST), false)]
    #[case::realtime_get("/v1/realtime", Some(Method::GET), true)]
    #[case::live_post("/live/sessions", Some(Method::POST), false)]
    #[case::chat_post("/v1/chat/completions", Some(Method::POST), true)]
    #[case::chat_get("/chat/completions", Some(Method::GET), false)]
    #[case::deployment_path("/openai/deployments/gpt/x/chat/completions", Some(Method::POST), true)]
    #[case::google_verb("/v1beta/models/gemini:generateContent", None, true)]
    #[case::path_model_with_newline("/engines/a\nb/chat/completions", Some(Method::POST), true)]
    #[case::empty_path_model("/engines//completions", Some(Method::POST), false)]
    #[case::google_model_with_colon("/models/a:b:generateContent", Some(Method::POST), false)]
    #[case::google_second_verb("/models/a:generateContent:countTokens", Some(Method::POST), false)]
    #[case::mcp_any_method("/mcp/tools/call", Some(Method::DELETE), true)]
    #[case::mcp_token_endpoint("/token", Some(Method::POST), false)]
    #[case::a2a("/a2a/agent-1/message/send", Some(Method::POST), true)]
    #[case::management("/key/generate", Some(Method::POST), false)]
    fn managed_agents_reach_inference_tools_and_discovery_only(
        #[case] route: &str,
        #[case] method: Option<Method>,
        #[case] allowed: bool,
    ) {
        assert_eq!(
            route_groups().managed_agent_route_allowed(route, method.as_ref()),
            allowed
        );
    }
}
