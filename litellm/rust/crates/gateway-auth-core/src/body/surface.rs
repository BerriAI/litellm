//! What a route reads its request body as. A name that is a deployment parameter on a model call
//! is the route's own field elsewhere, so the route decides before any name is judged.

use crate::routes::is_managed_model_route;

const DEPLOYMENT_MANAGEMENT_ROUTES: [&str; 4] = [
    "/model/new",
    "/model/update",
    "/model/delete",
    "/health/test_connection",
];
const MCP_SERVER_MANAGEMENT_ROUTE: &str = "/v1/mcp/server";
const TOKEN_INTROSPECTION_ROUTE: &str = "/introspect";
const TRANSCRIPTION_SESSION_ROUTES: [&str; 3] = [
    "/v1/realtime/transcription_sessions",
    "/realtime/transcription_sessions",
    "/openai/v1/realtime/transcription_sessions",
];
const ROUTES_THAT_NAME_THEIR_VERTEX_TARGET: [&str; 3] =
    ["/vertex_ai/live", "/rag/ingest", "/v1/rag/ingest"];

const OAUTH_CLIENT_FIELDS: &[&str] = &["client_id", "client_secret"];
const TRANSCRIPTION_SESSION_FIELDS: &[&str] = &["client_secret"];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Surface {
    /// The route configures a deployment instead of calling one. Its author is the admin the
    /// role gate behind it admits, so naming a credential or a secret reference is the point.
    pub manages_deployments: bool,
    /// The route registers an MCP server, whose OAuth token exchange settings are its own fields.
    pub manages_mcp_servers: bool,
    /// Override names that are this route's own fields.
    pub owned: &'static [&'static str],
    /// The route names the Vertex project and location it works in as part of its own request.
    pub names_vertex_target: bool,
    /// The body is the keyword arguments of a model call, so every key outside its content can
    /// reach a provider as a parameter.
    pub model_call: bool,
}

impl Surface {
    /// A body checked without a route is one a handler assembled for a model call.
    pub fn of(route: Option<&str>) -> Self {
        let Some(route) = route else {
            return Self {
                manages_deployments: false,
                manages_mcp_servers: false,
                owned: &[],
                names_vertex_target: false,
                model_call: true,
            };
        };
        let manages_mcp_servers = manages_mcp_servers(route);
        Self {
            manages_deployments: manages_deployments(route),
            manages_mcp_servers,
            owned: if TRANSCRIPTION_SESSION_ROUTES.contains(&route) {
                TRANSCRIPTION_SESSION_FIELDS
            } else if manages_mcp_servers || route == TOKEN_INTROSPECTION_ROUTE {
                OAUTH_CLIENT_FIELDS
            } else {
                &[]
            },
            names_vertex_target: ROUTES_THAT_NAME_THEIR_VERTEX_TARGET.contains(&route),
            model_call: is_managed_model_route(route),
        }
    }
}

fn manages_deployments(route: &str) -> bool {
    DEPLOYMENT_MANAGEMENT_ROUTES.contains(&route)
        || route
            .strip_prefix("/model/")
            .and_then(|rest| rest.strip_suffix("/update"))
            .is_some_and(|id| !id.is_empty() && !id.contains('/'))
}

fn manages_mcp_servers(route: &str) -> bool {
    route
        .strip_prefix(MCP_SERVER_MANAGEMENT_ROUTE)
        .is_some_and(|rest| rest.is_empty() || rest.starts_with('/'))
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    #[rstest]
    #[case::create("/model/new", true)]
    #[case::update("/model/update", true)]
    #[case::delete("/model/delete", true)]
    #[case::test_connection("/health/test_connection", true)]
    #[case::update_by_id("/model/abc-123/update", true)]
    #[case::update_with_empty_id("/model//update", false)]
    #[case::update_below_two_segments("/model/a/b/update", false)]
    #[case::info("/model/info", false)]
    #[case::chat("/v1/chat/completions", false)]
    fn a_route_manages_deployments_when_it_writes_or_tests_one(
        #[case] route: &str,
        #[case] expected: bool,
    ) {
        assert_eq!(Surface::of(Some(route)).manages_deployments, expected);
    }

    #[rstest]
    #[case::transcription_session("/v1/realtime/transcription_sessions", &["client_secret"])]
    #[case::mcp_server("/v1/mcp/server", &["client_id", "client_secret"])]
    #[case::mcp_server_by_id("/v1/mcp/server/abc", &["client_id", "client_secret"])]
    #[case::introspection("/introspect", &["client_id", "client_secret"])]
    #[case::mcp_server_lookalike("/v1/mcp/servers", &[])]
    #[case::chat("/v1/chat/completions", &[])]
    fn a_route_owns_the_credential_names_that_are_its_own_fields(
        #[case] route: &str,
        #[case] expected: &[&str],
    ) {
        assert_eq!(Surface::of(Some(route)).owned, expected);
    }

    #[rstest]
    #[case::chat(Some("/v1/chat/completions"), true)]
    #[case::messages(Some("/v1/messages"), true)]
    #[case::azure_deployment(Some("/openai/deployments/gpt-4o/chat/completions"), true)]
    #[case::no_route(None, true)]
    #[case::key_management(Some("/key/generate"), false)]
    #[case::credentials(Some("/credentials"), false)]
    #[case::provider_pass_through(Some("/vertex_ai/v1/projects/p/locations/l/endpoints/e"), false)]
    #[case::mcp_server(Some("/v1/mcp/server"), false)]
    fn a_body_is_a_model_call_on_a_model_route_or_with_no_route(
        #[case] route: Option<&str>,
        #[case] expected: bool,
    ) {
        assert_eq!(Surface::of(route).model_call, expected);
    }

    #[rstest]
    #[case::live("/vertex_ai/live", true)]
    #[case::rag_ingest("/v1/rag/ingest", true)]
    #[case::chat("/v1/chat/completions", false)]
    fn a_route_names_its_vertex_target_when_that_is_its_own_request(
        #[case] route: &str,
        #[case] expected: bool,
    ) {
        assert_eq!(Surface::of(Some(route)).names_vertex_target, expected);
    }
}
