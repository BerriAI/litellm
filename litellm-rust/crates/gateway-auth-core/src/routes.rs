use std::{collections::HashMap, sync::LazyLock};

// Copied from LiteLLMRoutes in litellm/proxy/_types.py, as is route_groups.json. TODO: pin
// route_groups.json against LiteLLMRoutes from a Python unit test, read PUBLIC_ROUTES from it, then
// build LiteLLMRoutes from the same file so the copies cannot drift.
pub const PUBLIC_ROUTES: &[&str] = &[
    "/routes",
    "/",
    "/health/liveliness",
    "/health/liveness",
    "/test",
    "/config/yaml",
    "/litellm/.well-known/litellm-ui-config",
    "/.well-known/litellm-ui-config",
    "/public/model_hub",
    "/public/v1/model_hub",
    "/public/v1/model_hub/providers",
    "/public/v1/model_hub/modes",
    "/public/v1/model_hub/features",
    "/public/model_hub/info",
    "/public/agent_hub",
    "/public/mcp_hub",
    "/public/skill_hub",
    "/public/litellm_model_cost_map",
    "/moyai/connect/exchange",
];

pub const MASTER_KEY_ONLY_ROUTES: &[&str] = &[
    "/global/spend/reset",
    "/memory-usage-in-mem-cache",
    "/memory-usage-in-mem-cache-items",
];

#[derive(Clone, Debug, PartialEq, Eq)]
enum Part {
    Literal(String),
    Segment,
    Path,
}

/// A route template compiled as starlette's `compile_path` does.
#[derive(Debug)]
struct Template(Box<[Part]>);

impl Template {
    fn compile(template: &str) -> Self {
        let is_host = !template.starts_with('/');
        let mut parts = Vec::new();
        let mut rest = template;
        while let Some((before, name, convertor, after)) = next_param(rest) {
            parts.push(Part::Literal(before.to_owned()));
            parts.push(match convertor {
                None | Some("str") => Part::Segment,
                Some("path") => Part::Path,
                Some(other) => {
                    panic!("route template {template} uses an unported convertor {other} on {name}")
                }
            });
            rest = after;
        }
        let tail = if is_host {
            rest.split(':').next().unwrap_or_default()
        } else {
            rest
        };
        parts.push(Part::Literal(tail.to_owned()));
        Self(
            parts
                .into_iter()
                .filter(|part| *part != Part::Literal(String::new()))
                .collect(),
        )
    }

    fn matches(&self, route: &str) -> bool {
        matches_parts(&self.0, route)
    }
}

fn is_identifier(text: &str) -> bool {
    let mut chars = text.chars();
    chars
        .next()
        .is_some_and(|first| first.is_ascii_alphabetic() || first == '_')
        && chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
}

/// The next `{name}` or `{name:convertor}`, as starlette's `PARAM_REGEX` finds it.
fn next_param(text: &str) -> Option<(&str, &str, Option<&str>, &str)> {
    text.match_indices('{').find_map(|(start, _)| {
        let end = start + text[start..].find('}')?;
        let inner = &text[start + 1..end];
        let (name, convertor) = match inner.split_once(':') {
            Some((name, convertor)) => (name, Some(convertor)),
            None => (inner, None),
        };
        (is_identifier(name) && convertor.is_none_or(is_identifier))
            .then(|| (&text[..start], name, convertor, &text[end + 1..]))
    })
}

fn matches_parts(parts: &[Part], route: &str) -> bool {
    let Some((part, rest)) = parts.split_first() else {
        return route.is_empty() || route == "\n";
    };
    let candidates = |limit: usize, min: usize| {
        route
            .char_indices()
            .map(|(index, _)| index)
            .chain([route.len()])
            .filter(move |index| *index >= min && *index <= limit)
    };
    match part {
        Part::Literal(literal) => route
            .strip_prefix(literal.as_str())
            .is_some_and(|after| matches_parts(rest, after)),
        Part::Segment => candidates(route.find('/').unwrap_or(route.len()), 1)
            .any(|index| matches_parts(rest, &route[index..])),
        Part::Path => candidates(route.find('\n').unwrap_or(route.len()), 0)
            .any(|index| matches_parts(rest, &route[index..])),
    }
}

#[derive(Debug)]
struct RouteGroup {
    routes: Box<[String]>,
    templates: Box<[Template]>,
}

static ROUTE_GROUPS: LazyLock<HashMap<String, RouteGroup>> = LazyLock::new(|| {
    let groups: HashMap<String, Box<[String]>> =
        serde_json::from_str(include_str!("route_groups.json"))
            .expect("route_groups.json is a map of route lists");
    groups
        .into_iter()
        .map(|(name, routes)| {
            let templates = routes
                .iter()
                .map(|route| Template::compile(route))
                .collect();
            (name, RouteGroup { routes, templates })
        })
        .collect()
});

/// `RouteChecks.route_matches_wildcard_pattern`
fn matches_wildcard(route: &str, pattern: &str) -> bool {
    match pattern.strip_suffix('*') {
        Some(prefix) => route.starts_with(prefix),
        None => route == pattern,
    }
}

/// `_allowed_routes_check`: each entry is a `LiteLLMRoutes` group name, an exact route, or a
/// prefix ending in `*`.
pub fn is_route_allowed<S: AsRef<str>>(route: &str, allowed_routes: &[S]) -> bool {
    allowed_routes
        .iter()
        .any(|allowed| match ROUTE_GROUPS.get(allowed.as_ref()) {
            Some(group) => group
                .templates
                .iter()
                .any(|template| template.matches(route)),
            None => matches_wildcard(route, allowed.as_ref()),
        })
}

const MANAGED_REALTIME_ROUTES: &[&str] = &[
    "/realtime",
    "/v1/realtime",
    "/openai/v1/realtime",
    "/realtime/translations",
    "/v1/realtime/translations",
];
const MANAGED_LIVE_ROUTES: &[&str] = &["/live/sessions", "/v1/live/sessions", "/openai/v1/live/sessions"];
const MANAGED_MODEL_OPERATIONS: &[&str] = &[
    "chat/completions",
    "completions",
    "embeddings",
    "responses",
    "messages",
    "messages/count_tokens",
    "images/generations",
    "images/edits",
    "audio/transcriptions",
    "audio/speech",
    "moderations",
    "rerank",
    "decisions",
    "systemone",
    "extraction",
    "ocr",
];
const MANAGED_MODEL_EXTRA_ROUTES: &[&str] = &[
    "/openai/v1/responses",
    "/v2/rerank",
    "/claude_code_gateway/v1/messages",
    "/claude_code_gateway/v1/messages/count_tokens",
    "/cursor/chat/completions",
];
const MANAGED_MODEL_PATHS: &[&str] = &[
    "/engines/{model:path}/chat/completions",
    "/engines/{model:path}/completions",
    "/engines/{model:path}/embeddings",
    "/openai/deployments/{model:path}/chat/completions",
    "/openai/deployments/{model:path}/completions",
    "/openai/deployments/{model:path}/embeddings",
    "/openai/deployments/{model:path}/images/generations",
    "/openai/deployments/{model:path}/images/edits",
    "/v1beta/models/{model_name:path}:countTokens",
    "/v1beta/models/{model_name:path}:generateContent",
    "/v1beta/models/{model_name:path}:streamGenerateContent",
    "/models/{model_name:path}:countTokens",
    "/models/{model_name:path}:generateContent",
    "/models/{model_name:path}:streamGenerateContent",
];
/// MCP inference routes a managed agent may call; the OAuth token endpoints are not among them.
const MANAGED_MCP_EXCLUDED: &[&str] = &["/token", "/introspect"];

static MANAGED_MODEL_TEMPLATES: LazyLock<Box<[Template]>> =
    LazyLock::new(|| MANAGED_MODEL_PATHS.iter().map(|path| Template::compile(path)).collect());

static MANAGED_MCP_TEMPLATES: LazyLock<Box<[Template]>> = LazyLock::new(|| {
    ROUTE_GROUPS["mcp_inference_routes"]
        .routes
        .iter()
        .filter(|route| !MANAGED_MCP_EXCLUDED.contains(&route.as_str()))
        .map(|route| Template::compile(route))
        .collect()
});

fn is_managed_model_route(route: &str) -> bool {
    MANAGED_MODEL_EXTRA_ROUTES.contains(&route)
        || ["", "/v1"].iter().any(|prefix| {
            route
                .strip_prefix(prefix)
                .and_then(|rest| rest.strip_prefix('/'))
                .is_some_and(|operation| MANAGED_MODEL_OPERATIONS.contains(&operation))
        })
        || MANAGED_MODEL_TEMPLATES.iter().any(|template| template.matches(route))
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

pub fn managed_route(route: &str) -> ManagedRoute {
    if route == "/agents" || route == "/v1/agents" {
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
    if MANAGED_MCP_TEMPLATES.iter().any(|template| template.matches(route))
        || is_route_allowed(route, &["agent_inference_routes"])
    {
        return ManagedRoute::Tool;
    }
    ManagedRoute::Other
}

/// `managed_agent_route_allowed`: agent discovery is read-only, realtime and live sessions open
/// with a GET, inference is a POST, and MCP and agent-to-agent calls take any method.
pub fn managed_agent_route_allowed(route: &str, method: Option<&str>) -> bool {
    match managed_route(route) {
        ManagedRoute::Discovery => matches!(method, None | Some("GET" | "HEAD")),
        ManagedRoute::Realtime | ManagedRoute::Live => matches!(method, None | Some("GET")),
        ManagedRoute::Model => matches!(method, None | Some("POST")),
        ManagedRoute::Tool => true,
        ManagedRoute::Other => false,
    }
}

/// `route in LiteLLMRoutes.<group>.value`: the route is one of the group's entries as written.
pub fn group_lists(route: &str, group: &str) -> bool {
    ROUTE_GROUPS
        .get(group)
        .is_some_and(|group| group.routes.iter().any(|entry| entry == route))
}

/// `RouteChecks.check_route_access` against routes written as starlette templates.
pub fn matches_any_template(route: &str, templates: &[&str]) -> bool {
    templates.iter().any(|template| Template::compile(template).matches(route))
}

/// `get_actual_routes`: group names expanded to their routes, anything else kept as given.
pub fn actual_routes<S: AsRef<str>>(allowed_routes: &[S]) -> Vec<String> {
    allowed_routes
        .iter()
        .flat_map(|allowed| match ROUTE_GROUPS.get(allowed.as_ref()) {
            Some(group) => group.routes.to_vec(),
            None => vec![allowed.as_ref().to_owned()],
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use rstest::rstest;

    use super::*;

    #[rstest]
    #[case::discovery_get("/agents", Some("GET"), true)]
    #[case::discovery_post("/v1/agents", Some("POST"), false)]
    #[case::realtime_get("/v1/realtime", Some("GET"), true)]
    #[case::live_post("/live/sessions", Some("POST"), false)]
    #[case::chat_post("/v1/chat/completions", Some("POST"), true)]
    #[case::chat_get("/chat/completions", Some("GET"), false)]
    #[case::deployment_path("/openai/deployments/gpt/x/chat/completions", Some("POST"), true)]
    #[case::google_verb("/v1beta/models/gemini:generateContent", None, true)]
    #[case::mcp_any_method("/mcp/tools/call", Some("DELETE"), true)]
    #[case::mcp_token_endpoint("/token", Some("POST"), false)]
    #[case::a2a("/a2a/agent-1/message/send", Some("POST"), true)]
    #[case::management("/key/generate", Some("POST"), false)]
    fn managed_agents_reach_inference_tools_and_discovery_only(
        #[case] route: &str,
        #[case] method: Option<&str>,
        #[case] allowed: bool,
    ) {
        assert_eq!(managed_agent_route_allowed(route, method), allowed);
    }

    #[rstest]
    #[case::exact("/models", "/models", true)]
    #[case::exact_is_whole_route("/models", "/models/x", false)]
    #[case::segment("/files/{file_id}", "/files/abc", true)]
    #[case::segment_needs_a_char("/files/{file_id}/content", "/files//content", false)]
    #[case::segment_stops_at_slash("/files/{file_id}", "/files/a/b", false)]
    #[case::path_spans_slashes(
        "/v1beta/models/{model_name:path}:generateContent",
        "/v1beta/models/a/b:generateContent",
        true
    )]
    #[case::literal_after_param(
        "/claude-code/plugins/{plugin_name}/archive/{digest}.zip",
        "/claude-code/plugins/p/archive/d.zip",
        true
    )]
    #[case::dot_is_literal(
        "/claude-code/plugins/{plugin_name}/archive/{digest}.zip",
        "/claude-code/plugins/p/archive/dxzip",
        false
    )]
    #[case::question_mark_is_literal("/realtime?{model}", "/realtime?gpt", true)]
    #[case::trailing_newline_like_the_router("/models", "/models\n", true)]
    #[case::star_is_literal_in_templates("/a/*", "/a/b", false)]
    #[case::host_template_drops_port("chat_completion:80", "chat_completion", true)]
    fn templates_match_like_starlette(
        #[case] template: &str,
        #[case] route: &str,
        #[case] expected: bool,
    ) {
        assert_eq!(Template::compile(template).matches(route), expected);
    }

    #[rstest]
    #[case::group("info_routes", "/key/info", true)]
    #[case::group_template("openai_routes", "/engines/gpt/chat/completions", true)]
    #[case::exact_route("/v1/messages", "/v1/messages", true)]
    #[case::wildcard_prefix("/scim/v2/*", "/scim/v2/Users/1", true)]
    #[case::wildcard_other("/scim/v2/*", "/scim/v3", false)]
    #[case::unknown_name_is_exact("not_a_group", "/not_a_group", false)]
    fn allowed_routes(#[case] allowed: &str, #[case] route: &str, #[case] expected: bool) {
        assert_eq!(is_route_allowed(route, &[allowed]), expected);
    }

    #[test]
    fn actual_routes_expand_groups_in_order() {
        assert_eq!(
            actual_routes(&["/first", "master_key_only_routes", "/last"]),
            [
                "/first",
                "/global/spend/reset",
                "/memory-usage-in-mem-cache",
                "/memory-usage-in-mem-cache-items",
                "/last",
            ]
        );
    }

    #[test]
    fn route_groups_agree_with_route_constants() {
        let group = |name: &str| {
            ROUTE_GROUPS[name]
                .routes
                .iter()
                .map(String::as_str)
                .collect::<HashSet<_>>()
        };
        assert_eq!(
            group("public_routes"),
            PUBLIC_ROUTES.iter().copied().collect()
        );
        assert_eq!(
            group("master_key_only_routes"),
            MASTER_KEY_ONLY_ROUTES.iter().copied().collect()
        );
    }
}
