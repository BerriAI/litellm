use crate::routes::RouteGroups;

pub fn route_groups() -> RouteGroups {
    RouteGroups::new(route_table()).expect("the stand-in table holds every group the core reads")
}

/// A small stand-in for the `LiteLLMRoutes` table the Python host sends.
pub fn route_table() -> Vec<(String, Vec<String>)> {
    let group = |name: &str, routes: &[&str]| {
        (
            name.to_owned(),
            routes.iter().map(|route| (*route).to_owned()).collect(),
        )
    };
    vec![
        group("public_routes", &["/", "/health/liveliness"]),
        group(
            "master_key_only_routes",
            &[
                "/global/spend/reset",
                "/memory-usage-in-mem-cache",
                "/memory-usage-in-mem-cache-items",
            ],
        ),
        group(
            "openai_routes",
            &[
                "/chat/completions",
                "/v1/chat/completions",
                "/engines/{model}/chat/completions",
                "/openai/deployments/{model}/chat/completions",
            ],
        ),
        group("info_routes", &["/key/info", "/user/info", "/team/list"]),
        group(
            "mcp_routes",
            &["/mcp/tools/list", "/mcp/tools/call", "/token"],
        ),
        group(
            "mcp_inference_routes",
            &["/mcp/{subpath}", "/mcp/tools/call", "/token", "/introspect"],
        ),
        group("agent_inference_routes", &["/a2a/{agent_id}/message/send"]),
        group("spend_tracking_routes", &["/spend/logs"]),
    ]
}
