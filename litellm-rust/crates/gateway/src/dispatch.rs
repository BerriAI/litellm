use std::sync::Arc;

use axum::{
    Router,
    body::HttpBody,
    extract::{Request, State},
    middleware::{self, Next},
    response::Response,
};
use litellm_config::{Config, GeneralSettings};
use litellm_gateway_auth::{ApiKey, Auth, AuthenticatedRequest, AuthenticationMethod};
use litellm_gateway_auth_core::plan::{self, Plan};
use litellm_gateway_forward::Forward;
use litellm_gateway_inference::Gateway;
use tower::ServiceExt;

/// The auth paths the gateway runs itself.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum NativePlan {
    Key,
}

/// Settings Python applies before or instead of the auth paths, which the gateway can't apply yet.
/// Any of them sends every request to Python.
const PYTHON_ONLY_SETTINGS: [&str; 5] = [
    "allowed_ips",
    "allowed_routes",
    "custom_auth",
    "litellm_key_header_name",
    "pass_through_endpoints",
];

/// Where a request runs, decided from the config at boot, then the request, then its caller.
struct Policy {
    forward_all: bool,
    auth: plan::Settings,
}

impl Policy {
    fn new(config: &Config, env: &dyn Fn(&str) -> Option<String>) -> Self {
        let general = &config.general_settings;
        let configured = |name: &str| {
            general
                .additional_fields
                .get(name)
                .is_some_and(|value| !value.is_null())
        };
        let forward_all = has_db(general, env)
            || env("SERVER_ROOT_PATH").is_some_and(|path| !path.is_empty())
            || general.master_key.is_none()
            || general.max_request_size_mb.is_some()
            // TODO: `premium` needs the `LITELLM_LICENSE` check, which only Python runs, so Rust
            // can't tell whether these routes are public.
            || general.public_routes.as_deref().is_some_and(|routes| !routes.is_empty())
            || PYTHON_ONLY_SETTINGS.into_iter().any(configured);
        if forward_all {
            tracing::info!(
                "the configuration needs python for every request, so the gateway forwards all of them"
            );
        }
        Self {
            forward_all,
            auth: crate::auth::settings(config, None, env).plan,
        }
    }

    /// The path to run in Rust, or `None` to forward. Only called on routes the native router matched.
    fn dispatch(&self, route: &str, key: &ApiKey, has_body: bool) -> Option<NativePlan> {
        // TODO: `is_request_body_safe` only runs in Python, and it applies to the master key too.
        if self.forward_all || has_body {
            return None;
        }
        let request = plan::Request {
            route,
            token: key.token()?,
            // Every native route is an LLM API route except MCP, and counting MCP too only widens
            // OAuth2 introspection, which forwards.
            is_llm_api_route: true,
            ..plan::Request::default()
        };
        match plan::plan(&request, &self.auth) {
            Ok(Plan::Key) => Some(NativePlan::Key),
            Ok(
                Plan::Public
                | Plan::OAuth2Introspection
                | Plan::OAuth2Proxy
                | Plan::Jwt
                | Plan::PassThroughOpen,
            )
            | Err(_) => None,
        }
    }

    /// Whether Rust serves the caller it authenticated.
    fn serves(&self, identity: &AuthenticatedRequest) -> bool {
        match identity.caller().authentication().method {
            AuthenticationMethod::MasterKey => true,
            AuthenticationMethod::Session | AuthenticationMethod::External(_) => false,
        }
    }
}

/// Python overlays `general_settings` from the DB at runtime, which a one-time config read can't see.
fn has_db(general: &GeneralSettings, env: &dyn Fn(&str) -> Option<String>) -> bool {
    general.database_url.is_some()
        || ["DATABASE_URL", "DIRECT_URL", "DATABASE_HOST"]
            .into_iter()
            .any(|name| env(name).is_some())
}

#[derive(Clone)]
struct Gate {
    policy: Arc<Policy>,
    auth: Auth,
    forward: Forward,
}

/// Serves a request natively only when the policy lets its config, the request and its
/// authenticated caller run in Rust. Authentication runs once, here, and a request it does not
/// let through goes to Python, which answers it as it would without the gateway.
async fn gate(State(gate): State<Gate>, request: Request, next: Next) -> Response {
    let (mut parts, body) = request.into_parts();
    let key = ApiKey::from_headers(&parts.headers);
    let identity = match gate
        .policy
        .dispatch(parts.uri.path(), &key, !body.is_end_stream())
    {
        Some(NativePlan::Key) => gate.auth.admit(&parts).await.ok(),
        None => None,
    }
    .filter(|identity| gate.policy.serves(identity));
    let Some(identity) = identity else {
        return to_python(gate.forward, Request::from_parts(parts, body)).await;
    };
    parts.extensions.insert(identity);
    next.run(Request::from_parts(parts, body)).await
}

async fn to_python(forward: Forward, request: Request) -> Response {
    forward
        .oneshot(request)
        .await
        .unwrap_or_else(|never| match never {})
}

/// The native routes in front of Python: a request the gate lets through runs in Rust, and one it
/// doesn't, or that no native route or method matches, goes to Python through `forward`.
pub fn dispatching_router(
    inference: Arc<Gateway>,
    mcp: Option<Router>,
    config: &Config,
    env: &dyn Fn(&str) -> Option<String>,
    auth: Auth,
    forward: Forward,
) -> Router {
    let gate_state = Gate {
        policy: Arc::new(Policy::new(config, env)),
        auth,
        forward: forward.clone(),
    };
    let wrong_method = forward.clone();
    crate::native_routes(inference, mcp)
        .layer(middleware::from_fn(crate::log_request))
        .route_layer(middleware::from_fn_with_state(gate_state, gate))
        .method_not_allowed_fallback(move |request: Request| {
            to_python(wrong_method.clone(), request)
        })
        .fallback_service(forward)
}

#[cfg(test)]
mod tests {
    use axum::http::{HeaderMap, HeaderValue};
    use rstest::rstest;

    use super::*;

    const MASTER_KEY: &str = "sk-master";
    const JWT: &str = "header.claims.signature";

    fn no_env(_: &str) -> Option<String> {
        None
    }

    fn config_with(line: &str) -> Config {
        Config::from_yaml(&format!(
            "model_list: []\ngeneral_settings:\n  master_key: {MASTER_KEY}\n  {line}\n"
        ))
        .unwrap()
    }

    fn policy(general_line: &str) -> Policy {
        Policy::new(&config_with(general_line), &no_env)
    }

    fn bearer(token: &str) -> ApiKey {
        let headers = HeaderMap::from_iter([(
            axum::http::header::AUTHORIZATION,
            HeaderValue::from_str(&format!("Bearer {token}")).unwrap(),
        )]);
        ApiKey::from_headers(&headers)
    }

    #[rstest]
    fn a_key_on_a_native_route_runs_in_rust() {
        assert_eq!(
            policy("").dispatch("/v1/chat/completions", &bearer(MASTER_KEY), false),
            Some(NativePlan::Key)
        );
    }

    #[rstest]
    #[case::jwt(JWT, "enable_jwt_auth: true")]
    #[case::oauth2_introspection(MASTER_KEY, "enable_oauth2_auth: true")]
    #[case::oauth2_proxy(MASTER_KEY, "enable_oauth2_proxy_auth: true")]
    fn an_auth_path_rust_does_not_run_forwards(#[case] token: &str, #[case] setting: &str) {
        assert_eq!(
            policy(setting).dispatch("/v1/chat/completions", &bearer(token), false),
            None
        );
    }

    #[rstest]
    #[case::allowed_ips("allowed_ips: ['10.0.0.1']")]
    #[case::allowed_routes("allowed_routes: ['/v1/chat/completions']")]
    #[case::custom_auth("custom_auth: my_module.auth")]
    #[case::key_header_name("litellm_key_header_name: x-my-key")]
    #[case::pass_through_endpoints("pass_through_endpoints: []")]
    #[case::max_request_size("max_request_size_mb: 10")]
    #[case::database_url("database_url: postgresql://db")]
    #[case::public_routes("public_routes: ['/v1/models']")]
    fn a_python_only_setting_forwards_everything(#[case] setting: &str) {
        assert_eq!(
            policy(setting).dispatch("/v1/chat/completions", &bearer(MASTER_KEY), false),
            None
        );
    }

    #[rstest]
    fn a_request_with_a_body_forwards() {
        assert_eq!(
            policy("").dispatch("/v1/chat/completions", &bearer(MASTER_KEY), true),
            None
        );
    }

    #[rstest]
    #[case::unset("public_routes: null")]
    #[case::empty("public_routes: []")]
    fn no_public_routes_does_not_forward(#[case] setting: &str) {
        assert_eq!(
            policy(setting).dispatch("/v1/chat/completions", &bearer(MASTER_KEY), false),
            Some(NativePlan::Key)
        );
    }

    #[rstest]
    fn an_unset_python_only_setting_does_not_forward() {
        assert_eq!(
            policy("allowed_ips: null").dispatch(
                "/v1/chat/completions",
                &bearer(MASTER_KEY),
                false
            ),
            Some(NativePlan::Key)
        );
    }

    #[rstest]
    #[case::database_url("DATABASE_URL", "postgresql://db")]
    #[case::direct_url("DIRECT_URL", "postgresql://db")]
    #[case::database_host("DATABASE_HOST", "db")]
    #[case::server_root_path("SERVER_ROOT_PATH", "/api")]
    fn a_python_only_variable_forwards_everything(#[case] name: &str, #[case] value: &str) {
        let env = |key: &str| (key == name).then(|| value.to_owned());
        let policy = Policy::new(&config_with(""), &env);
        assert_eq!(
            policy.dispatch("/v1/chat/completions", &bearer(MASTER_KEY), false),
            None
        );
    }

    #[rstest]
    fn an_empty_server_root_path_does_not_forward() {
        let env = |key: &str| (key == "SERVER_ROOT_PATH").then(String::new);
        let policy = Policy::new(&config_with(""), &env);
        assert_eq!(
            policy.dispatch("/v1/chat/completions", &bearer(MASTER_KEY), false),
            Some(NativePlan::Key)
        );
    }

    #[rstest]
    fn no_master_key_forwards_everything() {
        let config = Config::from_yaml("model_list: []\n").unwrap();
        let policy = Policy::new(&config, &no_env);
        assert_eq!(
            policy.dispatch("/v1/chat/completions", &bearer("sk-any"), false),
            None
        );
    }

    #[rstest]
    fn an_unreadable_key_forwards() {
        assert_eq!(
            policy("").dispatch("/v1/chat/completions", &ApiKey::Unreadable, false),
            None
        );
    }
}
