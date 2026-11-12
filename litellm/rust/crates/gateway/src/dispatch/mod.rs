mod config;

use std::sync::Arc;

use axum::{
    Router,
    body::HttpBody,
    extract::{Request, State},
    middleware::{self, Next},
    response::Response,
};
use litellm_config::Config;
use litellm_gateway_auth::Auth;
use litellm_gateway_forward::Forward;
use litellm_gateway_inference::Gateway;

#[derive(Clone)]
struct Gate {
    auth: Auth,
    forward: Forward,
}

/// Serves a request natively only when it has no body and the gateway authenticates its caller as
/// the master key. Every other request goes to Python, which answers it as it would without the
/// gateway. This only authenticates. What the caller may do is still each handler's check.
async fn gate(State(gate): State<Gate>, request: Request, next: Next) -> Response {
    let (mut parts, body) = request.into_parts();
    // TODO: no body is on the allowlist until the gateway runs `is_request_body_safe` and knows
    // which body fields mean something to Python alone
    let identity = match body.is_end_stream() {
        true => gate.auth.authenticate_parts(&parts).await,
        false => Err(litellm_gateway_auth::Error::Unsupported),
    };
    match identity {
        Ok(identity) => {
            parts.extensions.insert(identity);
            next.run(Request::from_parts(parts, body)).await
        }
        Err(error) => {
            if !error.is_denied() {
                tracing::debug!(
                    error = %litellm_error::Report::new(&error),
                    "gateway authentication failed, so python authenticates the request"
                );
            }
            gate.forward.relay(Request::from_parts(parts, body)).await
        }
    }
}

/// The inference routes in front of Python. The gateway serves a request only when the
/// configuration, the route and the request are all ones it is known to handle the way Python
/// does, and Python serves the rest through `forward`.
pub fn dispatching_router(
    inference: Arc<Gateway>,
    config: &Config,
    env: &dyn Fn(&str) -> Option<String>,
    auth: Auth,
    forward: Forward,
) -> Router {
    if let Err(unsupported) = config::supported(config, env) {
        tracing::info!(
            reason = %unsupported,
            "the configuration is off the gateway's allowlist, so python serves every request"
        );
        return Router::new().fallback_service(forward);
    }
    let gate_state = Gate {
        auth,
        forward: forward.clone(),
    };
    let wrong_method = forward.clone();
    litellm_gateway_inference::router(inference)
        .layer(middleware::from_fn(crate::log_request))
        .route_layer(middleware::from_fn_with_state(gate_state, gate))
        .method_not_allowed_fallback(move |request: Request| wrong_method.relay(request))
        .fallback_service(forward)
}
