use std::sync::Arc;

use axum::{
    extract::{Request, State},
    middleware::Next,
    response::Response,
};
use futures_util::future::BoxFuture;
use litellm_gateway_auth::AuthenticatedRequest;
use litellm_host::interceptors::DynInterceptors;
use litellm_inference::RouteError;
use litellm_router::RouterHooks;
use serde_json::{Map, Value};

use crate::{Error, Gateway};

pub trait GatewayHooks: Send + Sync {
    fn pre_call<'a>(
        &'a self,
        _identity: &'a AuthenticatedRequest,
        body: Map<String, Value>,
    ) -> BoxFuture<'a, Result<Map<String, Value>, Error>> {
        Box::pin(async move { Ok(body) })
    }

    fn post_call<'a>(
        &'a self,
        _identity: &'a AuthenticatedRequest,
        response: Response,
    ) -> BoxFuture<'a, Response> {
        Box::pin(async move { response })
    }
}

impl GatewayHooks for () {}

#[derive(Clone)]
pub struct Hooks {
    pub gateway: Arc<dyn GatewayHooks>,
    pub router: Arc<dyn RouterHooks>,
    pub inference: Arc<dyn DynInterceptors<RouteError>>,
}

impl Default for Hooks {
    fn default() -> Self {
        Self {
            gateway: Arc::new(()),
            router: Arc::new(()),
            inference: Arc::new(()),
        }
    }
}

pub(crate) async fn post_call(
    State(gateway): State<Arc<Gateway>>,
    identity: AuthenticatedRequest,
    request: Request,
    next: Next,
) -> Response {
    let response = next.run(request).await;
    gateway.hooks.gateway.post_call(&identity, response).await
}
