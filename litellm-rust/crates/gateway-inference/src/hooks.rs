use futures_util::future::BoxFuture;
use litellm_gateway_auth::AuthenticatedRequest;
use litellm_host::{hooks::NativeHooks, native_chain::NativeChain};
use serde_json::{Map, Value};

use crate::{Error, Gateway};

/// One layer of the gateway's hook stack. Each method is the layer's part at one scope: the
/// request as the gateway receives it, and the call the route runs. A layer overrides only
/// the scopes it takes part in.
pub trait GatewayLayer: Send + Sync {
    fn pre_call<'a>(
        &'a self,
        _identity: &'a AuthenticatedRequest,
        body: Map<String, Value>,
    ) -> BoxFuture<'a, Result<Map<String, Value>, Error>> {
        Box::pin(async { Ok(body) })
    }

    fn call_hooks(&self) -> Option<Box<dyn NativeHooks>> {
        None
    }
}

pub(crate) async fn pre_call(
    gateway: &Gateway,
    identity: &AuthenticatedRequest,
    body: Map<String, Value>,
) -> Result<Map<String, Value>, Error> {
    let mut body = body;
    for layer in &gateway.layers {
        body = layer.pre_call(identity, body).await?;
    }
    Ok(body)
}

/// The hooks around one native call: the route's own first, then each layer's, in stack
/// order.
pub(crate) fn call_chain(gateway: &Gateway, route: impl NativeHooks + 'static) -> NativeChain {
    NativeChain::new(
        std::iter::once(Box::new(route) as Box<dyn NativeHooks>)
            .chain(gateway.layers.iter().filter_map(|layer| layer.call_hooks())),
    )
}
