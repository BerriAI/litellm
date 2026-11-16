//! The gateway's answers to router-core's hostcalls, so it routes with no Python behind it, the
//! native counterpart of python-bridge's `PythonRouterHost`. [`NativeBootHost`] answers the
//! deployment build at boot, and [`NativeHost`] one inference call through [`NativeRouterHost`].

mod boot;
mod router;

pub use boot::NativeBootHost;
pub use router::NativeRouterHost;

use litellm_gateway_auth::AuthenticatedRequest;
use litellm_gateway_inference::routing::{Host, RouteAttempt};
use litellm_router_core::HostFacts;

use crate::Gateway;

/// What the native host serves: every route the gateway mounts, and none of the host steps a
/// setting may need.
pub const FACTS: HostFacts = HostFacts {
    capabilities: &[],
    supports: |_, _| true,
};

/// The host the gateway's routes are served through.
pub struct NativeHost;

impl Host for NativeHost {
    type RouterHost<'a, A: RouteAttempt + 'a> = NativeRouterHost<'a, A>;

    fn router_host<'a, A: RouteAttempt>(
        &'a self,
        gateway: &'a Gateway,
        identity: &'a AuthenticatedRequest,
        route: &'a A,
    ) -> NativeRouterHost<'a, A> {
        NativeRouterHost {
            gateway,
            identity,
            route,
        }
    }
}
