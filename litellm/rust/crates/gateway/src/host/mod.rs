//! The gateway's answers to router-core's hostcalls, so it routes with no Python behind it, the
//! native counterpart of python-bridge's `PythonRouterHost`. [`NativeBootHost`] answers the
//! deployment build at boot, `NativeModelInfoHost` one deployment's model-info lookups from the
//! live model table, and [`NativeHost`] one inference call through [`NativeRouterHost`].

mod boot;
mod model_info;
mod precall;
mod router;

pub use boot::{NativeBootHost, environment_value};
use model_info::NativeModelInfoHost;
pub use router::NativeRouterHost;

use litellm_gateway_auth::AuthenticatedRequest;
use litellm_gateway_inference::routing::{Host, RouteAttempt};
use litellm_router_core::{Capability, HostFacts};

use crate::Gateway;

/// What the native host serves: every route the gateway mounts, and the pre-call facts from the
/// model table.
pub const FACTS: HostFacts = HostFacts {
    capabilities: &[Capability::PreCallFacts],
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
