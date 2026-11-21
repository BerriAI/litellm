//! The gateway's answers to router-core's hostcalls, so it routes with no Python behind it, the
//! native counterpart of python-bridge's `PythonRouterHost`. [`NativeBootHost`] answers the
//! deployment build at boot, and [`NativeHost`] one inference call: router-core's hostcalls
//! through [`NativeRouterHost`], and the callback hooks through the [`Callbacks`] it holds.

pub mod auth;
mod boot;
mod router;

pub use auth::NativeRows;
pub use boot::NativeBootHost;
pub use router::{NativeRouterHost, answers};

use std::sync::Arc;

use litellm_gateway_auth::Authenticated;
use litellm_gateway_inference::{
    CallType,
    routing::{Host, RouteAttempt},
};

use crate::{Gateway, callbacks, callbacks::Callbacks};

/// The host the gateway's routes are served through.
#[derive(Default)]
pub struct NativeHost {
    callbacks: Arc<Callbacks>,
}

impl NativeHost {
    pub fn new(callbacks: Arc<Callbacks>) -> Self {
        Self { callbacks }
    }
}

impl Host for NativeHost {
    type RouterHost<'a, A: RouteAttempt + 'a> = NativeRouterHost<'a, A>;
    type Hooks = callbacks::Call;
    type Rows = NativeRows;

    fn hooks(&self, _identity: &Authenticated<NativeRows>, call_type: CallType) -> callbacks::Call {
        self.callbacks.begin(call_type)
    }

    fn router_host<'a, A: RouteAttempt>(
        &'a self,
        gateway: &'a Gateway,
        _identity: &'a Authenticated<NativeRows>,
        route: &'a A,
    ) -> NativeRouterHost<'a, A> {
        NativeRouterHost { gateway, route }
    }
}
