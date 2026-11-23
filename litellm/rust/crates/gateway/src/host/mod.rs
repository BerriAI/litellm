//! The gateway's answers to router-core's hostcalls, so it routes with no Python behind it, the
//! native counterpart of python-bridge's `PythonRouterHost`. [`NativeBootHost`] answers the
//! deployment build at boot, and [`NativeHost`] one inference call: router-core's hostcalls
//! through [`NativeRouterHost`], and the callback hooks through the [`Callbacks`] it holds.

pub mod auth;
mod boot;
mod capacity;
mod router;

pub use auth::NativeRows;
pub use boot::NativeBootHost;
pub use capacity::NativeCapacityHost;
pub use router::NativeRouterHost;

use std::sync::Arc;

use litellm_gateway_auth::Authenticated;
use litellm_gateway_auth_core::{Caller, authz::Layer};
use litellm_gateway_inference::{
    CallType,
    routing::{Host, RouteAttempt},
};

use crate::{Gateway, callbacks, callbacks::Callbacks};

/// The model lists the native host read for `identity`: a key's own, and none for the master
/// key. A key that names a team, an owner or anything else that carries a list was declined when
/// it was authenticated.
pub(crate) fn layers(identity: &Authenticated<NativeRows>) -> impl Iterator<Item = Layer<'_>> {
    let key = match &**identity {
        Caller::MasterKey => None,
        Caller::Key(key) => Some(Layer::Key(key.key.models())),
    };
    key.into_iter()
}

/// The host the gateway's routes are served through.
#[derive(Clone, Default)]
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

    fn layers<'a>(
        &self,
        identity: &'a Authenticated<NativeRows>,
    ) -> impl Iterator<Item = Layer<'a>> {
        layers(identity)
    }

    fn hooks(&self, identity: &Authenticated<NativeRows>, call_type: CallType) -> callbacks::Call {
        self.callbacks.begin(identity, call_type)
    }

    fn router_host<'a, A: RouteAttempt>(
        &'a self,
        gateway: &'a Gateway,
        identity: &'a Authenticated<NativeRows>,
        route: &'a A,
    ) -> NativeRouterHost<'a, A> {
        NativeRouterHost {
            gateway,
            identity,
            route,
        }
    }
}
