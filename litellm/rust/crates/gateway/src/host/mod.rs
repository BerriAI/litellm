//! The gateway's answers to router-core's hostcalls, so it routes with no Python behind it, the
//! native counterpart of python-bridge's `PythonRouterHost`. [`NativeBootHost`] answers the
//! deployment build at boot, and [`NativeHost`] one inference call: router-core's hostcalls
//! through [`NativeRouterHost`], and the callback hooks through the [`Callbacks`] it holds.

pub mod auth;
mod boot;
mod capacity;
mod free;
mod model_info;
mod precall;
mod router;
mod spend_log;

#[cfg(test)]
pub(crate) use auth::Key;
pub use auth::NativeRows;
pub use boot::NativeBootHost;
pub use capacity::NativeCapacityHost;
pub use free::FreeModels;
pub use precall::ContextWindows;
pub use router::NativeRouterHost;
pub use spend_log::{Rows, RustSpendLogHost};

use std::{collections::BTreeSet, sync::Arc};

use litellm_gateway_auth::Authenticated;
use litellm_gateway_auth_core::{
    authz::Layer,
    body::{self, Credentials},
};
use litellm_gateway_inference::routing::{Host, RouteAttempt};
use tokio::sync::OnceCell;

use crate::{Gateway, callbacks, callbacks::Callbacks};

/// The host the gateway's routes are served through, over each deployment's context window as
/// read at boot.
#[derive(Clone, Default)]
pub struct NativeHost {
    callbacks: Arc<Callbacks>,
    windows: Arc<ContextWindows>,
}

impl NativeHost {
    pub fn new(callbacks: Arc<Callbacks>, windows: Arc<ContextWindows>) -> Self {
        Self { callbacks, windows }
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
        auth::layers(identity)
    }

    fn allowed_models(&self, identity: &Authenticated<NativeRows>) -> Vec<String> {
        auth::allowed_models(identity)
    }

    async fn credentials(
        &self,
        names: &BTreeSet<String>,
    ) -> Result<Credentials, litellm_gateway_auth_core::Error> {
        body::credentials(&auth::NativeAuthHost::credential_store(), names).await
    }

    fn hooks(&self, identity: &Authenticated<NativeRows>) -> callbacks::Call {
        self.callbacks.begin(identity)
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
            prompt: route.prompt(),
            windows: &self.windows,
            prompt_tokens: OnceCell::new(),
        }
    }
}
