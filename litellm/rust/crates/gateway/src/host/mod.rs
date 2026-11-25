//! The gateway's answers to router-core's hostcalls, so it routes with no Python behind it, the
//! native counterpart of python-bridge's `PythonRouterHost`. [`NativeBootHost`] answers the
//! deployment build at boot, and [`NativeHost`] one inference call: router-core's hostcalls
//! through [`NativeRouterHost`], and the callback hooks through the [`Callbacks`] it holds.

pub mod auth;
mod boot;
mod capacity;
mod hold;
mod model_info;
mod precall;
mod prompt_estimate;
mod router;
mod spend_log;

#[cfg(test)]
pub(crate) use auth::Key;
pub use auth::NativeRows;
pub use boot::NativeBootHost;
pub use capacity::NativeCapacityHost;
#[cfg(test)]
pub(crate) use hold::test_prices;
pub use hold::{Hold, HoldPrices, Unheld};
pub use precall::ContextWindows;
pub use router::NativeRouterHost;
pub use spend_log::{Rows, RustSpendLogHost};

use std::{collections::BTreeSet, sync::Arc};

use litellm_gateway_auth::Authenticated;
use litellm_gateway_auth_core::{
    Credential,
    authz::Layer,
    body::{self, Credentials},
};
use litellm_gateway_inference::routing::{Host, RouteAttempt};
use tokio::sync::OnceCell;

use crate::{Gateway, callbacks, callbacks::Callbacks};

/// The model lists the native host read for `identity`: a key's own, and none for the master
/// key. A key that names a team, an owner or anything else that carries a list was declined when
/// it was authenticated.
pub(crate) fn layers(identity: &Authenticated<NativeRows>) -> impl Iterator<Item = Layer<'_>> {
    let key = match &identity.credential {
        Credential::MasterKey => None,
        Credential::Key { key } => Some(Layer::Key(key.models())),
    };
    key.into_iter()
}

/// The host the gateway's routes are served through, over each deployment's context window as
/// read at boot.
#[derive(Clone, Default)]
pub struct NativeHost {
    callbacks: Arc<Callbacks>,
    windows: Arc<ContextWindows>,
    /// What a budget hold costs for each model group the gateway holds natively.
    prices: Arc<HoldPrices>,
}

impl NativeHost {
    pub fn new(callbacks: Arc<Callbacks>, windows: Arc<ContextWindows>) -> Self {
        Self {
            callbacks,
            windows,
            prices: Arc::default(),
        }
    }

    pub fn with_prices(self, prices: Arc<HoldPrices>) -> Self {
        Self { prices, ..self }
    }

    pub(crate) fn prices(&self) -> &HoldPrices {
        &self.prices
    }
}

impl Host for NativeHost {
    type RouterHost<'a, A: RouteAttempt + 'a> = NativeRouterHost<'a, A>;
    type Hooks = callbacks::Call;

    type Admitted = crate::Admitted;
    type Rows = NativeRows;

    fn layers<'a>(
        &self,
        identity: &'a Authenticated<NativeRows>,
    ) -> impl Iterator<Item = Layer<'a>> {
        layers(identity)
    }

    async fn credentials(
        &self,
        names: &BTreeSet<String>,
    ) -> Result<Credentials, litellm_gateway_auth_core::Error> {
        body::credentials(&auth::NativeAuthHost::credential_store(), names).await
    }

    fn hooks(
        &self,
        identity: &Authenticated<NativeRows>,
        admitted: Option<&crate::Admitted>,
    ) -> callbacks::Call {
        self.callbacks.begin(identity, admitted)
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
