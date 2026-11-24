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
pub use router::{NativeRouterHost, answers};

use std::sync::Arc;

use litellm_gateway_auth::Authenticated;
use litellm_gateway_auth_core::{
    Caller,
    authz::{Layer, ModelList},
    identity::key::TeamGrant,
};
use litellm_gateway_inference::routing::{Host, RouteAttempt};

use crate::{Gateway, callbacks, callbacks::Callbacks};

/// The model lists the native host read for `identity`: a key's own and its team's, and none for
/// the master key. A key that names an owner, a project or anything else that carries a list was
/// declined when it was authenticated. So was a key whose team has no row, and the copy of the
/// team the core leaves such a key is read here as a list that reaches nothing.
pub(crate) fn layers(identity: &Authenticated<NativeRows>) -> impl Iterator<Item = Layer<'_>> {
    let Caller::Key(context) = &**identity else {
        return None.into_iter().chain(None);
    };
    let team = match &context.team {
        TeamGrant::NoTeam => None,
        TeamGrant::Loaded(team) => Some(Layer::Team(ModelList::read(team.models()))),
        TeamGrant::KeyCopy(_) => Some(Layer::Team(ModelList::Nothing)),
    };
    let key = Layer::Key(ModelList::read(context.key.models()));
    Some(key).into_iter().chain(team)
}

/// Every entry of the model lists the native host read for `identity`, as their rows hold them:
/// the key's and then its team's. The router narrows a caller that reaches a model only through
/// an access group by these, as Python's does by a key's `models` plus its `team_models`,
/// sentinels included.
pub(crate) fn allowed_models(identity: &Authenticated<NativeRows>) -> Vec<String> {
    let Caller::Key(context) = &**identity else {
        return Vec::new();
    };
    let team = match &context.team {
        TeamGrant::Loaded(team) => team.models(),
        TeamGrant::NoTeam | TeamGrant::KeyCopy(_) => &[],
    };
    [context.key.models(), team].concat()
}

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

    fn layers<'a>(
        &self,
        identity: &'a Authenticated<NativeRows>,
    ) -> impl Iterator<Item = Layer<'a>> {
        layers(identity)
    }

    fn allowed_models(&self, identity: &Authenticated<NativeRows>) -> Vec<String> {
        allowed_models(identity)
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
        }
    }
}
