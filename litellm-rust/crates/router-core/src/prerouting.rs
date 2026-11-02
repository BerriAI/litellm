//! Pre-routing: before an async pick resolves its name, the host may route the request to another
//! model group (Python's `_core_async_pre_routing_hook`). The complexity, auto, adaptive and
//! quality routers registered from `auto_router/...` marker deployments, and the Claude Code
//! session binding, live host-side; the core asks once per pick, before resolution, and resolves
//! the name the host answers. The attempt still serves the requested name (its `model_group`):
//! only the deployments come from the routed group.
//!
//! A custom routing strategy (`register_plugin`, `set_custom_routing_strategy`, lar1) replaces the
//! whole pick (`_core_async_get_available_deployment` and `get_available_deployment`): the host
//! answers the same step with the deployment document the strategy returned, which may name a
//! deployment the registry does not hold, and the core serves that deployment as is.

use std::sync::Arc;

use crate::{Core, config::Op, error::Error, host::RouterRequest, registry::Deployment};

/// What the host's pre-routing step sees.
#[derive(Clone, Copy, Debug)]
pub struct PreRoute<'a> {
    pub op: &'a Op,
    pub request: &'a RouterRequest,
    /// The name this pick resolves: the requested name or the fallback target.
    pub name: &'a str,
    /// How many fallbacks deep the walk is (Python's `fallback_depth`): only a first attempt
    /// binds a Claude Code session.
    pub fallback_depth: u32,
    /// 0 for the first attempt in this group, then the retry number.
    pub retry: u32,
}

/// The host's answer to a pre-routing step.
#[derive(Clone, Debug)]
pub enum PreRouted {
    /// Resolve this model group instead of the requested name.
    Model(String),
    /// A custom strategy's pick: serve this deployment, without resolving or filtering.
    Deployment(Arc<Deployment>),
}

impl Core {
    /// What a pick of `ask.name` serves: the host's pre-routed group or custom pick, else the
    /// name itself. Only picks that ask for it pre-route (`request.pre_routes`).
    pub(crate) async fn pre_routed(&self, ask: PreRoute<'_>) -> Result<PreRouted, Error> {
        if !ask.request.pre_routes {
            return Ok(PreRouted::Model(ask.name.to_owned()));
        }
        Ok(self
            .host
            .pre_route(&ask)
            .await?
            .unwrap_or_else(|| PreRouted::Model(ask.name.to_owned())))
    }
}
