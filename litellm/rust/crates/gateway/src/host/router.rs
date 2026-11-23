use std::{sync::Arc, time::Duration};

use litellm_gateway_inference::{
    Error, RoutingCache,
    routing::{self, RouteAttempt},
};
use litellm_router_core::HostStep;
use litellm_router_core::{
    Attempt, AttemptError, AttemptFailure, ConfigError, Deployment, RouterEvent, RouterHost,
    RoutingContext, SelectStage, Served,
};

use crate::Gateway;

/// Whether [`NativeRouterHost`] answers `step` as the proxy does. `admit` leaves to Python every
/// call that would make a step it does not.
pub fn answers(step: &HostStep) -> bool {
    match step {
        HostStep::Pick(..)
        | HostStep::PreCallFacts
        | HostStep::HealthStates
        | HostStep::FallbackParams(_) => false,
    }
}

/// router-core's host for one inference call: each attempt runs the request's route, and the
/// rest is answered from the gateway's own state.
pub struct NativeRouterHost<'a, A> {
    pub(super) gateway: &'a Gateway,
    pub(super) route: &'a A,
}

impl<A: RouteAttempt> RouterHost for NativeRouterHost<'_, A> {
    type Response = A::Outcome;
    type Failure = Error;
    type Cache = RoutingCache;

    async fn attempt(
        &self,
        attempt: &Attempt,
    ) -> Result<Served<A::Outcome>, AttemptFailure<Error>> {
        routing::attempt(self.gateway, self.route, attempt).await
    }

    async fn select(
        &self,
        _stage: &SelectStage,
        _ctx: &RoutingContext<'_>,
        _candidates: Vec<Arc<Deployment>>,
    ) -> Result<Vec<Arc<Deployment>>, litellm_router_core::Error> {
        Err(ConfigError::Unanswered {
            step: "a selection stage",
        }
        .into())
    }

    /// Every caller this host serves may be served every model, since a key that holds a model
    /// list is declined when it is authenticated.
    async fn allow_fallback(&self, _target: &str) -> Result<bool, AttemptError> {
        Ok(true)
    }

    fn cache(&self) -> &RoutingCache {
        self.gateway.routing()
    }

    async fn sleep(&self, duration: Duration) {
        tokio::time::sleep(duration).await;
    }

    fn observe(&self, event: &RouterEvent) {
        tracing::debug!(target: "litellm::router", ?event, "routing event");
    }
}
