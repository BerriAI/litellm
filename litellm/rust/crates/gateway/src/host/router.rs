use std::{sync::Arc, time::Duration};

use litellm_gateway_auth::AuthenticatedRequest;
use litellm_gateway_inference::{
    Error,
    routing::{self, RouteAttempt},
};
use litellm_router_core::{
    Attempt, AttemptError, AttemptFailure, ConfigError, Deployment, InMemoryRouterCache,
    RouterEvent, RouterHost, RoutingContext, SelectStage, Served,
};

use crate::Gateway;

/// router-core's host for one inference call: each attempt runs the request's route, and the
/// rest is answered from the gateway's own state.
pub struct NativeRouterHost<'a, A> {
    pub(super) gateway: &'a Gateway,
    pub(super) identity: &'a AuthenticatedRequest,
    pub(super) route: &'a A,
}

impl<A: RouteAttempt> RouterHost for NativeRouterHost<'_, A> {
    type Response = A::Outcome;
    type Failure = Error;
    type Cache = InMemoryRouterCache;

    async fn attempt(
        &self,
        attempt: &Attempt,
    ) -> Result<Served<A::Outcome>, AttemptFailure<Error>> {
        self.route
            .run(attempt)
            .await
            .map_err(|error| routing::attempt_failure(attempt, error))
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

    async fn allow_fallback(&self, target: &str) -> Result<bool, AttemptError> {
        Ok(routing::allows(self.gateway, self.identity, target).await)
    }

    fn cache(&self) -> &InMemoryRouterCache {
        self.gateway.routing()
    }

    async fn sleep(&self, duration: Duration) {
        tokio::time::sleep(duration).await;
    }

    fn observe(&self, event: &RouterEvent) {
        tracing::debug!(target: "litellm::router", ?event, "routing event");
    }
}
