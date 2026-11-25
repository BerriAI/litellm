use std::{sync::Arc, time::Duration};

use litellm_gateway_auth::Authenticated;
use litellm_gateway_inference::{
    Error, RoutingCache,
    routing::{self, Prompt, RouteAttempt},
};
use litellm_router_core::{
    Attempt, AttemptError, AttemptFailure, ConfigError, Deployment, HostStep, Reached, RouterEvent,
    RouterHost, RoutingContext, SelectStage, Served, precall::PreCallFacts,
};
use tokio::sync::OnceCell;

use super::precall::{self, ContextWindows};
use crate::{
    Gateway,
    host::{NativeHost, NativeRows},
};

impl NativeHost {
    /// What a call served by `reached` is over: the first of its steps [`NativeRouterHost`] does
    /// not answer as the proxy does, and `None` when it answers them all. `admit` leaves every
    /// such call to Python.
    ///
    /// The pre-call facts are answered for a group whose every deployment has a window in
    /// [`ContextWindows`], since the proxy's own cost map may hold a window the native host reads none
    /// for.
    pub(crate) fn unanswered(&self, reached: &Reached) -> Option<Box<str>> {
        reached.steps.iter().find_map(|step| match step {
            HostStep::PreCallFacts => (!self.windows.cover(&reached.group))
                .then(|| "model_list.model_info.max_input_tokens".into()),
            HostStep::Pick(..) | HostStep::HealthStates | HostStep::FallbackParams(_) => {
                Some(format!("router_settings.{}", step.setting()).into())
            }
        })
    }
}

/// router-core's host for one inference call: each attempt runs the request's route, and the
/// rest is answered from the gateway's own state.
pub struct NativeRouterHost<'a, A> {
    pub(super) gateway: &'a Gateway,
    pub(super) identity: &'a Authenticated<NativeRows>,
    pub(super) route: &'a A,
    pub(super) prompt: Option<Prompt<'a>>,
    pub(super) windows: &'a ContextWindows,
    pub(super) prompt_tokens: OnceCell<Option<i64>>,
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

    async fn pre_call_facts(
        &self,
        _ctx: &RoutingContext<'_>,
        candidates: &[Arc<Deployment>],
    ) -> Result<PreCallFacts, litellm_router_core::Error> {
        Ok(precall::facts(self.windows, self.prompt, &self.prompt_tokens, candidates).await)
    }

    async fn allow_fallback(&self, target: &str) -> Result<bool, AttemptError> {
        Ok(routing::allows(self.gateway, self.identity, target))
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
