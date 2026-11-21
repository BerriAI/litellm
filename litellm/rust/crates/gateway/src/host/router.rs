use std::{sync::Arc, time::Duration};

use litellm_gateway_auth::Authenticated;
use litellm_gateway_inference::{
    Error, RoutingCache,
    routing::{self, RouteAttempt},
};
use litellm_router_core::HostStep;
use litellm_router_core::{
    Attempt, AttemptError, AttemptFailure, ConfigError, Deployment, RouterEvent, RouterHost,
    RoutingContext, SelectStage, Served, precall::PreCallFacts,
};
use tokio::sync::OnceCell;

use super::precall::{self, Windows};
use crate::{Gateway, host::NativeRows};

/// What a call served by `group` is over when [`NativeRouterHost`] does not answer its `step` as
/// the proxy does, and `None` when it does. `admit` leaves every such call to Python.
///
/// The pre-call facts are answered for a group whose every deployment has a window in `windows`,
/// since the proxy's own cost map may hold a window the native host reads none for.
pub fn unanswered(step: &HostStep, group: &str, windows: &Windows) -> Option<Box<str>> {
    match step {
        HostStep::PreCallFacts => {
            (!windows.cover(group)).then(|| "model_list.model_info.max_input_tokens".into())
        }
        HostStep::Pick(..) | HostStep::HealthStates | HostStep::FallbackParams(_) => {
            Some(format!("router_settings.{}", step.setting()).into())
        }
    }
}

/// router-core's host for one inference call: each attempt runs the request's route, and the
/// rest is answered from the gateway's own state.
pub struct NativeRouterHost<'a, A> {
    pub(super) gateway: &'a Gateway,
    pub(super) identity: &'a Authenticated<NativeRows>,
    pub(super) route: &'a A,
    pub(super) windows: &'a Windows,
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
        let prompt = self.route.prompt();
        Ok(precall::facts(self.windows, prompt, &self.prompt_tokens, candidates).await)
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
