//! [`Host`]: what the embedding process provides. `PythonHost` serves it from the existing Python
//! code; `NativeHost` serves it in the Rust server.

use std::{collections::BTreeMap, sync::Arc, time::Duration};

use async_trait::async_trait;
use serde_json::Value;

use crate::{
    cache::RouterCache,
    config::{Op, RoutingStrategy},
    error::{AttemptError, Error, FailureKind, Payload},
    health::Health,
    registry::{Deployment, Registry},
    settings::Settings,
};

/// One call into the router.
#[derive(Clone, Debug, Default)]
pub struct RouterRequest {
    /// The requested name: a model group, alias, routing group, deployment id or wildcard match.
    pub model: String,
    /// The request params, for native executors and plugins that read them.
    pub body: Value,
    /// `user_api_key_team_id`.
    pub team_id: Option<String>,
    pub specific_deployment: bool,
    /// A per-request `num_retries`, which wins over the router's.
    pub num_retries: Option<u32>,
    pub disable_fallbacks: bool,
    /// `_router_weights`: model group to deployment id to weight.
    pub router_weights: Option<BTreeMap<String, BTreeMap<String, f64>>>,
    /// What the host needs to run the call itself, such as the Python kwargs.
    pub payload: Option<Payload>,
}

/// One attempt against one deployment.
#[derive(Clone, Debug)]
pub struct Attempt {
    pub op: Op,
    pub request: Arc<RouterRequest>,
    pub deployment: Arc<Deployment>,
    /// The name this attempt resolved: the requested name, or the fallback target.
    pub name: String,
    /// The model group being served, after fallbacks.
    pub model_group: String,
    /// The `order` level an order fallback targets in the same group.
    pub target_order: Option<i64>,
    /// Every deployment `name` resolved to for this caller, the picked one included.
    pub candidate_ids: Vec<String>,
    /// 0 for the first attempt in this group, then the retry number.
    pub retry: u32,
    pub fallback_depth: u32,
    /// How many fallbacks deep the walk is: 0 for the requested name.
    pub fallback_level: u32,
    pub timeout: Option<Duration>,
}

/// What a successful attempt returns.
#[derive(Clone, Debug)]
pub struct Output {
    pub value: Payload,
    /// Prompt plus completion tokens, counted against the deployment's per-minute usage.
    pub total_tokens: u64,
}

/// What a plugin or strategy sees: never the router itself.
pub struct RoutingContext<'a> {
    pub registry: &'a Registry,
    pub settings: &'a Settings,
    pub health: &'a Health,
    pub request: &'a RouterRequest,
    pub op: &'a Op,
    pub model_group: &'a str,
}

/// Which selection step the host is asked to run.
#[derive(Clone, Debug, PartialEq)]
pub enum SelectStage {
    /// Pre-call checks and sub-routers registered as plugins: return the deployments that remain.
    Plugins,
    /// A strategy not yet ported: return the candidates with the pick first.
    Strategy(RoutingStrategy),
}

/// Routing events, delivered through [`Host::observe`].
#[derive(Clone, Debug, PartialEq)]
pub enum RouterEvent {
    Selected {
        model_group: String,
        deployment_id: String,
    },
    Succeeded {
        deployment_id: String,
    },
    AttemptFailed {
        deployment_id: String,
        kind: FailureKind,
        status: Option<u16>,
    },
    CooldownSet {
        deployment_id: String,
        cooldown: Duration,
    },
    Retry {
        model_group: String,
        retry: u32,
        delay: Duration,
    },
    Fallback {
        from: String,
        to: String,
        /// The order level of `to` a same-group hop walks to.
        order: Option<i64>,
        depth: u32,
        /// The fallback level `to` is tried at: 1 for the request's own fallbacks.
        level: u32,
    },
    /// A group's retry loop ran every retry and the last one failed: Python stamps
    /// `num_retries` and `max_retries` on that error.
    RetriesExhausted {
        model_group: String,
        attempted: u32,
        max: u32,
    },
}

#[async_trait]
pub trait Host: Send + Sync {
    /// Run a non-native attempt.
    async fn invoke(&self, attempt: &Attempt) -> Result<Output, AttemptError>;

    /// Run a selection stage the core does not implement.
    async fn select(
        &self,
        stage: &SelectStage,
        ctx: &RoutingContext<'_>,
        candidates: Vec<Arc<Deployment>>,
    ) -> Result<Vec<Arc<Deployment>>, Error>;

    /// Access and budget checks before a cross-group fallback target.
    async fn allow_fallback(&self, request: &RouterRequest, target: &str) -> bool;

    async fn sleep(&self, duration: Duration);

    fn observe(&self, event: &RouterEvent);

    /// The shared store routing state lives in.
    fn cache(&self) -> Arc<dyn RouterCache>;

    /// Whether [`Self::invoke`] can run an attempt. A `NativeHost` cannot, so every served
    /// operation needs a native executor.
    fn can_invoke(&self) -> bool {
        true
    }

    /// Whether [`Self::select`] can run strategies and plugins.
    fn can_select(&self) -> bool {
        true
    }

    /// Whether [`Self::invoke`] records each attempt's outcome itself (cooldowns, allowed-fails
    /// counters, usage), as the Python router's own callbacks do. The core then leaves the
    /// health store to the host for host-run attempts, so no failure is counted twice.
    fn records_outcomes(&self) -> bool {
        false
    }
}
