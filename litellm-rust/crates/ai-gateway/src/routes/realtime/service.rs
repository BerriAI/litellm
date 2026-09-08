//! Business logic: select a deployment with the (pure) core router, then call the
//! provider splice. The seam between `core::router` (selection only) and
//! `io` (the actual WebSocket I/O).
//!
//! On connect we try a pre-warmed upstream from the pool (handshake already paid,
//! `session.created` buffered) and relay it instantly. On a pool miss or dead warm
//! socket we fresh-dial exactly as before — the pool is never on the critical path
//! for correctness, only latency.

use std::sync::Arc;
use std::time::Duration;

use crate::io::realtime_pool::{RealtimePool, upstream_key};
use futures_util::{Sink, Stream};
use litellm_core::error::Error;
use litellm_core::integrations::custom_logger::{CustomLogger, CustomLoggerRunner, LogFuture};
use litellm_core::integrations::types::{RequestMetadata, StandardLoggingMetadata};
use litellm_core::lifecycle::{
    CallLifecycleContext, Clock, ExecutedCall, TerminalDispatcher, TerminalRecord,
};
use litellm_core::realtime::types::RealtimeEvent;
use litellm_core::realtime::{RealtimeRequest, realtime};
use litellm_core::router::Router;

struct GatewayRealtimeServices {
    runner: CustomLoggerRunner,
}

impl Clock for GatewayRealtimeServices {
    fn now(&self) -> f64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|duration| duration.as_secs_f64())
            .unwrap_or(0.0)
    }
}

impl TerminalDispatcher for GatewayRealtimeServices {
    fn dispatch<'a>(&'a self, terminal: &'a TerminalRecord) -> LogFuture<'a> {
        self.runner.dispatch(terminal)
    }
}

/// Select a deployment for `model` and splice the client stream to the provider.
///
/// `pool` supplies a pre-warmed upstream when one is available; otherwise we
/// fresh-dial. A disabled pool always misses, so this collapses to the original
/// fresh-dial behavior.
pub async fn run<In, Out>(
    router: &Router,
    pool: &RealtimePool,
    model: &str,
    idle_timeout: Option<Duration>,
    loggers: Arc<Vec<Arc<dyn CustomLogger>>>,
    call_id: String,
    metadata: RequestMetadata,
    client_in: In,
    client_out: Out,
) -> Result<ExecutedCall<(), Error>, Error>
where
    In: Stream<Item = RealtimeEvent> + Unpin + Send,
    Out: Sink<RealtimeEvent> + Unpin + Send,
    <Out as Sink<RealtimeEvent>>::Error: std::fmt::Display,
{
    let deployment = router
        .get_available_deployment(model)
        .ok_or_else(|| Error::Routing(format!("no deployment available for model '{model}'")))?;
    let params = &deployment.litellm_params;
    let connection = upstream_key(
        &params.model,
        params.api_key.as_deref(),
        params.api_base.as_deref(),
    );
    let warm = connection.as_ref().and_then(|key| pool.take(key));
    let context = CallLifecycleContext::new("realtime", model, "openai", call_id).with_metadata(
        StandardLoggingMetadata {
            user_api_key_hash: metadata.user_api_key_hash,
            user_api_key_user_id: metadata.user_api_key_user_id,
            user_api_key_team_id: metadata.user_api_key_team_id,
            ..Default::default()
        },
    );
    Ok(realtime(
        &GatewayRealtimeServices {
            runner: CustomLoggerRunner::new(loggers.as_ref().clone()),
        },
        RealtimeRequest {
            model: params.model.clone(),
            api_key: params.api_key.clone(),
            api_base: params.api_base.clone(),
            warm,
            idle_timeout,
        },
        context,
        client_in,
        client_out,
    )
    .await)
}
