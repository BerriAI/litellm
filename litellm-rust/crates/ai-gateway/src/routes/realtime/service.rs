//! Business logic: select a deployment with the (pure) core router, then call the
//! provider splice. The seam between `core::router` (selection only) and
//! `io` (the actual WebSocket I/O).
//!
//! On connect we try a pre-warmed upstream from the pool (handshake already paid,
//! `session.created` buffered) and relay it instantly. On a pool miss or dead warm
//! socket we fresh-dial exactly as before — the pool is never on the critical path
//! for correctness, only latency.

use std::time::Duration;

use crate::io::realtime_pool::{RealtimePool, upstream_key};
use futures_util::{Sink, Stream};
use litellm_core::CoreResult;
use litellm_core::error::CoreError;
use litellm_core::realtime::types::RealtimeEvent;
use litellm_core::router::Router;

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
    observe: impl FnMut(&RealtimeEvent) + Send,
    client_in: In,
    client_out: Out,
) -> CoreResult<()>
where
    In: Stream<Item = RealtimeEvent> + Unpin + Send,
    Out: Sink<RealtimeEvent> + Unpin + Send,
    <Out as Sink<RealtimeEvent>>::Error: std::fmt::Display,
{
    let deployment = router.get_available_deployment(model).ok_or_else(|| {
        CoreError::Routing(format!("no deployment available for model '{model}'"))
    })?;
    let params = &deployment.litellm_params;
    // Strip a leading `openai/` so the OpenAI-only realtime fn gets the bare model.
    let provider_model = params
        .model
        .strip_prefix("openai/")
        .unwrap_or(&params.model);

    // Warm path: take a pooled upstream (handshake already paid) and relay its
    // buffered session.created immediately. On miss/dead socket fall through.
    if let Some(key) = upstream_key(
        provider_model,
        params.api_key.as_deref(),
        params.api_base.as_deref(),
    ) {
        if let Some(handoff) = pool.take(&key) {
            return crate::io::realtime::realtime_warm(
                provider_model,
                handoff,
                idle_timeout,
                observe,
                client_in,
                client_out,
            )
            .await;
        }
    }

    // Cold path: fresh dial (the original behavior).
    crate::io::realtime::realtime(
        provider_model,
        params.api_key.as_deref(),
        params.api_base.as_deref(),
        idle_timeout,
        observe,
        client_in,
        client_out,
    )
    .await
}
