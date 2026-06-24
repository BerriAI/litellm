//! Gateway orchestration: pick a deployment with the (pure) core router, then
//! perform the provider call. This is the seam that bridges `core::router`
//! (selection only) and `providers` (the actual I/O), which is why it lives in
//! the gateway rather than in `core`.

use std::time::Duration;

use futures_util::{Sink, Stream};
use litellm_core::error::CoreError;
use litellm_core::realtime::types::RealtimeEvent;
use litellm_core::router::Router;
use litellm_core::CoreResult;

/// Select a deployment for `model` and splice the client stream to the provider.
pub async fn realtime<In, Out>(
    router: &Router,
    model: &str,
    timeout: Option<Duration>,
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
    let provider_model = params
        .model
        .strip_prefix("openai/")
        .unwrap_or(&params.model);

    litellm_providers::realtime::realtime(
        provider_model,
        params.api_key.as_deref(),
        params.api_base.as_deref(),
        timeout,
        client_in,
        client_out,
    )
    .await
}
