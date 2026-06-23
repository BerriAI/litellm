//! Gateway orchestration: pick a deployment with the (pure) core router, then
//! perform the provider call. This is the seam that bridges `core::router`
//! (selection only) and `providers` (the actual I/O), which is why it lives in
//! the gateway rather than in `core`.

use std::time::Duration;

use litellm_core::error::CoreError;
use litellm_core::realtime::types::RealtimeEvent;
use litellm_core::router::Router;
use litellm_core::CoreResult;

/// Select a deployment for `model` and invoke the realtime provider call.
pub async fn realtime(
    router: &Router,
    model: &str,
    input_events: Vec<RealtimeEvent>,
    timeout: Option<Duration>,
) -> CoreResult<Vec<RealtimeEvent>> {
    let deployment = router.get_available_deployment(model).ok_or_else(|| {
        CoreError::Routing(format!("no deployment available for model '{model}'"))
    })?;
    let params = &deployment.litellm_params;
    // Strip a leading `openai/` so the OpenAI-only realtime fn gets the bare model.
    let provider_model = params
        .model
        .strip_prefix("openai/")
        .unwrap_or(&params.model);

    litellm_providers::realtime::realtime(
        provider_model,
        input_events,
        params.api_key.as_deref(),
        params.api_base.as_deref(),
        timeout,
    )
    .await
}
