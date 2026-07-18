use std::time::Duration;

use futures_util::{Sink, Stream};
use litellm_core::responses::types::ResponsesWsEvent;
use litellm_core::{CoreError, CoreResult};

pub async fn run<In, Out>(
    router: &litellm_core::router::Router,
    model: &str,
    first_frame: Option<ResponsesWsEvent>,
    idle_timeout: Option<Duration>,
    observe: impl FnMut(&ResponsesWsEvent) + Send,
    client_in: In,
    client_out: Out,
) -> CoreResult<()>
where
    In: Stream<Item = ResponsesWsEvent> + Unpin + Send,
    Out: Sink<ResponsesWsEvent> + Unpin + Send,
    Out::Error: std::fmt::Display,
{
    let deployment = router.get_available_deployment(model).ok_or_else(|| {
        CoreError::Routing(format!("no deployment available for model '{model}'"))
    })?;
    let params = &deployment.litellm_params;
    let provider_model = params
        .model
        .strip_prefix("openai/")
        .unwrap_or(&params.model);
    if params.model.contains('/') && !params.model.starts_with("openai/") {
        return Err(CoreError::InvalidProvider(
            "Responses WebSocket route supports OpenAI deployments only".to_string(),
        ));
    }
    crate::io::responses_ws::responses_ws(
        provider_model,
        params.api_key.as_deref(),
        params.api_base.as_deref(),
        first_frame,
        idle_timeout,
        observe,
        client_in,
        client_out,
    )
    .await
}
