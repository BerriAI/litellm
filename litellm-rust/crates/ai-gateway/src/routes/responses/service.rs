use std::sync::Arc;
use std::time::Duration;

use futures_util::{Sink, Stream};
use litellm_core::Error;
use litellm_core::integrations::custom_logger::{CustomLogger, CustomLoggerRunner, LogFuture};
use litellm_core::integrations::types::{RequestMetadata, StandardLoggingMetadata};
use litellm_core::lifecycle::{
    CallLifecycleContext, Clock, ExecutedCall, TerminalDispatcher, TerminalRecord,
};
use litellm_core::responses::types::ResponsesWsEvent;
use litellm_core::responses::websocket::{ResponsesWebSocketRequest, responses_websocket};

struct GatewayResponsesServices {
    runner: CustomLoggerRunner,
}

impl GatewayResponsesServices {
    fn new(loggers: Arc<Vec<Arc<dyn CustomLogger>>>) -> Self {
        Self {
            runner: CustomLoggerRunner::new(loggers.as_ref().clone()),
        }
    }
}

impl Clock for GatewayResponsesServices {
    fn now(&self) -> f64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|duration| duration.as_secs_f64())
            .unwrap_or(0.0)
    }
}

impl TerminalDispatcher for GatewayResponsesServices {
    fn dispatch<'a>(&'a self, terminal: &'a TerminalRecord) -> LogFuture<'a> {
        self.runner.dispatch(terminal)
    }
}

#[allow(clippy::too_many_arguments)]
pub async fn run<In, Out>(
    router: &litellm_core::router::Router,
    model: &str,
    first_frame: Option<ResponsesWsEvent>,
    idle_timeout: Option<Duration>,
    loggers: Arc<Vec<Arc<dyn CustomLogger>>>,
    call_id: String,
    metadata: RequestMetadata,
    client_in: In,
    client_out: Out,
) -> Result<ExecutedCall<(), Error>, Error>
where
    In: Stream<Item = Result<ResponsesWsEvent, Error>> + Unpin + Send,
    Out: Sink<ResponsesWsEvent> + Unpin + Send,
    Out::Error: std::fmt::Display,
{
    let deployment = router
        .get_available_deployment(model)
        .ok_or_else(|| Error::Routing(format!("no deployment available for model '{model}'")))?;
    let params = &deployment.litellm_params;
    let provider_model = params
        .model
        .strip_prefix("openai/")
        .unwrap_or(&params.model);
    if params.model.contains('/') && !params.model.starts_with("openai/") {
        return Err(Error::InvalidProvider(
            "Responses WebSocket route supports OpenAI deployments only".to_string(),
        ));
    }
    let context = CallLifecycleContext::new("responses_websocket", model, "openai", call_id)
        .with_metadata(StandardLoggingMetadata {
            user_api_key_hash: metadata.user_api_key_hash,
            user_api_key_user_id: metadata.user_api_key_user_id,
            user_api_key_team_id: metadata.user_api_key_team_id,
            ..Default::default()
        });
    responses_websocket(
        Arc::new(GatewayResponsesServices::new(loggers)),
        ResponsesWebSocketRequest {
            model: provider_model.to_string(),
            api_key: params.api_key.clone(),
            api_base: params.api_base.clone(),
            first_frame,
            idle_timeout,
        },
        context,
        client_in,
        client_out,
    )
    .await
}
