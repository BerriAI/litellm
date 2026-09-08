use std::sync::Arc;

use litellm_core::Error;
use litellm_core::constants::ANTHROPIC_MESSAGES_PROVIDER;
use litellm_core::integrations::custom_logger::{CustomLogger, CustomLoggerRunner, LogFuture};
use litellm_core::lifecycle::{
    ActionResult, CallLifecycleContext, Clock, RequestPolicy, StreamingCall, TerminalDispatcher,
    TerminalRecord,
};
use litellm_core::messages::lifecycle::{self, Options};
use litellm_core::messages::types::{AnthropicMessagesRequest, MessagesRequest};
use litellm_core::router::Router;
use serde_json::{Map, Value};
use std::future::{Ready, ready};

pub(crate) struct GatewayTerminalDispatcher {
    runner: CustomLoggerRunner,
}

impl GatewayTerminalDispatcher {
    pub(crate) fn new(loggers: Arc<Vec<Arc<dyn CustomLogger>>>) -> Self {
        Self {
            runner: CustomLoggerRunner::new(loggers.as_ref().clone()),
        }
    }
}

impl Clock for GatewayTerminalDispatcher {
    fn now(&self) -> f64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|duration| duration.as_secs_f64())
            .unwrap_or(0.0)
    }
}

impl RequestPolicy<MessagesRequest, MessagesRequest> for GatewayTerminalDispatcher {
    type PreCallFuture<'a> = Ready<ActionResult<MessagesRequest, Error>>;
    type DuringCallFuture<'a> = Ready<ActionResult<MessagesRequest, Error>>;

    fn async_pre_call_hook<'a>(
        &'a self,
        _: &'a CallLifecycleContext,
        request: MessagesRequest,
    ) -> Self::PreCallFuture<'a> {
        ready(ActionResult::Continue(request))
    }

    fn async_during_call_hook<'a>(
        &'a self,
        _: &'a CallLifecycleContext,
        request: MessagesRequest,
    ) -> Self::DuringCallFuture<'a> {
        ready(ActionResult::Continue(request))
    }
}

impl TerminalDispatcher for GatewayTerminalDispatcher {
    fn dispatch<'a>(&'a self, terminal: &'a TerminalRecord) -> LogFuture<'a> {
        self.runner.dispatch(terminal)
    }
}

pub(crate) enum MessagesResponse {
    Json(Value),
    Stream(Box<StreamingCall>),
}

#[tracing::instrument(
    name = "messages_gateway_service",
    target = "litellm::function_trace",
    level = "trace",
    skip_all
)]
pub async fn run(
    router: &Arc<Router>,
    loggers: Arc<Vec<Arc<dyn CustomLogger>>>,
    body: AnthropicMessagesRequest,
    extra_headers: Option<Map<String, Value>>,
) -> Result<MessagesResponse, Error> {
    let model = body.model.trim();
    if model.is_empty() {
        return Err(Error::InvalidRequest(
            "messages body requires a model".to_string(),
        ));
    }
    let deployment = router
        .get_available_deployment(model)
        .ok_or_else(|| Error::Routing(format!("no deployment available for model '{model}'")))?;
    let provider_model = deployment.litellm_params.model.as_str();
    let upstream_model = provider_model
        .split_once('/')
        .map_or(provider_model, |(_, model)| model);
    let custom_llm_provider = if provider_model.contains('/') {
        None
    } else {
        Some(ANTHROPIC_MESSAGES_PROVIDER)
    };
    let stream = body.stream == Some(true);
    let body = serde_json::to_value(AnthropicMessagesRequest {
        model: upstream_model.to_string(),
        ..body
    })
    .map_err(|error| {
        Error::InvalidRequest(format!(
            "failed to serialize Anthropic messages request: {error}"
        ))
    })?;

    let request = MessagesRequest {
        model: provider_model.to_string(),
        body,
        api_key: deployment.litellm_params.api_key.clone(),
        api_base: deployment.litellm_params.api_base.clone(),
        custom_llm_provider: custom_llm_provider.map(str::to_string),
        extra_headers,
        timeout: None,
    };
    let services = Arc::new(GatewayTerminalDispatcher::new(loggers));
    let context = CallLifecycleContext::new(
        "messages",
        provider_model,
        custom_llm_provider.unwrap_or(ANTHROPIC_MESSAGES_PROVIDER),
        format!(
            "messages-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|duration| duration.as_nanos())
                .unwrap_or(0)
        ),
    );
    if stream {
        return lifecycle::messages_stream(services, request, Options::default(), context)
            .await
            .map(Box::new)
            .map(MessagesResponse::Stream);
    }

    let response = lifecycle::messages(&*services, request, Options::default(), context)
        .await
        .into_result()?;
    serde_json::to_value(response)
        .map(MessagesResponse::Json)
        .map_err(|err| {
            Error::InvalidResponse(format!("failed to serialize messages response: {err}"))
        })
}
