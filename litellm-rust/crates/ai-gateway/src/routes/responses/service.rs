use std::sync::Arc;
use std::time::Duration;

use futures_util::{Sink, Stream};
use litellm_core::call_lifecycle::{CallLifecycle, CallLifecycleContext};
use litellm_core::responses::instrumentation::{
    ResponsesWsCallbackPayload, ResponsesWsInstrumentation, ResponsesWsLogOutcome,
    ResponsesWsMetadata,
};
use litellm_core::responses::types::ResponsesWsEvent;
use litellm_core::{CoreError, CoreResult};

use crate::integrations::custom_logger::{
    CallbackTiming, CallbackValue, CustomLogger, CustomLoggerRunner, LoggingError, ModelCallDetails,
};
use crate::integrations::types::RequestMetadata;

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
    let instrumentation = Arc::new(ResponsesWsInstrumentation::new(
        call_id.clone(),
        model,
        ResponsesWsMetadata {
            user_api_key_hash: metadata.user_api_key_hash,
            user_api_key_user_id: metadata.user_api_key_user_id,
            user_api_key_team_id: metadata.user_api_key_team_id,
        },
    ));
    let observer_instrumentation = Arc::clone(&instrumentation);
    let context = CallLifecycleContext::new("responses_websocket", model, "openai", call_id);
    let result = CallLifecycle::default()
        .run(context, (), instrumentation.as_ref(), |_| async move {
            crate::io::responses_ws::async_responses_websocket(
                provider_model,
                params.api_key.as_deref(),
                params.api_base.as_deref(),
                first_frame,
                idle_timeout,
                move |event| {
                    observer_instrumentation.observe(event);
                },
                client_in,
                client_out,
            )
            .await
        })
        .await;
    if let Some(outcome) = instrumentation.take_outcome() {
        dispatch_outcome(loggers, outcome).await;
    }
    result
}

async fn dispatch_outcome(
    loggers: Arc<Vec<Arc<dyn CustomLogger>>>,
    outcome: ResponsesWsLogOutcome,
) {
    let runner = CustomLoggerRunner::new(loggers.as_ref().clone());
    match outcome {
        ResponsesWsLogOutcome::Success { payload, callback } => {
            let (details, response, start_time, end_time) = logging_values(payload, callback, None);
            let _ = runner
                .async_log_success_event(
                    &details,
                    &response,
                    CallbackTiming::new(start_time, end_time),
                )
                .await;
        }
        ResponsesWsLogOutcome::Failure {
            payload,
            callback,
            error_message,
            error_kind,
        } => {
            let error = LoggingError {
                message: error_message,
                kind: error_kind,
            };
            let (details, response, start_time, end_time) =
                logging_values(payload, callback, Some(error));
            let _ = runner
                .async_log_failure_event(
                    &details,
                    Some(&response),
                    CallbackTiming::new(start_time, end_time),
                )
                .await;
        }
    }
}

fn logging_values(
    payload: litellm_core::responses::instrumentation::ResponsesWsLogPayload,
    callback: ResponsesWsCallbackPayload,
    error: Option<LoggingError>,
) -> (ModelCallDetails, CallbackValue, f64, f64) {
    let start_time = payload.start_time;
    let end_time = payload.end_time;
    let callback = CallbackValue::new(callback.object, callback.value);
    let details = ModelCallDetails::from_standard_logging_payload(
        crate::integrations::types::StandardLoggingPayload {
            id: payload.id,
            litellm_call_id: payload.litellm_call_id,
            call_type: payload.call_type,
            model: payload.model,
            custom_llm_provider: payload.custom_llm_provider,
            response_cost: payload.response_cost,
            prompt_tokens: payload.usage.prompt_tokens,
            completion_tokens: payload.usage.completion_tokens,
            total_tokens: payload.usage.total_tokens,
            start_time: payload.start_time,
            end_time: payload.end_time,
            stream: payload.stream,
            metadata: crate::integrations::types::StandardLoggingMetadata {
                user_api_key_hash: payload.metadata.user_api_key_hash,
                user_api_key_user_id: payload.metadata.user_api_key_user_id,
                user_api_key_team_id: payload.metadata.user_api_key_team_id,
                ..Default::default()
            },
            messages: None,
        },
    );
    let details = match error {
        Some(error) => details.with_failure_error(error),
        None => details,
    };
    (details, callback, start_time, end_time)
}
