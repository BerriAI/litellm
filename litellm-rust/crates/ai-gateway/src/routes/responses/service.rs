use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use futures_util::{Sink, Stream};
use litellm_core::call_lifecycle::{
    CallLifecycle, CallLifecycleContext, CallLifecycleHooks, CallLifecycleTiming,
};
use litellm_core::responses::types::ResponsesWsEvent;
use litellm_core::{CoreError, CoreResult};

use crate::integrations::custom_logger::{CallbackTiming, CustomLogger, CustomLoggerRunner};
use crate::integrations::types::RequestMetadata;
use crate::responses::streaming::ResponsesWsStreaming;

type BoxLifecycleFuture<'a, T> = Pin<Box<dyn Future<Output = CoreResult<T>> + Send + 'a>>;

struct ResponsesWebSocketLifecycleHooks {
    collector: Arc<Mutex<ResponsesWsStreaming>>,
    loggers: Arc<Vec<Arc<dyn CustomLogger>>>,
}

impl CallLifecycleHooks<(), (), ()> for ResponsesWebSocketLifecycleHooks {
    type PreCallFuture<'a> = BoxLifecycleFuture<'a, ()>;
    type DuringCallFuture<'a> = BoxLifecycleFuture<'a, ()>;
    type SuccessFuture<'a> = Pin<Box<dyn Future<Output = ()> + Send + 'a>>;
    type FailureFuture<'a> = Pin<Box<dyn Future<Output = ()> + Send + 'a>>;

    fn async_pre_call_hook<'a>(
        &'a self,
        _context: &'a CallLifecycleContext,
        request: (),
    ) -> Self::PreCallFuture<'a> {
        Box::pin(async move { Ok(request) })
    }

    fn async_during_call_hook<'a>(
        &'a self,
        _context: &'a CallLifecycleContext,
        request: (),
    ) -> Self::DuringCallFuture<'a> {
        Box::pin(async move { Ok(request) })
    }

    fn async_log_success_event<'a>(
        &'a self,
        _context: &'a CallLifecycleContext,
        _response: &'a (),
        timing: &'a CallLifecycleTiming,
    ) -> Self::SuccessFuture<'a> {
        Box::pin(async move {
            let details = {
                let Ok(mut collector) = self.collector.lock() else {
                    return;
                };
                collector.success_details()
            };
            let (details, response) = details;
            let runner = CustomLoggerRunner::new(self.loggers.as_ref().clone());
            let _ = runner
                .async_log_success_event(
                    &details,
                    &response,
                    CallbackTiming::new(timing.start_time, timing.end_time),
                )
                .await;
        })
    }

    fn async_log_failure_event<'a>(
        &'a self,
        _context: &'a CallLifecycleContext,
        _error: &'a CoreError,
        timing: &'a CallLifecycleTiming,
    ) -> Self::FailureFuture<'a> {
        Box::pin(async move {
            let details = {
                let Ok(mut collector) = self.collector.lock() else {
                    return;
                };
                collector.failure_details()
            };
            let (details, response) = details;
            let runner = CustomLoggerRunner::new(self.loggers.as_ref().clone());
            let _ = runner
                .async_log_failure_event(
                    &details,
                    Some(&response),
                    CallbackTiming::new(timing.start_time, timing.end_time),
                )
                .await;
        })
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
    let collector = Arc::new(Mutex::new(ResponsesWsStreaming::new(
        call_id.clone(),
        model.to_string(),
        metadata,
    )));
    let observer_collector = Arc::clone(&collector);
    let lifecycle_hooks = ResponsesWebSocketLifecycleHooks { collector, loggers };
    let context = CallLifecycleContext::new("responses_websocket", model, "openai", call_id);
    CallLifecycle::default()
        .run(context, (), &lifecycle_hooks, |_| async move {
            crate::io::responses_ws::async_responses_websocket(
                provider_model,
                params.api_key.as_deref(),
                params.api_base.as_deref(),
                first_frame,
                idle_timeout,
                move |event| {
                    if let Ok(mut collector) = observer_collector.lock() {
                        collector.observe(event);
                    }
                },
                client_in,
                client_out,
            )
            .await
        })
        .await
}
