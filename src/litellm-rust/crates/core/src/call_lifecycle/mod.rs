use std::future::Future;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use crate::{CoreError, CoreResult};

pub mod types;

pub use types::{
    CallLifecycleContext, CallLifecyclePhase, CallLifecyclePhaseTiming, CallLifecycleRequest,
    CallLifecycleTiming,
};

pub trait CallLifecycleHooks<InitialReq, ProviderReq, Resp>: Send + Sync {
    type PreCallFuture<'a>: Future<Output = CoreResult<InitialReq>> + Send + 'a
    where
        Self: 'a,
        InitialReq: 'a,
        ProviderReq: 'a,
        Resp: 'a;

    type DuringCallFuture<'a>: Future<Output = CoreResult<ProviderReq>> + Send + 'a
    where
        Self: 'a,
        InitialReq: 'a,
        ProviderReq: 'a,
        Resp: 'a;

    type SuccessFuture<'a>: Future<Output = ()> + Send + 'a
    where
        Self: 'a,
        Resp: 'a;

    type FailureFuture<'a>: Future<Output = ()> + Send + 'a
    where
        Self: 'a;

    fn async_pre_call_hook<'a>(
        &'a self,
        context: &'a CallLifecycleContext,
        request: InitialReq,
    ) -> Self::PreCallFuture<'a>;

    fn async_during_call_hook<'a>(
        &'a self,
        context: &'a CallLifecycleContext,
        request: InitialReq,
    ) -> Self::DuringCallFuture<'a>;

    fn async_log_success_event<'a>(
        &'a self,
        context: &'a CallLifecycleContext,
        response: &'a Resp,
        timing: &'a CallLifecycleTiming,
    ) -> Self::SuccessFuture<'a>;

    fn async_log_failure_event<'a>(
        &'a self,
        context: &'a CallLifecycleContext,
        error: &'a CoreError,
        timing: &'a CallLifecycleTiming,
    ) -> Self::FailureFuture<'a>;
}

pub trait CallLifecycleObserver: Send + Sync {
    fn on_phase_start(&self, _context: &CallLifecycleContext, _phase: CallLifecyclePhase) {}

    fn on_phase_end(&self, _context: &CallLifecycleContext, _timing: &CallLifecyclePhaseTiming) {}
}

#[derive(Default)]
pub struct NoopCallLifecycleObserver;

impl CallLifecycleObserver for NoopCallLifecycleObserver {}

pub struct CallLifecycle<'a> {
    observer: &'a dyn CallLifecycleObserver,
}

impl<'a> CallLifecycle<'a> {
    pub fn new(observer: &'a dyn CallLifecycleObserver) -> Self {
        Self { observer }
    }

    pub async fn run_request<InitialReq, ProviderReq, Resp, Hooks, ProviderCall, ProviderFuture>(
        &self,
        request: InitialReq,
        hooks: &Hooks,
        provider_call: ProviderCall,
    ) -> CoreResult<Resp>
    where
        InitialReq: CallLifecycleRequest,
        Hooks: CallLifecycleHooks<InitialReq, ProviderReq, Resp>,
        ProviderCall: FnOnce(ProviderReq) -> ProviderFuture,
        ProviderFuture: Future<Output = CoreResult<Resp>>,
    {
        let context = request.lifecycle_context();
        self.run(context, request, hooks, provider_call).await
    }

    pub async fn run<InitialReq, ProviderReq, Resp, Hooks, ProviderCall, ProviderFuture>(
        &self,
        context: CallLifecycleContext,
        request: InitialReq,
        hooks: &Hooks,
        provider_call: ProviderCall,
    ) -> CoreResult<Resp>
    where
        Hooks: CallLifecycleHooks<InitialReq, ProviderReq, Resp>,
        ProviderCall: FnOnce(ProviderReq) -> ProviderFuture,
        ProviderFuture: Future<Output = CoreResult<Resp>>,
    {
        let call_start = epoch_seconds();
        let mut phases = Vec::new();

        let pre_call = self.start_phase(&context, CallLifecyclePhase::PreCall);
        let request = match hooks.async_pre_call_hook(&context, request).await {
            Ok(request) => {
                phases.push(self.finish_phase(&context, pre_call));
                request
            }
            Err(error) => {
                phases.push(self.finish_phase(&context, pre_call));
                self.log_failure(&context, hooks, &error, call_start, &mut phases)
                    .await;
                return Err(error);
            }
        };

        let during_call = self.start_phase(&context, CallLifecyclePhase::DuringCall);
        let provider_request = match hooks.async_during_call_hook(&context, request).await {
            Ok(request) => {
                phases.push(self.finish_phase(&context, during_call));
                request
            }
            Err(error) => {
                phases.push(self.finish_phase(&context, during_call));
                self.log_failure(&context, hooks, &error, call_start, &mut phases)
                    .await;
                return Err(error);
            }
        };

        let provider_phase = self.start_phase(&context, CallLifecyclePhase::ProviderCall);
        let result = provider_call(provider_request).await;
        phases.push(self.finish_phase(&context, provider_phase));

        match &result {
            Ok(response) => {
                let success_phase = self.start_phase(&context, CallLifecyclePhase::SuccessCallback);
                let timing = CallLifecycleTiming::new(call_start, epoch_seconds(), phases.clone());
                hooks
                    .async_log_success_event(&context, response, &timing)
                    .await;
                phases.push(self.finish_phase(&context, success_phase));
            }
            Err(error) => {
                self.log_failure(&context, hooks, error, call_start, &mut phases)
                    .await;
            }
        }

        result
    }

    async fn log_failure<InitialReq, ProviderReq, Resp, Hooks>(
        &self,
        context: &CallLifecycleContext,
        hooks: &Hooks,
        error: &CoreError,
        call_start: f64,
        phases: &mut Vec<CallLifecyclePhaseTiming>,
    ) where
        Hooks: CallLifecycleHooks<InitialReq, ProviderReq, Resp>,
    {
        let failure_phase = self.start_phase(context, CallLifecyclePhase::FailureCallback);
        let timing = CallLifecycleTiming::new(call_start, epoch_seconds(), phases.clone());
        hooks.async_log_failure_event(context, error, &timing).await;
        phases.push(self.finish_phase(context, failure_phase));
    }

    fn start_phase(&self, context: &CallLifecycleContext, phase: CallLifecyclePhase) -> PhaseStart {
        self.observer.on_phase_start(context, phase);
        PhaseStart {
            phase,
            start_time: epoch_seconds(),
            started_at: Instant::now(),
        }
    }

    fn finish_phase(
        &self,
        context: &CallLifecycleContext,
        phase_start: PhaseStart,
    ) -> CallLifecyclePhaseTiming {
        let timing = CallLifecyclePhaseTiming {
            phase: phase_start.phase,
            start_time: phase_start.start_time,
            end_time: epoch_seconds(),
            duration: phase_start.started_at.elapsed(),
        };
        self.observer.on_phase_end(context, &timing);
        timing
    }
}

impl Default for CallLifecycle<'static> {
    fn default() -> Self {
        static OBSERVER: NoopCallLifecycleObserver = NoopCallLifecycleObserver;
        Self::new(&OBSERVER)
    }
}

struct PhaseStart {
    phase: CallLifecyclePhase,
    start_time: f64,
    started_at: Instant,
}

fn epoch_seconds() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs_f64())
        .unwrap_or(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::pin::Pin;
    use std::sync::Mutex;

    type BoxFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

    #[derive(Default)]
    struct RecordingHooks {
        events: Mutex<Vec<&'static str>>,
    }

    struct RecordingRequest(String);

    impl CallLifecycleRequest for RecordingRequest {
        fn lifecycle_context(&self) -> CallLifecycleContext {
            CallLifecycleContext::new("ocr", "mistral-ocr-latest", "mistral", "call_1")
        }
    }

    impl RecordingHooks {
        fn events(&self) -> Vec<&'static str> {
            self.events.lock().unwrap().clone()
        }
    }

    impl CallLifecycleHooks<String, String, String> for RecordingHooks {
        type PreCallFuture<'a> = BoxFuture<'a, CoreResult<String>>;
        type DuringCallFuture<'a> = BoxFuture<'a, CoreResult<String>>;
        type SuccessFuture<'a> = BoxFuture<'a, ()>;
        type FailureFuture<'a> = BoxFuture<'a, ()>;

        fn async_pre_call_hook<'a>(
            &'a self,
            _context: &'a CallLifecycleContext,
            request: String,
        ) -> Self::PreCallFuture<'a> {
            Box::pin(async move {
                self.events.lock().unwrap().push("pre_call");
                Ok(format!("{request}:pre"))
            })
        }

        fn async_during_call_hook<'a>(
            &'a self,
            _context: &'a CallLifecycleContext,
            request: String,
        ) -> Self::DuringCallFuture<'a> {
            Box::pin(async move {
                self.events.lock().unwrap().push("during_call");
                Ok(format!("{request}:during"))
            })
        }

        fn async_log_success_event<'a>(
            &'a self,
            _context: &'a CallLifecycleContext,
            _response: &'a String,
            timing: &'a CallLifecycleTiming,
        ) -> Self::SuccessFuture<'a> {
            Box::pin(async move {
                assert!(timing.end_time >= timing.start_time);
                assert_eq!(timing.phases.len(), 3);
                self.events.lock().unwrap().push("success");
            })
        }

        fn async_log_failure_event<'a>(
            &'a self,
            _context: &'a CallLifecycleContext,
            _error: &'a CoreError,
            _timing: &'a CallLifecycleTiming,
        ) -> Self::FailureFuture<'a> {
            Box::pin(async move {
                self.events.lock().unwrap().push("failure");
            })
        }
    }

    impl CallLifecycleHooks<RecordingRequest, String, String> for RecordingHooks {
        type PreCallFuture<'a> = BoxFuture<'a, CoreResult<RecordingRequest>>;
        type DuringCallFuture<'a> = BoxFuture<'a, CoreResult<String>>;
        type SuccessFuture<'a> = BoxFuture<'a, ()>;
        type FailureFuture<'a> = BoxFuture<'a, ()>;

        fn async_pre_call_hook<'a>(
            &'a self,
            _context: &'a CallLifecycleContext,
            request: RecordingRequest,
        ) -> Self::PreCallFuture<'a> {
            Box::pin(async move {
                self.events.lock().unwrap().push("pre_call");
                Ok(RecordingRequest(format!("{}:pre", request.0)))
            })
        }

        fn async_during_call_hook<'a>(
            &'a self,
            _context: &'a CallLifecycleContext,
            request: RecordingRequest,
        ) -> Self::DuringCallFuture<'a> {
            Box::pin(async move {
                self.events.lock().unwrap().push("during_call");
                Ok(format!("{}:during", request.0))
            })
        }

        fn async_log_success_event<'a>(
            &'a self,
            _context: &'a CallLifecycleContext,
            _response: &'a String,
            _timing: &'a CallLifecycleTiming,
        ) -> Self::SuccessFuture<'a> {
            Box::pin(async move {
                self.events.lock().unwrap().push("success");
            })
        }

        fn async_log_failure_event<'a>(
            &'a self,
            _context: &'a CallLifecycleContext,
            _error: &'a CoreError,
            _timing: &'a CallLifecycleTiming,
        ) -> Self::FailureFuture<'a> {
            Box::pin(async move {
                self.events.lock().unwrap().push("failure");
            })
        }
    }

    #[tokio::test]
    async fn lifecycle_runs_hooks_around_provider_call() {
        let hooks = RecordingHooks::default();
        let response = CallLifecycle::default()
            .run(
                CallLifecycleContext::new("ocr", "mistral-ocr-latest", "mistral", "call_1"),
                "request".to_string(),
                &hooks,
                |request| async move {
                    assert_eq!(request, "request:pre:during");
                    Ok("response".to_string())
                },
            )
            .await
            .expect("call succeeds");

        assert_eq!(response, "response");
        assert_eq!(hooks.events(), vec!["pre_call", "during_call", "success"]);
    }

    #[tokio::test]
    async fn lifecycle_logs_failure_when_provider_fails() {
        let hooks = RecordingHooks::default();
        let error = CallLifecycle::default()
            .run(
                CallLifecycleContext::new("ocr", "mistral-ocr-latest", "mistral", "call_1"),
                "request".to_string(),
                &hooks,
                |_request| async move {
                    Err::<String, CoreError>(CoreError::Network("provider down".to_string()))
                },
            )
            .await
            .expect_err("call fails");

        assert_eq!(error, CoreError::Network("provider down".to_string()));
        assert_eq!(hooks.events(), vec!["pre_call", "during_call", "failure"]);
    }

    #[tokio::test]
    async fn lifecycle_can_run_any_request_with_embedded_context() {
        let hooks = RecordingHooks::default();
        let response = CallLifecycle::default()
            .run_request(
                RecordingRequest("request".to_string()),
                &hooks,
                |request| async move {
                    assert_eq!(request, "request:pre:during");
                    Ok("response".to_string())
                },
            )
            .await
            .expect("call succeeds");

        assert_eq!(response, "response");
        assert_eq!(hooks.events(), vec!["pre_call", "during_call", "success"]);
    }
}
