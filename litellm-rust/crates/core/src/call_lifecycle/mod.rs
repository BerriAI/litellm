use std::future::Future;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use crate::CoreResult;

pub mod types;

pub use types::{CallContext, CallOutcome, CallPhase, CallPhaseTiming, CallTiming};

pub trait CallSpec: 'static {
    const NAME: &'static str;

    type BeforeCall: Send + 'static;
    type BeforeSend: Send + 'static;
    type Response: Sync + 'static;
}

pub trait CallInterceptor<Call: CallSpec>: Send + Sync {
    fn before_call<'a>(
        &'a self,
        context: &'a CallContext,
        input: Call::BeforeCall,
    ) -> impl Future<Output = CoreResult<Call::BeforeCall>> + Send + 'a;

    fn before_send<'a>(
        &'a self,
        context: &'a CallContext,
        input: Call::BeforeSend,
    ) -> impl Future<Output = CoreResult<Call::BeforeSend>> + Send + 'a;

    fn complete<'a>(
        &'a self,
        context: &'a CallContext,
        outcome: CallOutcome<'a, Call::Response>,
        timing: &'a CallTiming,
    ) -> impl Future<Output = ()> + Send + 'a;
}

#[derive(Default)]
pub struct NoopCallInterceptor;

impl<Call: CallSpec> CallInterceptor<Call> for NoopCallInterceptor {
    async fn before_call<'a>(
        &'a self,
        _context: &'a CallContext,
        input: Call::BeforeCall,
    ) -> CoreResult<Call::BeforeCall> {
        Ok(input)
    }

    async fn before_send<'a>(
        &'a self,
        _context: &'a CallContext,
        input: Call::BeforeSend,
    ) -> CoreResult<Call::BeforeSend> {
        Ok(input)
    }

    async fn complete<'a>(
        &'a self,
        _context: &'a CallContext,
        _outcome: CallOutcome<'a, Call::Response>,
        _timing: &'a CallTiming,
    ) {
    }
}

pub trait CallObserver: Send + Sync {
    fn on_phase_start(&self, _call: &'static str, _context: &CallContext, _phase: CallPhase) {}

    fn on_phase_end(&self, _call: &'static str, _context: &CallContext, _timing: &CallPhaseTiming) {
    }
}

#[derive(Default)]
pub struct NoopCallObserver;

impl CallObserver for NoopCallObserver {}

pub struct CallRuntime<'a, Interceptor> {
    interceptor: &'a Interceptor,
    observer: &'a dyn CallObserver,
}

impl<'a, Interceptor> CallRuntime<'a, Interceptor> {
    pub fn new(interceptor: &'a Interceptor) -> Self {
        static OBSERVER: NoopCallObserver = NoopCallObserver;
        Self {
            interceptor,
            observer: &OBSERVER,
        }
    }

    pub fn with_observer(interceptor: &'a Interceptor, observer: &'a dyn CallObserver) -> Self {
        Self {
            interceptor,
            observer,
        }
    }

    pub async fn run<Call, Prepare, PrepareFuture, Provider, ProviderFuture>(
        &self,
        context: CallContext,
        input: Call::BeforeCall,
        prepare: Prepare,
        provider: Provider,
    ) -> CoreResult<Call::Response>
    where
        Call: CallSpec,
        Interceptor: CallInterceptor<Call>,
        Prepare: FnOnce(Call::BeforeCall) -> PrepareFuture,
        PrepareFuture: Future<Output = CoreResult<Call::BeforeSend>>,
        Provider: FnOnce(Call::BeforeSend) -> ProviderFuture,
        ProviderFuture: Future<Output = CoreResult<Call::Response>>,
    {
        let call_start = epoch_seconds();
        let mut phases = Vec::new();

        let before_call = self.start_phase::<Call>(&context, CallPhase::BeforeCall);
        let input = match self.interceptor.before_call(&context, input).await {
            Ok(input) => {
                phases.push(self.finish_phase::<Call>(&context, before_call));
                input
            }
            Err(error) => {
                phases.push(self.finish_phase::<Call>(&context, before_call));
                let result = Err(error);
                self.complete::<Call>(&context, &result, call_start, &mut phases)
                    .await;
                return result;
            }
        };

        let prepare_phase = self.start_phase::<Call>(&context, CallPhase::Prepare);
        let provider_input = match prepare(input).await {
            Ok(input) => {
                phases.push(self.finish_phase::<Call>(&context, prepare_phase));
                input
            }
            Err(error) => {
                phases.push(self.finish_phase::<Call>(&context, prepare_phase));
                let result = Err(error);
                self.complete::<Call>(&context, &result, call_start, &mut phases)
                    .await;
                return result;
            }
        };

        let before_send = self.start_phase::<Call>(&context, CallPhase::BeforeSend);
        let provider_input = match self.interceptor.before_send(&context, provider_input).await {
            Ok(input) => {
                phases.push(self.finish_phase::<Call>(&context, before_send));
                input
            }
            Err(error) => {
                phases.push(self.finish_phase::<Call>(&context, before_send));
                let result = Err(error);
                self.complete::<Call>(&context, &result, call_start, &mut phases)
                    .await;
                return result;
            }
        };

        let provider_phase = self.start_phase::<Call>(&context, CallPhase::Provider);
        let result = provider(provider_input).await;
        phases.push(self.finish_phase::<Call>(&context, provider_phase));
        self.complete::<Call>(&context, &result, call_start, &mut phases)
            .await;
        result
    }

    async fn complete<Call: CallSpec>(
        &self,
        context: &CallContext,
        result: &CoreResult<Call::Response>,
        call_start: f64,
        phases: &mut Vec<CallPhaseTiming>,
    ) where
        Interceptor: CallInterceptor<Call>,
    {
        let complete = self.start_phase::<Call>(context, CallPhase::Complete);
        let timing = CallTiming::new(call_start, epoch_seconds(), phases.clone());
        self.interceptor
            .complete(context, CallOutcome::from_result(result), &timing)
            .await;
        phases.push(self.finish_phase::<Call>(context, complete));
    }

    fn start_phase<Call: CallSpec>(&self, context: &CallContext, phase: CallPhase) -> PhaseStart {
        self.observer.on_phase_start(Call::NAME, context, phase);
        PhaseStart {
            phase,
            start_time: epoch_seconds(),
            started_at: Instant::now(),
        }
    }

    fn finish_phase<Call: CallSpec>(
        &self,
        context: &CallContext,
        phase_start: PhaseStart,
    ) -> CallPhaseTiming {
        let timing = CallPhaseTiming {
            phase: phase_start.phase,
            start_time: phase_start.start_time,
            end_time: epoch_seconds(),
            duration: phase_start.started_at.elapsed(),
        };
        self.observer.on_phase_end(Call::NAME, context, &timing);
        timing
    }
}

impl Default for CallRuntime<'static, NoopCallInterceptor> {
    fn default() -> Self {
        static INTERCEPTOR: NoopCallInterceptor = NoopCallInterceptor;
        Self::new(&INTERCEPTOR)
    }
}

struct PhaseStart {
    phase: CallPhase,
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
    use std::sync::Mutex;

    use crate::{CoreError, RequestError};

    use super::*;

    enum TestCall {}

    impl CallSpec for TestCall {
        const NAME: &'static str = "test";
        type BeforeCall = String;
        type BeforeSend = String;
        type Response = String;
    }

    #[derive(Default)]
    struct RecordingInterceptor {
        events: Mutex<Vec<&'static str>>,
    }

    impl CallInterceptor<TestCall> for RecordingInterceptor {
        async fn before_call<'a>(
            &'a self,
            _context: &'a CallContext,
            input: String,
        ) -> CoreResult<String> {
            self.events.lock().unwrap().push("before_call");
            Ok(format!("{input}:before_call"))
        }

        async fn before_send<'a>(
            &'a self,
            _context: &'a CallContext,
            input: String,
        ) -> CoreResult<String> {
            self.events.lock().unwrap().push("before_send");
            Ok(format!("{input}:before_send"))
        }

        async fn complete<'a>(
            &'a self,
            _context: &'a CallContext,
            outcome: CallOutcome<'a, String>,
            timing: &'a CallTiming,
        ) {
            assert!(!timing.phases.is_empty());
            self.events.lock().unwrap().push(match outcome {
                CallOutcome::Success(_) => "success",
                CallOutcome::Failure(_) => "failure",
            });
        }
    }

    #[tokio::test]
    async fn runtime_runs_typed_phases_in_order() {
        let interceptor = RecordingInterceptor::default();
        let result = CallRuntime::new(&interceptor)
            .run::<TestCall, _, _, _, _>(
                CallContext::new("model", "provider", "call-1"),
                "request".to_string(),
                |input| async move { Ok(format!("{input}:prepare")) },
                |input| async move {
                    assert_eq!(input, "request:before_call:prepare:before_send");
                    Ok("response".to_string())
                },
            )
            .await
            .unwrap();

        assert_eq!(result, "response");
        assert_eq!(
            interceptor.events.lock().unwrap().as_slice(),
            ["before_call", "before_send", "success"]
        );
    }

    #[tokio::test]
    async fn runtime_completes_once_when_preparation_fails() {
        let interceptor = RecordingInterceptor::default();
        let result = CallRuntime::new(&interceptor)
            .run::<TestCall, _, _, _, _>(
                CallContext::new("model", "provider", "call-1"),
                "request".to_string(),
                |_input| async { Err(CoreError::invalid_request("invalid".to_string())) },
                |_input| async { Ok("unreachable".to_string()) },
            )
            .await;

        assert!(matches!(
            result,
            Err(CoreError::Request(RequestError::InvalidRequest(_)))
        ));
        assert_eq!(
            interceptor.events.lock().unwrap().as_slice(),
            ["before_call", "failure"]
        );
    }
}
