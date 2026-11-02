use std::{sync::Arc, time::Duration};

use litellm_circuit_breaker::{State, Transition};
use litellm_host_python::{PythonContext, attach_blocking};
use litellm_redis_client::{CallEvent, FailureKind, Operation, Outcome, RedisEvents};
use pyo3::{prelude::*, types::PyModule};

tokio::task_local! {
    static CALL: RedisCall;
}

/// The Python caller of one store operation. Its context carries `with_service_target`, and its
/// `call_type` and `caller` are the labels `RedisCache` reports, so dashboards keep their series.
#[derive(Clone)]
pub(crate) struct RedisCall {
    pub(crate) context: PythonContext,
    pub(crate) call_type: &'static str,
    pub(crate) caller: Option<String>,
}

impl RedisCall {
    pub(crate) async fn scope<F: Future>(self, operation: F) -> F::Output {
        CALL.scope(self, operation).await
    }
}

/// Reports every native Redis call and breaker transition to `ServiceLogging`.
pub(crate) struct ServiceLoggingEvents {
    record_call: Arc<Py<PyAny>>,
    record_breaker: Arc<Py<PyAny>>,
    context: PythonContext,
}

impl ServiceLoggingEvents {
    pub(crate) fn new(py: Python<'_>) -> PyResult<Self> {
        const MODULE: &str = "litellm.rust_bridge.host.redis_events";
        Self::from_module(py, &py.import(MODULE)?)
    }

    fn from_module(py: Python<'_>, module: &Bound<'_, PyModule>) -> PyResult<Self> {
        Ok(Self {
            record_call: Arc::new(module.getattr("record_redis_call")?.unbind()),
            record_breaker: Arc::new(module.getattr("record_redis_breaker")?.unbind()),
            context: PythonContext::capture(py)?,
        })
    }

    fn report(&self, context: PythonContext, record: Arc<Py<PyAny>>, args: ReportArgs) {
        let Ok(runtime) = tokio::runtime::Handle::try_current() else {
            return;
        };
        runtime.spawn(async move {
            let reported = attach_blocking(context, move |py| match args {
                ReportArgs::Call(call_type, duration, failure, caller) => record
                    .bind(py)
                    .call1((call_type, duration.as_secs_f64(), failure, caller))
                    .map(drop),
                ReportArgs::Breaker(state) => record.bind(py).call1((state,)).map(drop),
            })
            .await
            .and_then(|reported| reported);
            if let Err(error) = reported {
                tracing::warn!(error = %error, "reporting a Redis event to ServiceLogging failed");
            }
        });
    }
}

enum ReportArgs {
    Call(&'static str, Duration, Option<&'static str>, Option<String>),
    Breaker(&'static str),
}

fn operation_call_type(operation: Operation) -> &'static str {
    match operation {
        Operation::Command(name) => name,
        Operation::Script => "eval",
        Operation::Pipeline { .. } => "pipeline",
    }
}

fn failure_label(outcome: Outcome) -> Option<&'static str> {
    match outcome {
        Outcome::Succeeded => None,
        Outcome::Failed(FailureKind::Timeout) => Some("timeout"),
        Outcome::Failed(FailureKind::Transport) => Some("transport"),
        Outcome::Failed(FailureKind::Server) => Some("server"),
        Outcome::Failed(FailureKind::Decode) => Some("decode"),
        Outcome::Failed(FailureKind::BreakerOpen) => Some("breaker_open"),
    }
}

pub(crate) fn state_label(state: &State) -> &'static str {
    match state {
        State::Closed { .. } => "closed",
        State::Open { .. } => "open",
        State::HalfOpen => "half_open",
    }
}

impl RedisEvents for ServiceLoggingEvents {
    fn call(&self, event: &CallEvent) {
        let caller = CALL.try_with(Clone::clone).ok();
        let (context, call_type, caller) = match caller {
            Some(call) => (call.context, call.call_type, call.caller),
            None => (
                self.context.clone(),
                operation_call_type(event.operation),
                None,
            ),
        };
        self.report(
            context,
            self.record_call.clone(),
            ReportArgs::Call(
                call_type,
                event.duration,
                failure_label(event.outcome),
                caller,
            ),
        );
    }

    fn breaker(&self, transition: &Transition<State>) {
        self.report(
            self.context.clone(),
            self.record_breaker.clone(),
            ReportArgs::Breaker(state_label(&transition.to)),
        );
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use litellm_host_python::PythonContext;
    use litellm_redis_client::{CallEvent, FailureKind, Operation, Outcome, RedisEvents};
    use pyo3::{prelude::*, types::PyModule};
    use rstest::rstest;

    use super::{RedisCall, ServiceLoggingEvents};

    const RECORDER: &str = "
import contextvars
target = contextvars.ContextVar('target', default=None)
seen = []
def record_redis_call(call_type, duration, failure, caller):
    seen.append((call_type, round(duration, 3), failure, caller, target.get()))
def record_redis_breaker(state):
    seen.append(('breaker', state))
";

    fn recorder(name: &str) -> (ServiceLoggingEvents, Py<PyModule>) {
        Python::initialize();
        Python::attach(|py| {
            let name = std::ffi::CString::new(format!("redis_events_recorder_{name}")).unwrap();
            let module =
                PyModule::from_code(py, &std::ffi::CString::new(RECORDER).unwrap(), &name, &name)
                    .unwrap();
            (
                ServiceLoggingEvents::from_module(py, &module).unwrap(),
                module.unbind(),
            )
        })
    }

    async fn seen(module: &Py<PyModule>, count: usize) -> String {
        let deadline = tokio::time::Instant::now() + Duration::from_secs(10);
        loop {
            let seen: (usize, String) = Python::attach(|py| {
                let seen = module.bind(py).getattr("seen").unwrap();
                (seen.len().unwrap(), seen.repr().unwrap().to_string())
            });
            if seen.0 >= count || tokio::time::Instant::now() > deadline {
                return seen.1;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    }

    #[rstest]
    #[tokio::test(flavor = "multi_thread")]
    async fn a_scoped_call_reports_the_store_labels_in_the_callers_context() {
        let (events, namespace) = recorder("scoped");
        let call = Python::attach(|py| {
            namespace
                .bind(py)
                .getattr("target")
                .unwrap()
                .call_method1("set", ("spend_counter",))
                .unwrap();
            RedisCall {
                context: PythonContext::capture(py).unwrap(),
                call_type: "async_increment",
                caller: Some("router.py:42".to_owned()),
            }
        });
        let event = CallEvent {
            operation: Operation::Command("INCRBYFLOAT"),
            duration: Duration::from_millis(250),
            outcome: Outcome::Failed(FailureKind::Timeout),
        };

        call.scope(async { events.call(&event) }).await;

        assert_eq!(
            seen(&namespace, 1).await,
            "[('async_increment', 0.25, 'timeout', 'router.py:42', 'spend_counter')]"
        );
    }

    #[rstest]
    #[case::opened(litellm_circuit_breaker::State::Open { since: std::time::Instant::now() }, "open")]
    #[case::probing(litellm_circuit_breaker::State::HalfOpen, "half_open")]
    #[tokio::test(flavor = "multi_thread")]
    async fn breaker_transitions_report_python_state_names(
        #[case] to: litellm_circuit_breaker::State,
        #[case] expected: &str,
    ) {
        let (events, namespace) = recorder(expected);

        events.breaker(&litellm_circuit_breaker::Transition {
            from: litellm_circuit_breaker::State::HalfOpen,
            to,
            generation: 1,
        });

        assert_eq!(
            seen(&namespace, 1).await,
            format!("[('breaker', '{expected}')]")
        );
    }

    #[rstest]
    #[tokio::test(flavor = "multi_thread")]
    async fn an_unscoped_call_falls_back_to_the_command_name() {
        let (events, namespace) = recorder("unscoped");

        events.call(&CallEvent {
            operation: Operation::Pipeline { ops: 3 },
            duration: Duration::from_millis(5),
            outcome: Outcome::Succeeded,
        });

        assert_eq!(
            seen(&namespace, 1).await,
            "[('pipeline', 0.005, None, None, None)]"
        );
    }
}
