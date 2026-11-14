use std::{
    sync::Arc,
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use litellm_circuit_breaker::{State, Transition};
use litellm_host_python::{PythonContext, attach_blocking};
use litellm_redis_client::{
    CallEvent, CallTag, FailureKind, Operation, Outcome, RedisEvents, current_tag, tagged,
};
use pyo3::{prelude::*, types::PyModule};
use tokio::runtime::Handle;

/// The asyncio loop a Python caller ran on. `RedisCache` scheduled its service events there with
/// `asyncio.create_task`, so the native events go there too instead of a new loop per event.
#[derive(Clone)]
pub(crate) struct EventLoop(Arc<Py<PyAny>>);

impl EventLoop {
    pub(crate) fn running(py: Python<'_>) -> Option<Self> {
        pyo3_async_runtimes::get_running_loop(py)
            .ok()
            .map(|event_loop| Self(Arc::new(event_loop.unbind())))
    }
}

/// The Python context a Redis event is reported in. Each constructor names whose context it is,
/// so no event is reported in the context of whoever happened to build the thing reporting it.
#[derive(Clone)]
pub(crate) struct ReportContext(PythonContext);

impl ReportContext {
    /// The caller of a store op, on the stack now: its `with_service_target` and the span its
    /// request parents service events under.
    pub(crate) fn of_caller(py: Python<'_>) -> PyResult<Self> {
        Ok(Self(PythonContext::capture(py)?))
    }

    /// The request whose post-call batch its deadline sends, with no caller awaiting the flush.
    fn of_deadline_flush(py: Python<'_>) -> PyResult<Self> {
        let context = py
            .import(ServiceLoggingEvents::MODULE)?
            .call_method0("deadline_flush_context")?;
        Ok(Self(PythonContext::built(context)))
    }

    /// Nobody's, for breaker transitions and calls no store op scoped.
    fn background(py: Python<'_>) -> PyResult<Self> {
        Ok(Self(PythonContext::empty(py)?))
    }
}

/// The Python caller of one store operation. Its context carries `with_service_target` and the
/// ambient OTel span, and its `call_type` is the label `RedisCache` reports, so dashboards keep
/// their series.
pub(crate) struct RedisCall {
    pub(crate) context: ReportContext,
    pub(crate) event_loop: Option<EventLoop>,
    pub(crate) call_type: &'static str,
}

impl RedisCall {
    /// The deadline's flush of a request's post-call batch. Built when a post-call declaration
    /// arms the deadline, so it is the declaring caller's request the flush is reported under.
    pub(crate) fn deadline_flush(py: Python<'_>) -> PyResult<Self> {
        Ok(Self {
            context: ReportContext::of_deadline_flush(py)?,
            event_loop: EventLoop::running(py),
            call_type: "post_call_redis_batch",
        })
    }

    pub(crate) fn tag(self) -> CallTag {
        Arc::new(self)
    }

    pub(crate) async fn scope<F: Future>(self, operation: F) -> F::Output {
        tagged(self.tag(), operation).await
    }
}

/// Reports every native Redis call and breaker transition to `ServiceLogging`.
pub(crate) struct ServiceLoggingEvents {
    record_call: Arc<Py<PyAny>>,
    record_breaker: Arc<Py<PyAny>>,
    context: ReportContext,
    /// The loop the client was built on, for calls whose caller had none running.
    event_loop: Option<EventLoop>,
    runtime: Handle,
}

impl ServiceLoggingEvents {
    const MODULE: &str = "litellm.rust_bridge.host.redis_events";

    pub(crate) fn new(py: Python<'_>, runtime: Handle) -> PyResult<Self> {
        Self::from_module(
            py,
            &py.import(Self::MODULE)?,
            runtime,
            EventLoop::running(py),
        )
    }

    /// For a breaker, whose transitions never run on the caller's loop, so building it never looks one up.
    pub(crate) fn transitions_only(py: Python<'_>, runtime: Handle) -> PyResult<Self> {
        Self::from_module(py, &py.import(Self::MODULE)?, runtime, None)
    }

    fn from_module(
        py: Python<'_>,
        module: &Bound<'_, PyModule>,
        runtime: Handle,
        event_loop: Option<EventLoop>,
    ) -> PyResult<Self> {
        Ok(Self {
            record_call: Arc::new(module.getattr("record_redis_call")?.unbind()),
            record_breaker: Arc::new(module.getattr("record_redis_breaker")?.unbind()),
            context: ReportContext::background(py)?,
            event_loop,
            runtime,
        })
    }

    fn report(&self, context: ReportContext, record: Arc<Py<PyAny>>, args: ReportArgs) {
        self.runtime.spawn(async move {
            let reported = attach_blocking(context.0, move |py| match args {
                ReportArgs::Call(call) => record
                    .bind(py)
                    .call1((
                        call.event_loop
                            .as_ref()
                            .map(|event_loop| event_loop.0.bind(py)),
                        call.call_type,
                        call.start_time,
                        call.end_time,
                        call.failure,
                    ))
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

/// One call as `record_redis_call` takes it, its times in `time.time()` seconds.
struct CallReport {
    event_loop: Option<EventLoop>,
    call_type: &'static str,
    start_time: f64,
    end_time: f64,
    failure: Option<&'static str>,
}

enum ReportArgs {
    Call(CallReport),
    Breaker(&'static str),
}

/// The wall-clock start and end of a call that just finished after `duration`.
fn wall_clock_span(duration: Duration) -> (f64, f64) {
    let end = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    (
        end.saturating_sub(duration).as_secs_f64(),
        end.as_secs_f64(),
    )
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
        let scoped = current_tag().and_then(|tag| tag.downcast::<RedisCall>().ok());
        let (context, event_loop, call_type) = match scoped {
            Some(call) => (
                call.context.clone(),
                call.event_loop.clone().or_else(|| self.event_loop.clone()),
                call.call_type,
            ),
            None => (
                self.context.clone(),
                self.event_loop.clone(),
                operation_call_type(event.operation),
            ),
        };
        let (start_time, end_time) = wall_clock_span(event.duration);
        self.report(
            context,
            self.record_call.clone(),
            ReportArgs::Call(CallReport {
                event_loop,
                call_type,
                start_time,
                end_time,
                failure: failure_label(event.outcome),
            }),
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
    use std::{sync::Arc, time::Duration};

    use litellm_redis_client::{CallEvent, FailureKind, Operation, Outcome, RedisEvents};
    use pyo3::{
        prelude::*,
        types::{PyCFunction, PyModule, PyString},
    };
    use rstest::rstest;
    use tokio::runtime::Handle;

    use super::{EventLoop, RedisCall, ReportContext, ServiceLoggingEvents};

    const RECORDER: &str = "
import contextvars, time
target = contextvars.ContextVar('target', default=None)
seen = []
def record_redis_call(event_loop, call_type, start_time, end_time, failure):
    just_now = abs(time.time() - end_time) < 60
    seen.append((event_loop, call_type, round(end_time - start_time, 3), just_now, failure, target.get()))
def record_redis_breaker(state):
    seen.append(('breaker', state))
";

    fn stand_in_loop(py: Python<'_>, name: &str) -> EventLoop {
        EventLoop(Arc::new(PyString::new(py, name).into_any().unbind()))
    }

    fn recorder(name: &str, client_loop: Option<&str>) -> (ServiceLoggingEvents, Py<PyModule>) {
        Python::initialize();
        Python::attach(|py| {
            let name = std::ffi::CString::new(format!("redis_events_recorder_{name}")).unwrap();
            let module =
                PyModule::from_code(py, &std::ffi::CString::new(RECORDER).unwrap(), &name, &name)
                    .unwrap();
            let events = ServiceLoggingEvents::from_module(
                py,
                &module,
                Handle::current(),
                client_loop.map(|client_loop| stand_in_loop(py, client_loop)),
            )
            .unwrap();
            (events, module.unbind())
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

    fn scoped_call(namespace: &Py<PyModule>, caller_loop: Option<&str>) -> RedisCall {
        Python::attach(|py| {
            namespace
                .bind(py)
                .getattr("target")
                .unwrap()
                .call_method1("set", ("spend_counter",))
                .unwrap();
            RedisCall {
                context: ReportContext::of_caller(py).unwrap(),
                event_loop: caller_loop.map(|caller_loop| stand_in_loop(py, caller_loop)),
                call_type: "async_increment",
            }
        })
    }

    #[rstest]
    #[case::on_the_callers_loop(Some("caller loop"), "caller loop")]
    #[case::on_the_clients_loop_when_the_caller_had_none(None, "client loop")]
    #[tokio::test(flavor = "multi_thread")]
    async fn a_scoped_call_reports_the_store_labels_and_its_times_in_the_callers_context(
        #[case] caller_loop: Option<&str>,
        #[case] expected_loop: &str,
    ) {
        let (events, namespace) = recorder(
            expected_loop.replace(' ', "_").as_str(),
            Some("client loop"),
        );
        let call = scoped_call(&namespace, caller_loop);
        let event = CallEvent {
            operation: Operation::Command("INCRBYFLOAT"),
            duration: Duration::from_millis(250),
            outcome: Outcome::Failed(FailureKind::Timeout),
        };

        call.scope(async { events.call(&event) }).await;

        assert_eq!(
            seen(&namespace, 1).await,
            format!(
                "[('{expected_loop}', 'async_increment', 0.25, True, 'timeout', 'spend_counter')]"
            )
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn a_call_nobody_scoped_is_reported_outside_the_context_the_client_was_built_in() {
        Python::initialize();
        let (events, namespace) = Python::attach(|py| {
            let name = c"redis_events_recorder_built_inside_a_request";
            let module =
                PyModule::from_code(py, &std::ffi::CString::new(RECORDER).unwrap(), name, name)
                    .unwrap();
            module
                .getattr("target")
                .unwrap()
                .call_method1("set", ("the request that connected",))
                .unwrap();
            let events = ServiceLoggingEvents::from_module(
                py,
                &module,
                Handle::current(),
                Some(stand_in_loop(py, "client loop")),
            )
            .unwrap();
            (events, module.unbind())
        });

        events.call(&CallEvent {
            operation: Operation::Command("INCRBYFLOAT"),
            duration: Duration::from_millis(250),
            outcome: Outcome::Succeeded,
        });

        assert_eq!(
            seen(&namespace, 1).await,
            "[('client loop', 'INCRBYFLOAT', 0.25, True, None, None)]"
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
        let (events, namespace) = recorder(expected, None);

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
    async fn a_transition_on_a_thread_without_a_runtime_is_still_reported() {
        let (events, namespace) = recorder("plain_thread", None);

        std::thread::spawn(move || {
            events.breaker(&litellm_circuit_breaker::Transition {
                from: litellm_circuit_breaker::State::HalfOpen,
                to: litellm_circuit_breaker::State::HalfOpen,
                generation: 1,
            });
        })
        .join()
        .unwrap();

        assert_eq!(seen(&namespace, 1).await, "[('breaker', 'half_open')]");
    }

    #[rstest]
    #[tokio::test(flavor = "multi_thread")]
    async fn an_unscoped_call_falls_back_to_the_command_name_and_the_clients_loop() {
        let (events, namespace) = recorder("unscoped", None);

        events.call(&CallEvent {
            operation: Operation::Pipeline { ops: 3 },
            duration: Duration::from_millis(5),
            outcome: Outcome::Succeeded,
        });

        assert_eq!(
            seen(&namespace, 1).await,
            "[(None, 'pipeline', 0.005, True, None, None)]"
        );
    }

    #[rstest]
    fn running_is_the_loop_the_caller_runs_on_and_none_outside_one() {
        Python::initialize();
        Python::attach(|py| {
            let module = PyModule::from_code(
                py,
                c"import asyncio\nasync def probe_inside(probe):\n    return probe() is asyncio.get_running_loop()\n",
                c"event_loop_probe",
                c"event_loop_probe",
            )
            .unwrap();
            let probe = PyCFunction::new_closure(py, None, None, |_, _| {
                Python::attach(|py| -> PyResult<Option<Py<PyAny>>> {
                    Ok(EventLoop::running(py).map(|event_loop| event_loop.0.clone_ref(py)))
                })
            })
            .unwrap();
            let coroutine = module
                .getattr("probe_inside")
                .unwrap()
                .call1((probe,))
                .unwrap();

            let inside: bool = py
                .import("asyncio")
                .unwrap()
                .call_method1("run", (coroutine,))
                .unwrap()
                .extract()
                .unwrap();

            assert!(inside);
            assert!(EventLoop::running(py).is_none());
        });
    }
}
