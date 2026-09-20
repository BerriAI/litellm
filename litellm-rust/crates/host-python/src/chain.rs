//! Several lifecycles consuming one call as one [`PythonLifecycle`]. The driver keeps a
//! single consumer; which contracts a call runs under is decided where the chain is
//! built, never by the driver and never by a route.

use litellm_host::event::{FailureOrigin, MachineEvent, RequestContext, Timing, WireRequest};
use pyo3::exceptions::PyBaseException;
use pyo3::gc::{PyTraverseError, PyVisit};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::adapter::{LifecycleEvent, LifecycleStep, PythonLifecycle, missing_state};

/// An event kept across a suspension, so the lifecycles after the suspended one still
/// receive it. A failure keeps the raised exception object itself.
enum RetainedEvent {
    Machine(MachineEvent),
    Succeeded {
        timing: Timing,
        response: Py<PyAny>,
    },
    Failed {
        timing: Timing,
        origin: FailureOrigin,
        error: Py<PyBaseException>,
    },
}

/// The step a suspended lifecycle was answering, with what the later lifecycles need to
/// answer it too. The value being threaded stays with the suspended lifecycle.
enum Step {
    Begin { start_time: f64 },
    BeforeSend { context: RequestContext },
    AfterSuccess { timing: Timing },
    Emit(RetainedEvent),
}

struct Suspended {
    index: usize,
    step: Step,
}

/// Runs every step through its lifecycles in the order they were given.
///
/// `begin`, `before_send` and `after_success` thread their value: each lifecycle
/// receives what the one before it returned, so a later lifecycle sees an earlier one's
/// rewrites and never the reverse. Events reach each lifecycle in turn. An error from
/// any lifecycle ends the step at once, with no dispatch to the ones after it.
///
/// A lifecycle observes a call only once its `begin` was reached: when an earlier
/// lifecycle fails the call in `begin`, the later ones never learn of the call, and the
/// terminal event goes to those that did.
pub struct LifecycleChain {
    lifecycles: Vec<Box<dyn PythonLifecycle>>,
    /// How many lifecycles `begin` has reached.
    entered: usize,
    pending: Option<Suspended>,
}

impl RetainedEvent {
    fn retain(py: Python<'_>, event: LifecycleEvent<'_>) -> Self {
        match event {
            LifecycleEvent::Machine(event) => Self::Machine(event.clone()),
            LifecycleEvent::Succeeded { timing, response } => Self::Succeeded {
                timing,
                response: response.clone_ref(py),
            },
            LifecycleEvent::Failed {
                timing,
                origin,
                error,
            } => Self::Failed {
                timing,
                origin,
                error: error.clone_ref(py).into_value(py),
            },
        }
    }

    fn emit(&self, py: Python<'_>, lifecycle: &mut dyn PythonLifecycle) -> PyResult<LifecycleStep> {
        match self {
            Self::Machine(event) => lifecycle.emit(py, LifecycleEvent::Machine(event)),
            Self::Succeeded { timing, response } => lifecycle.emit(
                py,
                LifecycleEvent::Succeeded {
                    timing: *timing,
                    response,
                },
            ),
            Self::Failed {
                timing,
                origin,
                error,
            } => {
                let error = PyErr::from_value(error.bind(py).clone().into_any());
                lifecycle.emit(
                    py,
                    LifecycleEvent::Failed {
                        timing: *timing,
                        origin: *origin,
                        error: &error,
                    },
                )
            }
        }
    }

    fn traverse(&self, visit: &PyVisit<'_>) -> Result<(), PyTraverseError> {
        match self {
            Self::Succeeded { response, .. } => visit.call(response),
            Self::Failed { error, .. } => visit.call(error),
            Self::Machine(_) => Ok(()),
        }
    }
}

impl LifecycleChain {
    pub fn new(lifecycles: Vec<Box<dyn PythonLifecycle>>) -> Self {
        Self {
            lifecycles,
            entered: 0,
            pending: None,
        }
    }

    /// How many lifecycles a step reaches: `begin` tries all of them, every later step
    /// reaches the ones `begin` reached.
    fn audience(&self, step: &Step) -> usize {
        match step {
            Step::Begin { .. } => self.lifecycles.len(),
            Step::BeforeSend { .. } | Step::AfterSuccess { .. } | Step::Emit(_) => self.entered,
        }
    }

    fn call(
        &mut self,
        py: Python<'_>,
        index: usize,
        step: &Step,
        carried: LifecycleStep,
    ) -> PyResult<LifecycleStep> {
        let lifecycle = self.lifecycles[index].as_mut();
        match (step, carried) {
            (Step::Begin { start_time }, LifecycleStep::Arguments(arguments)) => {
                self.entered = index + 1;
                lifecycle.begin(py, arguments, *start_time)
            }
            (Step::BeforeSend { context }, LifecycleStep::Wire(wire)) => {
                lifecycle.before_send(py, wire, context)
            }
            (Step::AfterSuccess { timing }, LifecycleStep::Response(response)) => {
                lifecycle.after_success(py, response, *timing)
            }
            (Step::Emit(event), LifecycleStep::Done) => event.emit(py, lifecycle),
            _ => Err(missing_state()),
        }
    }

    /// Continues `step` from the lifecycle at `next`, handing each one what the previous
    /// produced, until one suspends or the last has answered.
    fn proceed(
        &mut self,
        py: Python<'_>,
        mut next: usize,
        mut carried: LifecycleStep,
        step: Step,
    ) -> PyResult<LifecycleStep> {
        while next < self.audience(&step) {
            carried = match self.call(py, next, &step, carried)? {
                LifecycleStep::Await(awaitable) => {
                    self.pending = Some(Suspended { index: next, step });
                    return Ok(LifecycleStep::Await(awaitable));
                }
                produced => produced,
            };
            next += 1;
        }
        Ok(carried)
    }
}

impl PythonLifecycle for LifecycleChain {
    fn begin(
        &mut self,
        py: Python<'_>,
        arguments: Py<PyDict>,
        start_time: f64,
    ) -> PyResult<LifecycleStep> {
        self.proceed(
            py,
            0,
            LifecycleStep::Arguments(arguments),
            Step::Begin { start_time },
        )
    }

    fn before_send(
        &mut self,
        py: Python<'_>,
        wire: Box<WireRequest>,
        context: &RequestContext,
    ) -> PyResult<LifecycleStep> {
        self.proceed(
            py,
            0,
            LifecycleStep::Wire(wire),
            Step::BeforeSend {
                context: context.clone(),
            },
        )
    }

    fn after_success(
        &mut self,
        py: Python<'_>,
        response: Py<PyAny>,
        timing: Timing,
    ) -> PyResult<LifecycleStep> {
        self.proceed(
            py,
            0,
            LifecycleStep::Response(response),
            Step::AfterSuccess { timing },
        )
    }

    fn emit(&mut self, py: Python<'_>, event: LifecycleEvent<'_>) -> PyResult<LifecycleStep> {
        for index in 0..self.entered {
            match self.lifecycles[index].emit(py, event)? {
                LifecycleStep::Await(awaitable) => {
                    self.pending = Some(Suspended {
                        index,
                        step: Step::Emit(RetainedEvent::retain(py, event)),
                    });
                    return Ok(LifecycleStep::Await(awaitable));
                }
                LifecycleStep::Done => {}
                _ => return Err(missing_state()),
            }
        }
        Ok(LifecycleStep::Done)
    }

    fn opened(&mut self, py: Python<'_>) -> PyResult<()> {
        self.lifecycles[..self.entered]
            .iter_mut()
            .try_for_each(|lifecycle| lifecycle.opened(py))
    }

    fn delivered(&mut self, py: Python<'_>, chunk: &Py<PyAny>) -> PyResult<()> {
        self.lifecycles[..self.entered]
            .iter_mut()
            .try_for_each(|lifecycle| lifecycle.delivered(py, chunk))
    }

    fn resume(&mut self, py: Python<'_>, result: PyResult<Py<PyAny>>) -> PyResult<LifecycleStep> {
        let Suspended { index, step } = self.pending.take().ok_or_else(missing_state)?;
        match self.lifecycles[index].resume(py, result)? {
            LifecycleStep::Await(awaitable) => {
                self.pending = Some(Suspended { index, step });
                Ok(LifecycleStep::Await(awaitable))
            }
            produced => self.proceed(py, index + 1, produced, step),
        }
    }

    fn close(&mut self, py: Python<'_>) {
        self.pending = None;
        for lifecycle in &mut self.lifecycles {
            lifecycle.close(py);
        }
    }

    fn traverse(&self, visit: &PyVisit<'_>) -> Result<(), PyTraverseError> {
        for lifecycle in &self.lifecycles {
            lifecycle.traverse(visit)?;
        }
        match &self.pending {
            Some(Suspended {
                step: Step::Emit(event),
                ..
            }) => event.traverse(visit),
            _ => Ok(()),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use pyo3::exceptions::PyValueError;

    use super::*;

    type Log = Arc<Mutex<Vec<String>>>;

    #[derive(Clone, Copy, PartialEq)]
    enum Script {
        Plain,
        FailBegin,
        SuspendBegin,
        SuspendTerminal,
    }

    struct Recorder {
        name: &'static str,
        script: Script,
        log: Log,
        suspended: Option<LifecycleStep>,
    }

    impl Recorder {
        fn record(&self, entry: &str) {
            self.log
                .lock()
                .unwrap()
                .push(format!("{}.{entry}", self.name));
        }

        fn suspend(&mut self, py: Python<'_>, then: LifecycleStep) -> PyResult<LifecycleStep> {
            self.suspended = Some(then);
            Ok(LifecycleStep::Await(py.None()))
        }
    }

    impl PythonLifecycle for Recorder {
        fn begin(
            &mut self,
            py: Python<'_>,
            arguments: Py<PyDict>,
            _: f64,
        ) -> PyResult<LifecycleStep> {
            let seen: Vec<String> = arguments
                .bind(py)
                .keys()
                .iter()
                .map(|key| key.extract().unwrap())
                .collect();
            self.record(&format!("begin{seen:?}"));
            arguments.bind(py).set_item(self.name, true)?;
            match self.script {
                Script::FailBegin => Err(PyValueError::new_err("begin failed")),
                Script::SuspendBegin => self.suspend(py, LifecycleStep::Arguments(arguments)),
                _ => Ok(LifecycleStep::Arguments(arguments)),
            }
        }

        fn before_send(
            &mut self,
            _: Python<'_>,
            mut wire: Box<WireRequest>,
            _: &RequestContext,
        ) -> PyResult<LifecycleStep> {
            self.record(&format!("before_send:{}", wire.url));
            wire.url.push_str(self.name);
            Ok(LifecycleStep::Wire(wire))
        }

        fn after_success(
            &mut self,
            _: Python<'_>,
            response: Py<PyAny>,
            _: Timing,
        ) -> PyResult<LifecycleStep> {
            self.record("after_success");
            Ok(LifecycleStep::Response(response))
        }

        fn emit(&mut self, py: Python<'_>, event: LifecycleEvent<'_>) -> PyResult<LifecycleStep> {
            let terminal = match event {
                LifecycleEvent::Machine(_) => {
                    self.record("machine");
                    false
                }
                LifecycleEvent::Succeeded { response, .. } => {
                    self.record(&format!("succeeded:{}", response.bind(py)));
                    true
                }
                LifecycleEvent::Failed { error, .. } => {
                    self.record(&format!("failed:{}", error.value(py)));
                    true
                }
            };
            if terminal && self.script == Script::SuspendTerminal {
                return self.suspend(py, LifecycleStep::Done);
            }
            Ok(LifecycleStep::Done)
        }

        fn opened(&mut self, _: Python<'_>) -> PyResult<()> {
            self.record("opened");
            Ok(())
        }

        fn delivered(&mut self, _: Python<'_>, _: &Py<PyAny>) -> PyResult<()> {
            self.record("delivered");
            Ok(())
        }

        fn resume(&mut self, _: Python<'_>, _: PyResult<Py<PyAny>>) -> PyResult<LifecycleStep> {
            self.record("resume");
            self.suspended.take().ok_or_else(missing_state)
        }

        fn close(&mut self, _: Python<'_>) {
            self.record("close");
        }

        fn traverse(&self, _: &PyVisit<'_>) -> Result<(), PyTraverseError> {
            Ok(())
        }
    }

    fn chain(scripts: [(&'static str, Script); 2]) -> (LifecycleChain, Log) {
        let log = Log::default();
        let lifecycles = scripts
            .into_iter()
            .map(|(name, script)| {
                Box::new(Recorder {
                    name,
                    script,
                    log: Arc::clone(&log),
                    suspended: None,
                }) as Box<dyn PythonLifecycle>
            })
            .collect();
        (LifecycleChain::new(lifecycles), log)
    }

    fn entries(log: &Log) -> Vec<String> {
        std::mem::take(&mut *log.lock().unwrap())
    }

    fn timing() -> Timing {
        Timing {
            start_time: 1.0,
            end_time: 2.0,
        }
    }

    fn context() -> RequestContext {
        RequestContext {
            model: "model".to_string(),
            custom_llm_provider: "provider".to_string(),
            optional_params: serde_json::Value::Null,
            secret_fields: Vec::new(),
            api_key: None,
        }
    }

    #[test]
    fn a_later_lifecycle_receives_what_an_earlier_one_returned() {
        crate::initialize_python();
        Python::attach(|py| {
            let (mut chain, log) = chain([("first", Script::Plain), ("second", Script::Plain)]);
            let arguments = PyDict::new(py).unbind();
            let step = chain.begin(py, arguments.clone_ref(py), 1.0).unwrap();
            assert!(matches!(step, LifecycleStep::Arguments(returned) if returned.is(&arguments)));
            let wire = Box::new(WireRequest {
                url: "url:".to_string(),
                headers: Vec::new(),
                body: serde_json::Value::Null,
            });
            let step = chain.before_send(py, wire, &context()).unwrap();
            assert!(matches!(step, LifecycleStep::Wire(wire) if wire.url == "url:firstsecond"));
            assert_eq!(
                entries(&log),
                [
                    "first.begin[]",
                    "second.begin[\"first\"]",
                    "first.before_send:url:",
                    "second.before_send:url:first",
                ]
            );
        });
    }

    #[test]
    fn a_lifecycle_begin_never_reached_observes_nothing_of_the_call() {
        crate::initialize_python();
        Python::attach(|py| {
            let (mut chain, log) = chain([("first", Script::FailBegin), ("second", Script::Plain)]);
            let error = chain
                .begin(py, PyDict::new(py).unbind(), 1.0)
                .err()
                .expect("begin fails");
            let failed = LifecycleEvent::Failed {
                timing: timing(),
                origin: FailureOrigin::Host,
                error: &error,
            };
            assert!(matches!(
                chain.emit(py, failed).unwrap(),
                LifecycleStep::Done
            ));
            chain.close(py);
            assert_eq!(
                entries(&log),
                [
                    "first.begin[]",
                    "first.failed:begin failed",
                    "first.close",
                    "second.close",
                ]
            );
        });
    }

    #[test]
    fn a_suspended_step_resumes_its_lifecycle_and_then_reaches_the_later_ones() {
        crate::initialize_python();
        Python::attach(|py| {
            let (mut chain, log) = chain([
                ("first", Script::SuspendBegin),
                ("second", Script::SuspendTerminal),
            ]);
            let step = chain.begin(py, PyDict::new(py).unbind(), 1.0).unwrap();
            assert!(matches!(step, LifecycleStep::Await(_)));
            assert_eq!(entries(&log), ["first.begin[]"]);
            let step = chain.resume(py, Ok(py.None())).unwrap();
            assert!(matches!(step, LifecycleStep::Arguments(_)));
            assert_eq!(entries(&log), ["first.resume", "second.begin[\"first\"]"]);
        });
    }

    #[test]
    fn an_event_kept_across_a_suspension_reaches_the_later_lifecycles_unchanged() {
        crate::initialize_python();
        Python::attach(|py| {
            let (mut chain, log) = chain([
                ("first", Script::SuspendTerminal),
                ("second", Script::Plain),
            ]);
            chain.begin(py, PyDict::new(py).unbind(), 1.0).unwrap();
            entries(&log);
            let raised = PyValueError::new_err("provider failed");
            let failed = LifecycleEvent::Failed {
                timing: timing(),
                origin: FailureOrigin::Call,
                error: &raised,
            };
            assert!(matches!(
                chain.emit(py, failed).unwrap(),
                LifecycleStep::Await(_)
            ));
            assert!(matches!(
                chain.resume(py, Ok(py.None())).unwrap(),
                LifecycleStep::Done
            ));
            assert_eq!(
                entries(&log),
                [
                    "first.failed:provider failed",
                    "first.resume",
                    "second.failed:provider failed",
                ]
            );
        });
    }
}
