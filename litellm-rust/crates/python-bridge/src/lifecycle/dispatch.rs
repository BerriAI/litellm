use litellm_core::call_lifecycle::{
    CallbackFamily, CallbackId, CallbackInvocation, CallbackKind, CallbackMethod, Delivery,
    DispatchCursor, DispatchFacts, DispatchStep, InvocationOutcome, LoggedMarker,
};
use pyo3::exceptions::{PyBaseException, PyException, PyRuntimeError};
use pyo3::gc::{PyTraverseError, PyVisit};
use pyo3::prelude::*;
use pyo3::types::PyString;

use super::bindings::PythonLogger;

const LEAVES: &str = "litellm.rust_bridge.leaves";

pub(super) fn leaves(py: Python<'_>) -> PyResult<Bound<'_, PyModule>> {
    py.import(LEAVES)
}

#[derive(Clone, Copy)]
pub(super) enum Outcome {
    Success,
    Failure,
}

pub(super) struct Targets {
    objects: Vec<Py<PyAny>>,
    kinds: Vec<CallbackKind>,
}

impl Targets {
    pub(super) fn read(
        py: Python<'_>,
        lists: &[Bound<'_, PyAny>],
    ) -> PyResult<(Self, Vec<Vec<CallbackId>>)> {
        let custom_logger = py
            .import("litellm.integrations.custom_logger")?
            .getattr("CustomLogger")?;
        let known = py
            .import("litellm")?
            .getattr("_known_custom_logger_compatible_callbacks")?;
        let mut targets = Self {
            objects: Vec::new(),
            kinds: Vec::new(),
        };
        let mut ids = Vec::with_capacity(lists.len());
        for list in lists {
            let mut family = Vec::new();
            for object in list.try_iter()? {
                let object = object?;
                family.push(targets.intern(py, &object, &custom_logger, &known)?);
            }
            ids.push(family);
        }
        Ok((targets, ids))
    }

    fn intern(
        &mut self,
        py: Python<'_>,
        object: &Bound<'_, PyAny>,
        custom_logger: &Bound<'_, PyAny>,
        known: &Bound<'_, PyAny>,
    ) -> PyResult<CallbackId> {
        for (index, existing) in self.objects.iter().enumerate() {
            let existing = existing.bind(py);
            if existing.is(object) || existing.eq(object)? {
                return Ok(CallbackId(index as u64));
            }
        }
        let kind = if object.is_instance(custom_logger)? {
            CallbackKind::CustomLogger
        } else if let Ok(name) = object.cast::<PyString>() {
            CallbackKind::Named {
                known: known.contains(name)?,
            }
        } else if object.is_callable() {
            CallbackKind::Callable {
                internal: internal_callable(object)?,
            }
        } else {
            CallbackKind::Opaque
        };
        self.objects.push(object.clone().unbind());
        self.kinds.push(kind);
        Ok(CallbackId((self.objects.len() - 1) as u64))
    }

    pub(super) fn kinds(&self, ids: &[CallbackId]) -> Vec<CallbackKind> {
        ids.iter().map(|id| self.kinds[id.0 as usize]).collect()
    }

    fn object<'py>(&self, py: Python<'py>, id: CallbackId) -> &Bound<'py, PyAny> {
        self.objects[id.0 as usize].bind(py)
    }

    fn kind(&self, id: CallbackId) -> CallbackKind {
        self.kinds[id.0 as usize]
    }

    pub(super) fn traverse(&self, visit: &PyVisit<'_>) -> Result<(), PyTraverseError> {
        for object in &self.objects {
            visit.call(object)?;
        }
        Ok(())
    }
}

fn internal_callable(object: &Bound<'_, PyAny>) -> PyResult<bool> {
    let name = if let Ok(name) = object.getattr("__name__") {
        name.extract::<String>()?
    } else if let Ok(func) = object.getattr("__func__") {
        func.getattr("__name__")?.extract::<String>()?
    } else {
        object.get_type().name()?.to_string()
    };
    Ok([
        "_PROXY",
        "_service_logger.ServiceLogging",
        "sync_deployment_callback_on_success",
    ]
    .iter()
    .any(|prefix| name.contains(prefix)))
}

pub(super) struct Job {
    pub logger: PythonLogger,
    pub targets: Targets,
    pub ids: Vec<CallbackId>,
    pub family: CallbackFamily,
    pub response: Option<Py<PyAny>>,
    pub error: Option<Py<PyBaseException>>,
    pub start: Py<PyAny>,
    pub end: Py<PyAny>,
    pub stream: bool,
}

impl Job {
    pub(super) fn traverse(&self, visit: &PyVisit<'_>) -> Result<(), PyTraverseError> {
        self.logger.traverse(visit)?;
        self.targets.traverse(visit)?;
        visit.call(&self.response)?;
        visit.call(&self.error)?;
        visit.call(&self.start)?;
        visit.call(&self.end)
    }

    fn family_name(&self) -> &'static str {
        match self.family {
            CallbackFamily::SyncSuccess => "sync_success",
            CallbackFamily::AsyncSuccess => "async_success",
            CallbackFamily::SyncFailure => "sync_failure",
            CallbackFamily::AsyncFailure => "async_failure",
            _ => "request",
        }
    }

    fn outcome(&self) -> Outcome {
        match self.family {
            CallbackFamily::SyncSuccess | CallbackFamily::AsyncSuccess => Outcome::Success,
            _ => Outcome::Failure,
        }
    }
}

struct Eligibility<'a, 'py> {
    py: Python<'py>,
    job: &'a Job,
    leaves: &'a Bound<'py, PyModule>,
}

impl DispatchFacts for Eligibility<'_, '_> {
    fn eligible(&mut self, target: CallbackId, method: CallbackMethod) -> bool {
        let object = self.job.targets.object(self.py, target);
        let kind = self.job.targets.kind(target);
        let result = match method {
            CallbackMethod::LoggingHook | CallbackMethod::AsyncLoggingHook => {
                if kind != CallbackKind::CustomLogger {
                    return false;
                }
                self.leaves
                    .getattr("should_run_guardrail_hook")
                    .and_then(|f| f.call1((self.job.logger.object(self.py), object)))
                    .and_then(|v| v.extract::<bool>())
            }
            CallbackMethod::LogSuccessEvent
            | CallbackMethod::AsyncLogSuccessEvent
            | CallbackMethod::LogFailureEvent
            | CallbackMethod::AsyncLogFailureEvent => {
                if kind == CallbackKind::Opaque {
                    return false;
                }
                self.leaves
                    .getattr("should_run_callback")
                    .and_then(|f| {
                        f.call1((
                            self.job.logger.object(self.py),
                            object,
                            self.job.family_name(),
                        ))
                    })
                    .and_then(|v| v.extract::<bool>())
            }
            _ => Ok(kind != CallbackKind::Opaque),
        };
        match result {
            Ok(value) => value,
            Err(error) => {
                error.write_unraisable(self.py, Some(object));
                false
            }
        }
    }
}

pub(super) enum Step {
    Await(Py<PyAny>),
    Done,
}

pub(super) struct Runner {
    job: Job,
    cursor: DispatchCursor,
    result: Option<Py<PyAny>>,
    formatted: Option<Py<PyAny>>,
    pending: Option<CallbackInvocation>,
}

impl Runner {
    pub(super) fn start(py: Python<'_>, job: Job) -> PyResult<Self> {
        let leaves = leaves(py)?;
        let already = match job.family.marker() {
            Some(marker) => leaves
                .getattr("already_logged")?
                .call1((job.logger.object(py), marker.key()))?
                .extract::<bool>()?,
            None => false,
        };
        let cursor = DispatchCursor::start(job.family, job.ids.clone(), already, job.stream);
        let result = job.response.as_ref().map(|value| value.clone_ref(py));
        Ok(Self {
            job,
            cursor,
            result,
            formatted: None,
            pending: None,
        })
    }

    pub(super) fn traverse(&self, visit: &PyVisit<'_>) -> Result<(), PyTraverseError> {
        self.job.traverse(visit)?;
        visit.call(&self.result)?;
        visit.call(&self.formatted)
    }

    pub(super) fn logger<'py>(&self, py: Python<'py>) -> &Bound<'py, PyAny> {
        self.job.logger.object(py)
    }

    pub(super) fn resume(
        &mut self,
        py: Python<'_>,
        awaited: Option<PyResult<Py<PyAny>>>,
    ) -> PyResult<Step> {
        let leaves = leaves(py)?;
        if let Some(awaited) = awaited {
            let invocation = self.pending.take().ok_or_else(super::missing_state)?;
            let result = awaited.map(|value| {
                (invocation.method == CallbackMethod::AsyncLoggingHook).then_some(value)
            });
            self.accept(py, &leaves, invocation.target, result)?;
        }
        loop {
            let step = self.cursor.next(&mut Eligibility {
                py,
                job: &self.job,
                leaves: &leaves,
            });
            match step {
                DispatchStep::PrepareLogging => self.prepare(py, &leaves)?,
                DispatchStep::MarkLogged(marker) => self.mark(py, &leaves, marker)?,
                DispatchStep::Invoke(invocation) => {
                    let outcome = self.invoke(py, &leaves, invocation);
                    match outcome {
                        Ok(Some(awaitable)) => {
                            self.pending = Some(invocation);
                            return Ok(Step::Await(awaitable));
                        }
                        Ok(None) => self.accept(py, &leaves, invocation.target, Ok(None))?,
                        Err(error) => self.accept(py, &leaves, invocation.target, Err(error))?,
                    }
                }
                DispatchStep::Complete { .. } => {
                    leaves
                        .getattr("restore_correlation_context")?
                        .call1((self.job.logger.object(py),))?;
                    return Ok(Step::Done);
                }
            }
        }
    }

    fn prepare(&mut self, py: Python<'_>, leaves: &Bound<'_, PyModule>) -> PyResult<()> {
        let logger = self.job.logger.object(py);
        let result = match self.job.outcome() {
            Outcome::Success => leaves
                .getattr("prepare_success_logging")?
                .call1((logger, &self.result, &self.job.start, &self.job.end))
                .map(|redacted| self.result = Some(redacted.unbind())),
            Outcome::Failure => leaves
                .getattr("prepare_failure_logging")?
                .call1((logger, &self.job.error, &self.job.start, &self.job.end))
                .map(|formatted| self.formatted = Some(formatted.unbind())),
        };
        match result {
            Err(error) if error.is_instance_of::<PyException>(py) => {
                error.write_unraisable(py, Some(logger));
                Ok(())
            }
            result => result,
        }
    }

    fn mark(
        &self,
        py: Python<'_>,
        leaves: &Bound<'_, PyModule>,
        marker: LoggedMarker,
    ) -> PyResult<()> {
        leaves
            .getattr("mark_logged")?
            .call1((self.job.logger.object(py), marker.key()))?;
        Ok(())
    }

    fn invoke(
        &mut self,
        py: Python<'_>,
        leaves: &Bound<'_, PyModule>,
        invocation: CallbackInvocation,
    ) -> PyResult<Option<Py<PyAny>>> {
        let logger = self.job.logger.object(py);
        let target = self.job.targets.object(py, invocation.target);
        let kind = self.job.targets.kind(invocation.target);
        let value = match (invocation.method, kind) {
            (CallbackMethod::LoggingHook, CallbackKind::CustomLogger) => {
                let replaced =
                    leaves
                        .getattr("logging_hook")?
                        .call1((logger, target, &self.result))?;
                self.result = Some(replaced.unbind());
                return Ok(None);
            }
            (CallbackMethod::AsyncLoggingHook, CallbackKind::CustomLogger) => leaves
                .getattr("async_logging_hook")?
                .call1((logger, target, &self.result))?,
            (CallbackMethod::LogSuccessEvent, CallbackKind::CustomLogger) => leaves
                .getattr("log_success_event")?
                .call1((logger, target, &self.result, &self.job.start, &self.job.end))?,
            (CallbackMethod::AsyncLogSuccessEvent, CallbackKind::CustomLogger) => leaves
                .getattr("async_log_success_event")?
                .call1((logger, target, &self.result, &self.job.start, &self.job.end))?,
            (CallbackMethod::LogFailureEvent, CallbackKind::CustomLogger) => leaves
                .getattr("log_failure_event")?
                .call1((logger, target, &self.job.start, &self.job.end))?,
            (CallbackMethod::AsyncLogFailureEvent, CallbackKind::CustomLogger) => leaves
                .getattr("async_log_failure_event")?
                .call1((logger, target, &self.job.start, &self.job.end))?,
            (
                CallbackMethod::LogSuccessEvent
                | CallbackMethod::AsyncLogSuccessEvent
                | CallbackMethod::LogFailureEvent
                | CallbackMethod::AsyncLogFailureEvent,
                CallbackKind::Callable { .. },
            ) => leaves.getattr("dispatch_callable")?.call1((
                logger,
                target,
                self.job.family_name(),
                &self.result,
                &self.job.start,
                &self.job.end,
            ))?,
            (
                CallbackMethod::LogSuccessEvent | CallbackMethod::AsyncLogSuccessEvent,
                CallbackKind::Named { .. },
            ) => leaves.getattr("dispatch_named_success")?.call1((
                logger,
                target,
                &self.result,
                &self.job.start,
                &self.job.end,
            ))?,
            (
                CallbackMethod::LogFailureEvent | CallbackMethod::AsyncLogFailureEvent,
                CallbackKind::Named { .. },
            ) => leaves.getattr("dispatch_named_failure")?.call1((
                logger,
                target,
                &self.job.error,
                &self.formatted,
                &self.job.start,
                &self.job.end,
            ))?,
            _ => return Ok(None),
        };
        match invocation.delivery {
            Delivery::Inline | Delivery::Worker => Ok(None),
            Delivery::Await | Delivery::Background if value.is_none() => Ok(None),
            Delivery::Await | Delivery::Background if value.hasattr("__await__")? => {
                Ok(Some(value.unbind()))
            }
            Delivery::Await | Delivery::Background => Err(PyRuntimeError::new_err(format!(
                "{:?} leaf for {:?} returned a non-awaitable {}",
                invocation.method,
                self.job.family,
                value.get_type().name()?
            ))),
        }
    }

    fn accept(
        &mut self,
        py: Python<'_>,
        leaves: &Bound<'_, PyModule>,
        target: CallbackId,
        result: PyResult<Option<Py<PyAny>>>,
    ) -> PyResult<()> {
        match result {
            Ok(Some(replaced)) => {
                self.result = Some(replaced);
                self.cursor.accept(InvocationOutcome::Completed);
            }
            Ok(None) => self.cursor.accept(InvocationOutcome::Completed),
            Err(error) if error.is_instance_of::<PyException>(py) => {
                let object = self.job.targets.object(py, target);
                if let Err(report) = leaves.getattr("report_target_failure").and_then(|f| {
                    f.call1((
                        self.job.logger.object(py),
                        object,
                        self.job.family_name(),
                        error.value(py),
                    ))
                }) {
                    report.write_unraisable(py, Some(object));
                }
                self.cursor.accept(InvocationOutcome::Failed);
            }
            Err(error) => return Err(error),
        }
        Ok(())
    }
}

#[pyclass]
pub(super) struct WorkerJob {
    runner: Option<Runner>,
}

impl WorkerJob {
    pub(super) fn new(runner: Runner) -> Self {
        Self {
            runner: Some(runner),
        }
    }
}

#[pymethods]
impl WorkerJob {
    fn __call__(slf: &Bound<'_, Self>, py: Python<'_>) -> PyResult<()> {
        let Some(mut runner) = slf.borrow_mut().runner.take() else {
            return Ok(());
        };
        match runner.resume(py, None) {
            Ok(Step::Done) => Ok(()),
            Ok(Step::Await(_)) => Err(PyRuntimeError::new_err(
                "worker dispatch selected an awaiting delivery",
            )),
            Err(error) if error.is_instance_of::<PyException>(py) => {
                error.write_unraisable(py, Some(runner.job.logger.object(py)));
                Ok(())
            }
            Err(error) => Err(error),
        }
    }

    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        match &self.runner {
            Some(runner) => runner.traverse(&visit),
            None => Ok(()),
        }
    }

    fn __clear__(&mut self) {
        self.runner = None;
    }
}

pub(super) struct AwaitingBody {
    runner: Option<Runner>,
}

impl AwaitingBody {
    pub(super) fn new(runner: Runner) -> Self {
        Self {
            runner: Some(runner),
        }
    }
}

impl super::handle::ExecutionBody for AwaitingBody {
    fn resume(
        &mut self,
        result: Option<PyResult<Py<PyAny>>>,
    ) -> PyResult<super::handle::ExecutionStep> {
        Python::attach(|py| {
            let runner = self.runner.as_mut().ok_or_else(super::missing_state)?;
            match runner.resume(py, result) {
                Ok(Step::Await(awaitable)) => Ok(super::handle::ExecutionStep::Await(awaitable)),
                Ok(Step::Done) => {
                    self.runner = None;
                    Ok(super::handle::ExecutionStep::Return(py.None()))
                }
                Err(error) if error.is_instance_of::<PyException>(py) => {
                    error.write_unraisable(py, Some(runner.job.logger.object(py)));
                    self.runner = None;
                    Ok(super::handle::ExecutionStep::Return(py.None()))
                }
                Err(error) => Err(error),
            }
        })
    }

    fn traverse(&self, visit: &PyVisit<'_>) -> Result<(), PyTraverseError> {
        match &self.runner {
            Some(runner) => runner.traverse(visit),
            None => Ok(()),
        }
    }
}

pub(super) fn coroutine(py: Python<'_>, runner: Runner) -> PyResult<Py<PyAny>> {
    let execution = Py::new(py, super::handle::Execution::new(AwaitingBody::new(runner)))?;
    py.import("litellm.rust_bridge.lifecycle")?
        .getattr("drive")?
        .call1((execution,))
        .map(Bound::unbind)
}

pub(super) struct RequestJob<'a> {
    pub logger: &'a PythonLogger,
    pub family: CallbackFamily,
}

pub(super) fn dispatch_request(py: Python<'_>, job: RequestJob<'_>) -> PyResult<()> {
    let leaves = leaves(py)?;
    let (targets, ids) = family_targets(py, job.logger, job.family)?;
    let mut cursor = DispatchCursor::start(job.family, ids, false, false);
    let logger = job.logger.object(py);
    let event = match job.family {
        CallbackFamily::RequestPreCall => "pre_api_call",
        _ => "post_api_call",
    };
    let mut facts = RequestEligibility;
    loop {
        match cursor.next(&mut facts) {
            DispatchStep::Invoke(invocation) => {
                let target = targets.object(py, invocation.target);
                let result = match (invocation.method, targets.kind(invocation.target)) {
                    (CallbackMethod::LogPreApiCall, CallbackKind::CustomLogger) => {
                        leaves.getattr("log_pre_api_call")?.call1((logger, target))
                    }
                    (CallbackMethod::LogPostApiCall, CallbackKind::CustomLogger) => {
                        leaves.getattr("log_post_api_call")?.call1((logger, target))
                    }
                    (_, CallbackKind::Named { .. }) => leaves
                        .getattr("dispatch_named_request")?
                        .call1((logger, target, event)),
                    (CallbackMethod::LogPreApiCall, CallbackKind::Callable { .. }) => leaves
                        .getattr("dispatch_callable_request")?
                        .call1((logger, target)),
                    _ => Ok(py.None().into_bound(py)),
                };
                match result {
                    Ok(_) => cursor.accept(InvocationOutcome::Completed),
                    Err(error) if error.is_instance_of::<PyException>(py) => {
                        if let Err(report) = leaves
                            .getattr("report_target_failure")
                            .and_then(|f| f.call1((logger, target, "request", error.value(py))))
                        {
                            report.write_unraisable(py, Some(target));
                        }
                        cursor.accept(InvocationOutcome::Failed);
                    }
                    Err(error) => return Err(error),
                }
            }
            DispatchStep::Complete { .. } => return Ok(()),
            DispatchStep::PrepareLogging | DispatchStep::MarkLogged(_) => {}
        }
    }
}

struct RequestEligibility;

impl DispatchFacts for RequestEligibility {
    fn eligible(&mut self, _: CallbackId, _: CallbackMethod) -> bool {
        true
    }
}

pub(super) fn read_lists<'py>(
    py: Python<'py>,
    logger: &PythonLogger,
    family: CallbackFamily,
) -> PyResult<(Bound<'py, PyAny>, Option<Bound<'py, PyAny>>)> {
    let litellm = py.import("litellm")?;
    let logger = logger.object(py);
    let (global, dynamic) = match family {
        CallbackFamily::RequestPreCall | CallbackFamily::RequestPostCall => {
            ("input_callback", "dynamic_input_callbacks")
        }
        CallbackFamily::SyncSuccess => ("success_callback", "dynamic_success_callbacks"),
        CallbackFamily::AsyncSuccess => {
            ("_async_success_callback", "dynamic_async_success_callbacks")
        }
        CallbackFamily::SyncFailure => ("failure_callback", "dynamic_failure_callbacks"),
        CallbackFamily::AsyncFailure => {
            ("_async_failure_callback", "dynamic_async_failure_callbacks")
        }
        CallbackFamily::DeploymentPreCall
        | CallbackFamily::DeploymentPostCall
        | CallbackFamily::DeploymentFailure => ("callbacks", ""),
    };
    let global = litellm.getattr(global)?;
    let dynamic = if dynamic.is_empty() {
        None
    } else {
        let value = logger.getattr(dynamic)?;
        (!value.is_none()).then_some(value)
    };
    Ok((global, dynamic))
}

pub(super) fn family_targets(
    py: Python<'_>,
    logger: &PythonLogger,
    family: CallbackFamily,
) -> PyResult<(Targets, Vec<CallbackId>)> {
    let (global, dynamic) = read_lists(py, logger, family)?;
    let lists: Vec<Bound<'_, PyAny>> = std::iter::once(global).chain(dynamic.clone()).collect();
    let (targets, ids) = Targets::read(py, &lists)?;
    let global_ids = ids.first().cloned().unwrap_or_default();
    let dynamic_ids = ids.get(1).map(Vec::as_slice);
    let ordered = family.targets(
        &global_ids,
        dynamic.is_some().then_some(dynamic_ids.unwrap_or(&[])),
    );
    Ok((targets, ordered))
}

pub(super) enum DeploymentEvent {
    PreCall,
    PostCall,
    Failure {
        request: Py<PyAny>,
        exception: Py<PyBaseException>,
        fallback_depth: Py<PyAny>,
    },
}

pub(super) struct DeploymentBody {
    logger: PythonLogger,
    targets: Targets,
    cursor: DispatchCursor,
    event: DeploymentEvent,
    call_type: &'static str,
    current: Py<PyAny>,
    kwargs: Py<pyo3::types::PyDict>,
    pending: Option<CallbackId>,
}

impl DeploymentBody {
    pub(super) fn start(
        py: Python<'_>,
        logger: &PythonLogger,
        family: CallbackFamily,
        call_type: &'static str,
        kwargs: &Py<pyo3::types::PyDict>,
        current: Py<PyAny>,
        error: Option<&Py<PyBaseException>>,
    ) -> PyResult<Self> {
        let (targets, ids) = family_targets(py, logger, family)?;
        let event = match family {
            CallbackFamily::DeploymentPreCall => DeploymentEvent::PreCall,
            CallbackFamily::DeploymentPostCall => DeploymentEvent::PostCall,
            CallbackFamily::DeploymentFailure => {
                let exception = error.ok_or_else(super::missing_state)?;
                let view = leaves(py)?
                    .getattr("failure_deployment_hook_view")?
                    .call1((kwargs, exception))?;
                let (request, snapshot, fallback_depth): (Py<PyAny>, Py<PyAny>, Py<PyAny>) =
                    view.extract()?;
                DeploymentEvent::Failure {
                    request,
                    exception: snapshot
                        .into_bound(py)
                        .cast_into::<PyBaseException>()?
                        .unbind(),
                    fallback_depth,
                }
            }
            _ => return Err(super::missing_state()),
        };
        Ok(Self {
            logger: logger.clone_ref(py),
            targets,
            cursor: DispatchCursor::start(family, ids, false, false),
            event,
            call_type,
            current,
            kwargs: kwargs.clone_ref(py),
            pending: None,
        })
    }

    fn invoke(&self, py: Python<'_>, target: CallbackId) -> PyResult<Py<PyAny>> {
        let leaves = leaves(py)?;
        let object = self.targets.object(py, target);
        let awaitable = match &self.event {
            DeploymentEvent::PreCall => leaves.getattr("pre_call_deployment_hook")?.call1((
                object,
                &self.current,
                self.call_type,
            ))?,
            DeploymentEvent::PostCall => leaves
                .getattr("post_call_success_deployment_hook")?
                .call1((object, &self.kwargs, &self.current, self.call_type))?,
            DeploymentEvent::Failure {
                request,
                exception,
                fallback_depth,
            } => leaves
                .getattr("post_call_failure_deployment_hook")?
                .call1((object, request, exception, self.call_type, fallback_depth))?,
        };
        Ok(awaitable.unbind())
    }

    fn accept(
        &mut self,
        py: Python<'_>,
        target: CallbackId,
        result: PyResult<Py<PyAny>>,
    ) -> PyResult<()> {
        match result {
            Ok(value) => {
                if !value.is_none(py) && !matches!(self.event, DeploymentEvent::Failure { .. }) {
                    self.current = value;
                }
                self.cursor.accept(InvocationOutcome::Completed);
                Ok(())
            }
            Err(error)
                if matches!(self.event, DeploymentEvent::Failure { .. })
                    && error.is_instance_of::<PyException>(py) =>
            {
                let object = self.targets.object(py, target);
                leaves(py)?
                    .getattr("report_deployment_failure_hook_error")?
                    .call1((object, error.value(py)))?;
                self.cursor.accept(InvocationOutcome::Failed);
                Ok(())
            }
            Err(error) => Err(error),
        }
    }
}

impl super::handle::ExecutionBody for DeploymentBody {
    fn resume(
        &mut self,
        result: Option<PyResult<Py<PyAny>>>,
    ) -> PyResult<super::handle::ExecutionStep> {
        Python::attach(|py| {
            if let Some(result) = result {
                let target = self.pending.take().ok_or_else(super::missing_state)?;
                self.accept(py, target, result)?;
            }
            let mut facts = DeploymentEligibility(&self.targets);
            loop {
                match self.cursor.next(&mut facts) {
                    DispatchStep::Invoke(invocation) => {
                        let awaitable = self.invoke(py, invocation.target)?;
                        self.pending = Some(invocation.target);
                        return Ok(super::handle::ExecutionStep::Await(awaitable));
                    }
                    DispatchStep::Complete { .. } => {
                        return Ok(super::handle::ExecutionStep::Return(
                            self.current.clone_ref(py),
                        ));
                    }
                    DispatchStep::PrepareLogging | DispatchStep::MarkLogged(_) => {}
                }
            }
        })
    }

    fn traverse(&self, visit: &PyVisit<'_>) -> Result<(), PyTraverseError> {
        self.logger.traverse(visit)?;
        self.targets.traverse(visit)?;
        visit.call(&self.current)?;
        visit.call(&self.kwargs)?;
        if let DeploymentEvent::Failure {
            request,
            exception,
            fallback_depth,
        } = &self.event
        {
            visit.call(request)?;
            visit.call(exception)?;
            visit.call(fallback_depth)?;
        }
        Ok(())
    }
}

struct DeploymentEligibility<'a>(&'a Targets);

impl DispatchFacts for DeploymentEligibility<'_> {
    fn eligible(&mut self, target: CallbackId, _: CallbackMethod) -> bool {
        self.0.kind(target) == CallbackKind::CustomLogger
    }
}

pub(super) fn deployment_coroutine(py: Python<'_>, body: DeploymentBody) -> PyResult<Py<PyAny>> {
    let execution = Py::new(py, super::handle::Execution::new(body))?;
    py.import("litellm.rust_bridge.lifecycle")?
        .getattr("drive")?
        .call1((execution,))
        .map(Bound::unbind)
}

#[cfg(test)]
mod tests {
    use super::*;
    use pyo3::types::{PyDict, PyList};

    fn fixture(py: Python<'_>) -> Bound<'_, PyDict> {
        let locals = PyDict::new(py);
        py.run(
            pyo3::ffi::c_str!(
                r#"
import sys, types
for name in ("litellm", "litellm.integrations", "litellm.integrations.custom_logger"):
    sys.modules.setdefault(name, types.ModuleType(name))
class CustomLogger: pass
sys.modules["litellm.integrations.custom_logger"].CustomLogger = CustomLogger
sys.modules["litellm"]._known_custom_logger_compatible_callbacks = []
class Logger: pass
logger = Logger()
target = CustomLogger()
"#
            ),
            Some(&locals),
            Some(&locals),
        )
        .unwrap();
        locals
    }

    fn runner(py: Python<'_>, locals: &Bound<'_, PyDict>, family: CallbackFamily) -> Runner {
        let logger = locals.get_item("logger").unwrap().unwrap();
        let target = locals.get_item("target").unwrap().unwrap();
        let list = PyList::new(py, [&target]).unwrap().into_any();
        let (targets, ids) = Targets::read(py, &[list]).unwrap();
        let ids: Vec<CallbackId> = ids.into_iter().flatten().collect();
        Runner {
            cursor: DispatchCursor::start(family, ids.clone(), false, false),
            job: Job {
                logger: logger.extract().unwrap(),
                targets,
                ids,
                family,
                response: Some(target.clone().unbind()),
                error: None,
                start: py.None(),
                end: py.None(),
                stream: false,
            },
            result: Some(target.unbind()),
            formatted: None,
            pending: None,
        }
    }

    fn assert_collectable(py: Python<'_>, locals: &Bound<'_, PyDict>, handle: Py<PyAny>) {
        locals.set_item("handle", handle).unwrap();
        py.run(
            pyo3::ffi::c_str!(
                r#"
import gc
import weakref
target.handle = handle
reference = weakref.ref(target)
del logger, target, handle
gc.collect()
assert reference() is None, "cycle through the retained target was not collected"
"#
            ),
            Some(locals),
            Some(locals),
        )
        .unwrap();
    }

    #[test]
    fn worker_job_collects_cycles_through_logger_targets_and_response() {
        Python::initialize();
        Python::attach(|py| {
            let locals = fixture(py);
            let runner = runner(py, &locals, CallbackFamily::SyncSuccess);
            let handle = Py::new(py, WorkerJob::new(runner)).unwrap().into_any();
            assert_collectable(py, &locals, handle);
        });
    }

    #[test]
    fn deferred_success_collects_cycles_and_close_is_idempotent() {
        Python::initialize();
        Python::attach(|py| {
            let locals = fixture(py);
            let runner = runner(py, &locals, CallbackFamily::AsyncSuccess);
            let deferred = Py::new(
                py,
                super::super::DeferredSuccess {
                    runner: Some(runner),
                },
            )
            .unwrap();
            assert_collectable(py, &locals, deferred.into_any());
        });
    }

    #[test]
    fn deferred_success_releases_at_most_once_and_close_prevents_release() {
        Python::initialize();
        Python::attach(|py| {
            let locals = fixture(py);
            let runner = runner(py, &locals, CallbackFamily::AsyncSuccess);
            let deferred = Py::new(
                py,
                super::super::DeferredSuccess {
                    runner: Some(runner),
                },
            )
            .unwrap();
            deferred.call_method0(py, "close").unwrap();
            deferred.call_method0(py, "close").unwrap();
            deferred.call0(py).unwrap();
            assert!(deferred.borrow(py).runner.is_none());
        });
    }
}
