use std::cell::RefCell;
use std::collections::HashMap;
use std::future::Future;
use std::io::{ErrorKind, Read, Write};
use std::os::fd::AsRawFd;
use std::os::unix::net::UnixStream;
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::task::{Poll, Wake, Waker};

use pyo3::exceptions::asyncio::{CancelledError, InvalidStateError};
use pyo3::exceptions::{PyNotImplementedError, PyRuntimeError};
use pyo3::gc::{PyTraverseError, PyVisit};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::missing_state;
use crate::runtime::poll_with_waker;

const POISONED: &str = "native step slot poisoned";

pub(crate) type StepFuture = Pin<Box<dyn Future<Output = PyResult<()>> + Send>>;

enum Outcome {
    Pending,
    Ready,
    Failed(PyErr),
    Cancelled(Option<Py<PyAny>>),
}

struct Slot {
    future: Option<StepFuture>,
    outcome: Outcome,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum StepState {
    Pending,
    Ready,
    Failed,
    Cancelled,
}

pub(crate) struct StepCore {
    id: u64,
    woken: AtomicBool,
    target: OnceLock<Arc<WakeQueue>>,
    slot: Mutex<Slot>,
}

impl StepCore {
    pub(crate) fn new() -> Arc<Self> {
        static NEXT_ID: AtomicU64 = AtomicU64::new(0);
        Arc::new(Self {
            id: NEXT_ID.fetch_add(1, Ordering::Relaxed),
            woken: AtomicBool::new(false),
            target: OnceLock::new(),
            slot: Mutex::new(Slot {
                future: None,
                outcome: Outcome::Pending,
            }),
        })
    }

    pub(crate) fn id(&self) -> u64 {
        self.id
    }

    pub(crate) fn bound(&self) -> bool {
        self.target.get().is_some()
    }

    pub(crate) fn bind(&self, queue: Arc<WakeQueue>) {
        let _ = self.target.set(queue);
        if self.woken.load(Ordering::SeqCst)
            && let Some(queue) = self.target.get()
        {
            queue.push(self.id);
        }
    }

    pub(crate) fn arm(&self, future: StepFuture) {
        self.woken.store(false, Ordering::SeqCst);
        let discarded = {
            let mut slot = self.slot.lock().expect(POISONED);
            (
                slot.future.replace(future),
                std::mem::replace(&mut slot.outcome, Outcome::Pending),
            )
        };
        drop(discarded);
    }

    pub(crate) fn state(&self) -> StepState {
        match &self.slot.lock().expect(POISONED).outcome {
            Outcome::Pending => StepState::Pending,
            Outcome::Ready => StepState::Ready,
            Outcome::Failed(_) => StepState::Failed,
            Outcome::Cancelled(_) => StepState::Cancelled,
        }
    }

    pub(crate) fn error(&self, py: Python<'_>) -> Option<PyErr> {
        match &self.slot.lock().expect(POISONED).outcome {
            Outcome::Failed(error) => Some(error.clone_ref(py)),
            Outcome::Cancelled(Some(message)) => {
                Some(CancelledError::new_err((message.clone_ref(py),)))
            }
            Outcome::Cancelled(None) => Some(CancelledError::new_err(())),
            _ => None,
        }
    }

    pub(crate) fn take_future(&self) -> Option<StepFuture> {
        self.slot.lock().expect(POISONED).future.take()
    }

    pub(crate) fn close(&self) {
        let discarded = {
            let mut slot = self.slot.lock().expect(POISONED);
            let future = slot.future.take();
            if matches!(slot.outcome, Outcome::Pending) {
                slot.outcome = Outcome::Failed(PyRuntimeError::new_err("native execution closed"));
            }
            future
        };
        drop(discarded);
        if let Some(queue) = self.target.get() {
            queue.push(self.id);
        }
    }
}

impl Wake for StepCore {
    fn wake(self: Arc<Self>) {
        self.wake_by_ref();
    }

    fn wake_by_ref(self: &Arc<Self>) {
        self.woken.store(true, Ordering::SeqCst);
        if let Some(queue) = self.target.get() {
            queue.push(self.id);
        }
    }
}

pub(crate) struct WakeQueue {
    ready: Mutex<Vec<u64>>,
    armed: AtomicBool,
    writer: UnixStream,
}

impl WakeQueue {
    fn push(&self, id: u64) {
        self.ready.lock().expect(POISONED).push(id);
        if !self.armed.swap(true, Ordering::AcqRel) {
            let _ = (&self.writer).write(&[1]);
        }
    }
}

#[pyclass]
pub(crate) struct LoopDispatcher {
    event_loop: Py<PyAny>,
    reader: UnixStream,
    queue: Arc<WakeQueue>,
    waiting: HashMap<u64, Py<NativeStep>>,
}

#[pymethods]
impl LoopDispatcher {
    fn _drain(slf: &Bound<'_, Self>, py: Python<'_>) -> PyResult<()> {
        let woken = {
            let mut dispatcher = slf.borrow_mut();
            dispatcher.consume_signals()?;
            dispatcher.queue.armed.store(false, Ordering::SeqCst);
            let mut ready = std::mem::take(&mut *dispatcher.queue.ready.lock().expect(POISONED));
            ready.sort_unstable();
            ready.dedup();
            ready
                .into_iter()
                .filter_map(|id| {
                    dispatcher
                        .waiting
                        .get(&id)
                        .map(|step| (id, step.clone_ref(py)))
                })
                .collect::<Vec<_>>()
        };
        let mut first_error = None;
        for (id, step) in woken {
            match NativeStep::advance(step.bind(py), py) {
                Ok(true) => {
                    let removed = slf.borrow_mut().waiting.remove(&id);
                    drop(removed);
                }
                Err(error) => {
                    if first_error.is_none() {
                        first_error = Some(error);
                    }
                    let removed = slf.borrow_mut().waiting.remove(&id);
                    drop(removed);
                }
                Ok(false) => {}
            }
        }
        first_error.map_or(Ok(()), Err)
    }

    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        for step in self.waiting.values() {
            visit.call(step)?;
        }
        visit.call(&self.event_loop)
    }

    fn __clear__(slf: &Bound<'_, Self>) {
        let py = slf.py();
        let (waiting, event_loop) = {
            let mut dispatcher = slf.borrow_mut();
            (
                std::mem::take(&mut dispatcher.waiting),
                std::mem::replace(&mut dispatcher.event_loop, py.None()),
            )
        };
        drop(waiting);
        drop(event_loop);
    }
}

impl LoopDispatcher {
    fn consume_signals(&mut self) -> PyResult<()> {
        let mut signals = [0_u8; 64];
        loop {
            match self.reader.read(&mut signals) {
                Ok(0) => return Ok(()),
                Ok(_) => {}
                Err(error) if error.kind() == ErrorKind::Interrupted => {}
                Err(error) if error.kind() == ErrorKind::WouldBlock => return Ok(()),
                Err(error) => return Err(error.into()),
            }
        }
    }
}

thread_local! {
    static DISPATCHERS: RefCell<Vec<Py<LoopDispatcher>>> = const { RefCell::new(Vec::new()) };
    static UNSUPPORTED_LOOPS: RefCell<Vec<Py<PyAny>>> = const { RefCell::new(Vec::new()) };
}

pub(crate) fn dispatcher_for(
    py: Python<'_>,
    event_loop: &Bound<'_, PyAny>,
) -> PyResult<Option<Py<LoopDispatcher>>> {
    let known = DISPATCHERS.with_borrow(|dispatchers| {
        dispatchers
            .iter()
            .find(|dispatcher| {
                dispatcher
                    .bind(py)
                    .borrow()
                    .event_loop
                    .bind(py)
                    .is(event_loop)
            })
            .map(|dispatcher| dispatcher.clone_ref(py))
    });
    if let Some(dispatcher) = known {
        return Ok(Some(dispatcher));
    }
    let unsupported = UNSUPPORTED_LOOPS
        .with_borrow(|loops| loops.iter().any(|loop_| loop_.bind(py).is(event_loop)));
    if unsupported {
        return Ok(None);
    }
    prune_closed_loops(py)?;

    let (reader, writer) = UnixStream::pair()?;
    reader.set_nonblocking(true)?;
    writer.set_nonblocking(true)?;
    let descriptor = reader.as_raw_fd();
    let dispatcher = Py::new(
        py,
        LoopDispatcher {
            event_loop: event_loop.clone().unbind(),
            reader,
            queue: Arc::new(WakeQueue {
                ready: Mutex::new(Vec::new()),
                armed: AtomicBool::new(false),
                writer,
            }),
            waiting: HashMap::new(),
        },
    )?;
    let drain = dispatcher.bind(py).getattr("_drain")?;
    if let Err(error) = event_loop.call_method1("add_reader", (descriptor, drain)) {
        if error.is_instance_of::<PyNotImplementedError>(py) {
            UNSUPPORTED_LOOPS.with_borrow_mut(|loops| {
                loops.push(event_loop.clone().unbind());
            });
            return Ok(None);
        }
        return Err(error);
    }
    DISPATCHERS.with_borrow_mut(|dispatchers| dispatchers.push(dispatcher.clone_ref(py)));
    Ok(Some(dispatcher))
}

fn prune_closed_loops(py: Python<'_>) -> PyResult<()> {
    let known = DISPATCHERS.with_borrow(|dispatchers| {
        dispatchers
            .iter()
            .map(|dispatcher| dispatcher.clone_ref(py))
            .collect::<Vec<_>>()
    });
    let mut closed_dispatchers = Vec::new();
    for dispatcher in &known {
        let event_loop = dispatcher.bind(py).borrow().event_loop.clone_ref(py);
        if event_loop.bind(py).call_method0("is_closed")?.is_truthy()? {
            closed_dispatchers.push(dispatcher.as_ptr());
        }
    }
    DISPATCHERS.with_borrow_mut(|dispatchers| {
        dispatchers.retain(|dispatcher| !closed_dispatchers.contains(&dispatcher.as_ptr()));
    });
    let unsupported = UNSUPPORTED_LOOPS.with_borrow(|loops| {
        loops
            .iter()
            .map(|event_loop| event_loop.clone_ref(py))
            .collect::<Vec<_>>()
    });
    let mut closed_loops = Vec::new();
    for event_loop in &unsupported {
        if event_loop.bind(py).call_method0("is_closed")?.is_truthy()? {
            closed_loops.push(event_loop.as_ptr());
        }
    }
    UNSUPPORTED_LOOPS.with_borrow_mut(|loops| {
        loops.retain(|event_loop| !closed_loops.contains(&event_loop.as_ptr()));
    });
    Ok(())
}

#[pyclass]
pub(crate) struct NativeStep {
    core: Arc<StepCore>,
    dispatcher: Py<LoopDispatcher>,
    event_loop: Py<PyAny>,
    callbacks: Vec<(Py<PyAny>, Option<Py<PyAny>>)>,
    blocking: bool,
}

impl NativeStep {
    fn new(core: Arc<StepCore>, dispatcher: Py<LoopDispatcher>, event_loop: Py<PyAny>) -> Self {
        Self {
            core,
            dispatcher,
            event_loop,
            callbacks: Vec::new(),
            blocking: false,
        }
    }

    fn advance(slf: &Bound<'_, Self>, py: Python<'_>) -> PyResult<bool> {
        let core = Arc::clone(&slf.borrow().core);
        poll_step(py, &core)?;
        if core.state() == StepState::Pending {
            return Ok(false);
        }
        Self::schedule_callbacks(slf, py)?;
        Ok(true)
    }

    fn rearm(slf: &Bound<'_, Self>) -> PyResult<()> {
        let mut step = slf.borrow_mut();
        if !step.callbacks.is_empty() {
            return Err(missing_state());
        }
        step.blocking = false;
        Ok(())
    }

    fn schedule_callbacks(slf: &Bound<'_, Self>, py: Python<'_>) -> PyResult<()> {
        let (callbacks, event_loop) = {
            let mut step = slf.borrow_mut();
            (
                std::mem::take(&mut step.callbacks),
                step.event_loop.clone_ref(py),
            )
        };
        for (callback, context) in callbacks {
            schedule_callback(py, event_loop.bind(py), slf, &callback, context.as_ref())?;
        }
        Ok(())
    }
}

fn schedule_callback(
    py: Python<'_>,
    event_loop: &Bound<'_, PyAny>,
    step: &Bound<'_, NativeStep>,
    callback: &Py<PyAny>,
    context: Option<&Py<PyAny>>,
) -> PyResult<()> {
    match context {
        Some(context) => {
            let keywords = PyDict::new(py);
            keywords.set_item("context", context)?;
            event_loop.call_method("call_soon", (callback, step), Some(&keywords))?;
        }
        None => {
            event_loop.call_method1("call_soon", (callback, step))?;
        }
    }
    Ok(())
}

#[pymethods]
impl NativeStep {
    fn __await__(slf: Py<Self>) -> Py<Self> {
        slf
    }

    fn __iter__(slf: Py<Self>) -> Py<Self> {
        slf
    }

    fn __next__(slf: &Bound<'_, Self>, py: Python<'_>) -> PyResult<Option<Py<PyAny>>> {
        let core = Arc::clone(&slf.borrow().core);
        match core.state() {
            StepState::Pending => {
                slf.borrow_mut().blocking = true;
                Ok(Some(slf.clone().into_any().unbind()))
            }
            StepState::Ready => Ok(None),
            _ => Err(core.error(py).unwrap_or_else(missing_state)),
        }
    }

    #[getter]
    fn _asyncio_future_blocking(&self) -> bool {
        self.blocking
    }

    #[setter(_asyncio_future_blocking)]
    fn set_blocking(&mut self, blocking: bool) {
        self.blocking = blocking;
    }

    fn get_loop(&self, py: Python<'_>) -> Py<PyAny> {
        self.event_loop.clone_ref(py)
    }

    fn done(&self) -> bool {
        self.core.state() != StepState::Pending
    }

    fn cancelled(&self) -> bool {
        self.core.state() == StepState::Cancelled
    }

    fn result(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        match self.core.state() {
            StepState::Ready => Ok(py.None()),
            StepState::Pending => Err(InvalidStateError::new_err("Result is not set.")),
            _ => Err(self.core.error(py).unwrap_or_else(missing_state)),
        }
    }

    fn exception(&self, py: Python<'_>) -> PyResult<Option<Py<PyAny>>> {
        match self.core.state() {
            StepState::Ready => Ok(None),
            StepState::Pending => Err(InvalidStateError::new_err("Exception is not set.")),
            StepState::Failed => Ok(self
                .core
                .error(py)
                .map(|error| error.into_value(py).into_any())),
            StepState::Cancelled => Err(self.core.error(py).unwrap_or_else(missing_state)),
        }
    }

    #[pyo3(signature = (callback, *, context = None))]
    fn add_done_callback(
        slf: &Bound<'_, Self>,
        py: Python<'_>,
        callback: Py<PyAny>,
        context: Option<Py<PyAny>>,
    ) -> PyResult<()> {
        let (settled, event_loop) = {
            let step = slf.borrow();
            (
                step.core.state() != StepState::Pending,
                step.event_loop.clone_ref(py),
            )
        };
        if settled {
            return schedule_callback(py, event_loop.bind(py), slf, &callback, context.as_ref());
        }
        slf.borrow_mut().callbacks.push((callback, context));
        Ok(())
    }

    fn remove_done_callback(
        slf: &Bound<'_, Self>,
        py: Python<'_>,
        callback: &Bound<'_, PyAny>,
    ) -> PyResult<usize> {
        let registered = std::mem::take(&mut slf.borrow_mut().callbacks);
        let matching = registered
            .iter()
            .map(|(registered_callback, _)| registered_callback.bind(py).eq(callback))
            .collect::<PyResult<Vec<_>>>();
        let matching = match matching {
            Ok(matching) => matching,
            Err(error) => {
                slf.borrow_mut().callbacks.extend(registered);
                return Err(error);
            }
        };
        let (removed, kept): (Vec<_>, Vec<_>) = registered
            .into_iter()
            .zip(matching)
            .partition(|(_, matches)| *matches);
        let removed = removed.len();
        let kept = kept
            .into_iter()
            .map(|(callback, _)| callback)
            .collect::<Vec<_>>();
        drop(slf.borrow_mut().callbacks.splice(..0, kept));
        Ok(removed)
    }

    #[pyo3(signature = (msg = None))]
    fn cancel(slf: &Bound<'_, Self>, py: Python<'_>, msg: Option<Py<PyAny>>) -> PyResult<bool> {
        let core = Arc::clone(&slf.borrow().core);
        let discarded = {
            let mut slot = core.slot.lock().expect(POISONED);
            if !matches!(slot.outcome, Outcome::Pending) {
                return Ok(false);
            }
            slot.outcome = Outcome::Cancelled(msg);
            slot.future.take()
        };
        drop(discarded);
        let dispatcher = slf.borrow().dispatcher.clone_ref(py);
        let removed = dispatcher.bind(py).borrow_mut().waiting.remove(&core.id());
        drop(removed);
        Self::schedule_callbacks(slf, py)?;
        Ok(true)
    }

    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        for (callback, context) in &self.callbacks {
            visit.call(callback)?;
            if let Some(context) = context {
                visit.call(context)?;
            }
        }
        visit.call(&self.dispatcher)?;
        visit.call(&self.event_loop)
    }

    fn __clear__(slf: &Bound<'_, Self>) {
        let py = slf.py();
        let (callbacks, event_loop, dispatcher, id) = {
            let mut step = slf.borrow_mut();
            (
                std::mem::take(&mut step.callbacks),
                std::mem::replace(&mut step.event_loop, py.None()),
                step.dispatcher.clone_ref(py),
                step.core.id(),
            )
        };
        let removed = dispatcher.bind(py).borrow_mut().waiting.remove(&id);
        drop((callbacks, event_loop, removed));
    }
}

pub(crate) fn poll_step(py: Python<'_>, core: &Arc<StepCore>) -> PyResult<()> {
    let Some(mut future) = core.take_future() else {
        return Ok(());
    };
    let waker = Waker::from(Arc::clone(core));
    let polled = poll_with_waker(py, future.as_mut(), &waker);
    let discarded = {
        let mut slot = core.slot.lock().expect(POISONED);
        let settled = !matches!(slot.outcome, Outcome::Pending);
        match polled {
            Ok(Poll::Pending) if !settled => {
                slot.future = Some(future);
                None
            }
            Ok(Poll::Pending) => Some(future),
            Ok(Poll::Ready(())) => {
                if !settled {
                    slot.outcome = Outcome::Ready;
                }
                Some(future)
            }
            Err(error) => {
                if !settled {
                    slot.outcome = Outcome::Failed(error);
                }
                Some(future)
            }
        }
    };
    drop(discarded);
    Ok(())
}

pub(crate) fn running_loop(py: Python<'_>) -> PyResult<Bound<'_, PyAny>> {
    static GET_RUNNING_LOOP: OnceLock<Py<PyAny>> = OnceLock::new();
    let get_running_loop = match GET_RUNNING_LOOP.get() {
        Some(cached) => cached,
        None => {
            let resolved = py.import("asyncio")?.getattr("get_running_loop")?.unbind();
            GET_RUNNING_LOOP.get_or_init(|| resolved)
        }
    };
    get_running_loop.bind(py).call0()
}

pub(crate) fn park(py: Python<'_>, step: &Py<NativeStep>, id: u64) {
    let dispatcher = step.bind(py).borrow().dispatcher.clone_ref(py);
    dispatcher
        .bind(py)
        .borrow_mut()
        .waiting
        .insert(id, step.clone_ref(py));
}

pub(crate) fn unpark(py: Python<'_>, step: &Py<NativeStep>, id: u64) {
    let dispatcher = step.bind(py).borrow().dispatcher.clone_ref(py);
    let removed = dispatcher.bind(py).borrow_mut().waiting.remove(&id);
    drop(removed);
}

pub(crate) fn reuse<'py>(
    py: Python<'py>,
    step: &Py<NativeStep>,
    event_loop: &Bound<'py, PyAny>,
) -> PyResult<bool> {
    let bound = step.bind(py);
    if !bound.borrow().event_loop.bind(py).is(event_loop) {
        return Ok(false);
    }
    NativeStep::rearm(bound)?;
    Ok(true)
}

pub(crate) fn new_step(
    py: Python<'_>,
    core: &Arc<StepCore>,
    dispatcher: &Py<LoopDispatcher>,
    event_loop: &Bound<'_, PyAny>,
) -> PyResult<Py<NativeStep>> {
    core.bind(Arc::clone(&dispatcher.bind(py).borrow().queue));
    Py::new(
        py,
        NativeStep::new(
            Arc::clone(core),
            dispatcher.clone_ref(py),
            event_loop.clone().unbind(),
        ),
    )
}
