use std::future::Future;
use std::io::{self, Read, Write};
use std::os::fd::AsRawFd;
use std::os::unix::net::UnixStream;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll, Wake, Waker};

use pyo3::exceptions::{PyRuntimeError, PyStopIteration};
use pyo3::prelude::*;
use pyo3::types::{PyCFunction, PyIterator};

use crate::execution::runtime;
use crate::{panic_to_pyerr, release_gil};

type Native = Pin<Box<dyn Future<Output = PyResult<()>> + Send>>;

/// The first poll of a native future: complete, or parked behind an awaitable.
pub enum Started {
    Ready,
    Suspended(Py<InlineAwait>),
}

enum Stage {
    /// The last poll returned pending with this waker registered; the next `__next__` parks
    /// the task on it without polling again.
    Registered(Arc<LoopWaker>),
    /// The task is parked, or was released with `None`; the next `__next__` polls again.
    Parked,
}

/// A Python awaitable that polls a native future on the event loop's own thread. A pending
/// poll parks the awaiting task on one `asyncio.Future`, and the native waker completes it
/// through `call_soon_threadsafe`, so a suspension costs one Future and one loop callback
/// rather than a Tokio spawn, a blocking-pool hop and a cancellation channel.
#[pyclass]
pub struct InlineAwait {
    future: Mutex<Option<Native>>,
    stage: Stage,
}

impl InlineAwait {
    /// Polls `future` once on the caller's thread: ready, or the awaitable that parks the
    /// caller's task until its waker fires.
    pub fn start(
        py: Python<'_>,
        future: impl Future<Output = PyResult<()>> + Send + 'static,
    ) -> PyResult<Started> {
        let mut future: Native = Box::pin(future);
        let waker = Arc::new(LoopWaker::default());
        match poll_native(py, &mut future, &waker)? {
            Poll::Ready(()) => Ok(Started::Ready),
            Poll::Pending => Py::new(
                py,
                Self {
                    future: Mutex::new(Some(future)),
                    stage: Stage::Registered(waker),
                },
            )
            .map(Started::Suspended),
        }
    }

    fn step(&mut self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let waker = match std::mem::replace(&mut self.stage, Stage::Parked) {
            Stage::Registered(waker) => waker,
            Stage::Parked => {
                let slot = self.future.get_mut().expect("native future slot poisoned");
                let Some(future) = slot.as_mut() else {
                    return Err(PyRuntimeError::new_err(
                        "native awaitable already completed",
                    ));
                };
                let waker = Arc::new(LoopWaker::default());
                let polled = poll_native(py, future, &waker);
                if !matches!(polled, Ok(Poll::Pending)) {
                    *slot = None;
                }
                match polled? {
                    Poll::Ready(()) => return Err(PyStopIteration::new_err((py.None(),))),
                    Poll::Pending => waker,
                }
            }
        };
        match waker.park(py)? {
            Some(parked) => Ok(parked),
            None => Ok(py.None()),
        }
    }
}

fn poll_native(py: Python<'_>, future: &mut Native, waker: &Arc<LoopWaker>) -> PyResult<Poll<()>> {
    let task_waker = Waker::from(Arc::clone(waker));
    let runtime = runtime()?;
    let polled = release_gil(py, || {
        let _runtime = runtime.enter();
        catch_unwind(AssertUnwindSafe(|| {
            future.as_mut().poll(&mut Context::from_waker(&task_waker))
        }))
        .map_err(panic_to_pyerr)
    })?;
    match polled {
        Poll::Ready(result) => result.map(Poll::Ready),
        Poll::Pending => Ok(Poll::Pending),
    }
}

#[pymethods]
impl InlineAwait {
    fn __await__(slf: Py<Self>) -> Py<Self> {
        slf
    }

    fn __next__(&mut self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        self.step(py)
    }

    fn send(&mut self, py: Python<'_>, _value: Py<PyAny>) -> PyResult<Py<PyAny>> {
        self.step(py)
    }

    fn throw(&mut self, py: Python<'_>, error: Py<PyAny>) -> PyResult<Py<PyAny>> {
        self.close();
        Err(PyErr::from_value(error.into_bound(py)))
    }

    fn close(&mut self) {
        drop(
            self.future
                .get_mut()
                .expect("native future slot poisoned")
                .take(),
        );
    }
}

enum Parking {
    Fresh,
    Woken,
    Parked(Parked),
}

/// The waker for one poll: parks the task on an `asyncio.Future` created lazily, so a wake
/// that lands before the park costs nothing more than yielding `None` to the loop. A wake
/// after the park never touches Python: it queues this waker on the loop's [`LoopSignal`]
/// and writes one byte, and the loop thread resolves the Future when it drains the signal.
struct LoopWaker(Mutex<Parking>);

impl Default for LoopWaker {
    fn default() -> Self {
        Self(Mutex::new(Parking::Fresh))
    }
}

impl LoopWaker {
    /// The `asyncio.Future` for the task to await, as its own `__iter__` yields it, or `None`
    /// when the wake already arrived.
    fn park(&self, py: Python<'_>) -> PyResult<Option<Py<PyAny>>> {
        let parked = Parked::new(py)?;
        let future = parked.future.clone_ref(py);
        {
            let mut parking = self.0.lock().expect("parking state poisoned");
            match *parking {
                Parking::Fresh => *parking = Parking::Parked(parked),
                Parking::Woken => return Ok(None),
                Parking::Parked(_) => {
                    return Err(PyRuntimeError::new_err("native read parked twice"));
                }
            }
        }
        PyIterator::from_object(future.bind(py))?
            .next()
            .transpose()
            .map(|yielded| yielded.map(Bound::unbind))
    }

    /// The parked Future, if any, for the loop thread to resolve.
    fn parked_future(&self, py: Python<'_>) -> Option<Py<PyAny>> {
        match &*self.0.lock().expect("parking state poisoned") {
            Parking::Parked(parked) => Some(parked.future.clone_ref(py)),
            Parking::Fresh | Parking::Woken => None,
        }
    }
}

impl Wake for LoopWaker {
    fn wake(self: Arc<Self>) {
        self.wake_by_ref();
    }

    fn wake_by_ref(self: &Arc<Self>) {
        let signal = {
            let mut parking = self.0.lock().expect("parking state poisoned");
            match &*parking {
                Parking::Parked(parked) => Arc::clone(&parked.signal),
                Parking::Fresh => {
                    *parking = Parking::Woken;
                    return;
                }
                Parking::Woken => return,
            }
        };
        signal.fire(Arc::clone(self));
    }
}

struct Parked {
    signal: Arc<LoopSignal>,
    future: Py<PyAny>,
}

impl Parked {
    fn new(py: Python<'_>) -> PyResult<Self> {
        let event_loop = py.import("asyncio")?.call_method0("get_running_loop")?;
        let signal = LoopSignal::for_loop(py, &event_loop)?;
        let future = event_loop.call_method0("create_future")?;
        Ok(Self {
            signal,
            future: future.unbind(),
        })
    }
}

/// One event loop's wake channel: a socketpair whose read end the loop watches. Wakes from
/// native threads queue their waker and write a byte; the loop's reader callback drains the
/// socket and resolves every queued Future in one loop turn, so concurrent wakes coalesce.
struct LoopSignal {
    loop_ref: Py<PyAny>,
    reader: UnixStream,
    writer: UnixStream,
    fired: Mutex<Vec<Arc<LoopWaker>>>,
    signalled: AtomicBool,
}

static SIGNALS: Mutex<Vec<Arc<LoopSignal>>> = Mutex::new(Vec::new());

impl LoopSignal {
    fn for_loop(py: Python<'_>, event_loop: &Bound<'_, PyAny>) -> PyResult<Arc<Self>> {
        let mut signals = SIGNALS.lock().expect("loop signal registry poisoned");
        let mut existing = None;
        for signal in signals.iter() {
            if signal.loop_ref.bind(py).call0()?.is(event_loop) {
                existing = Some(Arc::clone(signal));
                break;
            }
        }
        if let Some(signal) = existing {
            return Ok(signal);
        }
        signals.retain(|signal| {
            signal
                .loop_ref
                .bind(py)
                .call0()
                .map(|alive| !alive.is_none())
                .unwrap_or(false)
        });
        let signal = Arc::new(Self::attach(py, event_loop)?);
        signal.watch(py, event_loop)?;
        signals.push(Arc::clone(&signal));
        Ok(signal)
    }

    fn attach(py: Python<'_>, event_loop: &Bound<'_, PyAny>) -> PyResult<Self> {
        let (reader, writer) = UnixStream::pair()?;
        reader.set_nonblocking(true)?;
        writer.set_nonblocking(true)?;
        let loop_ref = py
            .import("weakref")?
            .call_method1("ref", (event_loop,))?
            .unbind();
        Ok(Self {
            loop_ref,
            reader,
            writer,
            fired: Mutex::new(Vec::new()),
            signalled: AtomicBool::new(false),
        })
    }

    fn watch(self: &Arc<Self>, py: Python<'_>, event_loop: &Bound<'_, PyAny>) -> PyResult<()> {
        let signal = Arc::clone(self);
        let drain = PyCFunction::new_closure(py, None, None, move |_, _| {
            Python::attach(|py| signal.drain(py))
        })?;
        event_loop.call_method1("add_reader", (self.reader.as_raw_fd(), drain))?;
        Ok(())
    }

    /// Queues `waker` for the loop thread and wakes the loop, without touching Python.
    fn fire(&self, waker: Arc<LoopWaker>) {
        self.fired.lock().expect("fired list poisoned").push(waker);
        if self.signalled.swap(true, Ordering::AcqRel) {
            return;
        }
        match (&self.writer).write(&[1]) {
            Ok(_) => {}
            Err(error) if error.kind() == io::ErrorKind::WouldBlock => {}
            Err(error) => panic!("waking the event loop failed: {error}"),
        }
    }

    /// Runs on the loop thread when the socket is readable.
    fn drain(&self, py: Python<'_>) -> PyResult<()> {
        let mut sink = [0_u8; 64];
        loop {
            match (&self.reader).read(&mut sink) {
                Ok(0) => break,
                Ok(_) => continue,
                Err(error) if error.kind() == io::ErrorKind::WouldBlock => break,
                Err(error) => return Err(error.into()),
            }
        }
        self.signalled.store(false, Ordering::Release);
        let fired = std::mem::take(&mut *self.fired.lock().expect("fired list poisoned"));
        for waker in fired {
            let Some(future) = waker.parked_future(py) else {
                continue;
            };
            let future = future.bind(py);
            if !future.call_method0("done")?.extract::<bool>()? {
                future.call_method1("set_result", (py.None(),))?;
            }
        }
        Ok(())
    }
}
