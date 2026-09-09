use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use bytes::Bytes;
use futures_util::{Stream, StreamExt};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use tokio::sync::{Mutex, Notify};

pub type PythonByteStream = Pin<Box<dyn Stream<Item = PyResult<Bytes>> + Send>>;
pub type PythonStreamCompletion = Pin<Box<dyn Future<Output = PyResult<()>> + Send>>;

struct State {
    stream: Mutex<Option<PythonByteStream>>,
    completion: std::sync::Mutex<Option<PythonStreamCompletion>>,
    completing: Mutex<()>,
    closed: AtomicBool,
    polling: AtomicBool,
    notify: Notify,
    runtime: tokio::runtime::Handle,
    detached: AtomicBool,
}

impl State {
    fn close(&self) {
        self.closed.store(true, Ordering::Release);
        self.notify.notify_one();
        let stream = self
            .stream
            .try_lock()
            .ok()
            .and_then(|mut stream| stream.take());
        drop(stream);
    }

    fn detach(self: &Arc<Self>) {
        self.close();
        if !self.detached.swap(true, Ordering::AcqRel) {
            let state = self.clone();
            self.runtime.spawn(async move {
                let stream = state.stream.lock().await.take();
                drop(stream);
                let _ = state.finish().await;
            });
        }
    }

    async fn finish(&self) -> PyResult<()> {
        let _completing = self.completing.lock().await;
        let mut completion = CompletionLease {
            slot: &self.completion,
            future: self.completion.lock().unwrap().take(),
        };
        if let Some(future) = completion.future.as_mut() {
            let result = future.await;
            drop(completion.future.take());
            result?;
        }
        Ok(())
    }
}

struct CompletionLease<'a> {
    slot: &'a std::sync::Mutex<Option<PythonStreamCompletion>>,
    future: Option<PythonStreamCompletion>,
}

impl Drop for CompletionLease<'_> {
    fn drop(&mut self) {
        if let Some(future) = self.future.take() {
            let displaced = self.slot.lock().unwrap().replace(future);
            drop(displaced);
        }
    }
}

struct PollGuard {
    state: Arc<State>,
    completed: bool,
}

impl Drop for PollGuard {
    fn drop(&mut self) {
        if !self.completed {
            self.state.detach();
        }
        self.state.polling.store(false, Ordering::Release);
    }
}

#[derive(Clone)]
pub struct ByteStreamReader {
    state: Arc<State>,
}

impl ByteStreamReader {
    pub fn new(stream: PythonByteStream, completion: PythonStreamCompletion) -> Self {
        Self {
            state: Arc::new(State {
                stream: Mutex::new(Some(stream)),
                completion: std::sync::Mutex::new(Some(completion)),
                completing: Mutex::new(()),
                closed: AtomicBool::new(false),
                polling: AtomicBool::new(false),
                notify: Notify::new(),
                runtime: tokio::runtime::Handle::try_current()
                    .unwrap_or_else(|_| pyo3_async_runtimes::tokio::get_runtime().handle().clone()),
                detached: AtomicBool::new(false),
            }),
        }
    }

    pub async fn next_chunk(&self) -> PyResult<Option<Bytes>> {
        let state = &self.state;
        if state.closed.load(Ordering::Acquire) {
            return Ok(None);
        }
        if state.polling.swap(true, Ordering::AcqRel) {
            return Err(PyRuntimeError::new_err(
                "concurrent stream iteration is not supported",
            ));
        }
        let mut guard = PollGuard {
            state: state.clone(),
            completed: false,
        };
        let mut stream = state.stream.lock().await.take();
        let item = if state.closed.load(Ordering::Acquire) {
            None
        } else {
            match stream.as_mut() {
                Some(inner) => tokio::select! {
                    biased;
                    _ = state.notify.notified() => None,
                    item = inner.next() => item,
                },
                None => None,
            }
        };
        if matches!(item, Some(Ok(_))) {
            let mut slot = state.stream.lock().await;
            if !state.closed.load(Ordering::Acquire) {
                *slot = stream.take();
            }
        }
        drop(stream);
        match item {
            Some(Ok(bytes)) => {
                guard.completed = true;
                Ok(Some(bytes))
            }
            item => {
                let detached = state.closed.load(Ordering::Acquire);
                state.close();
                if !detached {
                    let completion = state.finish().await;
                    if item.is_none() {
                        completion?;
                    }
                }
                guard.completed = true;
                item.transpose()
            }
        }
    }

    pub fn close(&self) {
        self.state.detach();
    }

    pub async fn aclose(&self) -> PyResult<()> {
        self.close();
        Ok(())
    }
}
