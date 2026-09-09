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
    completion: Mutex<Option<PythonStreamCompletion>>,
    closed: AtomicBool,
    polling: AtomicBool,
    notify: Notify,
}

impl State {
    fn close(&self) {
        self.closed.store(true, Ordering::Release);
        self.notify.notify_one();
        if let Ok(mut stream) = self.stream.try_lock() {
            stream.take();
        }
    }

    async fn finish(&self) -> PyResult<()> {
        let mut completion = self.completion.lock().await;
        if let Some(future) = completion.as_mut() {
            let result = future.await;
            completion.take();
            result?;
        }
        Ok(())
    }
}

struct PollGuard {
    state: Arc<State>,
    completed: bool,
}

impl Drop for PollGuard {
    fn drop(&mut self) {
        if !self.completed {
            self.state.close();
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
                completion: Mutex::new(Some(completion)),
                closed: AtomicBool::new(false),
                polling: AtomicBool::new(false),
                notify: Notify::new(),
            }),
        }
    }

    pub async fn next_chunk(&self) -> PyResult<Option<Bytes>> {
        let state = &self.state;
        if state.polling.swap(true, Ordering::AcqRel) {
            return Err(PyRuntimeError::new_err(
                "concurrent stream iteration is not supported",
            ));
        }
        let mut guard = PollGuard {
            state: state.clone(),
            completed: false,
        };
        let item = {
            let mut stream = state.stream.lock().await;
            if state.closed.load(Ordering::Acquire) {
                stream.take();
            }
            match stream.as_mut() {
                Some(inner) => tokio::select! {
                    biased;
                    _ = state.notify.notified() => None,
                    item = inner.next() => item,
                },
                None => None,
            }
        };
        match item {
            Some(Ok(bytes)) => {
                guard.completed = true;
                Ok(Some(bytes))
            }
            item => {
                state.close();
                state.finish().await?;
                guard.completed = true;
                item.transpose()
            }
        }
    }

    pub fn close(&self) {
        self.state.close();
    }

    pub async fn aclose(&self) -> PyResult<()> {
        self.close();
        self.state.stream.lock().await.take();
        self.state.finish().await
    }
}
