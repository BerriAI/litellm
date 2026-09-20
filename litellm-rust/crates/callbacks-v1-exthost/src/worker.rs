//! The blocking reference client of one worker. It starts the worker on a unix socket it
//! listens on (so nothing but a path is handed over, and an attached sidecar connects the
//! same way), reads the hello, and then does exactly what a [`CallSession`] says: sends an
//! emission's envelope to its observers and asks each interceptor for its patch in turn.
//! The session, not this client, decides the protocol and validates every patch.

use std::{
    io,
    os::unix::net::{UnixListener, UnixStream},
    path::Path,
    process::{Child, Command},
};

use litellm_callbacks_v1::{Emission, Interception, PatchError, SCHEMA_V1, Subscription};
use thiserror::Error;

use crate::protocol::{FromWorker, ToWorker, read_frame, write_frame};

#[derive(Debug, Error)]
pub enum ExtHostError {
    #[error("extension host i/o: {0}")]
    Io(#[from] io::Error),
    #[error("extension host protocol: {0}")]
    Protocol(String),
    #[error("worker speaks schema {0}, this gateway speaks {SCHEMA_V1}")]
    Schema(u32),
    /// The interceptor raised. Whether that fails the call or skips the patch is the
    /// gateway's policy for this subscriber, not the worker's.
    #[error("callback {subscriber}: {error_class}: {message}")]
    Interceptor {
        subscriber: String,
        error_class: String,
        message: String,
    },
    #[error("callback {subscriber}: {error}")]
    Patch {
        subscriber: String,
        error: PatchError,
    },
}

/// An observer's swallowed failure, as the worker reported it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Report {
    pub subscriber: String,
    pub event: Option<String>,
    pub message: String,
}

pub struct Worker {
    stream: UnixStream,
    child: Option<Child>,
    subscriptions: Vec<Subscription>,
    reports: Vec<Report>,
    next_id: u64,
}

impl Worker {
    /// Starts `command` with `--connect <socket>` appended and waits for its hello. What
    /// the worker can reach is whatever `command` was given: its environment, its working
    /// directory, its user. This crate grants nothing.
    pub fn spawn(mut command: Command, socket: &Path) -> Result<Self, ExtHostError> {
        let listener = UnixListener::bind(socket)?;
        let child = command.arg("--connect").arg(socket).spawn()?;
        let (stream, _) = listener.accept()?;
        Self::greeted(stream, Some(child))
    }

    /// Takes a worker that connected by itself, a sidecar for example.
    pub fn attach(stream: UnixStream) -> Result<Self, ExtHostError> {
        Self::greeted(stream, None)
    }

    fn greeted(mut stream: UnixStream, child: Option<Child>) -> Result<Self, ExtHostError> {
        match read_frame(&mut stream)? {
            Some(FromWorker::Hello {
                schema: SCHEMA_V1,
                subscriptions,
            }) => Ok(Self {
                stream,
                child,
                subscriptions,
                reports: Vec::new(),
                next_id: 0,
            }),
            Some(FromWorker::Hello { schema, .. }) => Err(ExtHostError::Schema(schema)),
            other => Err(ExtHostError::Protocol(format!(
                "expected hello, got {other:?}"
            ))),
        }
    }

    /// What the worker's callbacks subscribed to: the session's subscriptions for every
    /// call this worker serves.
    pub fn subscriptions(&self) -> &[Subscription] {
        &self.subscriptions
    }

    fn name(&self, subscriber: usize) -> String {
        self.subscriptions
            .get(subscriber)
            .map_or_else(|| format!("#{subscriber}"), |found| found.name.clone())
    }

    fn id(&mut self) -> u64 {
        self.next_id += 1;
        self.next_id
    }

    /// Sends the envelope to each of its observers and returns at once: observers are
    /// asynchronous to the call.
    pub fn observe(&mut self, emission: &Emission) -> Result<(), ExtHostError> {
        for &subscriber in &emission.observers {
            write_frame(
                &mut self.stream,
                &ToWorker::Event {
                    subscriber,
                    envelope: &emission.envelope,
                },
            )?;
        }
        Ok(())
    }

    /// The next frame that is not a report; reports are kept for [`Worker::flush`].
    fn answer(&mut self) -> Result<FromWorker, ExtHostError> {
        loop {
            match read_frame(&mut self.stream)? {
                Some(FromWorker::Report {
                    subscriber,
                    event,
                    message,
                }) => {
                    let subscriber = self.name(subscriber);
                    self.reports.push(Report {
                        subscriber,
                        event,
                        message,
                    });
                }
                Some(frame) => return Ok(frame),
                None => return Err(ExtHostError::Protocol("worker closed".to_string())),
            }
        }
    }

    /// Asks every interceptor for its patch, in the session's order, and folds each into
    /// the interception, which validates it.
    pub fn intercept(&mut self, interception: Interception) -> Result<Interception, ExtHostError> {
        let mut current = interception;
        while let Some(turn) = current.turn() {
            let id = self.id();
            write_frame(
                &mut self.stream,
                &ToWorker::Intercept {
                    id,
                    subscriber: turn.subscriber,
                    request: &turn.request,
                },
            )?;
            let subscriber = self.name(turn.subscriber);
            current = match self.answer()? {
                FromWorker::Patch {
                    id: answered,
                    patch,
                } if answered == id => current
                    .patched(patch.unwrap_or_default())
                    .map_err(|error| ExtHostError::Patch { subscriber, error })?,
                FromWorker::Error {
                    id: answered,
                    error_class,
                    message,
                } if answered == id => {
                    return Err(ExtHostError::Interceptor {
                        subscriber,
                        error_class,
                        message,
                    });
                }
                other => {
                    return Err(ExtHostError::Protocol(format!(
                        "expected the answer to intercept {id}, got {other:?}"
                    )));
                }
            };
        }
        Ok(current)
    }

    /// Waits until the worker has handled every event sent so far, and returns what its
    /// observers reported since the last flush.
    pub fn flush(&mut self) -> Result<Vec<Report>, ExtHostError> {
        let id = self.id();
        write_frame(&mut self.stream, &ToWorker::Flush { id })?;
        match self.answer()? {
            FromWorker::Flushed { id: answered } if answered == id => {
                Ok(std::mem::take(&mut self.reports))
            }
            other => Err(ExtHostError::Protocol(format!(
                "expected flushed {id}, got {other:?}"
            ))),
        }
    }
}

impl Drop for Worker {
    /// Closing the socket is how a worker is told to stop; it exits once its queues drain.
    fn drop(&mut self) {
        let _ = self.stream.shutdown(std::net::Shutdown::Both);
        if let Some(child) = self.child.as_mut() {
            let _ = child.wait();
        }
    }
}
