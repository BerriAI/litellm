use std::{
    future::Future,
    sync::Arc,
    time::{SystemTime, UNIX_EPOCH},
};

use futures_util::TryStreamExt;

use crate::{call::CallOutput, hooks::MachineEvent};

/// Seconds since the Unix epoch, on one clock for every host.
pub fn epoch_seconds() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs_f64())
        .unwrap_or(0.0)
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Timing {
    pub start_time: f64,
    pub end_time: f64,
}

/// Whether a failure surfaced inside the call, including a host op the call asked for,
/// or in a host step around it (preparing the arguments, finalizing the response).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FailureOrigin {
    Call,
    Host,
}

/// What an in-process host observes: the machine's own events between the driver's
/// start and terminal ones.
#[derive(Clone, Debug, PartialEq)]
pub enum CallEvent {
    Started {
        start_time: f64,
    },
    Machine(MachineEvent),
    Succeeded {
        timing: Timing,
    },
    Failed {
        timing: Timing,
        origin: FailureOrigin,
    },
    Cancelled {
        timing: Timing,
    },
}

pub trait CallObserver: Send + Sync {
    fn observe(&self, event: CallEvent);
}

struct CallGuard {
    observer: Option<Arc<dyn CallObserver>>,
    started_at: f64,
}

impl CallGuard {
    fn new(observer: Arc<dyn CallObserver>) -> Self {
        let started_at = epoch_seconds();
        observer.observe(CallEvent::Started {
            start_time: started_at,
        });
        Self {
            observer: Some(observer),
            started_at,
        }
    }

    fn timing(&self) -> Timing {
        Timing {
            start_time: self.started_at,
            end_time: epoch_seconds(),
        }
    }

    fn finish(mut self, failed: bool) {
        if let Some(observer) = self.observer.take() {
            observer.observe(if failed {
                CallEvent::Failed {
                    timing: self.timing(),
                    origin: FailureOrigin::Call,
                }
            } else {
                CallEvent::Succeeded {
                    timing: self.timing(),
                }
            });
        }
    }
}

impl Drop for CallGuard {
    fn drop(&mut self) {
        if let Some(observer) = self.observer.take() {
            observer.observe(CallEvent::Cancelled {
                timing: self.timing(),
            });
        }
    }
}

pub async fn observe_call<R, H, C, E>(
    observer: Option<Arc<dyn CallObserver>>,
    execute: impl Future<Output = Result<CallOutput<R, H, C, E>, E>>,
) -> Result<CallOutput<R, H, C, E>, E>
where
    C: Send + 'static,
    E: Send + 'static,
{
    let Some(observer) = observer else {
        return execute.await;
    };
    let guard = CallGuard::new(observer);
    match execute.await {
        Err(error) => {
            guard.finish(true);
            Err(error)
        }
        Ok(CallOutput::Complete(response)) => {
            guard.finish(false);
            Ok(CallOutput::Complete(response))
        }
        Ok(CallOutput::Stream { head, chunks }) => {
            let stream = futures_util::stream::try_unfold(
                (chunks, guard),
                |(mut chunks, guard)| async move {
                    match chunks.try_next().await {
                        Ok(Some(chunk)) => Ok(Some((chunk, (chunks, guard)))),
                        Ok(None) => {
                            guard.finish(false);
                            Ok(None)
                        }
                        Err(error) => {
                            guard.finish(true);
                            Err(error)
                        }
                    }
                },
            );
            Ok(CallOutput::Stream {
                head,
                chunks: Box::pin(stream),
            })
        }
    }
}

pub async fn observe_unary<R, E>(
    observer: Option<Arc<dyn CallObserver>>,
    execute: impl Future<Output = Result<R, E>>,
) -> Result<R, E> {
    let Some(observer) = observer else {
        return execute.await;
    };
    let guard = CallGuard::new(observer);
    let result = execute.await;
    guard.finish(result.is_err());
    result
}
