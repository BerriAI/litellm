use std::{
    future::Future,
    time::{SystemTime, UNIX_EPOCH},
};

use futures_util::TryStreamExt;

use crate::{call::CallOutput, hooks::NativeHooks, interceptors::RawResponse};

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

#[derive(Clone, Debug, PartialEq)]
pub enum CallEvent<Response = (), Error = (), Raw = RawResponse> {
    Started {
        start_time: f64,
    },
    Execution(ExecutionEvent<Raw>),
    Succeeded {
        timing: Timing,
        response: Response,
    },
    Failed {
        timing: Timing,
        origin: FailureOrigin,
        error: Error,
    },
    Cancelled {
        timing: Timing,
    },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ExecutionEvent<Raw = RawResponse> {
    ResultReady {
        facts: crate::interceptors::ExecutionFacts,
    },
    ProviderResponseReceived {
        raw: Raw,
    },
}

impl<Response, Error, Raw: std::borrow::Borrow<RawResponse>> CallEvent<Response, Error, Raw> {
    pub fn snapshot(&self) -> CallEvent {
        match self {
            Self::Started { start_time } => CallEvent::Started {
                start_time: *start_time,
            },
            Self::Execution(ExecutionEvent::ProviderResponseReceived { raw }) => {
                CallEvent::Execution(ExecutionEvent::ProviderResponseReceived {
                    raw: raw.borrow().clone(),
                })
            }
            Self::Execution(ExecutionEvent::ResultReady { facts }) => {
                CallEvent::Execution(ExecutionEvent::ResultReady {
                    facts: facts.clone(),
                })
            }
            Self::Succeeded { timing, .. } => CallEvent::Succeeded {
                timing: *timing,
                response: (),
            },
            Self::Failed { timing, origin, .. } => CallEvent::Failed {
                timing: *timing,
                origin: *origin,
                error: (),
            },
            Self::Cancelled { timing } => CallEvent::Cancelled { timing: *timing },
        }
    }
}

struct CallGuard<H: NativeHooks> {
    hooks: Option<H>,
    started_at: f64,
}

impl<H: NativeHooks> CallGuard<H> {
    fn new(hooks: H) -> Self {
        let started_at = epoch_seconds();
        hooks.on_event(&CallEvent::Started {
            start_time: started_at,
        });
        Self {
            hooks: Some(hooks),
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
        if let Some(hooks) = self.hooks.take() {
            hooks.on_event(&if failed {
                CallEvent::Failed {
                    timing: self.timing(),
                    origin: FailureOrigin::Call,
                    error: (),
                }
            } else {
                CallEvent::Succeeded {
                    timing: self.timing(),
                    response: (),
                }
            });
        }
    }
}

impl<H: NativeHooks> Drop for CallGuard<H> {
    fn drop(&mut self) {
        if let Some(hooks) = self.hooks.take() {
            hooks.on_event(&CallEvent::Cancelled {
                timing: self.timing(),
            });
        }
    }
}

/// Reports `Started` and one terminal event around a call whose stream, if any, the caller
/// consumes later: a stream succeeds when it is exhausted and is cancelled when dropped.
pub async fn observe_call<R, H, C, E>(
    hooks: impl NativeHooks + 'static,
    execute: impl Future<Output = Result<CallOutput<R, H, C, E>, E>>,
) -> Result<CallOutput<R, H, C, E>, E>
where
    C: Send + 'static,
    E: Send + 'static,
{
    let guard = CallGuard::new(hooks);
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
    hooks: impl NativeHooks,
    execute: impl Future<Output = Result<R, E>>,
) -> Result<R, E> {
    let guard = CallGuard::new(hooks);
    let result = execute.await;
    guard.finish(result.is_err());
    result
}
