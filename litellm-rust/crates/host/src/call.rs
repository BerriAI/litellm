use std::{future::Future, sync::Arc};

use futures_util::{TryStreamExt, stream::BoxStream};

use crate::{
    event::{CallEvent, FailureOrigin, Timing, epoch_seconds},
    host::Demand,
    machine::{CallMachine, HostChannel, MachineFault},
    protocol::Protocol,
};

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

pub enum CallOutput<Response, Head, Chunk, Error> {
    Complete(Response),
    Stream {
        head: Head,
        chunks: BoxStream<'static, Result<Chunk, Error>>,
    },
}

#[derive(Debug, PartialEq, Eq)]
pub enum HostedCompletion<Response> {
    Complete(Response),
    StreamEnded,
    Detached,
}

impl<Response> From<Response> for HostedCompletion<Response> {
    fn from(response: Response) -> Self {
        Self::Complete(response)
    }
}

pub type OutputOf<P> = CallOutput<
    <P as Protocol>::Response,
    <P as Protocol>::StreamHead,
    <P as Protocol>::Chunk,
    <P as Protocol>::Error,
>;

pub type HostedMachine<P> = CallMachine<P, HostedCompletion<<P as Protocol>::Response>>;

pub fn hosted_call<P, F, Fut>(execute: F) -> HostedMachine<P>
where
    P: Protocol,
    P::Error: From<MachineFault>,
    F: FnOnce(P::Projection, HostChannel<P>) -> Fut + Send + 'static,
    Fut: Future<Output = Result<OutputOf<P>, P::Error>> + Send + 'static,
{
    CallMachine::new(move |host| {
        Box::pin(async move {
            let request = host.project().await?;
            match execute(request, host.clone()).await? {
                CallOutput::Complete(response) => Ok(HostedCompletion::Complete(response)),
                CallOutput::Stream { head, mut chunks } => {
                    if host.open(head).await? == Demand::Detached {
                        return Ok(HostedCompletion::Detached);
                    }
                    while let Some(chunk) = chunks.try_next().await? {
                        if host.deliver(chunk).await? == Demand::Detached {
                            return Ok(HostedCompletion::Detached);
                        }
                    }
                    Ok(HostedCompletion::StreamEnded)
                }
            }
        })
    })
}
