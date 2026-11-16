use crate::observation::ObservationSender;
use std::{
    future::Future,
    sync::atomic::{AtomicUsize, Ordering},
};

use futures_util::{TryStreamExt, stream::BoxStream};

use crate::{
    machine::{CallMachine, ChannelInterceptors, HostServices, MachineFault, StreamSender},
    protocol::{DetachedDrain, Protocol},
};

static DETACHED_DRAINS: AtomicUsize = AtomicUsize::new(0);

struct DrainSlot;

impl DrainSlot {
    fn claim(limit: usize) -> Option<Self> {
        DETACHED_DRAINS
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |running| {
                (running < limit).then_some(running + 1)
            })
            .ok()
            .map(|_| Self)
    }
}

impl Drop for DrainSlot {
    fn drop(&mut self) {
        DETACHED_DRAINS.fetch_sub(1, Ordering::AcqRel);
    }
}

async fn drain_detached<P>(
    stream: &StreamSender<P>,
    mut chunks: BoxStream<'static, Result<P::Chunk, P::Error>>,
    drain: DetachedDrain,
) where
    P: Protocol,
    P::Error: From<MachineFault>,
{
    let Some(_slot) = DrainSlot::claim(drain.limit) else {
        return;
    };
    let delivered = async {
        while let Ok(Some(chunk)) = chunks.try_next().await {
            if stream.send_chunk(chunk).await.is_err() {
                return;
            }
        }
    };
    let _ = tokio::time::timeout(drain.timeout, delivered).await;
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

pub fn hosted_call<P, F, Fut>(
    request: P::Request,
    observers: Option<ObservationSender>,
    execute: F,
) -> HostedMachine<P>
where
    P: Protocol,
    P::Error: From<MachineFault>,
    F: FnOnce(
            P::Request,
            HostServices<P>,
            ChannelInterceptors<P>,
            Option<ObservationSender>,
        ) -> Fut
        + Send
        + 'static,
    Fut: Future<Output = Result<OutputOf<P>, P::Error>> + Send + 'static,
{
    CallMachine::new(observers, move |host| {
        Box::pin(async move {
            match execute(request, host.services, host.interceptors, host.observers).await? {
                CallOutput::Complete(response) => Ok(HostedCompletion::Complete(response)),
                CallOutput::Stream { head, mut chunks } => {
                    let drain = P::detached_drain(&head);
                    if host.stream.open_stream(head).await?.is_break() {
                        return Ok(HostedCompletion::Detached);
                    }
                    while let Some(chunk) = chunks.try_next().await? {
                        if host.stream.send_chunk(chunk).await?.is_break() {
                            if let Some(drain) = drain {
                                drain_detached(&host.stream, chunks, drain).await;
                            }
                            return Ok(HostedCompletion::Detached);
                        }
                    }
                    Ok(HostedCompletion::StreamEnded)
                }
            }
        })
    })
}
