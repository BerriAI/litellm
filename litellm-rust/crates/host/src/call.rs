use std::future::Future;

use futures_util::{TryStreamExt, stream::BoxStream};

use crate::{
    machine::{CallMachine, ChannelInterceptors, HostServices, MachineFault},
    protocol::Protocol,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Operation {
    Completion,
    Responses,
    Messages,
    Ocr,
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

pub fn hosted_call<P, F, Fut>(request: P::Request, execute: F) -> HostedMachine<P>
where
    P: Protocol,
    P::Error: From<MachineFault>,
    F: FnOnce(P::Request, HostServices<P>, ChannelInterceptors<P>) -> Fut + Send + 'static,
    Fut: Future<Output = Result<OutputOf<P>, P::Error>> + Send + 'static,
{
    CallMachine::new(move |host| {
        Box::pin(async move {
            match execute(request, host.services, host.interceptors).await? {
                CallOutput::Complete(response) => Ok(HostedCompletion::Complete(response)),
                CallOutput::Stream { head, mut chunks } => {
                    if host.stream.open_stream(head).await?.is_break() {
                        return Ok(HostedCompletion::Detached);
                    }
                    while let Some(chunk) = chunks.try_next().await? {
                        if host.stream.send_chunk(chunk).await?.is_break() {
                            return Ok(HostedCompletion::Detached);
                        }
                    }
                    Ok(HostedCompletion::StreamEnded)
                }
            }
        })
    })
}
