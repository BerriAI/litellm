use std::{convert::Infallible, sync::Arc};

use axum::{body::Body, response::Response};
use bytes::Bytes;
use futures_util::{StreamExt, stream};
use litellm_host::{
    call::{CallOutput, HostedCompletion, HostedMachine},
    failure::Failure,
    interceptors::Interceptors,
    lifecycle::{observe_call, observe_unary},
    machine::MachineFault,
    observation::ObservationSender,
    protocol::Protocol,
};
use litellm_host_native::{Boundary, Driver, services::HostCallHandler};

use crate::{Error, ResponseEncoder, StreamEncoder};

type Output<E> = CallOutput<Response, http::Response<()>, Bytes, E>;
type HostedDriver<P, S, H> = Driver<HostedMachine<P>, S, H>;
type CallError<P> = Error<Failure<<P as Protocol>::Error>>;

pub async fn serve_unary<P, A, H, S>(
    machine: HostedMachine<P>,
    services: S,
    interceptors: H,
    encoder: A,
    observers: Option<ObservationSender>,
) -> Result<Response, CallError<P>>
where
    P: Protocol<Chunk = Infallible, StreamHead = Infallible>,
    P::Error: From<MachineFault>,
    H: Interceptors<P::Error>,
    S: HostCallHandler<P>,
    A: ResponseEncoder<Protocol = P>,
{
    let mut driver = Driver::new(machine, services, interceptors);
    observe_unary(observers, async move {
        match driver.advance().await.map_err(Error::Call)? {
            Boundary::Complete(HostedCompletion::Complete(value)) => encoder
                .encode_response(value)
                .map_err(|error| Error::Call(Failure::host(error))),
            _ => Err(Error::Protocol),
        }
    })
    .await
}

pub async fn serve<P, A, H, S>(
    machine: HostedMachine<P>,
    services: S,
    interceptors: H,
    encoder: A,
    observers: Option<ObservationSender>,
) -> Result<Response, CallError<P>>
where
    P: Protocol,
    P::Error: From<MachineFault>,
    A: StreamEncoder<Protocol = P>,
    H: Interceptors<P::Error> + 'static,
    S: HostCallHandler<P> + 'static,
{
    let encoder = Arc::new(encoder);
    let driver = Driver::new(machine, services, interceptors);
    match observe_call(observers, start(driver, encoder.clone())).await? {
        CallOutput::Complete(response) => Ok(response),
        CallOutput::Stream { head, chunks } => {
            let body = chunks.map(move |chunk| {
                Ok::<_, Infallible>(
                    chunk.unwrap_or_else(|error| encoder.encode_stream_error(error)),
                )
            });
            Ok(head.map(|()| Body::from_stream(body)))
        }
    }
}

async fn start<P, A, H, S>(
    mut driver: HostedDriver<P, S, H>,
    encoder: Arc<A>,
) -> Result<Output<CallError<P>>, CallError<P>>
where
    P: Protocol,
    P::Error: From<MachineFault>,
    A: StreamEncoder<Protocol = P>,
    H: Interceptors<P::Error> + 'static,
    S: HostCallHandler<P> + 'static,
{
    match driver.advance().await.map_err(Error::Call)? {
        Boundary::Complete(HostedCompletion::Complete(value)) => encoder
            .encode_response(value)
            .map(CallOutput::Complete)
            .map_err(|error| Error::Call(Failure::host(error))),
        Boundary::Open(head) => {
            let head = encoder
                .encode_stream_head(head)
                .map_err(|error| Error::Call(Failure::host(error)))?;
            let chunks =
                stream::try_unfold((driver, encoder), |(mut driver, encoder)| async move {
                    match driver.advance().await.map_err(Error::Call)? {
                        Boundary::Chunk(chunk) => {
                            let bytes = encoder
                                .encode_chunk(chunk)
                                .map_err(|error| Error::Call(Failure::host(error)))?;
                            Ok(Some((bytes, (driver, encoder))))
                        }
                        Boundary::Complete(HostedCompletion::StreamEnded) => Ok(None),
                        _ => Err(Error::Protocol),
                    }
                })
                .boxed();
            Ok(CallOutput::Stream { head, chunks })
        }
        _ => Err(Error::Protocol),
    }
}
