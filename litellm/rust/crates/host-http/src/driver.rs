use litellm_host::observation::ObservationSender;
use std::{convert::Infallible, sync::Arc};

use axum::{body::Body, response::Response};
use bytes::Bytes;
use futures_util::{StreamExt, stream};

use litellm_host::{
    call::{CallOutput, HostedCompletion, HostedMachine},
    interceptors::Interceptors,
    lifecycle::{observe_call, observe_unary},
    machine::MachineFault,
    protocol::Protocol,
};
use litellm_host_native::{Boundary, Driver, services::HostCallHandler};

use crate::{Error, ResponseEncoder, StreamEncoder};

type Output<E> = CallOutput<Response, http::Response<()>, Bytes, E>;
type HostedDriver<P, S, H> = Driver<HostedMachine<P>, S, H>;

pub async fn serve_unary<P, A, H, S>(
    machine: HostedMachine<P>,
    services: S,
    interceptors: H,
    encoder: A,
    observers: Option<ObservationSender>,
) -> Result<Response, Error<P::Error>>
where
    P: Protocol<Chunk = Infallible, StreamHead = Infallible>,
    P::Error: From<MachineFault>,
    H: Interceptors<P::Error>,
    S: HostCallHandler<P>,
    A: ResponseEncoder<Protocol = P>,
{
    let driver = Driver::new(machine, services, interceptors);
    observe_unary(observers, async move {
        encoder
            .encode_response(complete(driver).await?)
            .map_err(Error::Call)
    })
    .await
}

/// Run a unary hosted call to the protocol's response without rendering it. Observation covers
/// the call; rendering is the caller's.
pub async fn run_unary<P, H, S>(
    machine: HostedMachine<P>,
    services: S,
    interceptors: H,
    observers: Option<ObservationSender>,
) -> Result<P::Response, Error<P::Error>>
where
    P: Protocol<Chunk = Infallible, StreamHead = Infallible>,
    P::Error: From<MachineFault>,
    H: Interceptors<P::Error>,
    S: HostCallHandler<P>,
{
    observe_unary(
        observers,
        complete(Driver::new(machine, services, interceptors)),
    )
    .await
}

async fn complete<P, H, S>(
    mut driver: HostedDriver<P, S, H>,
) -> Result<P::Response, Error<P::Error>>
where
    P: Protocol<Chunk = Infallible, StreamHead = Infallible>,
    P::Error: From<MachineFault>,
    H: Interceptors<P::Error>,
    S: HostCallHandler<P>,
{
    match driver.advance().await.map_err(Error::Call)? {
        Boundary::Complete(HostedCompletion::Complete(value)) => Ok(value),
        _ => Err(Error::Protocol),
    }
}

pub async fn serve<P, A, H, S>(
    machine: HostedMachine<P>,
    services: S,
    interceptors: H,
    encoder: A,
    observers: Option<ObservationSender>,
) -> Result<Response, Error<P::Error>>
where
    P: Protocol,
    P::Error: From<MachineFault>,
    A: StreamEncoder<Protocol = P>,
    H: Interceptors<P::Error> + 'static,
    S: HostCallHandler<P> + 'static,
{
    let encoder = Arc::new(encoder);
    let driver = Driver::new(machine, services, interceptors);
    let encoded = observe_call(observers, async {
        encode(&encoder, advance(driver).await?)
    })
    .await?;
    Ok(body(encoded, encoder))
}

/// A hosted call run to its typed result: the protocol's response, or a stream's head and its
/// chunks. The caller renders it with [`render`].
pub type Ran<P> = CallOutput<
    <P as Protocol>::Response,
    <P as Protocol>::StreamHead,
    <P as Protocol>::Chunk,
    Error<<P as Protocol>::Error>,
>;

/// Run a hosted call to its typed result without rendering it. Observation covers the call and a
/// stream's delivery; rendering is the caller's.
pub async fn run<P, H, S>(
    machine: HostedMachine<P>,
    services: S,
    interceptors: H,
    observers: Option<ObservationSender>,
) -> Result<Ran<P>, Error<P::Error>>
where
    P: Protocol,
    P::Error: From<MachineFault>,
    H: Interceptors<P::Error> + 'static,
    S: HostCallHandler<P> + 'static,
{
    observe_call(
        observers,
        advance(Driver::new(machine, services, interceptors)),
    )
    .await
}

/// Render what [`run`] returned as the HTTP response [`serve`] would have sent.
pub fn render<A: StreamEncoder>(
    encoder: A,
    ran: Ran<A::Protocol>,
) -> Result<Response, Error<<A::Protocol as Protocol>::Error>> {
    let encoder = Arc::new(encoder);
    let encoded = encode(&encoder, ran)?;
    Ok(body(encoded, encoder))
}

type Encoded<P> = Output<Error<<P as Protocol>::Error>>;

fn encode<A: StreamEncoder>(
    encoder: &Arc<A>,
    ran: Ran<A::Protocol>,
) -> Result<Encoded<A::Protocol>, Error<<A::Protocol as Protocol>::Error>> {
    match ran {
        CallOutput::Complete(value) => encoder
            .encode_response(value)
            .map(CallOutput::Complete)
            .map_err(Error::Call),
        CallOutput::Stream { head, chunks } => {
            let head = encoder.encode_stream_head(head).map_err(Error::Call)?;
            let encoder = encoder.clone();
            let chunks = chunks
                .map(move |chunk| {
                    chunk.and_then(|chunk| encoder.encode_chunk(chunk).map_err(Error::Call))
                })
                .boxed();
            Ok(CallOutput::Stream { head, chunks })
        }
    }
}

fn body<A: StreamEncoder>(encoded: Encoded<A::Protocol>, encoder: Arc<A>) -> Response {
    match encoded {
        CallOutput::Complete(response) => response,
        CallOutput::Stream { head, chunks } => {
            let body = chunks.map(move |chunk| {
                Ok::<_, Infallible>(
                    chunk.unwrap_or_else(|error| encoder.encode_stream_error(error)),
                )
            });
            head.map(|()| Body::from_stream(body))
        }
    }
}

async fn advance<P, H, S>(mut driver: HostedDriver<P, S, H>) -> Result<Ran<P>, Error<P::Error>>
where
    P: Protocol,
    P::Error: From<MachineFault>,
    H: Interceptors<P::Error> + 'static,
    S: HostCallHandler<P> + 'static,
{
    match driver.advance().await.map_err(Error::Call)? {
        Boundary::Complete(HostedCompletion::Complete(value)) => Ok(CallOutput::Complete(value)),
        Boundary::Open(head) => {
            let chunks = stream::try_unfold(driver, |mut driver| async move {
                match driver.advance().await.map_err(Error::Call)? {
                    Boundary::Chunk(chunk) => Ok(Some((chunk, driver))),
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
