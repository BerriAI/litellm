use litellm_host::observation::ObservationSender;
use std::{convert::Infallible, sync::Arc};

use axum::{body::Body, response::Response};
use bytes::Bytes;
use futures_util::{StreamExt, stream};

use litellm_host::{
    call::{CallOutput, HostedCompletion, HostedMachine},
    hooks::{CallInterceptors, CallOutcome},
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
    let mut driver = Driver::new(machine, services, interceptors);
    observe_unary(observers, async move {
        match driver.advance().await.map_err(Error::Call)? {
            Boundary::Complete(HostedCompletion::Complete(value)) => {
                encoder.encode_response(value).map_err(Error::Call)
            }
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
) -> Result<Response, Error<P::Error>>
where
    P: Protocol,
    P::Error: From<MachineFault>,
    A: StreamEncoder<Protocol = P>,
    H: Interceptors<P::Error> + 'static,
    S: HostCallHandler<P> + 'static,
{
    let driver = Driver::new(machine, services, interceptors);
    serve_driver(driver, encoder, observers, ()).await
}

pub async fn serve_with_hooks<P, A, H, S>(
    machine: HostedMachine<P>,
    services: S,
    hooks: H,
    encoder: A,
    observers: Option<ObservationSender>,
) -> Result<Response, Error<P::Error>>
where
    P: Protocol,
    P::Error: From<MachineFault>,
    A: StreamEncoder<Protocol = P>,
    H: CallInterceptors<P> + Clone + 'static,
    S: HostCallHandler<P> + 'static,
{
    let driver = Driver::new(machine, services, hooks.clone());
    serve_driver(driver, encoder, observers, hooks).await
}

async fn serve_driver<P, A, H, S, L>(
    driver: HostedDriver<P, S, H>,
    encoder: A,
    observers: Option<ObservationSender>,
    hooks: L,
) -> Result<Response, Error<P::Error>>
where
    P: Protocol,
    P::Error: From<MachineFault>,
    A: StreamEncoder<Protocol = P>,
    H: Interceptors<P::Error> + 'static,
    S: HostCallHandler<P> + 'static,
    L: CallInterceptors<P> + 'static,
{
    let encoder = Arc::new(encoder);
    let hooks = HookGuard::<P, L> {
        hooks,
        finished: false,
        protocol: std::marker::PhantomData,
    };
    match observe_call(observers, start(driver, encoder.clone(), hooks)).await? {
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

struct HookGuard<P: Protocol, H: CallInterceptors<P>> {
    hooks: H,
    finished: bool,
    protocol: std::marker::PhantomData<fn() -> P>,
}

impl<P: Protocol, H: CallInterceptors<P>> HookGuard<P, H> {
    async fn terminal(&mut self, outcome: CallOutcome) -> Result<(), litellm_host::HookError> {
        let result = self.hooks.on_terminal(outcome).await;
        self.finished = true;
        result
    }

    async fn failed(&mut self) {
        let _ = self.terminal(CallOutcome::Failed).await;
    }
}

impl<P: Protocol, H: CallInterceptors<P>> Drop for HookGuard<P, H> {
    fn drop(&mut self) {
        if !self.finished {
            self.hooks.on_cancel();
        }
    }
}

async fn start<P, A, H, S, L>(
    mut driver: HostedDriver<P, S, H>,
    encoder: Arc<A>,
    mut hooks: HookGuard<P, L>,
) -> Result<Output<Error<P::Error>>, Error<P::Error>>
where
    P: Protocol,
    P::Error: From<MachineFault>,
    A: StreamEncoder<Protocol = P>,
    H: Interceptors<P::Error> + 'static,
    S: HostCallHandler<P> + 'static,
    L: CallInterceptors<P> + 'static,
{
    let boundary = match driver.advance().await {
        Ok(boundary) => boundary,
        Err(error) => {
            hooks.failed().await;
            return Err(Error::Call(error));
        }
    };
    match boundary {
        Boundary::Complete(HostedCompletion::Complete(value)) => {
            let value = match hooks.hooks.transform_response(value).await {
                Ok(value) => value,
                Err(error) => {
                    hooks.failed().await;
                    return Err(Error::Hook(error));
                }
            };
            let response = match encoder.encode_response(value) {
                Ok(response) => response,
                Err(error) => {
                    hooks.failed().await;
                    return Err(Error::Call(error));
                }
            };
            hooks.terminal(CallOutcome::Succeeded).await?;
            Ok(CallOutput::Complete(response))
        }
        Boundary::Open(head) => {
            let head = match encoder.encode_stream_head(head) {
                Ok(head) => head,
                Err(error) => {
                    hooks.failed().await;
                    return Err(Error::Call(error));
                }
            };
            let chunks = stream::try_unfold(
                (driver, encoder, hooks),
                |(mut driver, encoder, mut hooks)| async move {
                    let boundary = match driver.advance().await {
                        Ok(boundary) => boundary,
                        Err(error) => {
                            hooks.failed().await;
                            return Err(Error::Call(error));
                        }
                    };
                    match boundary {
                        Boundary::Chunk(chunk) => {
                            if let Err(error) = hooks.hooks.on_stream_chunk(&chunk).await {
                                hooks.failed().await;
                                return Err(Error::Hook(error));
                            }
                            let bytes = match encoder.encode_chunk(chunk) {
                                Ok(bytes) => bytes,
                                Err(error) => {
                                    hooks.failed().await;
                                    return Err(Error::Call(error));
                                }
                            };
                            Ok(Some((bytes, (driver, encoder, hooks))))
                        }
                        Boundary::Complete(HostedCompletion::StreamEnded) => {
                            hooks.terminal(CallOutcome::Succeeded).await?;
                            Ok(None)
                        }
                        _ => {
                            hooks.failed().await;
                            Err(Error::Protocol)
                        }
                    }
                },
            )
            .boxed();
            Ok(CallOutput::Stream { head, chunks })
        }
        _ => {
            hooks.failed().await;
            Err(Error::Protocol)
        }
    }
}
