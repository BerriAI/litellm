use std::{convert::Infallible, marker::PhantomData};

use bytes::Bytes;
use http::{HeaderValue, Response, header::CONTENT_TYPE};
use litellm_host::protocol::Protocol;

use crate::{Error, StreamAdapter};

pub struct Sse<P, F> {
    stream_error: F,
    protocol: PhantomData<fn() -> P>,
}

impl<P, F> Sse<P, F> {
    pub fn new(stream_error: F) -> Self {
        Self {
            stream_error,
            protocol: PhantomData,
        }
    }
}

impl<P, F> StreamAdapter for Sse<P, F>
where
    P: Protocol<Op = Infallible, Chunk = Bytes>,
    F: Fn(Error<P::Error>) -> Bytes + Send + Sync + 'static,
{
    type Protocol = P;

    async fn custom_op(&self, op: Infallible) -> Result<(), P::Error> {
        match op {}
    }

    fn head(&self, _: P::StreamHead) -> Result<Response<()>, P::Error> {
        let mut response = Response::new(());
        response
            .headers_mut()
            .insert(CONTENT_TYPE, HeaderValue::from_static("text/event-stream"));
        Ok(response)
    }

    fn chunk(&self, chunk: Bytes) -> Result<Bytes, P::Error> {
        Ok(chunk)
    }

    fn stream_error(&self, error: Error<P::Error>) -> Bytes {
        (self.stream_error)(error)
    }
}
