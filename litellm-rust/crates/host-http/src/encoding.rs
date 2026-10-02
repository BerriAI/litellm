use std::marker::PhantomData;

use axum::response::{IntoResponse, Response};
use bytes::Bytes;
use litellm_host::protocol::Protocol;

use crate::Error;

pub trait ResponseEncoder: Send + Sync {
    type Protocol: Protocol;

    fn encode_response(
        &self,
        response: <Self::Protocol as Protocol>::Response,
    ) -> Result<Response, <Self::Protocol as Protocol>::Error>;
}

pub trait StreamEncoder: ResponseEncoder + 'static {
    fn encode_stream_head(
        &self,
        head: <Self::Protocol as Protocol>::StreamHead,
    ) -> Result<http::Response<()>, <Self::Protocol as Protocol>::Error>;

    fn encode_chunk(
        &self,
        chunk: <Self::Protocol as Protocol>::Chunk,
    ) -> Result<Bytes, <Self::Protocol as Protocol>::Error>;

    fn encode_stream_error(&self, error: Error<<Self::Protocol as Protocol>::Error>) -> Bytes;
}

pub struct Unary<P, F> {
    response: F,
    protocol: PhantomData<fn() -> P>,
}

impl<P, F> Unary<P, F> {
    pub fn new(response: F) -> Self {
        Self {
            response,
            protocol: PhantomData,
        }
    }
}

impl<P, F, R> ResponseEncoder for Unary<P, F>
where
    P: Protocol,
    F: Fn(P::Response) -> R + Send + Sync,
    R: IntoResponse,
{
    type Protocol = P;

    fn encode_response(&self, response: P::Response) -> Result<Response, P::Error> {
        Ok((self.response)(response).into_response())
    }
}
