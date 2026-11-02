use axum::response::{IntoResponse, Response};
use bytes::Bytes;
use http::{HeaderValue, header::CONTENT_TYPE};
use litellm_host::protocol::Protocol;

use crate::{Error, ResponseEncoder, StreamEncoder, Unary};

pub struct Sse<P, C, F> {
    response: Unary<P, C>,
    stream_error: F,
}

impl<P, C, F> Sse<P, C, F> {
    pub fn new(response: C, stream_error: F) -> Self {
        Self {
            response: Unary::new(response),
            stream_error,
        }
    }
}

impl<P, C, F, R> ResponseEncoder for Sse<P, C, F>
where
    P: Protocol,
    C: Fn(P::Response) -> R + Send + Sync,
    F: Send + Sync,
    R: IntoResponse,
{
    type Protocol = P;

    fn encode_response(&self, response: P::Response) -> Result<Response, P::Error> {
        self.response.encode_response(response)
    }
}

impl<P, C, F, R> StreamEncoder for Sse<P, C, F>
where
    P: Protocol<Chunk = Bytes>,
    C: Fn(P::Response) -> R + Send + Sync + 'static,
    F: Fn(Error<P::Error>) -> Bytes + Send + Sync + 'static,
    R: IntoResponse,
{
    fn encode_stream_head(&self, _: P::StreamHead) -> Result<http::Response<()>, P::Error> {
        let mut response = http::Response::new(());
        response
            .headers_mut()
            .insert(CONTENT_TYPE, HeaderValue::from_static("text/event-stream"));
        Ok(response)
    }

    fn encode_chunk(&self, chunk: Bytes) -> Result<Bytes, P::Error> {
        Ok(chunk)
    }

    fn encode_stream_error(&self, error: Error<P::Error>) -> Bytes {
        (self.stream_error)(error)
    }
}
