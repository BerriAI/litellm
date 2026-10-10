use axum::response::{IntoResponse, Response};
use bytes::Bytes;
use http::{HeaderValue, header::CONTENT_TYPE};
use litellm_host::protocol::Protocol;

use crate::{Error, ResponseEncoder, StreamEncoder, Unary};

pub struct Sse<P: Protocol, C, F, H = fn(<P as Protocol>::StreamHead) -> http::HeaderMap> {
    response: Unary<P, C>,
    stream_error: F,
    stream_headers: H,
}

impl<P: Protocol, C, F> Sse<P, C, F> {
    pub fn new(response: C, stream_error: F) -> Self {
        Self {
            response: Unary::new(response),
            stream_error,
            stream_headers: |_| http::HeaderMap::new(),
        }
    }
}

impl<P: Protocol, C, F, H> Sse<P, C, F, H> {
    pub fn with_stream_headers<J: Fn(P::StreamHead) -> http::HeaderMap>(
        self,
        stream_headers: J,
    ) -> Sse<P, C, F, J> {
        Sse {
            response: self.response,
            stream_error: self.stream_error,
            stream_headers,
        }
    }
}

impl<P, C, F, H, R> ResponseEncoder for Sse<P, C, F, H>
where
    P: Protocol,
    C: Fn(P::Response) -> R + Send + Sync,
    F: Send + Sync,
    H: Send + Sync,
    R: IntoResponse,
{
    type Protocol = P;

    fn encode_response(&self, response: P::Response) -> Result<Response, P::Error> {
        self.response.encode_response(response)
    }
}

impl<P, C, F, H, R> StreamEncoder for Sse<P, C, F, H>
where
    P: Protocol<Chunk = Bytes>,
    C: Fn(P::Response) -> R + Send + Sync + 'static,
    F: Fn(Error<P::Error>) -> Bytes + Send + Sync + 'static,
    H: Fn(P::StreamHead) -> http::HeaderMap + Send + Sync + 'static,
    R: IntoResponse,
{
    fn encode_stream_head(&self, head: P::StreamHead) -> Result<http::Response<()>, P::Error> {
        let mut response = http::Response::new(());
        *response.headers_mut() = (self.stream_headers)(head);
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
