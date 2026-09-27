mod driver;
mod error;
mod sse;

use std::future::Future;

use bytes::Bytes;
use litellm_host::protocol::Protocol;

pub use driver::{serve, serve_unary};
pub use error::Error;
pub use sse::Sse;

pub trait StreamAdapter: Send + Sync + 'static {
    type Protocol: Protocol;

    fn custom_op(
        &self,
        op: <Self::Protocol as Protocol>::Op,
    ) -> impl Future<Output = Result<(), <Self::Protocol as Protocol>::Error>> + Send;

    fn head(
        &self,
        head: <Self::Protocol as Protocol>::StreamHead,
    ) -> Result<http::Response<()>, <Self::Protocol as Protocol>::Error>;

    fn chunk(
        &self,
        chunk: <Self::Protocol as Protocol>::Chunk,
    ) -> Result<Bytes, <Self::Protocol as Protocol>::Error>;

    fn stream_error(&self, error: Error<<Self::Protocol as Protocol>::Error>) -> Bytes;
}
