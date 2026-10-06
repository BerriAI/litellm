mod key;
mod plan;
mod session;
mod stream;

use litellm_host::protocol::Protocol;

use crate::RouteError;

pub use key::CacheKeyProjection;
pub use plan::CachePlan;
pub(crate) use session::CacheSession;
pub use session::{CachedOutput, execute_streaming, execute_unary};
pub use stream::StreamCachable;

pub trait Cachable: Protocol<Error = RouteError> {
    const SURFACE: &'static str;

    fn reusable(_response: &Self::Response) -> bool {
        true
    }
}
