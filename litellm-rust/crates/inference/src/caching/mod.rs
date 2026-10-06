mod key;
mod plan;
mod session;
mod stream;

pub use key::CacheKeyProjection;
use litellm_host::protocol::Protocol;
pub use plan::CachePlan;
pub use session::{CacheSession, CachedOutput, execute_streaming, execute_unary};
pub use stream::StreamCachable;

use crate::RouteError;

pub trait Cachable: Protocol<Error = RouteError> {
    const SURFACE: &'static str;

    fn reusable(_response: &Self::Response) -> bool {
        true
    }
}
