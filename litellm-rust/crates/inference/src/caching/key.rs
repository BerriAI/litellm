use litellm_cache_response::CacheKeyInput;

use crate::RouteError;

pub trait CacheKeyProjection {
    fn cache_key_input(&self) -> Result<CacheKeyInput, RouteError>;
}
