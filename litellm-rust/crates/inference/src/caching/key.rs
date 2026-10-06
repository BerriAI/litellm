use crate::RouteError;
use litellm_cache_response::CacheKeyInput;

pub trait CacheKeyProjection {
    fn cache_key_input(&self) -> Result<CacheKeyInput, RouteError>;
}
