use crate::RouteError;
use litellm_cache_response::CacheKeyInput;

pub trait CacheKeyProjection {
    fn cache_key_input(&self, model_group: Option<&str>) -> Result<CacheKeyInput, RouteError>;
}
