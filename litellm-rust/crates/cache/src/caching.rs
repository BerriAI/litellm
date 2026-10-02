use std::sync::Arc;

pub use crate::BaseCache as Cache;
use crate::{BaseCache, Error};

pub fn get_cache<B: BaseCache>(
    cache: &B,
    key: &str,
    context: &B::Context,
) -> Result<Option<B::Value>, Error> {
    cache.get_cache(key, context)
}

pub fn set_cache<B: BaseCache>(
    cache: &B,
    key: &str,
    value: B::Value,
    context: &B::Context,
) -> Result<(), Error> {
    cache.set_cache(key, value, context)
}

pub type CacheBackend<B> = Arc<B>;
