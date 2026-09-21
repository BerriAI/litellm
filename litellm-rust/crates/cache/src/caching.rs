use std::sync::Arc;

use crate::{BaseCache, CacheKwargs, Error};

pub use crate::BaseCache as Cache;

pub fn get_cache<B: BaseCache>(
    cache: &B,
    key: &str,
    kwargs: &CacheKwargs,
) -> Result<Option<B::Value>, Error> {
    cache.get_cache(key, kwargs)
}

pub fn set_cache<B: BaseCache>(
    cache: &B,
    key: &str,
    value: B::Value,
    kwargs: CacheKwargs,
) -> Result<(), Error> {
    cache.set_cache(key, value, kwargs)
}

pub type CacheBackend<B> = Arc<B>;
