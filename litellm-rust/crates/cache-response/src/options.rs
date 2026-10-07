use crate::{CachePolicy, CacheScope};

#[derive(Clone, Debug, Default)]
pub struct CacheOptions {
    pub policy: CachePolicy,
    pub scope: CacheScope,
}
