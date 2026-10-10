/// `ValkeySemanticCache.DEFAULT_VALKEY_INDEX_NAME`.
pub const DEFAULT_INDEX_NAME: &str = "litellm_semantic_cache_index";

#[derive(Clone, Debug, PartialEq)]
pub struct ValkeySemanticConfig {
    pub similarity_threshold: f64,
    pub index_name: String,
}
