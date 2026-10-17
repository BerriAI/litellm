/// `RedisSemanticCache.DEFAULT_REDIS_INDEX_NAME`.
pub const DEFAULT_INDEX_NAME: &str = "litellm_semantic_cache_index";

#[derive(Clone, Debug)]
pub struct RedisSemanticConfig {
    pub index_name: String,
    pub similarity_threshold: f32,
}
