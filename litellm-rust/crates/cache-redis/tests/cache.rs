use litellm_cache_redis::RedisCache;

#[test]
fn constructor_rejects_invalid_urls() {
    assert!(RedisCache::new("not a redis url", None).is_err());
}
