use litellm_router_types::RouterRedis;
use rstest::rstest;
use serde_json::json;

#[rstest]
fn debug_shows_where_the_redis_is_and_none_of_its_credentials() {
    let redis: RouterRedis = serde_json::from_value(json!({
        "redis_url": "redis://user:url-credential-value@router:6390",
        "redis_host": "router",
        "redis_password": "password-credential-value",
        "cache_kwargs": {"sentinel_password": "kwarg-credential-value"},
    }))
    .unwrap();

    let shown = format!("{redis:?}");

    assert!(!shown.contains("credential-value"), "{shown}");
    assert!(shown.contains("[REDACTED]"), "{shown}");
    assert!(shown.contains("router"), "{shown}");
}
