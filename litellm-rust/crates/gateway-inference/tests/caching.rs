mod support;

use std::{sync::Arc, time::Duration};

use axum::body::to_bytes;
use litellm_cache_memory::InMemoryCache;
use litellm_cache_response::{
    CacheKeyInput, ResponseCache, ResponseCacheConfig, ResponseCacheRequest, ResponseCacheService,
};
use rstest::rstest;
use serde_json::{Value, json};
use wiremock::{Mock, MockServer, ResponseTemplate, matchers::method};

#[rstest]
#[case::chat("/v1/chat/completions", "anthropic/test-model", false)]
#[case::messages("/v1/messages", "anthropic/test-model", false)]
#[case::responses("/v1/responses", "openai/test-model", false)]
#[case::messages_stream("/v1/messages", "anthropic/test-model", true)]
#[case::responses_stream("/v1/responses", "openai/test-model", true)]
#[tokio::test]
async fn all_inference_endpoints_share_native_cache(
    #[case] path: &str,
    #[case] model: &str,
    #[case] stream: bool,
    #[values("s-maxage", "s-max-age")] max_age: &str,
) {
    let upstream = MockServer::start().await;
    let is_responses = path.ends_with("responses");
    let provider_body = if is_responses {
        json!({"id":"response-1", "model":"test-model", "status":"completed", "output":[]})
    } else {
        json!({"id":"message-1", "model":"test-model", "type":"message", "role":"assistant", "content":[{"type":"text","text":"hello"}], "stop_reason":"end_turn", "usage":{"input_tokens":1,"output_tokens":1}})
    };
    let terminal = if is_responses {
        "response.completed"
    } else {
        "message_stop"
    };
    let events = format!("event: {terminal}\ndata: {{\"type\":\"{terminal}\"}}\n\n");
    let template = if stream {
        ResponseTemplate::new(200).set_body_raw(events.clone(), "text/event-stream")
    } else {
        ResponseTemplate::new(200).set_body_json(provider_body)
    };
    Mock::given(method("POST"))
        .respond_with(template)
        .expect(2)
        .mount(&upstream)
        .await;
    let cache: Arc<dyn ResponseCacheService> = Arc::new(
        ResponseCache::new(Arc::new(InMemoryCache::new(
            Some(100),
            Some(Duration::from_secs(60)),
        )))
        .with_config(ResponseCacheConfig {
            namespace: "gateway-test".into(),
            max_entry_bytes: 4096,
        }),
    );
    let app = support::app_with_cache(model, &upstream.uri(), cache.clone());
    let request = if is_responses {
        json!({"model":"public/model", "input":"hello", "stream":stream, "cache":{(max_age):600}})
    } else {
        json!({"model":"public/model", "messages":[{"role":"user","content":"hello"}], "max_tokens":16, "stream":stream, "cache":{(max_age):600}})
    };
    let first = support::post(app.clone(), path, request.clone()).await;
    assert_eq!(first.status(), 200);
    assert!(!first.headers().contains_key("x-litellm-cache-key"));
    let first = to_bytes(first.into_body(), 4096).await.unwrap();
    let second = support::post(app.clone(), path, request.clone()).await;
    assert_eq!(second.status(), 200);
    let cache_key = second.headers().get("x-litellm-cache-key").unwrap().clone();
    assert!(!cache_key.as_bytes().is_empty());
    let stored = cache
        .lookup(
            &ResponseCacheRequest::new(CacheKeyInput {
                preset: Some(cache_key.to_str().unwrap().into()),
                ..Default::default()
            }),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap(),
        )
        .await
        .unwrap();
    assert!(
        stored.is_some(),
        "the header must identify the stored entry"
    );
    let second = to_bytes(second.into_body(), 4096).await.unwrap();
    if stream {
        assert_eq!(first, events);
        assert_eq!(second, first);
    } else {
        assert_eq!(
            serde_json::from_slice::<Value>(&first).unwrap(),
            serde_json::from_slice::<Value>(&second).unwrap()
        );
    }
    let bypass_request = Value::Object(
        request
            .as_object()
            .unwrap()
            .iter()
            .map(|(name, value)| {
                (
                    name.clone(),
                    if name == "cache" {
                        json!({"no-cache": true, "no-store": true})
                    } else {
                        value.clone()
                    },
                )
            })
            .collect(),
    );
    let bypassed = support::post(app.clone(), path, bypass_request).await;
    assert_eq!(bypassed.status(), 200);
    assert!(!bypassed.headers().contains_key("x-litellm-cache-key"));
    to_bytes(bypassed.into_body(), 4096).await.unwrap();
    let restored = support::post(app, path, request).await;
    assert_eq!(restored.status(), 200);
    assert_eq!(
        restored.headers().get("x-litellm-cache-key"),
        Some(&cache_key)
    );
    assert_eq!(to_bytes(restored.into_body(), 4096).await.unwrap(), second);
}

#[rstest]
#[case::different_subject("issuer", "tenant-b")]
#[case::different_authority("other-issuer", "tenant-a")]
#[tokio::test]
async fn authenticated_callers_do_not_share_cached_responses(
    #[case] authority: &str,
    #[case] subject: &str,
) {
    use litellm_gateway_auth::{Principal, PrincipalKind};

    let upstream = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "id":"response-1", "model":"test-model", "status":"completed", "output":[]
        })))
        .expect(2)
        .mount(&upstream)
        .await;
    let cache: Arc<dyn ResponseCacheService> = Arc::new(ResponseCache::new(Arc::new(
        InMemoryCache::new(Some(100), Some(Duration::from_secs(60))),
    )));
    let first_caller = support::app_with_cache_for_principal(
        "openai/test-model",
        &upstream.uri(),
        cache.clone(),
        Principal::new("issuer".into(), "tenant-a".into(), PrincipalKind::Service),
    );
    let second_caller = support::app_with_cache_for_principal(
        "openai/test-model",
        &upstream.uri(),
        cache,
        Principal::new(authority.into(), subject.into(), PrincipalKind::Service),
    );
    let body = json!({"model":"public/model","input":"same prompt"});
    let first = support::post(first_caller.clone(), "/v1/responses", body.clone()).await;
    assert_eq!(first.status(), 200);
    assert!(!first.headers().contains_key("x-litellm-cache-key"));
    let first_hit = support::post(first_caller.clone(), "/v1/responses", body.clone()).await;
    assert_eq!(first_hit.status(), 200);
    let first_key = first_hit.headers().get("x-litellm-cache-key").unwrap();
    let second = support::post(second_caller.clone(), "/v1/responses", body.clone()).await;
    assert_eq!(second.status(), 200);
    assert!(!second.headers().contains_key("x-litellm-cache-key"));
    let second_hit = support::post(second_caller, "/v1/responses", body.clone()).await;
    assert_eq!(second_hit.status(), 200);
    assert_ne!(
        second_hit.headers().get("x-litellm-cache-key").unwrap(),
        first_key
    );
    let first_again = support::post(first_caller, "/v1/responses", body).await;
    assert_eq!(first_again.status(), 200);
    assert_eq!(
        first_again.headers().get("x-litellm-cache-key"),
        Some(first_key)
    );
}
