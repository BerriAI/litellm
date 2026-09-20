use std::{sync::Arc, time::Duration};

use litellm_cache::{CacheControls, CacheKwargs, LLMCachingHandler};
use litellm_cache_memory::InMemoryCache;
use litellm_core_utils::budget::Budget;
use litellm_token_counter::{TokenCounter, Tokenizer};
use serde_json::{Value, json};
use tokio::{io::AsyncWriteExt, net::TcpListener};

use crate::{
    client::ClientOptions,
    messages::{Error, messages_with_options, types::MessagesRequest},
};

fn options() -> ClientOptions {
    ClientOptions {
        cache: Some(LLMCachingHandler {
            backend: Arc::new(InMemoryCache::default()),
            controls: CacheControls {
                supported_call_type: true,
                configured: true,
                native_backend: true,
                default_on: true,
                ..Default::default()
            },
            key: "test-key".into(),
            kwargs: CacheKwargs::default(),
            max_age: None,
        }),
        budget: Some(Arc::new(Budget::new(Some(100.0), 0.0))),
        model_info: Some(
            json!({"input_cost_per_token": 2.0, "output_cost_per_token": 3.0,
            "cache_read_input_token_cost": 0.5, "cache_creation_input_token_cost": 4.0}),
        ),
        ..Default::default()
    }
}

fn request(url: &str) -> MessagesRequest<'_> {
    MessagesRequest {
        model: "anthropic/test-model",
        body: json!({"model":"anthropic/test-model", "max_tokens": 100,
        "messages": [{"role": "user", "content": "hi"}]}),
        api_key: Some("test"),
        api_base: Some(url),
        custom_llm_provider: Some("anthropic"),
        extra_headers: None,
        timeout: Some(Duration::from_secs(2)),
    }
}

async fn start_server() -> (String, tokio::task::JoinHandle<Value>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let server = tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.unwrap();
        let request = super::tests::read_http_request(&mut socket).await;
        let body = json!({"id":"msg_test", "type":"message", "role":"assistant", "model":"test-model",
            "content":[{"type":"text","text":"hello"}], "stop_reason":"end_turn", "stop_sequence":null,
            "usage":{"input_tokens":2,"output_tokens":3,"cache_read_input_tokens":4,"cache_creation_input_tokens":5}});
        socket
            .write_all(super::tests::write_response(&body.to_string()).as_bytes())
            .await
            .unwrap();
        serde_json::from_str(request.split_once("\r\n\r\n").unwrap().1).unwrap()
    });
    (url, server)
}

#[tokio::test]
async fn native_messages_cache_hit_skips_network_and_does_not_double_charge() {
    let options = options();
    let (url, server) = start_server().await;
    let first = messages_with_options(request(&url), options.clone())
        .await
        .unwrap();
    server.await.unwrap();
    let cost = 2.0 * 2.0 + 3.0 * 3.0 + 4.0 * 0.5 + 5.0 * 4.0;
    assert_eq!(options.budget.as_ref().unwrap().current_cost(), cost);
    let cached = messages_with_options(request(&url), options.clone())
        .await
        .unwrap();
    assert_eq!(first, cached);
    assert_eq!(options.budget.as_ref().unwrap().current_cost(), cost);

    let exceeded = ClientOptions {
        budget: Some(Arc::new(Budget::new(Some(cost - 1.0), cost))),
        ..options
    };
    assert!(matches!(
        messages_with_options(request(&url), exceeded).await,
        Err(Error::BudgetExceeded(_))
    ));
}

#[tokio::test]
async fn no_store_and_no_cache_control_different_operations() {
    let mut options = options();
    options.cache.as_mut().unwrap().controls.no_store = true;
    let (url, server) = start_server().await;
    messages_with_options(request(&url), options.clone())
        .await
        .unwrap();
    server.await.unwrap();
    assert!(options.cache.as_ref().unwrap().get().await.is_none());

    options.cache.as_mut().unwrap().controls.no_store = false;
    options
        .cache
        .as_ref()
        .unwrap()
        .set(json!({"id":"sentinel"}))
        .await;
    options.cache.as_mut().unwrap().controls.no_cache = true;
    assert!(options.cache.as_ref().unwrap().get().await.is_none());
    let (url, server) = start_server().await;
    let response = messages_with_options(request(&url), options.clone())
        .await
        .unwrap();
    server.await.unwrap();
    options.cache.as_mut().unwrap().controls.no_cache = false;
    assert_eq!(
        options.cache.as_ref().unwrap().get().await.unwrap()["id"],
        response.id
    );
}

struct Characters;
impl Tokenizer for Characters {
    fn count_tokens(&self, text: &str) -> Result<usize, litellm_token_counter::Error> {
        Ok(text.chars().count())
    }
}

#[tokio::test]
async fn opt_in_token_adjustment_reaches_the_wire_with_a_standard_counter_interface() {
    let (url, server) = start_server().await;
    let options = ClientOptions {
        token_counter: Some(Arc::new(TokenCounter::new(Characters))),
        modify_params: true,
        model_info: Some(json!({"max_input_tokens": 40, "max_output_tokens": 40})),
        ..Default::default()
    };
    messages_with_options(request(&url), options).await.unwrap();
    let sent = server.await.unwrap();
    assert_eq!(sent["max_tokens"], 40 - (3 + 4 + 2 + 3) - 10);
}
