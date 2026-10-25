use std::{sync::Arc, time::Duration};

use litellm_auth::{AuthServices, InputSource, Sourced};
use litellm_http::Client;
use litellm_inference_testing::no_secrets;
use litellm_inference_tokenize::{TokenizeRequest, TokenizeRoute};
use litellm_llms_types::formats::tokenize::TokenizeInclude;
use rstest::rstest;
use testcontainers::{
    GenericImage, ImageExt,
    bollard::models::DeviceRequest,
    core::{IntoContainerPort, WaitFor, wait::HttpWaitStrategy},
    runners::AsyncRunner,
};

#[rstest]
#[tokio::test]
#[ignore = "requires Docker, NVIDIA Container Toolkit, a GPU and model downloads"]
async fn hosted_vllm_returns_available_representations_and_respects_include() {
    let container = GenericImage::new("vllm/vllm-openai", "v0.15.1")
        .with_exposed_port(8000.tcp())
        .with_wait_for(WaitFor::http(
            HttpWaitStrategy::new("/health").with_expected_status_code(200_u16),
        ))
        .with_device_requests(vec![DeviceRequest {
            driver: Some("nvidia".into()),
            count: Some(1),
            capabilities: Some(vec![vec!["gpu".into()]]),
            ..Default::default()
        }])
        .with_shm_size(1 << 30)
        .with_startup_timeout(Duration::from_secs(600))
        .with_cmd([
            "--model",
            "Qwen/Qwen3-0.6B",
            "--served-model-name",
            "tokenizer",
            "--max-model-len",
            "2048",
            "--enforce-eager",
        ])
        .start()
        .await
        .unwrap();
    let api_base = format!(
        "http://{}:{}",
        container.get_host().await.unwrap(),
        container.get_host_port_ipv4(8000).await.unwrap(),
    );
    let route = TokenizeRoute::new(
        Client::plain_for_test(),
        litellm_http::CredentialClient::no_redirect_for_test(),
        Arc::new(AuthServices::default()),
        no_secrets(),
    );
    let request = TokenizeRequest {
        model: "hosted_vllm/tokenizer".into(),
        prompt: "Hello world".into(),
        api_base: Some(Sourced::new(api_base, InputSource::Request)),
        ..Default::default()
    };
    let available = route.execute(request.clone(), &(), None).await.unwrap();
    assert!(available.count > 0);
    assert_eq!(
        available.token_ids.as_ref().unwrap().len(),
        available.count as usize
    );
    assert_eq!(
        available.tokens.as_ref().unwrap().len(),
        available.count as usize
    );

    let ids = route
        .execute(
            TokenizeRequest {
                include: Some(TokenizeInclude::new(true, false)),
                ..request.clone()
            },
            &(),
            None,
        )
        .await
        .unwrap();
    assert_eq!(ids.count, available.count);
    assert_eq!(ids.token_ids, available.token_ids);
    assert_eq!(ids.tokens, None);

    let count = route
        .execute(
            TokenizeRequest {
                include: Some(TokenizeInclude::new(false, false)),
                ..request
            },
            &(),
            None,
        )
        .await
        .unwrap();
    assert_eq!(count.count, available.count);
    assert_eq!(count.token_ids, None);
    assert_eq!(count.tokens, None);
}
