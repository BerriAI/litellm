use std::{num::NonZeroUsize, sync::Arc, time::Duration};

use litellm_auth::{AuthServices, InputSource, Sourced};
use litellm_host::{
    call::HostedCompletion,
    interceptors::ResultSource,
    lifecycle::{CallEvent, ExecutionEvent},
    observation::observation_channel,
};
use litellm_host_native::in_process::{Host, run_hosted};
use litellm_http::{Client, CredentialClient};
use litellm_inference_testing::no_secrets;
use litellm_inference_tokenize::{TokenizeRequest, TokenizeRoute};
use litellm_llms_types::formats::tokenize::TokenizeInclude;
use rstest::{fixture, rstest};
use serde_json::json;
use tokio::sync::Notify;
use wiremock::{Mock, MockServer, ResponseTemplate, matchers::method};

fn route_with_credentials(
    http: Client,
    auth: Arc<AuthServices>,
    secrets: Arc<dyn litellm_secrets::source::SecretSource>,
) -> TokenizeRoute {
    TokenizeRoute::new(
        http,
        CredentialClient::no_redirect_for_test(),
        auth,
        secrets,
    )
}

#[fixture]
fn route() -> TokenizeRoute {
    route_with_credentials(
        Client::plain_for_test(),
        Arc::new(AuthServices::default()),
        no_secrets(),
    )
}

fn request_api_base(value: String) -> Sourced<String> {
    Sourced::new(value, InputSource::Request)
}

#[rstest]
#[case::direct(false)]
#[case::hosted(true)]
#[tokio::test]
async fn observers_receive_result_facts_before_one_terminal_event(
    route: TokenizeRoute,
    #[case] hosted: bool,
) {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({"count":1,"tokens":[42]})))
        .expect(1)
        .mount(&server)
        .await;
    let request = TokenizeRequest {
        model: "hosted_vllm/org/model".into(),
        prompt: "hello".into(),
        include: Some(TokenizeInclude::new(true, false)),
        api_base: Some(request_api_base(server.uri())),
        ..Default::default()
    };
    let (observer, mut events) = observation_channel(NonZeroUsize::new(8).unwrap());
    let response = if hosted {
        let output = run_hosted(
            route.machine(request, Some(observer.clone())),
            Host {
                services: &(),
                interceptors: &(),
                stream: &(),
                observers: Some(&observer),
            },
        )
        .await
        .unwrap();
        let HostedCompletion::Complete(response) = output else {
            panic!("tokenize returns a complete response")
        };
        response
    } else {
        route
            .execute(request, &(), Some(observer.clone()))
            .await
            .unwrap()
    };
    assert_eq!(response.token_ids, Some(vec![42]));
    assert!(matches!(
        events.try_recv().unwrap(),
        CallEvent::Started { .. }
    ));
    assert!(matches!(
        events.try_recv().unwrap(),
        CallEvent::Execution(ExecutionEvent::ProviderResponseReceived { .. })
    ));
    let CallEvent::Execution(ExecutionEvent::ResultReady { facts }) = events.try_recv().unwrap()
    else {
        panic!("provider facts must precede the terminal event")
    };
    assert_eq!(facts.provider.provider, "hosted_vllm");
    assert_eq!(facts.provider.model, "org/model");
    assert_eq!(facts.source, ResultSource::Provider);
    assert!(matches!(
        events.try_recv().unwrap(),
        CallEvent::Succeeded { .. }
    ));
    assert!(events.try_recv().is_err());
    assert_eq!(observer.dropped_events(), 0);
}

#[rstest]
#[tokio::test]
async fn cancellation_emits_one_cancelled_terminal_event(route: TokenizeRoute) {
    let server = MockServer::start().await;
    let request_received = Arc::new(Notify::new());
    let server_request_received = request_received.clone();
    Mock::given(method("POST"))
        .respond_with(move |_: &wiremock::Request| {
            server_request_received.notify_one();
            ResponseTemplate::new(200)
                .set_body_json(json!({"count":1,"tokens":[42]}))
                .set_delay(Duration::from_secs(30))
        })
        .expect(1)
        .mount(&server)
        .await;
    let request = TokenizeRequest {
        model: "hosted_vllm/org/model".into(),
        prompt: "hello".into(),
        include: Some(TokenizeInclude::new(true, false)),
        api_base: Some(request_api_base(server.uri())),
        ..Default::default()
    };
    let (observer, mut events) = observation_channel(NonZeroUsize::new(8).unwrap());
    let call_observer = observer.clone();
    let call = tokio::spawn(async move { route.execute(request, &(), Some(call_observer)).await });
    tokio::time::timeout(Duration::from_secs(5), request_received.notified())
        .await
        .expect("the provider request starts before cancellation");
    call.abort();
    assert!(call.await.unwrap_err().is_cancelled());
    assert!(matches!(
        events.try_recv().unwrap(),
        CallEvent::Started { .. }
    ));
    assert!(matches!(
        events.try_recv().unwrap(),
        CallEvent::Cancelled { .. }
    ));
    assert!(events.try_recv().is_err());
    assert_eq!(observer.dropped_events(), 0);
}
