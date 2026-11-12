use std::sync::atomic::{AtomicUsize, Ordering};

use futures_util::future::BoxFuture;
use litellm_inference_messages::{HostAnswer, MessagesRoute};
use rstest::rstest;

use super::*;

#[derive(Default)]
struct CountingGateway(AtomicUsize);

impl HostAnswer for CountingGateway {
    fn answer(&self, _stream: bool) -> BoxFuture<'_, Result<(), Error>> {
        self.0.fetch_add(1, Ordering::SeqCst);
        Box::pin(async { Ok(()) })
    }
}

fn gateway_call(call: MessagesCall, api_base: String, mcp_gateway: bool) -> MessagesCall {
    MessagesCall {
        api_key: Some("sk-ant-test".into()),
        api_base: Some(api_base),
        shaping: MessagesShaping {
            settings: MessagesSettings {
                mcp_gateway,
                ..MessagesSettings::default()
            },
            ..MessagesShaping::default()
        },
        ..call
    }
}

async fn run_through(route: MessagesRoute, call: MessagesCall) -> Result<MessagesOutput, Error> {
    let host = LocalMessagesHost::new(call);
    litellm_host_native::in_process::run_hosted(
        route.machine(host.request()?, None),
        host.runtime(),
    )
    .await
}

#[rstest]
#[tokio::test]
async fn a_call_the_host_hands_to_the_mcp_gateway_never_reaches_the_provider(call: MessagesCall) {
    let upstream = upstream([message_response()]).await;
    let gateway = Arc::new(CountingGateway::default());
    let route =
        messages_route(Arc::new(RecordingSecrets::empty())).with_host_answer(gateway.clone());

    let output = run_through(route, gateway_call(call, upstream.uri(), true))
        .await
        .unwrap();

    let MessagesOutput::Complete(MessagesBody::Unchecked(body)) = output else {
        panic!("the host answers in place of the provider");
    };
    assert!(body.is_empty());
    assert_eq!(gateway.0.load(Ordering::SeqCst), 1);
    assert!(received(&upstream).await.is_empty());
}

#[rstest]
#[tokio::test]
async fn a_call_the_host_keeps_reaches_the_provider_past_the_gateway(call: MessagesCall) {
    let upstream = upstream([message_response()]).await;
    let gateway = Arc::new(CountingGateway::default());
    let route =
        messages_route(Arc::new(RecordingSecrets::empty())).with_host_answer(gateway.clone());

    let output = run_through(route, gateway_call(call, upstream.uri(), false))
        .await
        .unwrap();

    assert!(matches!(
        output,
        MessagesOutput::Complete(MessagesBody::Message(_))
    ));
    assert_eq!(gateway.0.load(Ordering::SeqCst), 0);
    only_request(&upstream).await;
}

#[rstest]
#[tokio::test]
async fn a_mock_answers_before_the_gateway(call: MessagesCall) {
    let gateway = Arc::new(CountingGateway::default());
    let route =
        messages_route(Arc::new(RecordingSecrets::empty())).with_host_answer(gateway.clone());
    let mut mocked = gateway_call(call, UNREACHABLE_BASE.into(), true);
    mocked.shaping.settings.mock_response = Some(object(json!({"mocked": true})));

    let output = run_through(route, mocked).await.unwrap();

    let MessagesOutput::Complete(MessagesBody::Unchecked(body)) = output else {
        panic!("the mock is the call's response");
    };
    assert_eq!(body, object(json!({"mocked": true})));
    assert_eq!(gateway.0.load(Ordering::SeqCst), 0);
}
