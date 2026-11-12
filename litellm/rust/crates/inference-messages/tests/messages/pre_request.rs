use futures_util::future::BoxFuture;
use litellm_inference_messages::PreRequestHooks;
use rstest::rstest;

use super::*;

struct Hooks {
    rewrite: Box<dyn Fn(MessagesCall) -> Result<MessagesCall, Error> + Send + Sync>,
    runs: Mutex<usize>,
}

impl PreRequestHooks for Hooks {
    fn run(&self, call: MessagesCall) -> BoxFuture<'_, Result<MessagesCall, Error>> {
        *self.runs.lock().unwrap() += 1;
        let rewritten = (self.rewrite)(call);
        Box::pin(async move { rewritten })
    }
}

fn hooks(
    rewrite: impl Fn(MessagesCall) -> Result<MessagesCall, Error> + Send + Sync + 'static,
) -> Arc<Hooks> {
    Arc::new(Hooks {
        rewrite: Box::new(rewrite),
        runs: Mutex::new(0),
    })
}

async fn run_hooked(hooks: Arc<Hooks>, call: MessagesCall) -> Result<MessagesOutput, Error> {
    let host = LocalMessagesHost::new(call);
    let route = messages_route(Arc::new(RecordingSecrets::empty())).with_pre_request_hooks(hooks);
    litellm_host_native::in_process::run_hosted(
        route.machine(host.request()?, None),
        host.runtime(),
    )
    .await
}

fn mocked(call: MessagesCall, mock: Map<String, Value>) -> MessagesCall {
    MessagesCall {
        shaping: MessagesShaping {
            settings: MessagesSettings {
                mock_response: Some(mock),
                ..call.shaping.settings
            },
            ..call.shaping
        },
        ..call
    }
}

fn routed(call: MessagesCall, base: String) -> MessagesCall {
    MessagesCall {
        api_key: Some("sk-test".into()),
        api_base: Some(base),
        ..call
    }
}

#[rstest]
#[tokio::test]
async fn the_provider_receives_the_call_the_hooks_leave(call: MessagesCall) {
    let upstream = upstream([message_response()]).await;
    let hooks = hooks(|call| {
        Ok(with_fields(
            call,
            json!({"system": "hooked", "max_tokens": 7}),
        ))
    });

    run_hooked(hooks.clone(), routed(call, upstream.uri()))
        .await
        .expect("messages call succeeds");

    let sent: Value = only_request(&upstream).await.body_json().unwrap();
    assert_eq!(sent["system"], "hooked");
    assert_eq!(sent["max_tokens"], 7);
    assert_eq!(*hooks.runs.lock().unwrap(), 1);
}

#[rstest]
#[case::hook_sets_the_mock(false, true, true)]
#[case::hook_keeps_the_callers_mock(true, false, true)]
#[case::hook_clears_the_callers_mock(true, true, false)]
#[tokio::test]
async fn the_mock_is_read_from_the_hooked_call(
    call: MessagesCall,
    #[case] called_with_mock: bool,
    #[case] hook_rewrites_mock: bool,
    #[case] mocked_response: bool,
) {
    let upstream = upstream([message_response()]).await;
    let mock = object(json!({"id": "mocked"}));
    let call = routed(call, upstream.uri());
    let call = match called_with_mock {
        true => mocked(call, mock.clone()),
        false => call,
    };
    let hooked_mock = (!called_with_mock).then(|| mock.clone());
    let hooks = hooks(move |call| {
        Ok(match (hook_rewrites_mock, &hooked_mock) {
            (false, _) => call,
            (true, Some(mock)) => mocked(call, mock.clone()),
            (true, None) => MessagesCall {
                shaping: MessagesShaping {
                    settings: MessagesSettings {
                        mock_response: None,
                        ..call.shaping.settings
                    },
                    ..call.shaping
                },
                ..call
            },
        })
    });

    let output = run_hooked(hooks, call)
        .await
        .expect("messages call succeeds");

    let mocked_output = matches!(
        output,
        MessagesOutput::Complete(MessagesBody::Unchecked(ref body)) if *body == mock
    );
    assert_eq!(mocked_output, mocked_response);
    assert_eq!(received(&upstream).await.is_empty(), mocked_response);
}

#[rstest]
#[tokio::test]
async fn a_failing_hook_fails_the_call_before_the_provider(call: MessagesCall) {
    let upstream = upstream([message_response()]).await;
    let hooks = hooks(|_| Err(Error::InvalidProvider("refused".into())));

    let Err(error) = run_hooked(hooks, routed(call, upstream.uri())).await else {
        panic!("the hook's failure fails the call");
    };

    assert!(matches!(error, Error::InvalidProvider(ref provider) if provider == "refused"));
    assert!(received(&upstream).await.is_empty());
}
