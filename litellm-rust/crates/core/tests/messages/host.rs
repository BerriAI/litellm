use std::{convert::Infallible, sync::Mutex};

use litellm_core::messages::route::Messages;
use litellm_host::{
    event::{CallEvent, MachineEvent, RequestContext, WireRequest},
    host::Host,
};
use litellm_llms::anthropic::common_utils::AnthropicModelCapabilities;
use rstest::rstest;

use super::*;

type Rewrite = Box<dyn Fn(WireRequest) -> Result<WireRequest, Error> + Send + Sync>;

/// Projects like `LocalMessagesHost`, answers `before_send` through `rewrite`, and keeps
/// every event the driver emits.
struct RecordingHost {
    call: LocalMessagesHost,
    rewrite: Rewrite,
    events: Mutex<Vec<CallEvent>>,
    optional_params: Mutex<Vec<Value>>,
}

impl RecordingHost {
    fn new(call: MessagesCall, rewrite: Rewrite) -> Self {
        Self {
            call: LocalMessagesHost::new(call),
            rewrite,
            events: Mutex::new(Vec::new()),
            optional_params: Mutex::new(Vec::new()),
        }
    }

    fn passthrough(call: MessagesCall) -> Self {
        Self::new(call, Box::new(Ok))
    }

    fn raw_responses(&self) -> Vec<String> {
        self.events
            .lock()
            .unwrap()
            .iter()
            .filter_map(|event| match event {
                CallEvent::Machine(MachineEvent::ResponseReceived { raw }) => {
                    Some(raw.body.clone())
                }
                _ => None,
            })
            .collect()
    }
}

impl Host<Messages> for RecordingHost {
    async fn project(&self) -> Result<MessagesCall, Error> {
        self.call.project().await
    }

    async fn custom_op(&self, op: Infallible) -> Result<(), Error> {
        match op {}
    }

    async fn before_send(
        &self,
        wire: WireRequest,
        context: &RequestContext,
    ) -> Result<WireRequest, Error> {
        self.optional_params
            .lock()
            .unwrap()
            .push(context.optional_params.clone());
        (self.rewrite)(wire)
    }

    async fn emit(&self, event: &CallEvent) -> Result<(), Error> {
        self.events.lock().unwrap().push(event.clone());
        Ok(())
    }
}

async fn run_through(host: &RecordingHost) -> Result<MessagesOutput, Error> {
    litellm_host::run::run(messages_machine(Arc::new(RecordingSecrets::empty())), host).await
}

fn authenticated(call: MessagesCall, api_base: String) -> MessagesCall {
    MessagesCall {
        api_key: Some("sk-ant".into()),
        api_base: Some(api_base),
        ..call
    }
}

#[rstest]
#[tokio::test]
async fn what_before_send_returns_is_what_the_provider_receives(call: MessagesCall) {
    let upstream = upstream([message_response()]).await;
    let host = RecordingHost::new(
        authenticated(call, upstream.uri()),
        Box::new(|wire| {
            let mut body = wire.body;
            body["system"] = json!("added by the host");
            Ok(WireRequest {
                headers: wire
                    .headers
                    .into_iter()
                    .chain([("x-host".to_string(), "seen".to_string())])
                    .collect(),
                body,
                ..wire
            })
        }),
    );

    run_through(&host).await.expect("messages call succeeds");

    let request = only_request(&upstream).await;
    assert_eq!(request.json()["system"], "added by the host");
    assert_eq!(request.header("x-host"), Some("seen"));
    assert_eq!(request.header("x-api-key"), Some("sk-ant"));
}

#[rstest]
#[tokio::test]
async fn a_before_send_failure_never_sends(call: MessagesCall) {
    let upstream = upstream([message_response()]).await;
    let host = RecordingHost::new(
        authenticated(call, upstream.uri()),
        Box::new(|_| Err(Error::InvalidRequest("vetoed by the host".into()))),
    );

    let error = run_through(&host)
        .await
        .err()
        .expect("the host failure fails the call");

    assert_eq!(error, Error::InvalidRequest("vetoed by the host".into()));
    assert!(received(&upstream).await.is_empty());
    assert!(host.raw_responses().is_empty());
}

#[rstest]
#[tokio::test]
async fn the_raw_upstream_text_is_emitted_once_for_a_message(call: MessagesCall) {
    let raw = message_body();
    let upstream = upstream([json_response(raw.clone())]).await;
    let host = RecordingHost::passthrough(authenticated(call, upstream.uri()));

    let output = run_through(&host).await.expect("messages call succeeds");

    assert!(matches!(output, MessagesOutput::Message(_)));
    let [emitted] = <[String; 1]>::try_from(host.raw_responses())
        .unwrap_or_else(|raws| panic!("expected one raw response, got {}", raws.len()));
    assert_eq!(serde_json::from_str::<Value>(&emitted).unwrap(), raw);
}

#[rstest]
#[case::upstream_error(ResponseTemplate::new(500).set_body_string("boom"))]
#[case::stream(ResponseTemplate::new(200).set_body_raw("event: message_stop\ndata: {}\n\n", "text/event-stream"))]
#[tokio::test]
async fn no_raw_response_is_emitted_for_a_stream_or_a_failure(
    call: MessagesCall,
    #[case] response: ResponseTemplate,
) {
    let upstream = upstream([response]).await;
    let mut body = call.body.clone();
    body.insert("stream".into(), json!(true));
    let host =
        RecordingHost::passthrough(authenticated(MessagesCall { body, ..call }, upstream.uri()));

    let _ = run_through(&host).await;

    assert_eq!(received(&upstream).await.len(), 1);
    assert!(host.raw_responses().is_empty());
}

/// Python logs `optional_params` as what it is about to send, so a dropped param must
/// not resurface in callbacks.
#[rstest]
#[tokio::test]
async fn the_request_context_carries_the_shaped_params_without_model_or_messages(
    call: MessagesCall,
) {
    let upstream = upstream([message_response()]).await;
    let body: Map<String, Value> = call
        .body
        .clone()
        .into_iter()
        .chain([("temperature".to_string(), json!(0.2))])
        .collect();
    let host = RecordingHost::passthrough(authenticated(
        MessagesCall {
            body,
            shaping: MessagesShaping {
                capabilities: AnthropicModelCapabilities {
                    supports_sampling_params: false,
                    ..AnthropicModelCapabilities::default()
                },
                drop_params: true,
                ..MessagesShaping::default()
            },
            ..call
        },
        upstream.uri(),
    ));

    run_through(&host).await.expect("messages call succeeds");

    let [optional_params] = <[Value; 1]>::try_from(host.optional_params.into_inner().unwrap())
        .unwrap_or_else(|seen| panic!("before_send runs once, saw {}", seen.len()));
    assert_eq!(optional_params, json!({"max_tokens": 16}));
}
