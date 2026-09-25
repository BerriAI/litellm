use std::sync::Mutex;

use bytes::Bytes;
use litellm_core::messages::route::{Messages, MessagesStreamHead};
use litellm_host::{
    event::{PublicRequest, RequestContext, WireRequest},
    host::{Demand, Host, Verdict},
};
use rstest::rstest;

use super::*;

const SSE_BODY: &str = "event: message_start\ndata: {\"type\":\"message_start\"}\n\nevent: message_stop\ndata: {\"type\":\"message_stop\"}\n\n";

enum Decision {
    Return,
    Replace(Box<AnthropicMessagesResponse>),
    Resend(Map<String, Value>),
}

/// Projects like `LocalMessagesHost`, records what every hook op offered it, answers
/// `pre_request` with `edit` laid over the offered params, and answers `after_response`
/// from `decisions` in order, returning the message as is once they run out.
struct HookingHost {
    call: LocalMessagesHost,
    edit: Map<String, Value>,
    refuse_pre_request: bool,
    decisions: Mutex<Vec<Decision>>,
    pre_requests: Mutex<Vec<PublicRequest>>,
    wire_bodies: Mutex<Vec<Value>>,
    after_responses: Mutex<Vec<AnthropicMessagesResponse>>,
    opened: Mutex<usize>,
    delivered: Mutex<Vec<Bytes>>,
}

impl HookingHost {
    fn new(call: MessagesCall) -> Self {
        Self {
            call: LocalMessagesHost::new(call),
            edit: Map::new(),
            refuse_pre_request: false,
            decisions: Mutex::new(Vec::new()),
            pre_requests: Mutex::new(Vec::new()),
            wire_bodies: Mutex::new(Vec::new()),
            after_responses: Mutex::new(Vec::new()),
            opened: Mutex::new(0),
            delivered: Mutex::new(Vec::new()),
        }
    }

    fn editing(self, edit: Value) -> Self {
        Self {
            edit: object(edit),
            ..self
        }
    }

    fn refusing_pre_request(self) -> Self {
        Self {
            refuse_pre_request: true,
            ..self
        }
    }

    fn deciding(self, decisions: Vec<Decision>) -> Self {
        Self {
            decisions: Mutex::new(decisions),
            ..self
        }
    }

    fn offered_texts(&self) -> Vec<String> {
        self.after_responses
            .lock()
            .unwrap()
            .iter()
            .map(text_of)
            .collect()
    }

    fn delivered_text(&self) -> String {
        String::from_utf8(self.delivered.lock().unwrap().concat()).expect("utf-8 events")
    }
}

impl Host<Messages> for HookingHost {
    async fn project(&self) -> Result<MessagesCall, Error> {
        self.call.project().await
    }

    async fn custom_op(&self, op: std::convert::Infallible) -> Result<(), Error> {
        match op {}
    }

    async fn pre_request(&self, request: PublicRequest) -> Result<Map<String, Value>, Error> {
        self.pre_requests.lock().unwrap().push(request.clone());
        if self.refuse_pre_request {
            return Err(Error::InvalidRequest("hook refused".into()));
        }
        Ok(request
            .params
            .into_iter()
            .chain(self.edit.clone())
            .collect())
    }

    async fn before_send(
        &self,
        wire: WireRequest,
        _: &RequestContext,
    ) -> Result<WireRequest, Error> {
        self.wire_bodies.lock().unwrap().push(wire.body.clone());
        Ok(wire)
    }

    async fn after_response(&self, response: MessagesOutput) -> Result<Verdict<Messages>, Error> {
        let MessagesOutput::Message(message) = response else {
            panic!("a streamed response is never offered as a decoded message");
        };
        self.after_responses
            .lock()
            .unwrap()
            .push((*message).clone());
        let mut decisions = self.decisions.lock().unwrap();
        let decision = match decisions.is_empty() {
            true => Decision::Return,
            false => decisions.remove(0),
        };
        Ok(match decision {
            Decision::Return => Verdict::Return(MessagesOutput::Message(message)),
            Decision::Replace(replacement) => Verdict::Return(MessagesOutput::Message(replacement)),
            Decision::Resend(patch) => Verdict::Resend(patch),
        })
    }

    async fn open(&self, _: MessagesStreamHead) -> Result<Demand, Error> {
        *self.opened.lock().unwrap() += 1;
        Ok(Demand::More)
    }

    async fn deliver(&self, chunk: Bytes) -> Result<Demand, Error> {
        self.delivered.lock().unwrap().push(chunk);
        Ok(Demand::More)
    }
}

fn original_tool() -> Value {
    json!({"name": "original_tool", "description": "Lookup", "input_schema": {"type": "object"}})
}

fn original_messages() -> Value {
    json!([{"role": "user", "content": "hi"}])
}

fn message_with_text(text: &str) -> Value {
    let mut body = message_body();
    body["content"] = json!([{"type": "text", "text": text}]);
    body
}

fn response_of(text: &str) -> AnthropicMessagesResponse {
    serde_json::from_value(message_with_text(text)).expect("a well-formed message")
}

fn text_of(message: &AnthropicMessagesResponse) -> String {
    serde_json::to_value(message).expect("serializable")["content"][0]["text"]
        .as_str()
        .expect("a text block")
        .to_string()
}

/// The fixture call against `api_base`, authenticated, carrying one tool and `extra`.
fn hooked(call: MessagesCall, api_base: String, extra: Value) -> MessagesCall {
    let body: Map<String, Value> = call
        .body
        .into_iter()
        .chain(object(json!({"tools": [original_tool()]})))
        .chain(object(extra))
        .collect();
    MessagesCall {
        api_key: Some("sk-ant".into()),
        api_base: Some(api_base),
        body,
        ..call
    }
}

async fn run_hooked(host: &HookingHost) -> Result<MessagesOutput, Error> {
    litellm_host::run::run(messages_machine(Arc::new(RecordingSecrets::empty())), host).await
}

async fn message_through(host: &HookingHost) -> AnthropicMessagesResponse {
    match run_hooked(host).await.expect("messages call succeeds") {
        MessagesOutput::Message(message) => *message,
        MessagesOutput::Streamed => panic!("a non-streaming call returned a stream"),
    }
}

fn event_names(sse: &str) -> Vec<String> {
    sse.lines()
        .filter_map(|line| line.strip_prefix("event: "))
        .map(str::to_string)
        .collect()
}

fn delta_text(sse: &str) -> String {
    sse.lines()
        .filter_map(|line| line.strip_prefix("data: "))
        .map(|data| serde_json::from_str::<Value>(data).expect("event data is json"))
        .filter(|event| event["type"] == "content_block_delta")
        .filter_map(|event| event["delta"]["text"].as_str().map(str::to_string))
        .collect()
}

#[rstest]
#[case::tools("tools", json!([{"name": "renamed_tool", "input_schema": {"type": "object"}}]))]
#[case::messages("messages", json!([{"role": "user", "content": "edited"}]))]
#[tokio::test]
async fn the_pre_request_hook_sees_the_projected_request_and_its_edit_reaches_the_wire(
    call: MessagesCall,
    #[case] field: &str,
    #[case] edited: Value,
) {
    let upstream = upstream([message_response()]).await;
    let host =
        HookingHost::new(hooked(call, upstream.uri(), json!({}))).editing(json!({field: edited}));

    message_through(&host).await;

    let seen = host.pre_requests.into_inner().unwrap();
    let [seen] = <[PublicRequest; 1]>::try_from(seen)
        .unwrap_or_else(|seen| panic!("pre_request runs once, ran {} times", seen.len()));
    assert_eq!(seen.model, MODEL);
    assert_eq!(seen.custom_llm_provider, "anthropic");
    assert_eq!(seen.messages, original_messages());
    assert_eq!(
        seen.params,
        object(json!({"max_tokens": 16, "tools": [original_tool()]}))
    );
    assert!(seen.fields.contains(&"tools") && seen.fields.contains(&"stream"));
    assert!(!seen.fields.contains(&"model"));
    let sent = only_request(&upstream).await.json();
    assert_eq!(sent[field], edited);
    assert_eq!(host.wire_bodies.lock().unwrap()[0][field], edited);
}

#[rstest]
#[case::model(json!({"model": "claude-other"}))]
#[case::a_key_the_route_never_sends(json!({"max_agentic_loops": 3}))]
#[tokio::test]
async fn a_pre_request_answer_rewrites_nothing_outside_the_routes_fields(
    call: MessagesCall,
    #[case] extra: Value,
) {
    let upstream = upstream([message_response()]).await;
    let host = HookingHost::new(hooked(call, upstream.uri(), json!({}))).editing(extra);

    message_through(&host).await;

    let sent = only_request(&upstream).await.json();
    assert_eq!(sent["model"], json!(MODEL));
    assert_eq!(sent["messages"], original_messages());
    assert_eq!(sent.get("max_agentic_loops"), None);
}

#[rstest]
#[tokio::test]
async fn a_refused_pre_request_fails_the_call_before_any_request_is_sent(call: MessagesCall) {
    let upstream = upstream([message_response()]).await;
    let host = HookingHost::new(hooked(call, upstream.uri(), json!({}))).refusing_pre_request();

    let error = run_hooked(&host)
        .await
        .err()
        .expect("the hook's refusal is the call's failure");

    assert!(
        matches!(error, Error::InvalidRequest(ref message) if message == "hook refused"),
        "{error:?}"
    );
    assert!(received(&upstream).await.is_empty());
    assert!(host.wire_bodies.lock().unwrap().is_empty());
}

#[rstest]
#[tokio::test]
async fn a_caller_that_streams_still_receives_events_when_a_hook_turns_streaming_off(
    call: MessagesCall,
) {
    let upstream = upstream([message_response()]).await;
    let host = HookingHost::new(hooked(call, upstream.uri(), json!({"stream": true})))
        .editing(json!({"stream": false}));

    let outcome = run_hooked(&host).await.expect("streamed call succeeds");

    assert!(matches!(outcome, MessagesOutput::Streamed));
    assert_eq!(only_request(&upstream).await.json()["stream"], json!(false));
    assert_eq!(host.offered_texts(), ["hi"]);
    assert_eq!(*host.opened.lock().unwrap(), 1);
    let events = host.delivered_text();
    assert_eq!(
        event_names(&events),
        [
            "message_start",
            "content_block_start",
            "content_block_delta",
            "content_block_stop",
            "message_delta",
            "message_stop",
        ]
    );
    assert_eq!(delta_text(&events), "hi");
}

#[rstest]
#[tokio::test]
async fn the_decoded_message_is_offered_to_the_host_before_the_call_completes(call: MessagesCall) {
    let upstream = upstream([message_response()]).await;
    let host = HookingHost::new(hooked(call, upstream.uri(), json!({})));

    let message = message_through(&host).await;

    assert_eq!(host.after_responses.into_inner().unwrap(), [message]);
    assert_eq!(received(&upstream).await.len(), 1);
}

#[rstest]
#[tokio::test]
async fn a_replacement_completes_the_call_without_another_request(call: MessagesCall) {
    let upstream = upstream([message_response()]).await;
    let host = HookingHost::new(hooked(call, upstream.uri(), json!({})))
        .deciding(vec![Decision::Replace(Box::new(response_of("replaced")))]);

    let message = message_through(&host).await;

    assert_eq!(message, response_of("replaced"));
    assert_eq!(host.offered_texts(), ["hi"]);
    assert_eq!(received(&upstream).await.len(), 1);
}

#[rstest]
#[tokio::test]
async fn a_resend_sends_the_patched_request_and_offers_the_next_message(call: MessagesCall) {
    let upstream = upstream([
        json_response(message_with_text("first")),
        json_response(message_with_text("second")),
    ])
    .await;
    let followup = json!([
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": [{"type": "text", "text": "first"}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "found"}]},
    ]);
    let host = HookingHost::new(hooked(call, upstream.uri(), json!({}))).deciding(vec![
        Decision::Resend(object(json!({"messages": followup}))),
        Decision::Return,
    ]);

    let message = message_through(&host).await;

    assert_eq!(text_of(&message), "second");
    assert_eq!(host.offered_texts(), ["first", "second"]);
    let requests = received(&upstream).await;
    let [first, second] = <[wiremock::Request; 2]>::try_from(requests)
        .unwrap_or_else(|requests| panic!("expected two requests, got {}", requests.len()));
    assert_eq!(first.json()["messages"], original_messages());
    assert_eq!(second.json()["messages"], followup);
    assert_eq!(second.json()["tools"], first.json()["tools"]);
    assert_eq!(second.json()["max_tokens"], first.json()["max_tokens"]);
    assert_eq!(host.pre_requests.lock().unwrap().len(), 1);
    assert_eq!(host.wire_bodies.lock().unwrap().len(), 2);
}

#[rstest]
#[tokio::test]
async fn a_relayed_stream_is_never_offered_as_a_decoded_message(call: MessagesCall) {
    let upstream =
        upstream([ResponseTemplate::new(200).set_body_raw(SSE_BODY, "text/event-stream")]).await;
    let host = HookingHost::new(hooked(call, upstream.uri(), json!({"stream": true})));

    let outcome = run_hooked(&host).await.expect("streamed call succeeds");

    assert!(matches!(outcome, MessagesOutput::Streamed));
    assert!(host.after_responses.lock().unwrap().is_empty());
    assert_eq!(host.delivered_text(), SSE_BODY);
}

#[rstest]
#[tokio::test]
async fn a_null_hook_value_removes_the_projected_field(call: MessagesCall) {
    let upstream = upstream([message_response()]).await;
    let host =
        HookingHost::new(hooked(call, upstream.uri(), json!({}))).editing(json!({"tools": null}));
    message_through(&host).await;
    assert!(only_request(&upstream).await.json().get("tools").is_none());
}

#[rstest]
#[tokio::test]
async fn synthetic_events_preserve_tools_thinking_and_usage(call: MessagesCall) {
    let blocks = json!([
        {"type": "tool_use", "id": "tool-1", "name": "lookup", "input": {"query": "hello"}},
        {"type": "thinking", "thinking": "reasoning", "signature": "signed"},
        {"type": "redacted_thinking", "data": "opaque"},
    ]);
    let usage = json!({"input_tokens": 12, "output_tokens": 8, "cache_read_input_tokens": 7});
    let body = Value::Object(
        message_body()
            .as_object()
            .unwrap()
            .iter()
            .filter(|(key, _)| !matches!(key.as_str(), "content" | "usage"))
            .map(|(key, value)| (key.clone(), value.clone()))
            .chain([
                ("content".into(), blocks.clone()),
                ("usage".into(), usage.clone()),
            ])
            .collect(),
    );
    let upstream = upstream([json_response(body)]).await;
    let host = HookingHost::new(hooked(call, upstream.uri(), json!({"stream": true})))
        .editing(json!({"stream": false}));
    assert!(matches!(
        run_hooked(&host).await.unwrap(),
        MessagesOutput::Streamed
    ));
    let events: Vec<Value> = host
        .delivered_text()
        .lines()
        .filter_map(|line| line.strip_prefix("data: "))
        .map(|data| serde_json::from_str(data).unwrap())
        .collect();
    assert_eq!(
        events[0]["message"]["usage"]["cache_read_input_tokens"],
        usage["cache_read_input_tokens"]
    );
    assert_eq!(events[1]["content_block"]["id"], blocks[0]["id"]);
    assert_eq!(
        serde_json::from_str::<Value>(events[2]["delta"]["partial_json"].as_str().unwrap())
            .unwrap(),
        blocks[0]["input"]
    );
    assert_eq!(events[5]["delta"]["thinking"], blocks[1]["thinking"]);
    assert_eq!(events[6]["delta"]["signature"], blocks[1]["signature"]);
    assert_eq!(events[8]["content_block"], blocks[2]);
    assert_eq!(events[10]["usage"], usage);
}
