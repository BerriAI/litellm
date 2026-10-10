use std::{sync::OnceLock, time::Duration};

use futures_util::{FutureExt, TryStreamExt};
use litellm_host::interceptors::{
    ExecutionFacts, Interceptors, RawResponse, RequestContext, ResultSource, WireRequest,
};
use litellm_inference_messages::{Error, MessagesCall, MessagesCallResponse, MessagesRoute};
use litellm_inference_testing::live::{LiveResources, model, within_deadline};
use litellm_llms_types::formats::messages::MessagesResponse;
use litellm_router_types::{GithubCopilotSession, LitellmParams};
use serde_json::{Value, json};

pub(super) struct LiveCall {
    provider: &'static str,
    facts: OnceLock<ExecutionFacts>,
}

impl LiveCall {
    pub(super) fn new(provider: &'static str) -> Self {
        Self {
            provider,
            facts: OnceLock::new(),
        }
    }

    pub(super) fn facts(&self) -> &ExecutionFacts {
        self.facts
            .get()
            .expect("the route must deliver execution facts")
    }

    pub(super) fn assert_provider_result(&self) {
        let facts = self.facts();
        assert_eq!(facts.provider.provider, self.provider);
        assert_eq!(facts.source, ResultSource::Provider);
        println!(
            "{}",
            json!({"route": "messages", "provider": self.provider, "source": "provider"})
        );
    }
}

impl Interceptors<Error> for LiveCall {
    async fn result_ready(&self, facts: ExecutionFacts) -> Result<(), Error> {
        assert!(
            self.facts.set(facts).is_ok(),
            "result must be accepted exactly once"
        );
        Ok(())
    }

    async fn before_provider_request(
        &self,
        wire: WireRequest,
        context: RequestContext,
    ) -> Result<WireRequest, Error> {
        assert_eq!(context.custom_llm_provider, self.provider);
        println!(
            "{}",
            json!({"route": "messages", "provider": self.provider, "model": context.model})
        );
        Ok(wire)
    }

    async fn after_provider_response(&self, _: RawResponse) -> Result<(), Error> {
        Ok(())
    }
}

#[rstest::fixture]
pub(super) fn route() -> MessagesRoute {
    let resources = LiveResources::default();
    MessagesRoute::new(resources.http, resources.auth, resources.secrets)
}

pub(super) async fn request(
    route: &MessagesRoute,
    call: MessagesCall,
    host: &LiveCall,
) -> MessagesCallResponse {
    route
        .execute(call, host, None)
        .boxed()
        .await
        .expect("live request failed")
}

pub(super) fn call(provider: &str, payload: Value) -> MessagesCall {
    let Value::Object(fields) = payload else {
        panic!("live payload must be an object")
    };
    let body = serde_json::from_value(Value::Object(
        [("max_tokens".into(), json!(128))]
            .into_iter()
            .chain(fields)
            .chain([("model".into(), json!(model("messages", provider)))])
            .collect(),
    ))
    .expect("live payload must be a valid Messages request");
    MessagesCall {
        body,
        api_key: None,
        api_base: None,
        custom_llm_provider: Some(provider.into()),
        litellm_params: host_params(provider),
        extra_headers: None,
        provider_specific_header: None,
        timeout: Some(Duration::from_secs(60)),
        shaping: Default::default(),
    }
}

fn host_params(provider: &str) -> LitellmParams {
    if provider != "github_copilot" {
        return LitellmParams::default();
    }
    let required = |name: &str| {
        std::env::var(name)
            .ok()
            .filter(|value| !value.trim().is_empty())
            .unwrap_or_else(|| panic!("set {name} from the Copilot session Python resolved"))
    };
    LitellmParams {
        github_copilot_session: Some(GithubCopilotSession {
            token: litellm_auth::SecretValue::new(required("LITELLM_LIVE_GITHUB_COPILOT_TOKEN")),
            api_base: required("LITELLM_LIVE_GITHUB_COPILOT_API_BASE"),
        }),
        ..Default::default()
    }
}

pub(super) fn complete(response: MessagesCallResponse) -> Box<MessagesResponse> {
    match response {
        MessagesCallResponse::Complete(message) => message,
        MessagesCallResponse::Stream { .. } => panic!("a complete request returned a stream"),
    }
}

pub(super) fn assert_text(message: &MessagesResponse) {
    assert!(!message.id.is_empty());
    assert_eq!(message.role, "assistant");
    assert!(message.stop_reason.is_some());
    assert!(
        message
            .content
            .iter()
            .any(|block| block["text"].as_str().is_some_and(|text| !text.is_empty()))
    );
    assert!(
        message
            .usage
            .as_ref()
            .and_then(|usage| usage["output_tokens"].as_u64())
            .is_some_and(|count| count > 0)
    );
}

pub(super) async fn stream_events(response: MessagesCallResponse) -> Vec<Value> {
    let MessagesCallResponse::Stream { head, chunks } = response else {
        panic!("stream request returned a complete response")
    };
    let content_types: Vec<_> = head
        .headers
        .iter()
        .filter(|(name, _)| name.eq_ignore_ascii_case("content-type"))
        .map(|(_, value)| value.split(';').next().unwrap().trim())
        .collect();
    assert_eq!(content_types, ["text/event-stream"]);
    let frames = litellm_framer::frames(chunks, litellm_framer::sse::SseCodec::default())
        .try_collect::<Vec<_>>()
        .await
        .expect("live SSE framing failed");
    let data_frames = match frames.last() {
        Some(frame) if frame.data == "[DONE]" => &frames[..frames.len() - 1],
        _ => frames.as_slice(),
    };
    let events: Vec<Value> = data_frames
        .iter()
        .map(|frame| serde_json::from_str(&frame.data).expect("SSE data must be JSON"))
        .collect();
    assert_eq!(
        events.first().map(|event| event["type"].as_str()),
        Some(Some("message_start"))
    );
    assert_eq!(
        events.last().map(|event| event["type"].as_str()),
        Some(Some("message_stop"))
    );
    assert!(!events.iter().any(|event| event["type"] == "error"));
    assert!(events.iter().any(|event| {
        event["usage"]["output_tokens"]
            .as_u64()
            .is_some_and(|count| count > 0)
    }));
    println!("{}", json!({"events": events}));
    events
}

fn prompt() -> Value {
    json!({"role": "user", "content": "Use the echo tool with value live-check, then tell me the result."})
}

fn tools() -> Value {
    json!([{"name": "echo", "description": "Return a value", "input_schema": {
        "type": "object", "properties": {"value": {"type": "string", "enum": ["live-check"]}}, "required": ["value"]
    }}])
}

fn fragments(events: &[Value], index: u64, field: &str) -> String {
    events
        .iter()
        .filter(|event| event["index"].as_u64() == Some(index))
        .filter_map(|event| event["delta"][field].as_str())
        .collect()
}

fn streamed_block(start: &Value, events: &[Value]) -> Value {
    let index = start["index"]
        .as_u64()
        .expect("content block must have an index");
    let block = &start["content_block"];
    match block["type"].as_str() {
        Some("tool_use") => {
            let partial = fragments(events, index, "partial_json");
            let input = match partial.is_empty() {
                true => block["input"].clone(),
                false => {
                    serde_json::from_str(&partial).expect("streamed tool arguments must be JSON")
                }
            };
            json!({"type": "tool_use", "id": block["id"], "name": block["name"], "input": input})
        }
        Some("text") => {
            json!({"type": "text", "text": format!("{}{}", block["text"].as_str().unwrap_or_default(), fragments(events, index, "text"))})
        }
        Some("thinking") => {
            json!({"type": "thinking", "thinking": format!("{}{}", block["thinking"].as_str().unwrap_or_default(), fragments(events, index, "thinking")),
            "signature": format!("{}{}", block["signature"].as_str().unwrap_or_default(), fragments(events, index, "signature"))})
        }
        _ => block.clone(),
    }
}

pub(super) async fn tool_round_trip(
    route: MessagesRoute,
    provider: &'static str,
    stream: bool,
    tool_params: Value,
) {
    within_deadline(async {
        let first_host = LiveCall::new(provider);
        let payload = json!({"messages": [prompt()], "tools": tools(), "stream": stream});
        let fields = payload.as_object().expect("tool payload must be an object").clone();
        let params = tool_params.as_object().expect("tool parameters must be an object").clone();
        let first = request(&route, call(provider, Value::Object(fields.into_iter().chain(params).collect())), &first_host).await;
        let content = match stream {
            false => {
                let message = complete(first);
                assert_eq!(message.stop_reason.as_deref(), Some("tool_use"));
                message.content
            }
            true => {
                let events = stream_events(first).await;
                assert!(events.iter().any(|event| event["delta"]["stop_reason"] == "tool_use"));
                events.iter().filter(|event| event["type"] == "content_block_start")
                    .map(|event| streamed_block(event, &events)).collect()
            }
        };
        let tool = content.iter().find(|block| block["type"] == "tool_use").expect("requested tool must be returned");
        assert_eq!(tool["name"], "echo");
        assert_eq!(tool["input"]["value"], "live-check");
        let tool_id = tool["id"].as_str().filter(|id| !id.is_empty()).expect("tool must have an id");
        first_host.assert_provider_result();
        let second_host = LiveCall::new(provider);
        let second = complete(request(&route, call(provider, json!({
            "tools": tools(), "messages": [prompt(),
                {"role": "assistant", "content": content},
                {"role": "user", "content": [{"type": "tool_result", "tool_use_id": tool_id, "content": "live-check"}]}]
        })), &second_host).await);
        assert_text(&second);
        second_host.assert_provider_result();
        println!("{}", serde_json::to_string(&second).unwrap());
    })
    .await;
}
