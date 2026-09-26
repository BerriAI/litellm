use std::convert::Infallible;

use futures_util::StreamExt;
use litellm_auth::{CredentialPlacement, SecretValue};
use litellm_auth_aws::{
    AwsCredentialSource, bedrock_model_id_and_region,
    constants::{
        AWS_BEARER_TOKEN_BEDROCK, AWS_BEDROCK_RUNTIME_ENDPOINT, AWS_DEFAULT_REGION, AWS_REGION,
        AWS_REGION_NAME, BEDROCK_RUNTIME_ENDPOINT_TEMPLATE, BEDROCK_SERVICE,
    },
    resolve_bedrock_region,
};
use litellm_types::llms::anthropic_messages::anthropic_request::AnthropicMessagesRequest;
use serde_json::{Map, Value};

use crate::{
    Error,
    anthropic::messages::streaming_iterator::{AnthropicMessagesStreamEvent, AnthropicStreamUsage},
    base_llm::{
        anthropic_messages::{
            streaming::{ByteStream, EventStream, StreamDecoder},
            transformation::{
                BaseAnthropicMessagesConfig, Headers, MessagesTransformContext,
                ValidatedEnvironment,
            },
        },
        auth::AuthScheme,
        base_model_iterator::{StreamError, StreamTransformer, transform_stream},
    },
    bedrock::chat::invoke_handler::{decode_invoke_anthropic_chunk, invoke_chunk_stream},
};

const INVOCATION_METRICS_KEY: &str = "amazon-bedrock-invocationMetrics";

const METRICS_USAGE_KEYS: [(&str, &str); 4] = [
    ("input_tokens", "inputTokenCount"),
    ("output_tokens", "outputTokenCount"),
    ("cache_read_input_tokens", "cacheReadInputTokenCount"),
    ("cache_creation_input_tokens", "cacheWriteInputTokenCount"),
];

const INVOKE_PATH: &str = "invoke";
const INVOKE_STREAM_PATH: &str = "invoke-with-response-stream";
const INVOKE_MODEL_PREFIX: &str = "invoke/";

const SECRET_NAMES: &[&str] = &[
    AWS_BEARER_TOKEN_BEDROCK,
    AWS_BEDROCK_RUNTIME_ENDPOINT,
    AWS_REGION_NAME,
    AWS_REGION,
    AWS_DEFAULT_REGION,
];

pub struct AmazonAnthropicClaudeMessagesConfig;

pub const BEDROCK_ANTHROPIC_MESSAGES_CONFIG: AmazonAnthropicClaudeMessagesConfig =
    AmazonAnthropicClaudeMessagesConfig;

fn bearer_token(
    api_key: Option<&str>,
    env_lookup: &dyn Fn(&str) -> Option<String>,
) -> Option<String> {
    match api_key {
        Some(key) => Some(key.to_string()),
        None => env_lookup(AWS_BEARER_TOKEN_BEDROCK),
    }
    .filter(|token| !token.is_empty())
}

fn invoke_url(
    api_base: Option<&str>,
    model: &str,
    env_lookup: &dyn Fn(&str) -> Option<String>,
    path: &str,
) -> String {
    let (model_id, model_region) =
        bedrock_model_id_and_region(model.strip_prefix(INVOKE_MODEL_PREFIX).unwrap_or(model));
    let region = resolve_bedrock_region(model_region.as_deref(), &Map::new(), env_lookup);
    let endpoint = api_base
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_string)
        .or_else(|| env_lookup(AWS_BEDROCK_RUNTIME_ENDPOINT))
        .unwrap_or_else(|| BEDROCK_RUNTIME_ENDPOINT_TEMPLATE.replace("{region}", &region));
    format!("{}/model/{model_id}/{path}", endpoint.trim_end_matches('/'))
}

impl BaseAnthropicMessagesConfig for AmazonAnthropicClaudeMessagesConfig {
    fn get_complete_url(
        &self,
        api_base: Option<&str>,
        model: &str,
        env_lookup: &dyn Fn(&str) -> Option<String>,
    ) -> Result<String, Error> {
        Ok(invoke_url(api_base, model, env_lookup, INVOKE_PATH))
    }

    fn complete_stream_url(
        &self,
        api_base: Option<&str>,
        model: &str,
        env_lookup: &dyn Fn(&str) -> Option<String>,
    ) -> Result<String, Error> {
        Ok(invoke_url(api_base, model, env_lookup, INVOKE_STREAM_PATH))
    }

    fn transform_anthropic_messages_request(
        &self,
        _request: AnthropicMessagesRequest,
        _context: &MessagesTransformContext,
    ) -> Result<AnthropicMessagesRequest, Error> {
        Err(Error::Unsupported(
            "Bedrock invoke messages request shaping",
        ))
    }

    fn secret_names(&self) -> &'static [&'static str] {
        SECRET_NAMES
    }

    /// Python reads `api_key` as the Bedrock bearer token and consults the env only when the
    /// caller passed none. Without one the request is signed with SigV4.
    fn validate_environment(
        &self,
        headers: Headers,
        api_key: Option<&str>,
        model: &str,
        env_lookup: &dyn Fn(&str) -> Option<String>,
    ) -> Result<ValidatedEnvironment, Error> {
        if let Some(token) = bearer_token(api_key, env_lookup) {
            return Ok(ValidatedEnvironment {
                headers,
                auth: AuthScheme::Credential {
                    placement: CredentialPlacement::Bearer,
                    secret: SecretValue::new(token),
                },
            });
        }
        let (_, model_region) =
            bedrock_model_id_and_region(model.strip_prefix(INVOKE_MODEL_PREFIX).unwrap_or(model));
        let params = Map::new();
        Ok(ValidatedEnvironment {
            headers,
            auth: AuthScheme::AwsSigV4 {
                region: resolve_bedrock_region(model_region.as_deref(), &params, env_lookup),
                service: BEDROCK_SERVICE,
                credentials: Box::new(AwsCredentialSource::from_params(&params, env_lookup)),
            },
        })
    }

    fn default_headers(&self) -> &'static [(&'static str, &'static str)] {
        &[("content-type", "application/json")]
    }

    fn stream_decoder(&self) -> Option<StreamDecoder> {
        Some(bedrock_anthropic_messages_event_stream)
    }
}

fn with_invocation_usage(chunk: Value) -> Value {
    match chunk {
        Value::Object(fields) => Value::Object(with_metrics_usage(fields)),
        other => other,
    }
}

fn with_metrics_usage(mut fields: Map<String, Value>) -> Map<String, Value> {
    let Some(Value::Object(metrics)) = fields.remove(INVOCATION_METRICS_KEY) else {
        return fields;
    };
    if metrics.is_empty() {
        return fields;
    }
    let preserved = match fields.remove("usage") {
        Some(Value::Object(usage)) => usage,
        _ => Map::new(),
    };
    let usage: Map<String, Value> = METRICS_USAGE_KEYS
        .iter()
        .filter_map(|(anthropic, metric)| {
            Some((anthropic.to_string(), metrics.get(*metric)?.clone()))
        })
        .chain(preserved)
        .collect();
    fields.insert("usage".to_string(), Value::Object(usage));
    fields
}

pub fn bedrock_anthropic_messages_event_stream(bytes: ByteStream) -> EventStream {
    let events = invoke_chunk_stream(bytes)
        .map(|chunk| decode_invoke_anthropic_chunk(with_invocation_usage(chunk?)));
    Box::pin(
        transform_stream(events, MessageStopUsagePromoter::default())
            .map(|item| item.map_err(StreamError::into_decode)),
    )
}

#[derive(Default)]
pub struct MessageStopUsagePromoter {
    pending_delta: Option<AnthropicMessagesStreamEvent>,
    start_usage: Option<AnthropicStreamUsage>,
}

fn promoted_usage(
    delta: Option<AnthropicStreamUsage>,
    stop: Option<&AnthropicStreamUsage>,
    start: Option<&AnthropicStreamUsage>,
) -> Option<AnthropicStreamUsage> {
    let delta = delta.unwrap_or_default();
    let merged = AnthropicStreamUsage {
        input_tokens: stop
            .and_then(|stop| stop.input_tokens)
            .or(delta.input_tokens),
        cache_creation_input_tokens: stop
            .and_then(|stop| stop.cache_creation_input_tokens)
            .or(delta.cache_creation_input_tokens)
            .or_else(|| start.and_then(|start| start.cache_creation_input_tokens)),
        cache_read_input_tokens: stop
            .and_then(|stop| stop.cache_read_input_tokens)
            .or(delta.cache_read_input_tokens)
            .or_else(|| start.and_then(|start| start.cache_read_input_tokens)),
        extra: delta
            .extra
            .into_iter()
            .chain(
                start
                    .and_then(|start| start.extra.get_key_value("cache_creation"))
                    .map(|(key, value)| (key.clone(), value.clone())),
            )
            .fold(Map::new(), |mut extra, (key, value)| {
                extra.entry(key).or_insert(value);
                extra
            }),
        ..delta
    };
    (merged != AnthropicStreamUsage::default()).then_some(merged)
}

fn promoted(
    event: AnthropicMessagesStreamEvent,
    stop: Option<&AnthropicStreamUsage>,
    start: Option<&AnthropicStreamUsage>,
) -> AnthropicMessagesStreamEvent {
    match event {
        AnthropicMessagesStreamEvent::MessageDelta {
            delta,
            usage,
            context_management,
        } => AnthropicMessagesStreamEvent::MessageDelta {
            delta,
            usage: promoted_usage(usage, stop, start),
            context_management,
        },
        other => other,
    }
}

impl StreamTransformer for MessageStopUsagePromoter {
    type Input = AnthropicMessagesStreamEvent;
    type Output = AnthropicMessagesStreamEvent;
    type Error = Infallible;

    fn transform(
        &mut self,
        input: AnthropicMessagesStreamEvent,
    ) -> Result<Vec<AnthropicMessagesStreamEvent>, Infallible> {
        let pending = self.pending_delta.take();
        match input {
            AnthropicMessagesStreamEvent::MessageDelta { .. } => {
                self.pending_delta = Some(input);
                Ok(pending.into_iter().collect())
            }
            AnthropicMessagesStreamEvent::MessageStop { usage } => Ok(pending
                .map(|delta| promoted(delta, usage.as_ref(), self.start_usage.as_ref()))
                .into_iter()
                .chain([AnthropicMessagesStreamEvent::MessageStop { usage }])
                .collect()),
            AnthropicMessagesStreamEvent::MessageStart { message } => {
                self.start_usage = Some(message.usage.clone());
                Ok(pending
                    .into_iter()
                    .chain([AnthropicMessagesStreamEvent::MessageStart { message }])
                    .collect())
            }
            other => Ok(pending.into_iter().chain([other]).collect()),
        }
    }

    fn finish(&mut self) -> Result<Vec<AnthropicMessagesStreamEvent>, Infallible> {
        Ok(self
            .pending_delta
            .take()
            .map(|delta| promoted(delta, None, self.start_usage.as_ref()))
            .into_iter()
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use aws_smithy_eventstream::frame::write_message_to;
    use aws_smithy_types::event_stream::{Header, HeaderValue, Message};
    use base64::{Engine, engine::general_purpose::STANDARD};
    use bytes::Bytes;
    use futures_util::TryStreamExt;
    use rstest::rstest;
    use serde_json::json;

    use litellm_auth_aws::constants::DEFAULT_BEDROCK_REGION;

    use super::*;
    use crate::base_llm::anthropic_messages::streaming::encode_anthropic_sse;

    fn event(value: Value) -> AnthropicMessagesStreamEvent {
        serde_json::from_value(value).unwrap()
    }

    fn message_start(usage: Value) -> AnthropicMessagesStreamEvent {
        event(json!({
            "type": "message_start",
            "message": {
                "id": "msg_1", "type": "message", "role": "assistant", "model": "m",
                "content": [], "stop_reason": null, "stop_sequence": null, "usage": usage
            }
        }))
    }

    fn message_delta(usage: Value) -> AnthropicMessagesStreamEvent {
        event(json!({
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn"},
            "usage": usage
        }))
    }

    fn message_stop(usage: Option<Value>) -> AnthropicMessagesStreamEvent {
        match usage {
            Some(usage) => event(json!({"type": "message_stop", "usage": usage})),
            None => event(json!({"type": "message_stop"})),
        }
    }

    fn promote(events: Vec<AnthropicMessagesStreamEvent>) -> Vec<AnthropicMessagesStreamEvent> {
        let mut promoter = MessageStopUsagePromoter::default();
        let mut output: Vec<_> = events
            .into_iter()
            .flat_map(|event| promoter.transform(event).unwrap())
            .collect();
        output.extend(promoter.finish().unwrap());
        output
    }

    #[rstest]
    #[case::cache_fields_on_message_stop(
        json!({"input_tokens": 10, "output_tokens": 0}),
        json!({"output_tokens": 5}),
        Some(json!({"input_tokens": 3, "cache_read_input_tokens": 100, "cache_creation_input_tokens": 20})),
        json!({"input_tokens": 3, "output_tokens": 5, "cache_read_input_tokens": 100, "cache_creation_input_tokens": 20}),
    )]
    #[case::cache_only_on_message_start(
        json!({"input_tokens": 10, "output_tokens": 0, "cache_read_input_tokens": 80, "cache_creation_input_tokens": 4, "cache_creation": {"ephemeral_5m_input_tokens": 4}}),
        json!({"output_tokens": 5}),
        Some(json!({"input_tokens": 10})),
        json!({"input_tokens": 10, "output_tokens": 5, "cache_read_input_tokens": 80, "cache_creation_input_tokens": 4, "cache_creation": {"ephemeral_5m_input_tokens": 4}}),
    )]
    #[case::message_stop_wins_over_message_start(
        json!({"input_tokens": 10, "cache_read_input_tokens": 80}),
        json!({"output_tokens": 5}),
        Some(json!({"cache_read_input_tokens": 100})),
        json!({"output_tokens": 5, "cache_read_input_tokens": 100}),
    )]
    #[case::delta_cache_fields_are_kept(
        json!({"input_tokens": 10, "cache_read_input_tokens": 80}),
        json!({"output_tokens": 5, "cache_read_input_tokens": 7}),
        None,
        json!({"output_tokens": 5, "cache_read_input_tokens": 7}),
    )]
    fn message_delta_usage_is_completed_from_stop_then_start(
        #[case] start: Value,
        #[case] delta: Value,
        #[case] stop: Option<Value>,
        #[case] expected: Value,
    ) {
        let output = promote(vec![
            message_start(start),
            message_delta(delta),
            message_stop(stop.clone()),
        ]);

        assert_eq!(output.len(), 3);
        assert_eq!(output[1], message_delta(expected));
        assert_eq!(output[2], message_stop(stop));
    }

    #[test]
    fn a_delta_is_flushed_with_start_usage_when_the_stream_ends_without_a_stop() {
        let output = promote(vec![
            message_start(json!({"input_tokens": 10, "cache_read_input_tokens": 80})),
            message_delta(json!({"output_tokens": 5})),
        ]);

        assert_eq!(
            output[1],
            message_delta(json!({"output_tokens": 5, "cache_read_input_tokens": 80}))
        );
    }

    #[test]
    fn events_around_the_delta_keep_their_order() {
        let ping = event(json!({"type": "ping"}));
        let output = promote(vec![
            message_delta(json!({"output_tokens": 5})),
            ping.clone(),
            message_stop(None),
        ]);

        assert_eq!(
            output,
            vec![
                message_delta(json!({"output_tokens": 5})),
                ping,
                message_stop(None)
            ]
        );
    }

    #[rstest]
    #[case::metrics_fill_missing_usage(
        json!({"type": "message_stop", "amazon-bedrock-invocationMetrics": {"inputTokenCount": 3, "outputTokenCount": 9}}),
        json!({"type": "message_stop", "usage": {"input_tokens": 3, "output_tokens": 9}}),
    )]
    #[case::the_chunks_own_usage_wins(
        json!({"type": "message_stop", "usage": {"input_tokens": 1}, "amazon-bedrock-invocationMetrics": {"inputTokenCount": 3, "cacheReadInputTokenCount": 40}}),
        json!({"type": "message_stop", "usage": {"cache_read_input_tokens": 40, "input_tokens": 1}}),
    )]
    #[case::no_metrics_leaves_the_chunk(
        json!({"type": "message_stop"}),
        json!({"type": "message_stop"}),
    )]
    #[case::empty_metrics_are_dropped(
        json!({"type": "message_stop", "amazon-bedrock-invocationMetrics": {}}),
        json!({"type": "message_stop"}),
    )]
    fn invocation_metrics_become_anthropic_usage(#[case] chunk: Value, #[case] expected: Value) {
        assert_eq!(with_invocation_usage(chunk), expected);
    }

    fn aws_frame(chunk: &Value) -> Vec<u8> {
        let payload = json!({"bytes": STANDARD.encode(chunk.to_string())});
        let message = Message::new(Bytes::from(serde_json::to_vec(&payload).unwrap())).add_header(
            Header::new(":event-type", HeaderValue::String("chunk".into())),
        );
        let mut wire = Vec::new();
        write_message_to(&message, &mut wire).unwrap();
        wire
    }

    #[tokio::test]
    async fn bedrock_stream_yields_the_sse_an_anthropic_client_reads() {
        let chunks = [
            json!({"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 5}}),
            json!({"type": "message_stop", "amazon-bedrock-invocationMetrics": {"inputTokenCount": 3, "cacheReadInputTokenCount": 40}}),
        ];
        let wire: Vec<u8> = chunks.iter().flat_map(aws_frame).collect();
        let bytes: ByteStream = futures_util::stream::iter(
            wire.chunks(7)
                .map(|chunk| Ok(Bytes::copy_from_slice(chunk)))
                .collect::<Vec<_>>(),
        )
        .boxed();

        let sse = bedrock_anthropic_messages_event_stream(bytes)
            .map_ok(|event| encode_anthropic_sse(&event).unwrap())
            .try_collect::<Vec<_>>()
            .await
            .unwrap()
            .concat();

        let expected: Vec<u8> = [
            message_delta(
                json!({"output_tokens": 5, "cache_read_input_tokens": 40, "input_tokens": 3}),
            ),
            message_stop(Some(
                json!({"input_tokens": 3, "cache_read_input_tokens": 40}),
            )),
        ]
        .iter()
        .flat_map(|event| encode_anthropic_sse(event).unwrap())
        .collect();
        assert_eq!(sse, expected);
    }

    #[test]
    fn config_uses_the_streaming_url_only_for_streams() {
        let env = |_: &str| -> Option<String> { None };
        let config = AmazonAnthropicClaudeMessagesConfig;

        assert_eq!(
            config
                .get_complete_url(None, "anthropic.claude-3", &env)
                .unwrap(),
            config
                .complete_stream_url(None, "anthropic.claude-3", &env)
                .unwrap()
                .replace(INVOKE_STREAM_PATH, INVOKE_PATH)
        );
    }

    #[rstest]
    #[case::an_explicit_key_is_a_bearer_token(Some("token"), None, Some("token"))]
    #[case::the_env_token_is_a_bearer_token(None, Some("env-token"), Some("env-token"))]
    #[case::no_token_signs_with_sigv4(None, None, None)]
    fn requests_sign_only_without_a_bearer_token(
        #[case] api_key: Option<&str>,
        #[case] env_token: Option<&str>,
        #[case] expected_bearer: Option<&str>,
    ) {
        let env = |name: &str| {
            (name == AWS_BEARER_TOKEN_BEDROCK)
                .then(|| env_token.map(str::to_string))
                .flatten()
        };
        let validated = AmazonAnthropicClaudeMessagesConfig
            .validate_environment(
                vec![("authorization".into(), "Bearer forwarded".into())],
                api_key,
                "anthropic.claude-3",
                &env,
            )
            .unwrap();
        match (validated.auth, expected_bearer) {
            (
                AuthScheme::Credential {
                    placement: CredentialPlacement::Bearer,
                    secret,
                },
                Some(expected),
            ) => assert_eq!(secret.expose(), expected),
            (
                AuthScheme::AwsSigV4 {
                    region, service, ..
                },
                None,
            ) => {
                assert_eq!(
                    (region.as_str(), service),
                    (DEFAULT_BEDROCK_REGION, BEDROCK_SERVICE)
                );
            }
            (other, _) => panic!("unexpected auth {other:?}"),
        }
    }
}
