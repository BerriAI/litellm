use std::convert::Infallible;

use crate::{
    anthropic::messages::handler::shape_anthropic_messages_request,
    base_llm::messages::context::MessagesTransformContext,
};
use futures_util::StreamExt;
use litellm_auth::AwsParams;
use litellm_auth::{CredentialPlacement, SecretValue};
use litellm_auth_aws::{
    AwsCredentialSource, bedrock_model_id_and_region,
    constants::{
        AWS_BEARER_TOKEN_BEDROCK, AWS_BEDROCK_RUNTIME_ENDPOINT, AWS_DEFAULT_REGION, AWS_REGION,
        AWS_REGION_NAME, BEDROCK_RUNTIME_ENDPOINT_TEMPLATE, BEDROCK_SERVICE,
    },
    resolve_bedrock_region,
};
use litellm_llms_types::formats::messages::{
    ContextEdit, ContextManagement, MessagesOptionalParams, MessagesRequest,
    streaming::{MessagesStreamEvent, MessagesStreamUsage},
};
use litellm_llms_types::{
    providers::anthropic::{BetaProvider, BetaSet},
    recognized::Recognized,
};
use litellm_router_types::LitellmParams;
use serde_json::{Map, Value};

use crate::{
    Error,
    base_llm::{
        auth::AuthScheme,
        base_model_iterator::{StreamError, StreamTransformer, transform_stream},
        messages::{
            streaming::{ByteStream, EventStream, StreamDecoder},
            transformation::{BaseMessagesConfig, Headers, ValidatedEnvironment},
        },
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
pub const BEDROCK_ANTHROPIC_VERSION: &str = "bedrock-2023-05-31";
const BODY_FIELDS: &[&str] = &[
    "anthropic_version",
    "max_tokens",
    "messages",
    "anthropic_beta",
    "system",
    "stop_sequences",
    "temperature",
    "top_p",
    "top_k",
    "tools",
    "tool_choice",
    "thinking",
    "metadata",
    "output_config",
    "safeguards",
    "context_management",
];

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
    aws: &AwsParams,
    stream: bool,
    env_lookup: &dyn Fn(&str) -> Option<String>,
) -> String {
    let (model_id, model_region) =
        bedrock_model_id_and_region(model.strip_prefix(INVOKE_MODEL_PREFIX).unwrap_or(model));
    let region = resolve_bedrock_region(model_region.as_deref(), aws, env_lookup);
    let path = if stream {
        INVOKE_STREAM_PATH
    } else {
        INVOKE_PATH
    };
    let configured = |value: Option<&str>| {
        value
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(str::to_string)
    };
    let endpoint = configured(api_base)
        .or_else(|| configured(aws.aws_bedrock_runtime_endpoint.as_deref()))
        .or_else(|| env_lookup(AWS_BEDROCK_RUNTIME_ENDPOINT))
        .unwrap_or_else(|| BEDROCK_RUNTIME_ENDPOINT_TEMPLATE.replace("{region}", &region));
    let model_segment: String = url::form_urlencoded::byte_serialize(model_id.as_bytes())
        .map(|part| if part == "+" { "%20" } else { part })
        .collect();
    format!(
        "{}/model/{model_segment}/{path}",
        endpoint.trim_end_matches('/')
    )
}

impl BaseMessagesConfig for AmazonAnthropicClaudeMessagesConfig {
    fn shape_request(
        &self,
        request: MessagesRequest,
        reasoning_auto_summary: bool,
    ) -> Result<MessagesRequest, Error> {
        shape_anthropic_messages_request(request, reasoning_auto_summary)
    }

    fn get_complete_url(
        &self,
        api_base: Option<&str>,
        model: &str,
        litellm_params: &LitellmParams,
        stream: bool,
        env_lookup: &dyn Fn(&str) -> Option<String>,
    ) -> Result<String, Error> {
        Ok(invoke_url(
            api_base,
            model,
            &litellm_params.aws,
            stream,
            env_lookup,
        ))
    }

    fn transform_anthropic_messages_request(
        &self,
        request: MessagesRequest,
        context: &MessagesTransformContext,
    ) -> Result<MessagesRequest, Error> {
        let request = crate::base_llm::messages::normalization::normalize_system_role_messages(
            request,
            context
                .thinking
                .capabilities
                .supports_mid_conversation_system,
        );
        let request = crate::anthropic::messages::transformation::transform_messages_request(
            request, context,
        )?;
        if request.params.tools.as_ref().is_some_and(|tools| {
            tools.iter().any(|tool| {
                let Recognized::Unrecognized(tool) = tool else {
                    return false;
                };
                tool.get("type")
                    .and_then(Value::as_str)
                    .is_some_and(|kind| kind.starts_with("web_search"))
            })
        }) {
            return Err(Error::Unsupported("Bedrock server-side web search tools"));
        }
        let context_management = request
            .params
            .context_management
            .and_then(invoke_context_management);
        Ok(MessagesRequest {
            params: MessagesOptionalParams {
                context_management,
                ..request.params
            },
            ..request
        })
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
        litellm_params: &LitellmParams,
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
        let params = &litellm_params.aws;
        Ok(ValidatedEnvironment {
            headers,
            auth: AuthScheme::AwsSigV4 {
                region: resolve_bedrock_region(model_region.as_deref(), params, env_lookup),
                service: BEDROCK_SERVICE,
                credentials: Box::new(AwsCredentialSource::from_params(params, env_lookup)),
            },
        })
    }

    fn default_headers(&self) -> &'static [(&'static str, &'static str)] {
        &[("content-type", "application/json")]
    }

    fn request_headers(&self, headers: Headers, request: &MessagesRequest) -> Headers {
        crate::anthropic::common_utils::merge_beta_headers(
            headers,
            crate::anthropic::messages::transformation::provider_feature_betas(
                request,
                BetaProvider::Bedrock,
            ),
        )
    }

    fn wire_body(&self, body: Value, headers: Headers) -> (Value, Headers) {
        let betas: BetaSet = crate::anthropic::common_utils::existing_betas(&headers)
            .iter()
            .filter_map(|beta| beta.on(BetaProvider::Bedrock))
            .collect();
        let Value::Object(fields) = body else {
            return (body, headers);
        };
        let body = Value::Object(
            fields
                .into_iter()
                .filter(|(key, _)| {
                    BODY_FIELDS.contains(&key.as_str())
                        && key != "anthropic_beta"
                        && key != "anthropic_version"
                })
                .chain([(
                    "anthropic_version".into(),
                    Value::String(BEDROCK_ANTHROPIC_VERSION.into()),
                )])
                .chain((!betas.is_empty()).then(|| {
                    (
                        "anthropic_beta".into(),
                        Value::Array(
                            betas
                                .iter()
                                .map(|beta| Value::String(beta.as_str().into()))
                                .collect(),
                        ),
                    )
                }))
                .collect(),
        );
        (
            body,
            headers
                .into_iter()
                .filter(|(name, _)| !name.eq_ignore_ascii_case("anthropic-beta"))
                .collect(),
        )
    }

    fn stream_decoder(&self) -> Option<StreamDecoder> {
        Some(bedrock_anthropic_messages_event_stream)
    }
}

fn invoke_context_management(
    context: Recognized<ContextManagement>,
) -> Option<Recognized<ContextManagement>> {
    let Recognized::Known(context) = context else {
        return None;
    };
    let edits: Vec<_> = context
        .edits?
        .into_iter()
        .filter(|edit| {
            matches!(
                edit,
                Recognized::Known(ContextEdit::Compact { .. } | ContextEdit::ClearToolUses { .. })
            )
        })
        .collect();
    (!edits.is_empty()).then_some(Recognized::Known(ContextManagement {
        edits: Some(edits),
        ..context
    }))
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
    pending_delta: Option<MessagesStreamEvent>,
    start_usage: Option<MessagesStreamUsage>,
}

fn promoted_usage(
    delta: Option<MessagesStreamUsage>,
    stop: Option<&MessagesStreamUsage>,
    start: Option<&MessagesStreamUsage>,
) -> Option<MessagesStreamUsage> {
    let delta = delta.unwrap_or_default();
    let merged = MessagesStreamUsage {
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
    (merged != MessagesStreamUsage::default()).then_some(merged)
}

fn promoted(
    event: MessagesStreamEvent,
    stop: Option<&MessagesStreamUsage>,
    start: Option<&MessagesStreamUsage>,
) -> MessagesStreamEvent {
    match event {
        MessagesStreamEvent::MessageDelta {
            delta,
            usage,
            context_management,
        } => MessagesStreamEvent::MessageDelta {
            delta,
            usage: promoted_usage(usage, stop, start),
            context_management,
        },
        other => other,
    }
}

impl StreamTransformer for MessageStopUsagePromoter {
    type Input = MessagesStreamEvent;
    type Output = MessagesStreamEvent;
    type Error = Infallible;

    fn transform(
        &mut self,
        input: MessagesStreamEvent,
    ) -> Result<Vec<MessagesStreamEvent>, Infallible> {
        let pending = self.pending_delta.take();
        match input {
            MessagesStreamEvent::MessageDelta { .. } => {
                self.pending_delta = Some(input);
                Ok(pending.into_iter().collect())
            }
            MessagesStreamEvent::MessageStop { usage } => Ok(pending
                .map(|delta| promoted(delta, usage.as_ref(), self.start_usage.as_ref()))
                .into_iter()
                .chain([MessagesStreamEvent::MessageStop { usage }])
                .collect()),
            MessagesStreamEvent::MessageStart { message } => {
                self.start_usage = Some(message.usage.clone());
                Ok(pending
                    .into_iter()
                    .chain([MessagesStreamEvent::MessageStart { message }])
                    .collect())
            }
            other => Ok(pending.into_iter().chain([other]).collect()),
        }
    }

    fn finish(&mut self) -> Result<Vec<MessagesStreamEvent>, Infallible> {
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
    use crate::base_llm::messages::streaming::encode_anthropic_sse;

    fn event(value: Value) -> MessagesStreamEvent {
        serde_json::from_value(value).unwrap()
    }

    fn message_start(usage: Value) -> MessagesStreamEvent {
        event(json!({
            "type": "message_start",
            "message": {
                "id": "msg_1", "type": "message", "role": "assistant", "model": "m",
                "content": [], "stop_reason": null, "stop_sequence": null, "usage": usage
            }
        }))
    }

    fn message_delta(usage: Value) -> MessagesStreamEvent {
        event(json!({
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn"},
            "usage": usage
        }))
    }

    fn message_stop(usage: Option<Value>) -> MessagesStreamEvent {
        match usage {
            Some(usage) => event(json!({"type": "message_stop", "usage": usage})),
            None => event(json!({"type": "message_stop"})),
        }
    }

    fn promote(events: Vec<MessagesStreamEvent>) -> Vec<MessagesStreamEvent> {
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

    fn in_region(region: &str) -> LitellmParams {
        LitellmParams {
            aws: AwsParams {
                aws_region_name: Some(region.into()),
                ..AwsParams::default()
            },
            ..LitellmParams::default()
        }
    }

    fn with_runtime_endpoint(endpoint: &str) -> LitellmParams {
        LitellmParams {
            aws: AwsParams {
                aws_bedrock_runtime_endpoint: Some(endpoint.into()),
                ..in_region("us-west-2").aws
            },
            ..LitellmParams::default()
        }
    }

    #[rstest]
    #[case::invoke_in_the_params_region(
        None,
        in_region("us-west-2"),
        false,
        None,
        "https://bedrock-runtime.us-west-2.amazonaws.com/model/anthropic.claude-3/invoke"
    )]
    #[case::stream_in_the_params_region(
        None,
        in_region("us-west-2"),
        true,
        None,
        "https://bedrock-runtime.us-west-2.amazonaws.com/model/anthropic.claude-3/invoke-with-response-stream"
    )]
    #[case::api_base_outranks_the_params_endpoint(
        Some("https://base.test/"),
        with_runtime_endpoint("https://params.test"),
        false,
        None,
        "https://base.test/model/anthropic.claude-3/invoke"
    )]
    #[case::a_blank_api_base_does_not_hide_the_params_endpoint(
        Some("  "),
        with_runtime_endpoint("https://params.test"),
        false,
        Some("https://env.test"),
        "https://params.test/model/anthropic.claude-3/invoke"
    )]
    #[case::params_endpoint_outranks_the_environment(
        None,
        with_runtime_endpoint("https://params.test"),
        false,
        Some("https://env.test"),
        "https://params.test/model/anthropic.claude-3/invoke"
    )]
    #[case::environment_outranks_the_region_template(
        None,
        in_region("us-west-2"),
        false,
        Some("https://env.test"),
        "https://env.test/model/anthropic.claude-3/invoke"
    )]
    fn url_follows_python_endpoint_precedence_and_the_stream_path(
        #[case] api_base: Option<&str>,
        #[case] litellm_params: LitellmParams,
        #[case] stream: bool,
        #[case] env_endpoint: Option<&str>,
        #[case] expected: &str,
    ) {
        let env = |name: &str| {
            (name == AWS_BEDROCK_RUNTIME_ENDPOINT)
                .then_some(env_endpoint)
                .flatten()
                .map(str::to_string)
        };
        assert_eq!(
            AmazonAnthropicClaudeMessagesConfig
                .get_complete_url(
                    api_base,
                    "anthropic.claude-3",
                    &litellm_params,
                    stream,
                    &env
                )
                .unwrap(),
            expected
        );
    }

    #[test]
    fn sigv4_scope_and_credentials_come_from_the_litellm_params() {
        let litellm_params = LitellmParams {
            aws: AwsParams {
                aws_access_key_id: Some("AKIAPARAMS".into()),
                aws_secret_access_key: Some("params-secret".into()),
                ..in_region("eu-central-1").aws
            },
            ..LitellmParams::default()
        };
        let validated = AmazonAnthropicClaudeMessagesConfig
            .validate_environment(
                Vec::new(),
                None,
                "anthropic.claude-3",
                &litellm_params,
                &|_| None,
            )
            .unwrap();
        let AuthScheme::AwsSigV4 {
            region,
            credentials,
            ..
        } = validated.auth
        else {
            panic!("expected SigV4, got {:?}", validated.auth);
        };
        let AwsCredentialSource::HostSupplied(credentials) = *credentials else {
            panic!("expected the params' static keys, got {credentials:?}");
        };
        assert_eq!(
            (region.as_str(), credentials.access_key_id()),
            ("eu-central-1", "AKIAPARAMS")
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
                &LitellmParams::default(),
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
