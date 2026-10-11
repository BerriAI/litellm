use litellm_auth::AwsParams;
use litellm_auth::{CredentialPlacement, SecretValue};
use litellm_auth_aws::{
    AwsCredentialSource, bedrock_model_id_and_region,
    constants::{AWS_BEARER_TOKEN_BEDROCK, BEDROCK_RUNTIME_ENDPOINT_TEMPLATE, BEDROCK_SERVICE},
    resolve_bedrock_region,
};
use litellm_core_utils::{
    core_helpers::{finish_reason_for, unix_now, usage_from_parts},
    prompt_templates::factory::{Conversation, TurnRole, build_conversation},
};
use litellm_llms_types::{
    formats::chat_completions::{
        ChatCompletionsChoice, ChatCompletionsChoiceMessage, ChatCompletionsResponse,
        ChatCompletionsUsage, ChatMessage, ChatMessageContent,
    },
    providers::bedrock::{CONVERSE_PATH, ConverseResponse},
};
use serde_json::{Map, Value, json};

use crate::{
    Error,
    base_llm::{
        auth::AuthScheme,
        chat::{
            streaming::StreamShape,
            transformation::{
                BaseConfig, Headers, ProviderChatRequestData, ProviderChatResponseData,
                Unsupported, ValidatedEnvironment, unsupported_message, unsupported_param,
            },
        },
    },
};

/// Converse parameter names, post `map_openai_params`, that the Rust path can
/// place verbatim in `inferenceConfig`.
///
/// `topK` is deliberately absent: Python routes it to
/// `additionalModelRequestFields` for Anthropic base models and to
/// `inferenceConfig` otherwise, and that branch reads the model catalog the
/// core cannot see.
const SUPPORTED_PARAMS: &[(&str, &str)] = &[
    ("max_tokens", "maxTokens"),
    ("temperature", "temperature"),
    ("top_p", "topP"),
    ("stop", "stopSequences"),
];

const AWS_BEDROCK_RUNTIME_ENDPOINT: &str = "aws_bedrock_runtime_endpoint";

/// AWS call configuration a host passes down: consumed for signing and endpoint
/// resolution, never serialized into the Converse body.
const CONFIG_PARAMS: &[&str] = &[
    "aws_access_key_id",
    "aws_secret_access_key",
    "aws_session_token",
    "aws_region_name",
    "aws_session_name",
    "aws_profile_name",
    "aws_role_name",
    "aws_web_identity_token",
    "aws_sts_endpoint",
    "aws_external_id",
    AWS_BEDROCK_RUNTIME_ENDPOINT,
];

enum ConverseStopReason {
    Value(String),
    Unknown(String),
}

impl ConverseStopReason {
    fn parse(value: Option<String>) -> Option<Self> {
        value.map(|reason| match reason.as_str() {
            "end_turn"
            | "max_tokens"
            | "stop_sequence"
            | "content_filtered"
            | "guardrail_intervened" => Self::Value(reason),
            _ => Self::Unknown(reason),
        })
    }

    fn as_str(&self) -> &str {
        match self {
            Self::Value(value) | Self::Unknown(value) => value,
        }
    }
}

pub struct AmazonConverseConfig;

pub const BEDROCK_CHAT_COMPLETIONS_CONFIG: AmazonConverseConfig = AmazonConverseConfig;

impl BaseConfig for AmazonConverseConfig {
    fn secret_names(&self) -> Vec<&'static str> {
        litellm_auth::AwsParams::secret_names()
            .iter()
            .copied()
            .chain([AWS_BEARER_TOKEN_BEDROCK])
            .collect()
    }

    fn supported_openai_param_mappings(&self) -> &'static [(&'static str, &'static str)] {
        SUPPORTED_PARAMS
    }

    fn get_complete_url(
        &self,
        api_base: Option<&str>,
        model: &str,
        optional_params: &Map<String, Value>,
        env_lookup: &dyn Fn(&str) -> Option<String>,
    ) -> Result<String, Error> {
        let (model_id, model_region) = bedrock_model_id_and_region(model);
        let region = resolve_bedrock_region(
            model_region.as_deref(),
            &AwsParams::from_optional_params(optional_params),
            env_lookup,
        );
        let endpoint = optional_params
            .get(AWS_BEDROCK_RUNTIME_ENDPOINT)
            .and_then(Value::as_str)
            .or(api_base)
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(str::to_string)
            .unwrap_or_else(|| BEDROCK_RUNTIME_ENDPOINT_TEMPLATE.replace("{region}", &region));
        let endpoint = endpoint.trim_end_matches('/');
        // A host that already built the full Converse URL (LiteLLM's Python
        // path encodes the model id itself) passes it through untouched, the
        // way the Anthropic config leaves a complete `/v1/messages` URL alone.
        if endpoint.ends_with(CONVERSE_PATH) {
            return Ok(endpoint.to_string());
        }
        Ok(format!("{endpoint}/model/{model_id}{CONVERSE_PATH}"))
    }

    fn transform_request(
        &self,
        _model: &str,
        messages: Vec<ChatMessage>,
        optional_params: Map<String, Value>,
    ) -> Result<ProviderChatRequestData, Error> {
        Ok(ProviderChatRequestData {
            body: converse_body(&build_conversation(&messages), &optional_params),
            stream_shape: StreamShape::default(),
        })
    }

    fn transform_response(
        &self,
        model: &str,
        response: ProviderChatResponseData,
    ) -> Result<ChatCompletionsResponse, Error> {
        let body = response.body;
        if !body.is_object() {
            return Err(Error::InvalidResponse(
                "converse response is not an object".into(),
            ));
        }
        for field in ["output", "usage"] {
            if body.get(field).is_none() {
                return Err(Error::InvalidResponse(
                    format!("invalid Converse response: missing field `{field}`").into(),
                ));
            }
        }
        let response: ConverseResponse = serde_json::from_value(body).map_err(|error| {
            Error::InvalidResponse(format!("invalid Converse response: {error}").into())
        })?;
        // The route declines tool requests, so anything other than a text block
        // is something this path never asked for. Decline; the host falls back.
        if response.message_content_is_non_text() {
            return Err(Error::Unsupported("non-text response content block"));
        }
        let text = response.content_text();
        let computed = usage_from_parts(
            response.usage.input_tokens,
            response.usage.output_tokens,
            response.usage.cache_read_input_tokens,
            response.usage.cache_write_input_tokens,
        );
        // Converse reports `totalTokens` and Python passes it straight through,
        // where Anthropic has no such field and Python adds the two counts
        // instead, so only this provider overrides the computed total. Python
        // does a bare `usage["totalTokens"]` lookup, so a body without the key
        // raises there rather than reporting a zero; fall back to the computed
        // total, which is the closest thing to that without failing the call.
        let usage = ChatCompletionsUsage {
            total_tokens: response.usage.total_tokens.unwrap_or(computed.total_tokens),
            ..computed
        };

        Ok(ChatCompletionsResponse {
            created: unix_now(),
            // Converse echoes no model id, so Python reports the requested one.
            model: model.to_string(),
            choices: vec![ChatCompletionsChoice {
                index: 0,
                message: ChatCompletionsChoiceMessage {
                    role: "assistant".to_string(),
                    // Converse assigns the joined string unconditionally, so an
                    // empty response is `""` here and not `None` as it is on
                    // Anthropic. A caller calling `.strip()` on it would break
                    // on this path alone.
                    content: Some(text),
                },
                finish_reason: finish_reason_for(
                    ConverseStopReason::parse(response.stop_reason)
                        .as_ref()
                        .map(ConverseStopReason::as_str)
                        .unwrap_or(""),
                )
                .to_string(),
            }],
            usage,
        })
    }

    /// Python reads `api_key` as the Bedrock bearer token and consults the env only when
    /// the caller passed none, so a caller-supplied empty key falls through to SigV4
    /// without reaching for the environment. An all-whitespace token stays a bearer token
    /// here because Python sends it too: treating it as absent would sign as the host
    /// principal instead, which is the identity swap this branch exists to prevent.
    fn validate_environment(
        &self,
        headers: Headers,
        api_key: Option<&str>,
        model: &str,
        optional_params: &Map<String, Value>,
        env_lookup: &dyn Fn(&str) -> Option<String>,
    ) -> Result<ValidatedEnvironment, Error> {
        let bearer = match api_key {
            Some(key) => Some(key.to_string()),
            None => env_lookup(AWS_BEARER_TOKEN_BEDROCK),
        }
        .filter(|token| !token.is_empty());
        if let Some(token) = bearer {
            return Ok(ValidatedEnvironment {
                headers,
                auth: AuthScheme::Credential {
                    placement: CredentialPlacement::Bearer,
                    secret: SecretValue::new(token),
                },
            });
        }
        let (_, model_region) = bedrock_model_id_and_region(model);
        let params = AwsParams::from_optional_params(optional_params);
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
        &[("Content-Type", "application/json")]
    }

    fn config_params(&self) -> &'static [&'static str] {
        CONFIG_PARAMS
    }

    fn unsupported_reason(
        &self,
        messages: &[ChatMessage],
        optional_params: &Map<String, Value>,
    ) -> Option<Unsupported> {
        unsupported_param(
            self.supported_openai_param_mappings(),
            CONFIG_PARAMS,
            optional_params,
        )
        .or_else(|| messages.iter().find_map(unsupported_message))
        // Python's Converse translation drops blank text blocks instead of
        // substituting the placeholder the shared conversation builder
        // applies, so decline blank text rather than diverge.
        .or_else(|| {
            messages
                .iter()
                .any(has_blank_text)
                .then_some(Unsupported("blank message text"))
        })
        // Converse has no assistant prefill: Python inserts a continue turn
        // when a conversation opens or closes on an assistant message, and
        // only under `litellm.modify_params`, which the core cannot see.
        // Declining both ends also keeps the shared builder's final
        // assistant right-strip (an Anthropic rule) unreachable here.
        .or_else(|| {
            let conversation = build_conversation(messages);
            let ends_on_assistant = conversation
                .turns
                .last()
                .is_some_and(|turn| turn.role == TurnRole::Assistant);
            (!conversation.opens_on_user_turn() || ends_on_assistant).then_some(Unsupported(
                "conversation does not run user turn to user turn",
            ))
        })
    }
}

fn converse_body(conversation: &Conversation, optional_params: &Map<String, Value>) -> Value {
    let messages: Vec<Value> = conversation
        .turns
        .iter()
        .map(|turn| {
            json!({
                "role": <&'static str>::from(turn.role),
                "content": turn.texts.iter().map(|text| json!({"text": text})).collect::<Vec<_>>(),
            })
        })
        .collect();

    let inference_config = Map::from_iter(SUPPORTED_PARAMS.iter().filter_map(|(_, name)| {
        optional_params
            .get(*name)
            .map(|value| ((*name).to_string(), value.clone()))
    }));

    let system: Vec<Value> = conversation
        .system
        .iter()
        .map(|text| json!({"text": text}))
        .collect();

    Value::Object(Map::from_iter(
        [
            (
                "inferenceConfig".to_string(),
                Value::Object(inference_config),
            ),
            ("messages".to_string(), json!(messages)),
        ]
        .into_iter()
        .chain((!system.is_empty()).then(|| ("system".to_string(), json!(system)))),
    ))
}

fn has_blank_text(message: &ChatMessage) -> bool {
    match &message.content {
        None => false,
        Some(ChatMessageContent::Text(text)) => text.trim().is_empty(),
        Some(ChatMessageContent::Parts(parts)) => parts.iter().any(|part| {
            part.get("text")
                .and_then(Value::as_str)
                .is_none_or(|text| text.trim().is_empty())
        }),
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        Error,
        base_llm::{
            auth::AuthScheme,
            chat::transformation::{BaseConfig, ProviderChatResponseData, Unsupported},
        },
        bedrock::chat::converse_transformation::BEDROCK_CHAT_COMPLETIONS_CONFIG,
    };
    use litellm_auth::CredentialPlacement;
    use litellm_llms_types::formats::chat_completions::{ChatCompletionsResponse, ChatMessage};
    use rstest::rstest;
    use serde_json::{Map, Value, json};

    fn messages(value: Value) -> Vec<ChatMessage> {
        serde_json::from_value(value).expect("valid messages")
    }

    fn params(value: Value) -> Map<String, Value> {
        match value {
            Value::Object(map) => map,
            other => panic!("params must be an object, got {other}"),
        }
    }

    fn transform(msgs: Value, opts: Value) -> Value {
        BEDROCK_CHAT_COMPLETIONS_CONFIG
            .transform_request(
                "anthropic.claude-sonnet-4-5-v1:0",
                messages(msgs),
                params(opts),
            )
            .expect("request transforms")
            .body
    }

    fn transform_response(body: Value) -> Result<ChatCompletionsResponse, Error> {
        BEDROCK_CHAT_COMPLETIONS_CONFIG.transform_response(
            "anthropic.claude-sonnet-4-5-v1:0",
            ProviderChatResponseData { body },
        )
    }

    fn reason(msgs: Value, opts: Value) -> Option<Unsupported> {
        BEDROCK_CHAT_COMPLETIONS_CONFIG.unsupported_reason(&messages(msgs), &params(opts))
    }

    #[test]
    fn builds_the_converse_body_python_builds() {
        let body = transform(
            json!([
                {"role": "system", "content": "be terse"},
                {"role": "user", "content": "hi"}
            ]),
            json!({"maxTokens": 128, "temperature": 0.2}),
        );
        assert_eq!(
            body,
            json!({
                "inferenceConfig": {"maxTokens": 128, "temperature": 0.2},
                "messages": [{"role": "user", "content": [{"text": "hi"}]}],
                "system": [{"text": "be terse"}]
            })
        );
    }

    #[test]
    fn always_emits_inference_config_even_when_empty() {
        let body = transform(json!([{"role": "user", "content": "hi"}]), json!({}));
        assert_eq!(body["inferenceConfig"], json!({}));
        assert!(body.get("system").is_none());
    }

    #[test]
    fn places_only_inference_params_in_inference_config() {
        let body = transform(
            json!([{"role": "user", "content": "hi"}]),
            json!({
                "maxTokens": 64,
                "temperature": 0.1,
                "topP": 0.9,
                "stopSequences": ["STOP"]
            }),
        );
        assert_eq!(
            body["inferenceConfig"],
            json!({"maxTokens": 64, "temperature": 0.1, "topP": 0.9, "stopSequences": ["STOP"]})
        );
        assert!(body.get("additionalModelRequestFields").is_none());
    }

    #[test]
    fn merges_consecutive_user_turns_into_one_message() {
        let body = transform(
            json!([
                {"role": "user", "content": "one"},
                {"role": "user", "content": [{"type": "text", "text": "two"}]},
                {"role": "assistant", "content": "ack"},
                {"role": "user", "content": "three"}
            ]),
            json!({}),
        );
        assert_eq!(
            body["messages"],
            json!([
                {"role": "user", "content": [{"text": "one"}, {"text": "two"}]},
                {"role": "assistant", "content": [{"text": "ack"}]},
                {"role": "user", "content": [{"text": "three"}]}
            ])
        );
    }

    #[test]
    fn declines_streaming() {
        assert_eq!(
            reason(
                json!([{"role": "user", "content": "hi"}]),
                json!({"stream": true})
            ),
            Some(Unsupported("streaming"))
        );
    }

    #[test]
    fn declines_top_k_because_python_routes_it_by_base_model() {
        assert_eq!(
            reason(
                json!([{"role": "user", "content": "hi"}]),
                json!({"topK": 40})
            ),
            Some(Unsupported("unrecognized request parameter"))
        );
    }

    #[rstest]
    #[case::tools(json!({"tools": []}))]
    #[case::tool_choice(json!({"tool_choice": {"auto": {}}}))]
    #[case::thinking(json!({"thinking": {"type": "enabled"}}))]
    #[case::request_metadata(json!({"requestMetadata": {"k": "v"}}))]
    #[case::output_config(json!({"outputConfig": {}}))]
    #[case::parallel_tool_use_config(json!({"_parallel_tool_use_config": {}}))]
    fn declines_tools_and_other_params_outside_the_allowlist(#[case] param: Value) {
        assert_eq!(
            reason(json!([{"role": "user", "content": "hi"}]), param.clone()),
            Some(Unsupported("unrecognized request parameter")),
            "expected {param} to decline"
        );
    }

    #[rstest]
    #[case::empty_string(json!(""))]
    #[case::whitespace_string(json!("   "))]
    #[case::whitespace_text_block(json!([{"type": "text", "text": " "}]))]
    fn declines_blank_text_rather_than_substituting_the_anthropic_placeholder(
        #[case] content: Value,
    ) {
        assert_eq!(
            reason(
                json!([{"role": "user", "content": content}, {"role": "user", "content": "hi"}]),
                json!({})
            ),
            Some(Unsupported("blank message text")),
            "expected blank content {content} to decline"
        );
    }

    #[test]
    fn declines_a_message_whose_content_list_is_empty() {
        // The blank-text check scans parts, so an empty list clears it; Converse
        // rejects an empty `content` array, which is a decline the core owes the
        // host before the call rather than an error after it.
        assert_eq!(
            reason(json!([{"role": "user", "content": []}]), json!({})),
            Some(Unsupported("message without content"))
        );
        assert_eq!(
            reason(
                json!([{"role": "user", "content": [{"type": "text", "text": "hi"}]}]),
                json!({})
            ),
            None
        );
    }

    #[test]
    fn declines_a_conversation_that_opens_or_closes_on_an_assistant_turn() {
        assert_eq!(
            reason(
                json!([
                    {"role": "assistant", "content": "prefill"},
                    {"role": "user", "content": "hi"}
                ]),
                json!({})
            ),
            Some(Unsupported(
                "conversation does not run user turn to user turn"
            ))
        );
        assert_eq!(
            reason(
                json!([
                    {"role": "user", "content": "hi"},
                    {"role": "assistant", "content": "prefill"}
                ]),
                json!({})
            ),
            Some(Unsupported(
                "conversation does not run user turn to user turn"
            ))
        );
    }

    #[test]
    fn accepts_a_user_to_user_text_conversation() {
        assert_eq!(
            reason(
                json!([
                    {"role": "system", "content": "be terse"},
                    {"role": "user", "content": "hi"},
                    {"role": "assistant", "content": "hello"},
                    {"role": "user", "content": "again"}
                ]),
                json!({"maxTokens": 16})
            ),
            None
        );
    }

    #[test]
    fn builds_the_converse_url_from_the_region_in_the_model_id() {
        let config = &BEDROCK_CHAT_COMPLETIONS_CONFIG;
        assert_eq!(
            config
                .get_complete_url(None, "us-east-1/anthropic.claude-v2", &Map::new(), &|_| {
                    None
                })
                .expect("url builds"),
            "https://bedrock-runtime.us-east-1.amazonaws.com/model/anthropic.claude-v2/converse"
        );
    }

    #[test]
    fn falls_back_to_the_region_env_then_the_default_region() {
        let config = &BEDROCK_CHAT_COMPLETIONS_CONFIG;
        let with_env = |key: &str| (key == "AWS_REGION_NAME").then(|| "eu-west-1".to_string());
        assert_eq!(
            config
                .get_complete_url(None, "anthropic.claude-v2", &Map::new(), &with_env)
                .expect("url builds"),
            "https://bedrock-runtime.eu-west-1.amazonaws.com/model/anthropic.claude-v2/converse"
        );
        assert_eq!(
            config
                .get_complete_url(None, "anthropic.claude-v2", &Map::new(), &|_| None)
                .expect("url builds"),
            "https://bedrock-runtime.us-west-2.amazonaws.com/model/anthropic.claude-v2/converse"
        );
    }

    #[test]
    fn prefers_an_explicit_runtime_endpoint_over_the_api_base() {
        let config = &BEDROCK_CHAT_COMPLETIONS_CONFIG;
        let overrides = params(json!({"aws_bedrock_runtime_endpoint": "https://vpce.internal/"}));
        assert_eq!(
            config
                .get_complete_url(
                    Some("https://ignored.example"),
                    "anthropic.claude-v2",
                    &overrides,
                    &|_| None
                )
                .expect("url builds"),
            "https://vpce.internal/model/anthropic.claude-v2/converse"
        );
    }

    /// The bearer token a config named, or `None` for a SigV4 scheme in the given region.
    fn bearer_or_region(auth: AuthScheme) -> Result<String, String> {
        match auth {
            AuthScheme::Credential {
                placement: CredentialPlacement::Bearer,
                secret,
            } => Ok(secret.expose().to_string()),
            AuthScheme::AwsSigV4 {
                region,
                service: "bedrock",
                ..
            } => Err(region),
            other => panic!("unexpected auth {other:?}"),
        }
    }

    #[test]
    fn signs_with_sigv4_in_the_resolved_region() {
        let validated = BEDROCK_CHAT_COMPLETIONS_CONFIG
            .validate_environment(
                Vec::new(),
                None,
                "eu-central-1/anthropic.claude-v2",
                &Map::new(),
                &|_| None,
            )
            .expect("auth resolves");
        assert_eq!(
            bearer_or_region(validated.auth),
            Err("eu-central-1".to_string())
        );
    }

    #[test]
    fn a_bearer_token_outranks_sigv4_the_way_python_resolves_it() {
        // Python's get_request_headers reads `api_key` as the Bedrock bearer token
        // and only falls back to the env when the caller passed none, so each case
        // pins one of its precedence rules. Signing as the host principal when a
        // bearer identity is configured would cross an account and quota boundary.
        let bedrock_env =
            |key: &str| (key == "AWS_BEARER_TOKEN_BEDROCK").then(|| "from-env".to_string());
        let no_env = |_: &str| None;
        let resolve = |api_key, env: &dyn Fn(&str) -> Option<String>| {
            bearer_or_region(
                BEDROCK_CHAT_COMPLETIONS_CONFIG
                    .validate_environment(
                        Vec::new(),
                        api_key,
                        "eu-central-1/anthropic.claude-v2",
                        &Map::new(),
                        env,
                    )
                    .expect("auth resolves")
                    .auth,
            )
        };
        let bearer = |token: &str| Ok(token.to_string());
        let sigv4 = Err("eu-central-1".to_string());

        // A caller-supplied key is the bearer token, and outranks the env.
        assert_eq!(
            resolve(Some("bedrock-api-key"), &bedrock_env),
            bearer("bedrock-api-key")
        );
        // No key, so the env supplies it.
        assert_eq!(resolve(None, &bedrock_env), bearer("from-env"));
        // An empty key is not a bearer token, and deliberately does NOT reach for
        // the env, which is what Python's `is not None` check does.
        assert_eq!(resolve(Some(""), &bedrock_env), sigv4);
        // Whitespace is truthy in Python, so it stays a bearer token rather than
        // silently becoming a host-credentialed SigV4 request.
        assert_eq!(resolve(Some("  "), &no_env), bearer("  "));
        // Neither present, so SigV4 as before.
        assert_eq!(resolve(None, &no_env), sigv4);
    }

    #[test]
    fn normalizes_a_converse_response_into_openai_shape() {
        let response = transform_response(json!({
            "output": {"message": {"role": "assistant", "content": [
                {"text": "hello"}, {"text": " there"}
            ]}},
            "stopReason": "end_turn",
            "usage": {"inputTokens": 11, "outputTokens": 4, "totalTokens": 15}
        }))
        .expect("response transforms");

        assert_eq!(response.model, "anthropic.claude-sonnet-4-5-v1:0");
        assert_eq!(
            response.choices[0].message.content.as_deref(),
            Some("hello there")
        );
        assert_eq!(response.choices[0].finish_reason, "stop");
        assert_eq!(response.usage.prompt_tokens, 11);
        assert_eq!(response.usage.completion_tokens, 4);
        assert_eq!(response.usage.total_tokens, 15);
    }

    #[test]
    fn maps_converse_stop_reasons_python_maps() {
        for (provider_reason, expected) in [
            ("end_turn", "stop"),
            ("stop_sequence", "stop"),
            ("max_tokens", "length"),
            ("guardrail_intervened", "content_filter"),
            // Converse emits this one, and Python's `_FINISH_REASON_MAP` carries
            // it. Folding it into `stop` reports a filtered completion as a normal
            // one to anything keying on the finish reason.
            ("content_filtered", "content_filter"),
            ("content_filter", "content_filter"),
        ] {
            let response = transform_response(json!({
                "output": {"message": {"content": [{"text": "x"}]}},
                "stopReason": provider_reason,
                "usage": {"inputTokens": 1, "outputTokens": 1}
            }))
            .expect("response transforms");
            assert_eq!(
                response.choices[0].finish_reason, expected,
                "stopReason {provider_reason}"
            );
        }
    }

    #[test]
    fn reports_an_empty_converse_answer_as_an_empty_string_not_null() {
        // Converse assigns the joined text unconditionally
        // (`chat_completion_message["content"] = content_str`), unlike Anthropic's
        // `merged_text or None`, so an empty answer is `""` on both paths. A caller
        // calling `.strip()` on it would break on the Rust path alone. Reachable
        // through a filtered or guardrail-intervened response.
        for content in [json!([]), json!([{"text": ""}])] {
            let response = transform_response(json!({
                "output": {"message": {"content": content}},
                "stopReason": "content_filtered",
                "usage": {"inputTokens": 1, "outputTokens": 0}
            }))
            .expect("response transforms");
            assert_eq!(response.choices[0].message.content, Some(String::new()));
        }
    }

    #[test]
    fn reports_the_total_tokens_converse_sent_rather_than_recomputing_them() {
        // Python reads `usage["totalTokens"]` straight through here, where Anthropic
        // has no such field and adds the two counts instead. The two agree while the
        // gate declines every cache_control request, so this is what keeps them
        // agreeing if that ever widens.
        let response = transform_response(json!({
            "output": {"message": {"content": [{"text": "x"}]}},
            "stopReason": "end_turn",
            "usage": {"inputTokens": 10, "outputTokens": 4, "cacheReadInputTokens": 7, "totalTokens": 14}
        }))
        .expect("response transforms");
        assert_eq!(
            response.usage.total_tokens, 14,
            "provider total was recomputed"
        );
        assert_eq!(response.usage.prompt_tokens, 17);
        assert_eq!(response.usage.completion_tokens, 4);
    }

    #[test]
    fn falls_back_to_the_computed_total_when_converse_omits_it() {
        // Python raises a KeyError on a body with no `totalTokens`. Reporting a zero
        // instead would be a worse divergence than the one above, so the computed
        // total stands in.
        let response = transform_response(json!({
            "output": {"message": {"content": [{"text": "x"}]}},
            "stopReason": "end_turn",
            "usage": {"inputTokens": 10, "outputTokens": 4}
        }))
        .expect("response transforms");
        assert_eq!(response.usage.total_tokens, 14);
    }

    #[test]
    fn declines_a_cache_control_message_so_widening_the_gate_is_a_red_test() {
        // Converse only reports cache token counts when the request carries a
        // cachePoint block, which is why the provider total and the computed one
        // cannot disagree today. This is the tripwire: whoever widens the gate to
        // admit prompt caching has to come back and re-check the usage mapping
        // rather than discovering a silent number change in production.
        assert_eq!(
            reason(
                json!([{"role": "user", "content": [
                    {"type": "text", "text": "hi", "cache_control": {"type": "ephemeral"}}
                ]}]),
                json!({})
            ),
            Some(Unsupported("non-text message content"))
        );
    }

    #[test]
    fn folds_converse_cache_tokens_into_prompt_tokens() {
        let response = transform_response(json!({
            "output": {"message": {"content": [{"text": "x"}]}},
            "stopReason": "end_turn",
            "usage": {
                "inputTokens": 10,
                "outputTokens": 2,
                "cacheReadInputTokens": 5,
                "cacheWriteInputTokens": 3
            }
        }))
        .expect("response transforms");
        assert_eq!(response.usage.prompt_tokens, 18);
        assert_eq!(response.usage.prompt_tokens_details.cached_tokens, 5);
        assert_eq!(
            response.usage.prompt_tokens_details.cache_creation_tokens,
            3
        );
        assert_eq!(response.usage.prompt_tokens_details.text_tokens, 10);
    }

    #[test]
    fn declines_a_response_carrying_a_tool_use_block() {
        let err = transform_response(json!({
            "output": {"message": {"content": [
                {"toolUse": {"toolUseId": "t1", "name": "f", "input": {}}}
            ]}},
            "stopReason": "tool_use",
            "usage": {"inputTokens": 1, "outputTokens": 1}
        }))
        .expect_err("tool use block");
        assert_eq!(err, Error::Unsupported("non-text response content block"));
    }

    #[test]
    fn errors_on_a_response_missing_required_fields() {
        assert_eq!(
            transform_response(json!("nope")).expect_err("not an object"),
            Error::InvalidResponse("converse response is not an object".to_string().into())
        );
        assert_eq!(
            transform_response(json!({"usage": {}})).expect_err("no output"),
            Error::InvalidResponse("invalid Converse response: missing field `output`".into())
        );
        assert_eq!(
            transform_response(json!({"output": {"message": {"content": []}}}))
                .expect_err("no usage"),
            Error::InvalidResponse("invalid Converse response: missing field `usage`".into())
        );
    }

    #[test]
    fn rejects_malformed_text_and_token_counts() {
        for response in [
            json!({"output": {"message": {"content": [{"text": 123}]}}, "usage": {"inputTokens": 1, "outputTokens": 1}}),
            json!({"output": {"message": {"content": [{"text": "x"}]}}, "usage": {"inputTokens": "1", "outputTokens": 1}}),
        ] {
            assert!(matches!(
                transform_response(response),
                Err(Error::InvalidResponse(message)) if message.to_string().starts_with("invalid Converse response:")
            ));
        }
    }

    #[test]
    fn accepts_aws_call_configuration_without_serializing_it() {
        let call_config = json!({
            "maxTokens": 16,
            "aws_access_key_id": "AKIA",
            "aws_secret_access_key": "secret",
            "aws_session_token": "token",
            "aws_region_name": "us-east-1",
            "aws_profile_name": "litellm-stage",
            "aws_role_name": "role",
            "aws_session_name": "session",
            "aws_web_identity_token": "wit",
            "aws_sts_endpoint": "https://sts.example",
            "aws_external_id": "ext",
            "aws_bedrock_runtime_endpoint": "https://vpce.internal"
        });
        assert_eq!(
            reason(
                json!([{"role": "user", "content": "hi"}]),
                call_config.clone()
            ),
            None
        );
        let body = transform(json!([{"role": "user", "content": "hi"}]), call_config);
        assert_eq!(
            body,
            json!({
                "inferenceConfig": {"maxTokens": 16},
                "messages": [{"role": "user", "content": [{"text": "hi"}]}]
            }),
            "aws call configuration must not reach the Converse body"
        );
    }

    #[test]
    fn leaves_a_complete_converse_url_untouched() {
        let config = &BEDROCK_CHAT_COMPLETIONS_CONFIG;
        let already_built = "https://bedrock-runtime.us-east-1.amazonaws.com/model/us.anthropic.claude-v2%3A0/converse";
        assert_eq!(
            config
                .get_complete_url(
                    Some(already_built),
                    "anthropic.claude-v2",
                    &Map::new(),
                    &|_| None
                )
                .expect("url builds"),
            already_built,
            "a host that encoded the model id itself must not have it re-derived"
        );
    }

    #[rstest]
    #[case::full_static_pair(
        json!({"aws_access_key_id": "AKIAHOST", "aws_secret_access_key": "hostsecret", "aws_session_token": "hosttoken"}),
        Some(("AKIAHOST", "hostsecret", Some("hosttoken")))
    )]
    #[case::pair_without_session_token(
        json!({"aws_access_key_id": "AKIAHOST", "aws_secret_access_key": "hostsecret"}),
        Some(("AKIAHOST", "hostsecret", None))
    )]
    #[case::key_id_alone(json!({"aws_access_key_id": "AKIA"}), None)]
    #[case::blank_key_id(json!({"aws_access_key_id": "  ", "aws_secret_access_key": "s"}), None)]
    #[case::nothing(json!({}), None)]
    fn host_supplied_credentials_need_a_full_static_pair(
        #[case] optional_params: Value,
        #[case] expected: Option<(&str, &str, Option<&str>)>,
    ) {
        use litellm_auth::AwsParams;
        use litellm_auth_aws::host_supplied_credentials;

        let credentials =
            host_supplied_credentials(&AwsParams::from_optional_params(&params(optional_params)));

        assert_eq!(
            credentials.as_ref().map(|credentials| (
                credentials.access_key_id(),
                credentials.secret_access_key(),
                credentials.session_token(),
            )),
            expected
        );
    }
}
