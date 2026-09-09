use crate::chat_completions::types::ChatAuthorizationContext;
use serde::Deserialize;
use serde_json::{Map, Value, json};

use crate::chat_completions::conversation::{Conversation, TurnRole, build_conversation};
use crate::chat_completions::response_utils::{finish_reason_for, unix_now, usage_from_parts};
use crate::chat_completions::transformation::{
    ChatCompletionsAuth, ChatCompletionsProviderConfig, Unsupported, unsupported_message,
    unsupported_param,
};
use crate::chat_completions::types::{
    ChatCompletionsChoice, ChatCompletionsChoiceMessage, ChatCompletionsResponse,
    ChatCompletionsUsage, ChatMessage, ChatMessageContent, ProviderChatRequestData,
    ProviderChatResponseData,
};
use crate::error::Error;

use super::super::aws_base::{bedrock_model_id_and_region, resolve_bedrock_region};
use super::super::constants::{AWS_BEARER_TOKEN_BEDROCK, BEDROCK_RUNTIME_ENDPOINT_TEMPLATE};

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

const CONVERSE_PATH_SUFFIX: &str = "/converse";

pub struct BedrockChatCompletionsConfig;

pub const BEDROCK_CHAT_COMPLETIONS_CONFIG: BedrockChatCompletionsConfig =
    BedrockChatCompletionsConfig;

#[derive(Deserialize)]
struct ConverseResponse {
    output: ConverseOutput,
    usage: ConverseUsage,
    #[serde(rename = "stopReason")]
    stop_reason: Option<String>,
}

#[derive(Deserialize)]
struct ConverseOutput {
    message: ConverseMessage,
}

#[derive(Deserialize)]
struct ConverseMessage {
    content: Vec<Value>,
}

#[derive(Deserialize)]
struct ConverseUsage {
    #[serde(default, rename = "inputTokens")]
    input_tokens: u64,
    #[serde(default, rename = "outputTokens")]
    output_tokens: u64,
    #[serde(default, rename = "cacheReadInputTokens")]
    cache_read_input_tokens: u64,
    #[serde(default, rename = "cacheWriteInputTokens")]
    cache_write_input_tokens: u64,
    #[serde(rename = "totalTokens")]
    total_tokens: Option<u64>,
}

fn converse_body(conversation: &Conversation, params: &Map<String, Value>) -> Map<String, Value> {
    let messages: Vec<Value> = conversation
        .turns
        .iter()
        .map(|turn| {
            json!({
                "role": turn.role.as_ref(),
                "content": turn.texts.iter().map(|text| json!({"text": text})).collect::<Vec<_>>(),
            })
        })
        .collect();

    let inference_config = Map::from_iter(SUPPORTED_PARAMS.iter().filter_map(|(_, name)| {
        params
            .get(*name)
            .map(|value| ((*name).to_string(), value.clone()))
    }));

    let system: Vec<Value> = conversation
        .system
        .iter()
        .map(|text| json!({"text": text}))
        .collect();

    Map::from_iter(
        [
            (
                "inferenceConfig".to_string(),
                Value::Object(inference_config),
            ),
            ("messages".to_string(), json!(messages)),
        ]
        .into_iter()
        .chain((!system.is_empty()).then(|| ("system".to_string(), json!(system)))),
    )
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

impl ChatCompletionsProviderConfig for BedrockChatCompletionsConfig {
    fn authorize<'a>(
        &'a self,
        services: &'a dyn crate::providers::auth::ChatAuthorizationServices,
        request: ChatAuthorizationContext<'a>,
        body: crate::lifecycle::WireBody,
    ) -> crate::providers::AuthorizationFuture<'a> {
        Box::pin(signed_headers(services, request, body))
    }

    fn request_body_policy(&self) -> crate::lifecycle::RequestBodyPolicy {
        crate::lifecycle::RequestBodyPolicy::SerializedAtBuild
    }

    fn complete_url(
        &self,
        api_base: Option<&str>,
        model: &str,
        optional_params: &Map<String, Value>,
        env_lookup: &dyn Fn(&str) -> Option<String>,
    ) -> Result<String, Error> {
        let (model_id, model_region) = bedrock_model_id_and_region(model);
        let region = resolve_bedrock_region(model_region.as_deref(), optional_params, env_lookup);
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
        if endpoint.ends_with(CONVERSE_PATH_SUFFIX) {
            return Ok(endpoint.to_string());
        }
        Ok(format!("{endpoint}/model/{model_id}{CONVERSE_PATH_SUFFIX}"))
    }

    fn auth(
        &self,
        api_key: Option<&str>,
        model: &str,
        optional_params: &Map<String, Value>,
        env_lookup: &dyn Fn(&str) -> Option<String>,
    ) -> Result<ChatCompletionsAuth, Error> {
        // Python reads `api_key` as the Bedrock bearer token and consults the
        // env only when the caller passed none, so a caller-supplied empty key
        // falls through to SigV4 without reaching for the environment. An
        // all-whitespace token stays a bearer token here because Python sends
        // it too: treating it as absent would sign as the host principal
        // instead, which is the identity swap this branch exists to prevent.
        let bearer = match api_key {
            Some(key) => Some(key.to_string()),
            None => env_lookup(AWS_BEARER_TOKEN_BEDROCK),
        }
        .filter(|token| !token.is_empty());
        if let Some(token) = bearer {
            return Ok(ChatCompletionsAuth::Bearer { token });
        }
        let (_, model_region) = bedrock_model_id_and_region(model);
        Ok(ChatCompletionsAuth::AwsSigV4 {
            region: resolve_bedrock_region(model_region.as_deref(), optional_params, env_lookup),
        })
    }

    fn default_headers(&self) -> &'static [(&'static str, &'static str)] {
        &[("Content-Type", "application/json")]
    }

    #[tracing::instrument(target = "litellm::function_trace", level = "trace", skip_all)]
    fn supported_openai_params(&self) -> &'static [(&'static str, &'static str)] {
        SUPPORTED_PARAMS
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
            self.supported_openai_params(),
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

    fn transform_request(
        &self,
        _model: &str,
        messages: Vec<ChatMessage>,
        optional_params: Map<String, Value>,
    ) -> Result<ProviderChatRequestData, Error> {
        Ok(ProviderChatRequestData {
            body: converse_body(&build_conversation(&messages), &optional_params),
        })
    }

    fn transform_response(
        &self,
        model: &str,
        response: ProviderChatResponseData,
    ) -> Result<ChatCompletionsResponse, Error> {
        let raw_body = response
            .body
            .as_object()
            .ok_or_else(|| Error::InvalidResponse("converse response is not an object".into()))?;
        if raw_body
            .get("output")
            .and_then(|output| output.get("message"))
            .and_then(|message| message.get("content"))
            .and_then(Value::as_array)
            .is_none()
        {
            return Err(Error::MissingField("output.message.content"));
        }
        if raw_body.get("usage").and_then(Value::as_object).is_none() {
            return Err(Error::MissingField("usage"));
        }
        let body: ConverseResponse = serde_json::from_value(response.body)
            .map_err(|error| Error::InvalidResponse(error.to_string()))?;
        let content = body.output.message.content;
        // The route declines tool requests, so anything other than a text block
        // is something this path never asked for. Decline; the host falls back.
        if content.iter().any(|block| {
            block
                .as_object()
                .is_none_or(|block| block.len() != 1 || !block.contains_key("text"))
        }) {
            return Err(Error::Unsupported("non-text response content block"));
        }
        let text: String = content
            .iter()
            .filter_map(|block| block.get("text").and_then(Value::as_str))
            .collect();

        let computed = usage_from_parts(
            body.usage.input_tokens,
            body.usage.output_tokens,
            body.usage.cache_read_input_tokens,
            body.usage.cache_write_input_tokens,
        );
        // Converse reports `totalTokens` and Python passes it straight through,
        // where Anthropic has no such field and Python adds the two counts
        // instead, so only this provider overrides the computed total. Python
        // does a bare `usage["totalTokens"]` lookup, so a body without the key
        // raises there rather than reporting a zero; fall back to the computed
        // total, which is the closest thing to that without failing the call.
        let usage = ChatCompletionsUsage {
            total_tokens: body.usage.total_tokens.unwrap_or(computed.total_tokens),
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
                finish_reason: finish_reason_for(body.stop_reason.as_deref().unwrap_or(""))
                    .to_string(),
            }],
            usage,
        })
    }
}

async fn signed_headers(
    services: &dyn crate::providers::auth::ChatAuthorizationServices,
    request: ChatAuthorizationContext<'_>,
    body: crate::lifecycle::WireBody,
) -> Result<crate::lifecycle::AuthorizedBody, Error> {
    use std::collections::BTreeMap;

    use crate::providers::bedrock::aws_base::{
        aws_auth_config, aws_signature_headers, host_supplied_credentials,
        is_sigv4_computed_header, sign_bedrock_post,
    };

    let ChatCompletionsAuth::AwsSigV4 { region } = request.auth() else {
        return Ok(crate::lifecycle::AuthorizedBody::new(
            body,
            request.upstream_headers().to_vec(),
        ));
    };
    // Reattaching a header the signer also emits would put both copies on the
    // wire, and Bedrock rejects that pair. Python instead drops the caller's
    // copy and prefers a forwarded Authorization over the signature, so leave
    // the request to Python rather than serving it a different way here.
    if request
        .upstream_headers()
        .iter()
        .any(|(name, _)| is_sigv4_computed_header(name))
    {
        return Err(Error::Unsupported(
            "request forwards a header AWS SigV4 computes",
        ));
    }
    let env_lookup = |key: &str| services.environment(key);
    let unsigned: BTreeMap<String, String> = request.upstream_headers().iter().cloned().collect();
    // A host with its own resolution chain hands the result down; only fall
    // back to deriving credentials here when it supplied none.
    let credentials = match host_supplied_credentials(request.optional_params()) {
        Some(credentials) => credentials,
        None => {
            crate::providers::bedrock::aws_base::resolve_credentials_with_state(
                aws_auth_config(request.optional_params(), &env_lookup),
                &env_lookup,
                services.aws_credential_state(),
            )
            .await?
        }
    };
    let signature = sign_bedrock_post(
        request.url(),
        body.as_bytes(),
        &aws_signature_headers(&unsigned),
        region,
        &credentials,
        services.signing_time(),
    )?;
    // Every original header goes back on the wire alongside the computed ones,
    // as Python reattaches them. The guard above already rejected the names
    // that would collide, so no name appears twice.
    Ok(crate::lifecycle::AuthorizedBody::new(
        body,
        unsigned.into_iter().chain(signature).collect(),
    ))
}
