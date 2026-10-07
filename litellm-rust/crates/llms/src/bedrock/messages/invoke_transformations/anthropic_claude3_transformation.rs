use std::convert::Infallible;

use crate::{
    anthropic::messages::handler::shape_anthropic_messages_request,
    base_llm::messages::context::MessagesTransformContext,
};
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
use litellm_llms_types::formats::messages::{
    CacheControl, ContentBlock, ContextEdit, ContextManagement, CustomTool, EffortLevel,
    MessageContent, MessageRole, MessagesOptionalParams, MessagesRequest, MessagesTool,
    MessagesUsage, OutputConfig, ThinkingConfig, ToolDefinition, streaming::MessagesStreamEvent,
};
use litellm_llms_types::serde_compat::Nullable;
use litellm_llms_types::{
    json_schema::{JsonSchema, JsonSchemaObject, JsonSchemaType},
    providers::anthropic::{AnthropicBeta, BetaProvider, BetaSet},
    recognized::Recognized,
};
use serde_json::{Map, Value};

use crate::{
    Error,
    anthropic::{
        common_utils::{existing_betas, is_tool_search_used},
        messages::transformation::{
            PARTNER_HOST_REQUEST_POLICY, feature_betas, messages_carry_output_config,
            transform_messages_request_with,
        },
    },
    base_llm::{
        auth::AuthScheme,
        base_model_iterator::{StreamError, StreamTransformer, transform_stream},
        messages::{
            normalization::normalize_system_role_messages,
            streaming::{ByteStream, EventStream, StreamDecoder},
            transformation::{
                BaseMessagesConfig, Headers, MessagesWireRequest, ValidatedEnvironment,
                messages_request_body,
            },
        },
    },
    bedrock::{
        chat::invoke_handler::{decode_invoke_anthropic_chunk, invoke_chunk_stream},
        messages::connection::BedrockMessagesConnection,
    },
};

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

pub(crate) fn bedrock_body_field(key: &str) -> bool {
    BODY_FIELDS.contains(&key)
}

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
const MODEL_SEGMENT: &percent_encoding::AsciiSet = &percent_encoding::NON_ALPHANUMERIC
    .remove(b'-')
    .remove(b'_')
    .remove(b'.')
    .remove(b'~');

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

impl AmazonAnthropicClaudeMessagesConfig {
    pub fn get_complete_url_with_connection(
        &self,
        model: &str,
        connection: &BedrockMessagesConnection,
        stream: bool,
        env_lookup: &dyn Fn(&str) -> Option<String>,
    ) -> Result<String, Error> {
        let (model_id, model_region) =
            bedrock_model_id_and_region(model.strip_prefix(INVOKE_MODEL_PREFIX).unwrap_or(model));
        let region = connection
            .region
            .clone()
            .or(model_region)
            .unwrap_or_else(|| resolve_bedrock_region(None, &Map::new(), env_lookup));
        let endpoint = connection
            .api_base
            .as_deref()
            .map(str::trim)
            .filter(|base| !base.is_empty())
            .map(str::to_string)
            .or_else(|| env_lookup(AWS_BEDROCK_RUNTIME_ENDPOINT))
            .unwrap_or_else(|| BEDROCK_RUNTIME_ENDPOINT_TEMPLATE.replace("{region}", &region));
        let model_id = connection.model_id.as_deref().unwrap_or(&model_id);
        let model_id = percent_encoding::utf8_percent_encode(model_id, MODEL_SEGMENT);
        let path = if stream {
            INVOKE_STREAM_PATH
        } else {
            INVOKE_PATH
        };
        Ok(format!(
            "{}/model/{model_id}/{path}",
            endpoint.trim_end_matches('/')
        ))
    }

    pub fn validate_environment_with_connection(
        &self,
        headers: Headers,
        api_key: Option<&str>,
        model: &str,
        connection: &BedrockMessagesConnection,
        env_lookup: &dyn Fn(&str) -> Option<String>,
    ) -> Result<ValidatedEnvironment, Error> {
        let headers = connection.headers(headers);
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
        let region = connection
            .region
            .clone()
            .or(model_region)
            .unwrap_or_else(|| resolve_bedrock_region(None, &Map::new(), env_lookup));
        Ok(ValidatedEnvironment {
            headers,
            auth: AuthScheme::AwsSigV4 {
                region,
                service: BEDROCK_SERVICE,
                credentials: Box::new(AwsCredentialSource::from_params(&Map::new(), env_lookup)),
            },
        })
    }
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
        env_lookup: &dyn Fn(&str) -> Option<String>,
    ) -> Result<String, Error> {
        self.get_complete_url_with_connection(
            model,
            &BedrockMessagesConnection {
                api_base: api_base.map(str::to_string),
                ..Default::default()
            },
            false,
            env_lookup,
        )
    }

    fn complete_stream_url(
        &self,
        api_base: Option<&str>,
        model: &str,
        env_lookup: &dyn Fn(&str) -> Option<String>,
    ) -> Result<String, Error> {
        self.get_complete_url_with_connection(
            model,
            &BedrockMessagesConnection {
                api_base: api_base.map(str::to_string),
                ..Default::default()
            },
            true,
            env_lookup,
        )
    }

    fn transform_anthropic_messages_request(
        &self,
        request: MessagesRequest,
        context: &MessagesTransformContext,
    ) -> Result<MessagesRequest, Error> {
        transform_bedrock_messages_request(request, context, BedrockMessagesSurface::Invoke)
    }

    fn prepare_wire_request(
        &self,
        request: &MessagesRequest,
        headers: Headers,
    ) -> Result<MessagesWireRequest, Error> {
        let headers = crate::anthropic::common_utils::merge_beta_headers(
            headers,
            bedrock_feature_betas(request),
        );
        let betas = existing_betas(&headers)
            .into_iter()
            .chain(
                request
                    .params
                    .extra
                    .get("anthropic_beta")
                    .and_then(Value::as_array)
                    .into_iter()
                    .flatten()
                    .filter_map(Value::as_str)
                    .map(|beta| beta.parse().unwrap_or_else(|never| match never {})),
            )
            .filter_map(|beta| beta.on(BetaProvider::Bedrock))
            .collect::<BetaSet>();
        let Value::Object(fields) = messages_request_body(request)? else {
            unreachable!("MessagesRequest serializes as an object")
        };
        let version = request
            .params
            .extra
            .get("anthropic_version")
            .cloned()
            .unwrap_or_else(|| Value::String(BEDROCK_ANTHROPIC_VERSION.to_string()));
        let body = fields
            .into_iter()
            .filter(|(key, _)| {
                BODY_FIELDS.contains(&key.as_str())
                    && key != "anthropic_beta"
                    && key != "anthropic_version"
            })
            .chain([("anthropic_version".to_string(), version)])
            .chain((!betas.is_empty()).then(|| {
                (
                    "anthropic_beta".to_string(),
                    Value::Array(
                        betas
                            .iter()
                            .map(|beta| Value::String(beta.as_str().to_string()))
                            .collect(),
                    ),
                )
            }))
            .collect();
        Ok(MessagesWireRequest {
            body: Value::Object(body),
            headers: headers
                .into_iter()
                .filter(|(name, _)| !name.eq_ignore_ascii_case("anthropic-beta"))
                .collect(),
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
        env_lookup: &dyn Fn(&str) -> Option<String>,
    ) -> Result<ValidatedEnvironment, Error> {
        self.validate_environment_with_connection(
            headers,
            api_key,
            model,
            &BedrockMessagesConnection::default(),
            env_lookup,
        )
    }

    fn default_headers(&self) -> &'static [(&'static str, &'static str)] {
        &[("content-type", "application/json")]
    }

    fn stream_decoder(&self) -> Option<StreamDecoder> {
        Some(bedrock_anthropic_messages_event_stream)
    }
}

fn reject_web_search(tools: Option<&[Recognized<MessagesTool>]>) -> Result<(), Error> {
    let is_search = tools.into_iter().flatten().any(|tool| match tool {
        Recognized::Known(MessagesTool::Builtin(tool)) => tool.is_web_search(),
        Recognized::Unrecognized(Value::Object(fields)) => fields
            .get("type")
            .and_then(Value::as_str)
            .is_some_and(|kind| kind.starts_with("web_search")),
        _ => false,
    });
    if is_search {
        return Err(Error::Unsupported("Bedrock server-side web search tools"));
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum BedrockMessagesSurface {
    Invoke,
    Mantle,
}

pub(crate) fn transform_bedrock_messages_request(
    request: MessagesRequest,
    context: &MessagesTransformContext,
    surface: BedrockMessagesSurface,
) -> Result<MessagesRequest, Error> {
    reject_web_search(request.params.tools.as_deref())?;
    let display = match request.params.thinking.as_ref().and_then(Recognized::known) {
        Some(ThinkingConfig::Enabled(thinking)) => thinking.display.clone(),
        Some(ThinkingConfig::Adaptive(thinking)) => thinking.display.clone(),
        _ => None,
    };
    let request = normalize_system_role_messages(
        request,
        context
            .thinking
            .capabilities
            .supports_mid_conversation_system,
    );
    let request = clamp_bedrock_effort(request, context);
    let request = transform_messages_request_with(request, context, PARTNER_HOST_REQUEST_POLICY)?;
    let request = with_bedrock_clear_thinking(request, context);
    let request = clamp_bedrock_effort(request, context);
    let request = with_bedrock_output_format(request, context)?;
    let request = MessagesRequest {
        params: MessagesOptionalParams {
            thinking: request.params.thinking.map(|thinking| match thinking {
                Recognized::Known(ThinkingConfig::Adaptive(thinking)) => {
                    Recognized::Known(ThinkingConfig::Adaptive(
                        litellm_llms_types::formats::messages::AdaptiveThinking {
                            display: display.or(thinking.display),
                            ..thinking
                        },
                    ))
                }
                other => other,
            }),
            ..request.params
        },
        ..request
    };
    let betas = bedrock_tool_betas(request.params.tools.as_deref(), context)
        .union(
            ["betas", "anthropic_beta"]
                .into_iter()
                .flat_map(|key| {
                    request
                        .params
                        .extra
                        .get(key)
                        .and_then(Value::as_array)
                        .into_iter()
                        .flatten()
                })
                .filter_map(Value::as_str)
                .map(|beta| beta.parse().unwrap_or_else(|never| match never {}))
                .collect(),
        )
        .union(
            [
                request
                    .params
                    .safeguards
                    .is_some()
                    .then_some(AnthropicBeta::DangerousToolUse20260903),
                matches!(
                    request.params.thinking,
                    Some(Recognized::Known(
                        ThinkingConfig::Enabled(_) | ThinkingConfig::Adaptive(_)
                    ))
                )
                .then_some(AnthropicBeta::InterleavedThinking20250514),
            ]
            .into_iter()
            .flatten()
            .collect(),
        );
    let cache = |block: ContentBlock| ContentBlock {
        cache_control: block.cache_control.map(|cache| match cache {
            Nullable::Value(cache) => Nullable::Value(bedrock_cache(
                cache,
                context.thinking.capabilities.supports_cache_control_ttl,
            )),
            Nullable::Null => Nullable::Null,
        }),
        ..block
    };
    let request = MessagesRequest {
        messages: request
            .messages
            .into_iter()
            .map(|message| {
                let content = match message.content {
                    MessageContent::Blocks(blocks) => {
                        MessageContent::Blocks(blocks.into_iter().map(cache).collect())
                    }
                    text => text,
                };
                litellm_llms_types::formats::messages::Message { content, ..message }
            })
            .collect(),
        params: MessagesOptionalParams {
            system: request.params.system.map(|system| match system {
                litellm_llms_types::formats::messages::SystemPrompt::Blocks(blocks) => {
                    litellm_llms_types::formats::messages::SystemPrompt::Blocks(
                        blocks.into_iter().map(cache).collect(),
                    )
                }
                text => text,
            }),
            tools: request.params.tools.map(|tools| {
                tools
                    .into_iter()
                    .enumerate()
                    .map(|(index, tool)| {
                        bedrock_tool(
                            tool,
                            index,
                            context.thinking.capabilities.supports_cache_control_ttl,
                        )
                    })
                    .collect()
            }),
            context_management: request
                .params
                .context_management
                .and_then(|management| bedrock_context_management(management, surface)),
            extra: request
                .params
                .extra
                .into_iter()
                .filter(|(key, _)| key != "anthropic_beta")
                .chain((!betas.is_empty()).then(|| {
                    (
                        "anthropic_beta".to_string(),
                        Value::Array(
                            betas
                                .iter()
                                .map(|beta| Value::String(beta.as_str().to_string()))
                                .collect(),
                        ),
                    )
                }))
                .collect(),
            ..request.params
        },
        ..request
    };
    Ok(request)
}

pub(crate) fn bedrock_feature_betas(request: &MessagesRequest) -> BetaSet {
    feature_betas(request)
        .iter()
        .filter(|beta| **beta != AnthropicBeta::AdvancedToolUse20251120)
        .cloned()
        .chain(
            messages_carry_output_config(&request.messages)
                .then_some(AnthropicBeta::MidConversationOutputConfig20260701),
        )
        .collect()
}

fn bedrock_cache(cache: CacheControl, supports_ttl: bool) -> CacheControl {
    CacheControl {
        scope: None,
        ttl: cache
            .ttl
            .filter(|ttl| supports_ttl && matches!(ttl.as_deref(), Some("5m" | "1h"))),
        ..cache
    }
}

fn bedrock_context_management(
    context: Recognized<ContextManagement>,
    surface: BedrockMessagesSurface,
) -> Option<Recognized<ContextManagement>> {
    let Recognized::Known(context) = context else {
        return Some(context);
    };
    let edits: Vec<_> = context
        .edits?
        .into_iter()
        .filter(|edit| {
            matches!(
                edit,
                Recognized::Known(ContextEdit::Compact { .. } | ContextEdit::ClearToolUses { .. })
            ) || (surface == BedrockMessagesSurface::Mantle
                && matches!(edit, Recognized::Known(ContextEdit::ClearThinking { .. })))
        })
        .collect();
    (!edits.is_empty()).then(|| {
        Recognized::Known(ContextManagement {
            edits: Some(edits),
            ..context
        })
    })
}

fn tool_definition(tool: &Recognized<MessagesTool>) -> Option<&ToolDefinition> {
    tool.known().map(MessagesTool::definition)
}

fn bedrock_tool_betas(
    tools: Option<&[Recognized<MessagesTool>]>,
    context: &MessagesTransformContext,
) -> BetaSet {
    let eager = tools.into_iter().flatten().any(|tool| {
        tool_definition(tool).is_some_and(|definition| {
            definition.eager_input_streaming == Some(Recognized::Known(true))
        })
    });
    let search = context.thinking.capabilities.supports_tool_search && is_tool_search_used(tools);
    [
        eager.then_some(AnthropicBeta::FineGrainedToolStreaming20250514),
        search.then_some(AnthropicBeta::ToolSearchTool20251019),
        search.then_some(AnthropicBeta::ToolExamples20251029),
    ]
    .into_iter()
    .flatten()
    .collect()
}

fn normalized_schema(schema: Recognized<JsonSchema>) -> Recognized<JsonSchema> {
    match schema {
        Recognized::Known(JsonSchema::Object(schema)) => {
            let map_schemas = |schemas: Recognized<Vec<Recognized<JsonSchema>>>| match schemas {
                Recognized::Known(schemas) => {
                    Recognized::Known(schemas.into_iter().map(normalized_schema).collect())
                }
                other => other,
            };
            let map_properties = |properties: Recognized<
                std::collections::BTreeMap<String, Recognized<JsonSchema>>,
            >| match properties {
                Recognized::Known(properties) => Recognized::Known(
                    properties
                        .into_iter()
                        .map(|(key, schema)| (key, normalized_schema(schema)))
                        .collect(),
                ),
                other => other,
            };
            let map_box = |schema: Recognized<Box<JsonSchema>>| match schema {
                Recognized::Known(schema) => match normalized_schema(Recognized::Known(*schema)) {
                    Recognized::Known(schema) => Recognized::Known(Box::new(schema)),
                    Recognized::Unrecognized(value) => Recognized::Unrecognized(value),
                },
                other => other,
            };
            Recognized::Known(JsonSchema::Object(Box::new(JsonSchemaObject {
                schema_type: schema.schema_type.map(|kind| match kind {
                    Recognized::Known(JsonSchemaType::Name(name)) if name == "custom" => {
                        Recognized::Known(JsonSchemaType::Name("object".into()))
                    }
                    other => other,
                }),
                properties: schema.properties.map(map_properties),
                defs: schema.defs.map(map_properties),
                items: schema.items.map(map_box),
                additional_properties: schema.additional_properties.map(map_box),
                any_of: schema.any_of.map(map_schemas),
                all_of: schema.all_of.map(map_schemas),
                one_of: schema.one_of.map(map_schemas),
                ..*schema
            })))
        }
        other => other,
    }
}

fn bedrock_tool_definition(
    definition: ToolDefinition,
    index: usize,
    supports_ttl: bool,
) -> ToolDefinition {
    let deferred = definition
        .extra
        .get("custom")
        .and_then(|custom| custom.get("defer_loading"))
        .and_then(Value::as_bool)
        .map(Recognized::Known);
    let name = match definition.name {
        Some(Recognized::Known(name)) if !name.trim().is_empty() => name,
        _ => format!("litellm_unnamed_tool_{index}"),
    };
    ToolDefinition {
        name: Some(Recognized::Known(name)),
        cache_control: definition.cache_control.map(|cache| match cache {
            Recognized::Known(cache) => Recognized::Known(bedrock_cache(cache, supports_ttl)),
            other => other,
        }),
        input_schema: definition.input_schema.map(normalized_schema),
        defer_loading: definition.defer_loading.or(deferred),
        eager_input_streaming: None,
        extra: definition
            .extra
            .into_iter()
            .filter(|(key, _)| key != "custom")
            .collect(),
        ..definition
    }
}

fn bedrock_tool(
    tool: Recognized<MessagesTool>,
    index: usize,
    supports_ttl: bool,
) -> Recognized<MessagesTool> {
    match tool {
        Recognized::Known(tool) => {
            Recognized::Known(tool.map_definition(|definition| {
                bedrock_tool_definition(definition, index, supports_ttl)
            }))
        }
        Recognized::Unrecognized(Value::Object(fields)) => {
            match serde_json::from_value::<ToolDefinition>(Value::Object(fields.clone())) {
                Ok(definition) => {
                    let definition = bedrock_tool_definition(definition, index, supports_ttl);
                    if definition.extra.contains_key("type") {
                        Recognized::Unrecognized(
                            serde_json::to_value(definition).expect("tool definition serializes"),
                        )
                    } else {
                        Recognized::Known(MessagesTool::Custom(CustomTool { definition }))
                    }
                }
                Err(_) => Recognized::Unrecognized(Value::Object(fields)),
            }
        }
        other => other,
    }
}

fn effort_rank(effort: EffortLevel) -> u8 {
    match effort {
        EffortLevel::Low => 0,
        EffortLevel::Medium => 1,
        EffortLevel::High => 2,
        EffortLevel::Max => 3,
        EffortLevel::Xhigh => 4,
    }
}

fn clamp_bedrock_effort(
    request: MessagesRequest,
    context: &MessagesTransformContext,
) -> MessagesRequest {
    let Some(ceiling) = context.thinking.capabilities.effort_ceiling else {
        return request;
    };
    let clamp = |level: EffortLevel| {
        if effort_rank(level) > effort_rank(ceiling) {
            ceiling
        } else {
            level
        }
    };
    let reasoning_effort = request.params.reasoning_effort.map(|effort| match effort {
        Recognized::Known(effort) if context.thinking.capabilities.supports_adaptive_thinking => {
            use litellm_llms_types::formats::chat_completions::ReasoningEffort;
            match effort {
                ReasoningEffort::Low | ReasoningEffort::Minimal | ReasoningEffort::None => {
                    Recognized::Known(effort)
                }
                ReasoningEffort::Medium => Recognized::Known(clamp(EffortLevel::Medium).into()),
                ReasoningEffort::High => Recognized::Known(clamp(EffortLevel::High).into()),
                ReasoningEffort::Xhigh => Recognized::Known(clamp(EffortLevel::Xhigh).into()),
                ReasoningEffort::Max => Recognized::Known(clamp(EffortLevel::Max).into()),
            }
        }
        other => other,
    });
    let output_config = request.params.output_config.map(|output| match output {
        Recognized::Known(output) => Recognized::Known(OutputConfig {
            effort: output.effort.map(|effort| match effort {
                Recognized::Known(level) => Recognized::Known(clamp(level)),
                other => other,
            }),
            ..output
        }),
        other => other,
    });
    MessagesRequest {
        params: MessagesOptionalParams {
            reasoning_effort,
            output_config,
            ..request.params
        },
        ..request
    }
}

pub(crate) fn with_bedrock_clear_thinking(
    request: MessagesRequest,
    context: &MessagesTransformContext,
) -> MessagesRequest {
    let clears = request
        .params
        .context_management
        .as_ref()
        .and_then(Recognized::known)
        .and_then(|management| management.edits.as_deref())
        .into_iter()
        .flatten()
        .any(|edit| matches!(edit, Recognized::Known(ContextEdit::ClearThinking { .. })));
    let capabilities = context.thinking.capabilities;
    if !clears || !(capabilities.supports_reasoning || capabilities.supports_adaptive_thinking) {
        return request;
    }
    let thinking = request.params.thinking.as_ref().and_then(Recognized::known);
    if matches!(thinking, Some(ThinkingConfig::Adaptive(_)))
        || (!capabilities.supports_adaptive_thinking
            && matches!(thinking, Some(ThinkingConfig::Enabled(_))))
    {
        return request;
    }
    let budget = match thinking {
        Some(ThinkingConfig::Enabled(thinking)) => thinking
            .budget_tokens
            .as_ref()
            .and_then(Recognized::known)
            .copied()
            .unwrap_or(crate::anthropic::messages::thinking::ANTHROPIC_MIN_THINKING_BUDGET_TOKENS),
        _ if request.params.max_tokens.is_some_and(|max| {
            max <= crate::anthropic::messages::thinking::ANTHROPIC_MIN_THINKING_BUDGET_TOKENS
        }) =>
        {
            return request;
        }
        _ => crate::anthropic::messages::thinking::ANTHROPIC_MIN_THINKING_BUDGET_TOKENS,
    };
    if !capabilities.supports_adaptive_thinking {
        return MessagesRequest {
            params: MessagesOptionalParams {
                thinking: Some(Recognized::Known(ThinkingConfig::enabled(budget))),
                ..request.params
            },
            ..request
        };
    }
    let display = match thinking {
        Some(ThinkingConfig::Enabled(thinking)) => thinking.display.clone(),
        _ => None,
    };
    let budgets = context.thinking.budgets;
    let effort = if budget >= budgets.xhigh {
        EffortLevel::Xhigh
    } else if budget >= budgets.high {
        EffortLevel::High
    } else if budget >= budgets.medium {
        EffortLevel::Medium
    } else {
        EffortLevel::Low
    };
    let output = match request.params.output_config {
        Some(Recognized::Known(output)) => output,
        _ => OutputConfig::default(),
    };
    MessagesRequest {
        params: MessagesOptionalParams {
            thinking: Some(Recognized::Known(ThinkingConfig::Adaptive(
                litellm_llms_types::formats::messages::AdaptiveThinking {
                    display,
                    ..Default::default()
                },
            ))),
            output_config: Some(Recognized::Known(OutputConfig {
                effort: output.effort.or(Some(Recognized::Known(effort))),
                ..output
            })),
            ..request.params
        },
        ..request
    }
}

fn with_bedrock_output_format(
    request: MessagesRequest,
    context: &MessagesTransformContext,
) -> Result<MessagesRequest, Error> {
    if matches!(
        request.params.output_config,
        Some(Recognized::Unrecognized(_))
    ) && request.params.output_format.is_none()
    {
        return Ok(request);
    }
    let output = request
        .params
        .output_config
        .as_ref()
        .and_then(Recognized::known)
        .cloned()
        .unwrap_or_default();
    let format = request
        .params
        .output_format
        .as_ref()
        .and_then(Recognized::known)
        .cloned()
        .map(Recognized::Known)
        .or(output.format.clone());
    let native = context
        .thinking
        .capabilities
        .supports_native_structured_output;
    let supported_effort = context.thinking.capabilities.supports_output_config
        || context.thinking.capabilities.effort_tiers.any();
    let output = OutputConfig {
        format: if native { format.clone() } else { None },
        effort: if supported_effort {
            output.effort
        } else {
            None
        },
        extra: if supported_effort {
            output.extra
        } else {
            Map::new()
        },
    };
    let schema = format
        .as_ref()
        .and_then(Recognized::known)
        .and_then(|format| format.schema.as_ref())
        .filter(|_| !native);
    let schema_text = schema
        .map(serde_json::to_string)
        .transpose()
        .map_err(|error| {
            Error::InvalidRequest(crate::ErrorDetail::invalid(
                "Bedrock inline JSON schema",
                error,
            ))
        })?;
    let last_user = request
        .messages
        .iter()
        .rposition(|message| message.role == MessageRole::User);
    let messages = request
        .messages
        .into_iter()
        .enumerate()
        .map(|(index, message)| {
            if Some(index) != last_user {
                return message;
            }
            let Some(text) = schema_text.as_ref() else {
                return message;
            };
            let blocks = match message.content {
                MessageContent::Text(text) => vec![ContentBlock::text(text)],
                MessageContent::Blocks(blocks) => blocks,
            };
            litellm_llms_types::formats::messages::Message {
                content: MessageContent::Blocks(
                    blocks
                        .into_iter()
                        .chain([ContentBlock::text(text.clone())])
                        .collect(),
                ),
                ..message
            }
        })
        .collect();
    Ok(MessagesRequest {
        messages,
        params: MessagesOptionalParams {
            output_format: None,
            output_config: (!output.is_empty()).then_some(Recognized::Known(output)),
            ..request.params
        },
        ..request
    })
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
    let events = transform_stream(events, MessageStopUsagePromoter::default())
        .map(|item| item.map_err(StreamError::into_decode));
    Box::pin(
        transform_stream(events, BedrockMessageCompletion::default())
            .map(|item| item.map_err(StreamError::into_decode)),
    )
}

#[derive(Default)]
struct BedrockMessageCompletion {
    started: bool,
    terminal: bool,
}

impl StreamTransformer for BedrockMessageCompletion {
    type Input = MessagesStreamEvent;
    type Output = MessagesStreamEvent;
    type Error = Infallible;

    fn transform(
        &mut self,
        event: MessagesStreamEvent,
    ) -> Result<Vec<MessagesStreamEvent>, Infallible> {
        match &event {
            MessagesStreamEvent::MessageStart { .. } => self.started = true,
            MessagesStreamEvent::MessageStop { .. } | MessagesStreamEvent::Error { .. } => {
                self.terminal = true
            }
            _ => {}
        }
        Ok(vec![event])
    }

    fn finish(&mut self) -> Result<Vec<MessagesStreamEvent>, Infallible> {
        if !self.started || self.terminal {
            return Ok(Vec::new());
        }
        Ok(vec![MessagesStreamEvent::Error {
            error: litellm_llms_types::formats::messages::streaming::MessagesStreamError {
                error_type: "api_error".into(),
                message: "Bedrock Messages stream ended before message_stop".into(),
                details: None,
                extra: Map::new(),
            },
            extra: Map::new(),
        }])
    }
}

#[derive(Default)]
pub struct MessageStopUsagePromoter {
    pending_delta: Option<MessagesStreamEvent>,
    start_usage: Option<MessagesUsage>,
}

fn promoted_usage(
    delta: Option<Box<MessagesUsage>>,
    stop: Option<&MessagesUsage>,
    start: Option<&MessagesUsage>,
) -> Option<Box<MessagesUsage>> {
    let delta = delta.map(|usage| *usage).unwrap_or_default();
    let merged = MessagesUsage {
        input_tokens: stop
            .and_then(|stop| stop.input_tokens.and_then(Nullable::into_value))
            .or(delta.input_tokens.and_then(Nullable::into_value))
            .map(Nullable::Value),
        cache_creation_input_tokens: stop
            .and_then(|stop| {
                stop.cache_creation_input_tokens
                    .and_then(Nullable::into_value)
            })
            .or(delta
                .cache_creation_input_tokens
                .and_then(Nullable::into_value))
            .or_else(|| {
                start.and_then(|start| {
                    start
                        .cache_creation_input_tokens
                        .and_then(Nullable::into_value)
                })
            })
            .map(Nullable::Value),
        cache_read_input_tokens: stop
            .and_then(|stop| stop.cache_read_input_tokens.and_then(Nullable::into_value))
            .or(delta.cache_read_input_tokens.and_then(Nullable::into_value))
            .or_else(|| {
                start.and_then(|start| start.cache_read_input_tokens.and_then(Nullable::into_value))
            })
            .map(Nullable::Value),
        cache_creation: delta
            .cache_creation
            .or_else(|| start.and_then(|start| start.cache_creation.clone())),
        ..delta
    };
    (merged != MessagesUsage::default()).then(|| Box::new(merged))
}

fn promoted(
    event: MessagesStreamEvent,
    stop: Option<&MessagesUsage>,
    start: Option<&MessagesUsage>,
) -> MessagesStreamEvent {
    match event {
        MessagesStreamEvent::MessageDelta {
            delta,
            usage,
            context_management,
            extra,
        } => MessagesStreamEvent::MessageDelta {
            delta,
            usage: promoted_usage(usage, stop, start),
            context_management,
            extra,
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
            MessagesStreamEvent::MessageStop { usage, extra } => Ok(pending
                .map(|delta| promoted(delta, usage.as_deref(), self.start_usage.as_ref()))
                .into_iter()
                .chain([MessagesStreamEvent::MessageStop { usage, extra }])
                .collect()),
            MessagesStreamEvent::MessageStart { message, extra } => {
                self.start_usage = Some(message.usage.clone());
                Ok(pending
                    .into_iter()
                    .chain([MessagesStreamEvent::MessageStart { message, extra }])
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
    use litellm_llms_types::formats::messages::BuiltinMessagesTool;

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
    #[case::explicit_delta_cache_breakdown_takes_precedence(
        json!({"cache_creation":{"ephemeral_5m_input_tokens":4}}),
        json!({"output_tokens":5,"cache_creation":null}),
        None,
        json!({"output_tokens":5,"cache_creation":null}),
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
    #[rstest::fixture]
    fn native_request() -> MessagesRequest {
        MessagesRequest {
            model: "test-model".into(),
            messages: vec![litellm_llms_types::formats::messages::Message {
                role: MessageRole::User,
                content: MessageContent::Text("hello".into()),
                extra: Map::new(),
            }],
            params: MessagesOptionalParams {
                max_tokens: Some(4096),
                ..Default::default()
            },
        }
    }

    fn native_context(
        capabilities: crate::base_llm::messages::context::MessagesModelCapabilities,
    ) -> MessagesTransformContext {
        MessagesTransformContext::with_lookup(capabilities, false, &|_: &str| None)
    }

    #[rstest]
    #[case::unknown_beta("unsupported-client-beta", None)]
    #[case::alias_beta("advanced-tool-use-2025-11-20", Some("tool-search-tool-2025-10-19"))]
    #[case::known_rejected_beta("interleaved-thinking-2025-05-14", None)]
    #[case::supported_beta("context-1m-2025-08-07", Some("context-1m-2025-08-07"))]
    fn test_bedrock_messages_filters_client_betas_into_body(
        native_request: MessagesRequest,
        #[case] client_beta: &str,
        #[case] expected_beta: Option<&str>,
    ) {
        let request = MessagesRequest {
            params: MessagesOptionalParams {
                stream: Some(true),
                temperature: Some(0.5),
                output_config: Some(Recognized::Known(OutputConfig {
                    effort: Some(Recognized::Known(EffortLevel::Low)),
                    ..Default::default()
                })),
                mcp_servers: Some(vec![Recognized::Known(
                    litellm_llms_types::formats::messages::McpServer {
                        server_type: litellm_llms_types::formats::messages::McpServerType::Url,
                        url: Some(Recognized::Known("https://example.invalid".into())),
                        name: None,
                        authorization_token: None,
                        tool_configuration: None,
                        extra: Map::new(),
                    },
                )]),
                context_management: Some(Recognized::Known(ContextManagement {
                    edits: Some(vec![]),
                    extra: Map::new(),
                })),
                speed: Some(Recognized::Known(
                    litellm_llms_types::formats::messages::Speed::Fast,
                )),
                service_tier: Some("priority".into()),
                extra: [
                    ("future_extension".into(), json!({"opaque": true})),
                    ("container".into(), json!({"skills":[]})),
                    ("inference_geo".into(), json!("us")),
                ]
                .into_iter()
                .collect(),
                ..native_request.params
            },
            ..native_request
        };
        let transformed = BEDROCK_ANTHROPIC_MESSAGES_CONFIG
            .transform_anthropic_messages_request(
                request,
                &native_context(
                    crate::base_llm::messages::context::MessagesModelCapabilities {
                        supports_speed: true,
                        supports_output_config: true,
                        supports_adaptive_thinking: true,
                        ..Default::default()
                    },
                ),
            )
            .unwrap();
        let wire = BEDROCK_ANTHROPIC_MESSAGES_CONFIG
            .prepare_wire_request(
                &transformed,
                vec![
                    ("Anthropic-Beta".into(), client_beta.into()),
                    ("x-caller".into(), "keep".into()),
                ],
            )
            .unwrap();
        let expected = expected_beta.map(|beta| json!([beta]));
        assert_eq!(wire.body.get("anthropic_beta"), expected.as_ref());
        assert_eq!(wire.body["anthropic_version"], BEDROCK_ANTHROPIC_VERSION);
        assert_eq!(wire.body["max_tokens"], 4096);
        assert_eq!(wire.body["temperature"], 0.5);
        assert_eq!(wire.body["output_config"], json!({"effort":"low"}));
        assert!(
            wire.body
                .as_object()
                .unwrap()
                .keys()
                .all(|key| bedrock_body_field(key))
        );
        assert_eq!(
            wire.body["messages"],
            json!([{"role":"user", "content":"hello"}])
        );
        for unsupported in [
            "model",
            "stream",
            "speed",
            "service_tier",
            "future_extension",
            "mcp_servers",
            "container",
            "inference_geo",
            "context_management",
        ] {
            assert!(wire.body.get(unsupported).is_none(), "{unsupported}");
        }
        assert_eq!(wire.headers, vec![("x-caller".into(), "keep".into())]);
        assert_eq!(transformed.params.stream, Some(true));
    }

    #[rstest]
    #[case::legacy_hour(false, Some("1h"), None)]
    #[case::legacy_five_minutes(false, Some("5m"), None)]
    #[case::current_hour(true, Some("1h"), Some("1h"))]
    #[case::current_five_minutes(true, Some("5m"), Some("5m"))]
    #[case::unsupported_ttl(true, Some("2h"), None)]
    #[case::no_ttl(true, None, None)]
    fn test_remove_ttl_and_scope_from_cache_control_in_all_protocol_sites(
        native_request: MessagesRequest,
        #[case] supports_cache_control_ttl: bool,
        #[case] ttl: Option<&str>,
        #[case] expected_ttl: Option<&str>,
        #[values(false, true)] forwarded_beta: bool,
    ) {
        use litellm_llms_types::formats::messages::{Message, SystemPrompt};
        let cache = CacheControl {
            cache_type: Some(Nullable::Value("ephemeral".into())),
            ttl: ttl.map(|ttl| Nullable::Value(ttl.into())),
            scope: Some(Nullable::Value("global".into())),
            extra: Map::new(),
        };
        let block = ContentBlock {
            cache_control: Some(Nullable::Value(cache.clone())),
            ..ContentBlock::text("cached")
        };
        let request = MessagesRequest {
            messages: vec![Message {
                content: MessageContent::Blocks(vec![block.clone()]),
                ..native_request.messages[0].clone()
            }],
            params: MessagesOptionalParams {
                system: Some(SystemPrompt::Blocks(vec![block])),
                tools: Some(vec![Recognized::Known(MessagesTool::Custom(CustomTool {
                    definition: ToolDefinition {
                        name: Some(Recognized::Known("lookup".into())),
                        cache_control: Some(Recognized::Known(cache)),
                        ..Default::default()
                    },
                }))]),
                ..native_request.params
            },
            ..native_request
        };
        let result = BEDROCK_ANTHROPIC_MESSAGES_CONFIG
            .transform_anthropic_messages_request(
                request,
                &native_context(
                    crate::base_llm::messages::context::MessagesModelCapabilities {
                        supports_cache_control_ttl,
                        ..Default::default()
                    },
                ),
            )
            .unwrap();
        let wire = BEDROCK_ANTHROPIC_MESSAGES_CONFIG
            .prepare_wire_request(
                &result,
                forwarded_beta
                    .then(|| ("anthropic-beta".into(), "context-1m-2025-08-07".into()))
                    .into_iter()
                    .collect(),
            )
            .unwrap();
        assert_eq!(
            wire.body.get("anthropic_beta"),
            forwarded_beta
                .then(|| json!(["context-1m-2025-08-07"]))
                .as_ref()
        );
        let expected = CacheControl {
            cache_type: Some(Nullable::Value("ephemeral".into())),
            ttl: expected_ttl.map(|ttl| Nullable::Value(ttl.into())),
            scope: None,
            extra: Map::new(),
        };
        assert_eq!(
            result.messages[0].blocks()[0].cache_control,
            Some(Nullable::Value(expected.clone()))
        );
        let Some(litellm_llms_types::formats::messages::SystemPrompt::Blocks(system)) =
            result.params.system
        else {
            panic!("system blocks")
        };
        assert_eq!(
            system[0].cache_control,
            Some(Nullable::Value(expected.clone()))
        );
        let Some(Recognized::Known(MessagesTool::Custom(tool))) =
            result.params.tools.unwrap().into_iter().next()
        else {
            panic!("custom tool")
        };
        assert_eq!(
            tool.definition.cache_control,
            Some(Recognized::Known(expected))
        );
    }

    #[rstest]
    #[case::wrapped_true(None, json!({"defer_loading":true}), Some(true))]
    #[case::explicit_false(Some(false), json!({"defer_loading":true}), Some(false))]
    #[case::wrapped_string(None, json!({"defer_loading":"true"}), None)]
    #[case::wrapped_integer(None, json!({"defer_loading":1}), None)]
    #[case::wrapped_null(None, json!({"defer_loading":null}), None)]
    #[case::wrapped_object(None, json!({"defer_loading":{"nested":true}}), None)]
    #[case::nonobject_custom(None, json!("deferred"), None)]
    #[case::null_custom(None, Value::Null, None)]
    fn test_normalize_custom_field_on_tools(
        #[case] explicit: Option<bool>,
        #[case] custom: Value,
        #[case] expected: Option<bool>,
    ) {
        let definition = ToolDefinition {
            name: Some(Recognized::Known("Read".into())),
            defer_loading: explicit.map(Recognized::Known),
            extra: [
                ("custom".into(), custom),
                ("opaque".into(), json!([null, 1])),
            ]
            .into_iter()
            .collect(),
            ..Default::default()
        };
        let result = bedrock_tool_definition(definition, 0, false);
        assert_eq!(result.defer_loading, expected.map(Recognized::Known));
        assert_eq!(result.name, Some(Recognized::Known("Read".into())));
        assert_eq!(
            result.extra,
            [("opaque".into(), json!([null, 1]))]
                .into_iter()
                .collect::<serde_json::Map<String, serde_json::Value>>()
        );
    }

    #[rstest]
    #[case::missing(None, "litellm_unnamed_tool_3")]
    #[case::empty(Some(""), "litellm_unnamed_tool_3")]
    #[case::whitespace(Some("  "), "litellm_unnamed_tool_3")]
    #[case::existing(Some("KeepMe"), "KeepMe")]
    fn test_ensure_bedrock_anthropic_messages_tool_names(
        #[case] name: Option<&str>,
        #[case] expected: &str,
    ) {
        let result = bedrock_tool_definition(
            ToolDefinition {
                name: name.map(|name| Recognized::Known(name.into())),
                ..Default::default()
            },
            3,
            false,
        );
        assert_eq!(result.name, Some(Recognized::Known(expected.into())));
    }

    #[rstest]
    #[case::custom("custom", "object")]
    #[case::object("object", "object")]
    fn test_normalize_tool_input_schema_types_for_bedrock_invoke(
        #[case] kind: &str,
        #[case] expected: &str,
    ) {
        let schema = |kind: &str| {
            JsonSchema::Object(Box::new(JsonSchemaObject {
                schema_type: Some(Recognized::Known(JsonSchemaType::Name(kind.into()))),
                ..Default::default()
            }))
        };
        let input = JsonSchema::Object(Box::new(JsonSchemaObject {
            schema_type: Some(Recognized::Known(JsonSchemaType::Name(kind.into()))),
            properties: Some(Recognized::Known(
                [("nested".into(), Recognized::Known(schema(kind)))]
                    .into_iter()
                    .collect(),
            )),
            required: Some(Recognized::Known(vec!["nested".into()])),
            ..Default::default()
        }));
        let result = normalized_schema(Recognized::Known(input));
        assert_eq!(
            result,
            Recognized::Known(JsonSchema::Object(Box::new(JsonSchemaObject {
                schema_type: Some(Recognized::Known(JsonSchemaType::Name(expected.into()))),
                properties: Some(Recognized::Known(
                    [("nested".into(), Recognized::Known(schema(expected)))]
                        .into_iter()
                        .collect()
                )),
                required: Some(Recognized::Known(vec!["nested".into()])),
                ..Default::default()
            })))
        );
    }

    #[rstest]
    #[case::constant("const", json!({"type": "custom"}))]
    #[case::enumeration("enum", json!([{"type": "custom"}]))]
    #[case::examples("examples", json!([{"type": "custom"}]))]
    #[case::extension("future", json!({"type": "custom"}))]
    #[case::unrecognized_items("items", json!([{"type": "custom"}]))]
    fn bedrock_tool_schema_normalization_preserves_literal_and_opaque_values(
        #[case] field: &str,
        #[case] value: Value,
    ) {
        let definition: ToolDefinition = serde_json::from_value(json!({
            "name": "example",
            "input_schema": {
                "type": "custom",
                "properties": {
                    "type": {"type": "custom", field: value.clone()}
                }
            }
        }))
        .unwrap();
        let normalized = bedrock_tool_definition(definition, 0, false);
        assert_eq!(
            serde_json::to_value(normalized).unwrap(),
            json!({
                "name": "example",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "type": {"type": "object", field: value}
                    }
                }
            })
        );
    }

    #[rstest]
    #[case::known_2025(
        Recognized::Known(MessagesTool::Builtin(BuiltinMessagesTool::WebSearch(
            ToolDefinition::default()
        )))
    )]
    #[case::known_2026(Recognized::Known(MessagesTool::Builtin(
        BuiltinMessagesTool::WebSearch20260209(ToolDefinition::default())
    )))]
    #[case::future(Recognized::Unrecognized(json!({"type":"web_search_future"}))) ]
    fn test_bedrock_invoke_messages_rejects_server_web_search_tool(
        #[case] tool: Recognized<MessagesTool>,
    ) {
        assert_eq!(
            reject_web_search(Some(&[tool])),
            Err(Error::Unsupported("Bedrock server-side web search tools"))
        );
    }
    #[rstest]
    #[case::unsupported_edit(vec![Recognized::Known(ContextEdit::ClearThinking { extra: Map::new() })], None, vec![])]
    #[case::compact(vec![Recognized::Known(ContextEdit::Compact { trigger: None, extra: Map::new() })], Some(vec![Recognized::Known(ContextEdit::Compact { trigger: None, extra: Map::new() })]), vec!["compact-2026-01-12"])]
    #[case::clear_tools(vec![Recognized::Known(ContextEdit::ClearToolUses { extra: Map::new() })], Some(vec![Recognized::Known(ContextEdit::ClearToolUses { extra: Map::new() })]), vec!["context-management-2025-06-27"])]
    #[case::mixed(vec![Recognized::Known(ContextEdit::Compact { trigger: None, extra: Map::new() }), Recognized::Known(ContextEdit::ClearThinking { extra: Map::new() }), Recognized::Known(ContextEdit::ClearToolUses { extra: Map::new() })], Some(vec![Recognized::Known(ContextEdit::Compact { trigger: None, extra: Map::new() }), Recognized::Known(ContextEdit::ClearToolUses { extra: Map::new() })]), vec!["compact-2026-01-12", "context-management-2025-06-27"])]
    fn test_bedrock_messages_filters_context_management_and_adds_required_beta(
        native_request: MessagesRequest,
        #[case] edits: Vec<Recognized<ContextEdit>>,
        #[case] expected_edits: Option<Vec<Recognized<ContextEdit>>>,
        #[case] expected_betas: Vec<&str>,
    ) {
        let request = MessagesRequest {
            params: MessagesOptionalParams {
                context_management: Some(Recognized::Known(ContextManagement {
                    edits: Some(edits),
                    extra: [("opaque".into(), json!(1))].into_iter().collect(),
                })),
                ..native_request.params
            },
            ..native_request
        };
        let result = BEDROCK_ANTHROPIC_MESSAGES_CONFIG
            .transform_anthropic_messages_request(request, &MessagesTransformContext::default())
            .unwrap();
        assert_eq!(
            result.params.context_management,
            expected_edits.map(|edits| Recognized::Known(ContextManagement {
                edits: Some(edits),
                extra: [("opaque".into(), json!(1))].into_iter().collect()
            }))
        );
        let wire = BEDROCK_ANTHROPIC_MESSAGES_CONFIG
            .prepare_wire_request(&result, vec![])
            .unwrap();
        assert_eq!(
            wire.body.get("anthropic_beta"),
            (!expected_betas.is_empty())
                .then(|| json!(expected_betas))
                .as_ref()
        );
    }

    #[rstest]
    #[case::native_format(true, false, false)]
    #[case::inline_format(false, false, false)]
    #[case::native_legacy_precedence(true, true, false)]
    #[case::inline_legacy_precedence(false, true, false)]
    #[case::native_without_effort_support(true, false, true)]
    #[case::inline_without_effort_support(false, false, true)]
    #[case::inline_legacy_without_effort_support(false, true, true)]
    fn test_bedrock_messages_routes_output_config_format_by_capability(
        native_request: MessagesRequest,
        #[case] native: bool,
        #[case] legacy: bool,
        #[case] strip_effort: bool,
    ) {
        use litellm_llms_types::formats::messages::{OutputFormat, OutputFormatType};
        let format = |name: &str| OutputFormat {
            format_type: OutputFormatType::JsonSchema,
            schema: Some(Recognized::Known(JsonSchema::Object(Box::new(
                JsonSchemaObject {
                    properties: Some(Recognized::Known(
                        [(name.into(), Recognized::Known(JsonSchema::Boolean(true)))]
                            .into_iter()
                            .collect(),
                    )),
                    ..Default::default()
                },
            )))),
            strict: None,
            extra: Map::new(),
        };
        let expected_format = format(if legacy {
            "legacy_field"
        } else {
            "newer_field"
        });
        let request = MessagesRequest {
            params: MessagesOptionalParams {
                output_format: legacy.then(|| Recognized::Known(format("legacy_field"))),
                output_config: Some(Recognized::Known(OutputConfig {
                    effort: Some(Recognized::Known(EffortLevel::High)),
                    format: Some(Recognized::Known(format("newer_field"))),
                    ..Default::default()
                })),
                ..native_request.params
            },
            ..native_request
        };
        let context = native_context(
            crate::base_llm::messages::context::MessagesModelCapabilities {
                supports_native_structured_output: native,
                supports_output_config: !strip_effort,
                ..Default::default()
            },
        );
        let result = BEDROCK_ANTHROPIC_MESSAGES_CONFIG
            .transform_anthropic_messages_request(request, &context)
            .unwrap();
        let expected_output = OutputConfig {
            effort: (!strip_effort).then_some(Recognized::Known(EffortLevel::High)),
            format: native.then(|| Recognized::Known(expected_format.clone())),
            ..Default::default()
        };
        assert_eq!(
            result.params.output_config,
            (!expected_output.is_empty()).then_some(Recognized::Known(expected_output))
        );
        assert!(result.params.output_format.is_none());
        if native {
            assert_eq!(
                result.messages[0].content,
                MessageContent::Text("hello".into())
            );
        } else {
            assert_eq!(result.messages[0].blocks().len(), 2);
            assert_eq!(result.messages[0].blocks()[0], ContentBlock::text("hello"));
            let text = result.messages[0].blocks()[1]
                .text
                .as_ref()
                .and_then(Nullable::as_deref)
                .unwrap();
            assert_eq!(
                serde_json::from_str::<JsonSchema>(text).unwrap(),
                expected_format.schema.unwrap().known().unwrap().clone()
            );
        }
    }

    #[rstest]
    #[case::high_ceiling(EffortLevel::High, EffortLevel::High)]
    #[case::max_ceiling(EffortLevel::Max, EffortLevel::Max)]
    #[case::xhigh_ceiling(EffortLevel::Xhigh, EffortLevel::Xhigh)]
    fn test_bedrock_messages_normalizes_output_config_effort_using_injected_ceiling(
        native_request: MessagesRequest,
        #[case] ceiling: EffortLevel,
        #[case] expected_effort: EffortLevel,
    ) {
        let request = MessagesRequest {
            params: MessagesOptionalParams {
                output_config: Some(Recognized::Known(OutputConfig {
                    effort: Some(Recognized::Known(EffortLevel::Xhigh)),
                    ..Default::default()
                })),
                ..native_request.params
            },
            ..native_request
        };
        let context = native_context(
            crate::base_llm::messages::context::MessagesModelCapabilities {
                effort_ceiling: Some(ceiling),
                supports_output_config: true,
                ..Default::default()
            },
        );
        let result = BEDROCK_ANTHROPIC_MESSAGES_CONFIG
            .transform_anthropic_messages_request(request, &context)
            .unwrap();
        assert_eq!(
            result.params.output_config,
            Some(Recognized::Known(OutputConfig {
                effort: Some(Recognized::Known(expected_effort)),
                ..Default::default()
            }))
        );
    }

    #[rstest]
    #[case::eager(true, "fine-grained-tool-streaming-2025-05-14")]
    #[case::not_eager(false, "")]
    fn test_bedrock_invoke_eager_input_streaming_adds_beta_and_strips_key(
        native_request: MessagesRequest,
        #[case] eager: bool,
        #[case] expected_beta: &str,
    ) {
        let request = MessagesRequest {
            params: MessagesOptionalParams {
                tools: Some(vec![Recognized::Known(MessagesTool::Custom(CustomTool {
                    definition: ToolDefinition {
                        name: Some(Recognized::Known("Read".into())),
                        eager_input_streaming: Some(Recognized::Known(eager)),
                        ..Default::default()
                    },
                }))]),
                ..native_request.params
            },
            ..native_request
        };
        let result = BEDROCK_ANTHROPIC_MESSAGES_CONFIG
            .transform_anthropic_messages_request(request, &MessagesTransformContext::default())
            .unwrap();
        let wire = BEDROCK_ANTHROPIC_MESSAGES_CONFIG
            .prepare_wire_request(
                &result,
                vec![("anthropic-beta".into(), expected_beta.into())],
            )
            .unwrap();
        assert_eq!(wire.body["tools"], json!([{"name":"Read"}]));
        assert_eq!(
            wire.body.get("anthropic_beta"),
            eager.then(|| json!([expected_beta])).as_ref()
        );
    }
    #[rstest]
    #[case::caller_omits_beta(true, None)]
    #[case::caller_sends_only_other_beta(true, Some("interleaved-thinking-2025-05-14"))]
    #[case::caller_sends_beta(true, Some("dangerous-tool-use-2026-09-03"))]
    #[case::caller_sends_mixed_betas(
        true,
        Some("dangerous-tool-use-2026-09-03,interleaved-thinking-2025-05-14")
    )]
    #[case::no_safeguards(false, None)]
    fn test_messages_forwards_safeguards_with_one_dangerous_tool_use_beta(
        #[case] enabled: bool,
        #[case] beta_header: Option<&str>,
    ) {
        use litellm_llms_types::formats::messages::{
            Message, MessageContent, MessageRole, Safeguard,
        };
        let safeguards = Recognized::Known(vec![Recognized::Known(Safeguard {
            safeguard_type: "dangerous_tool_use".into(),
            classifier_context: Some(Recognized::Known(
                [
                    ("v".into(), json!(1)),
                    ("permission_mode".into(), json!("auto")),
                ]
                .into_iter()
                .collect(),
            )),
            extra: Map::new(),
        })]);
        let request = MessagesRequest {
            model: "test-model".into(),
            messages: vec![Message {
                role: MessageRole::User,
                content: MessageContent::Text("hello".into()),
                extra: Map::new(),
            }],
            params: MessagesOptionalParams {
                max_tokens: Some(64),
                safeguards: enabled.then_some(safeguards.clone()),
                ..Default::default()
            },
        };
        let result = BEDROCK_ANTHROPIC_MESSAGES_CONFIG
            .transform_anthropic_messages_request(request, &MessagesTransformContext::default())
            .unwrap();
        assert_eq!(result.params.safeguards, enabled.then_some(safeguards));
        let headers = beta_header
            .map(|beta| ("anthropic-beta".into(), beta.into()))
            .into_iter()
            .collect();
        let wire = BEDROCK_ANTHROPIC_MESSAGES_CONFIG
            .prepare_wire_request(&result, headers)
            .unwrap();
        let betas = wire
            .body
            .get("anthropic_beta")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter_map(Value::as_str)
            .map(|beta| beta.parse().unwrap_or_else(|never| match never {}))
            .collect::<BetaSet>();
        assert_eq!(wire.body.get("safeguards"), enabled.then(|| json!([{ "type":"dangerous_tool_use", "classifier_context":{"v":1, "permission_mode":"auto"} }])).as_ref());
        assert_eq!(
            betas
                .iter()
                .filter(|beta| beta.as_str() == "dangerous-tool-use-2026-09-03")
                .count(),
            usize::from(enabled)
        );
        let response: litellm_llms_types::formats::messages::MessagesResponse = serde_json::from_value(json!({
            "id":"msg_1", "type":"message", "role":"assistant", "model":"model",
            "content":[{"type":"text","text":"ok"}], "stop_reason":"end_turn", "stop_sequence":null,
            "usage":{"input_tokens":1,"output_tokens":2},
            "safeguard_results":[{"type":"dangerous_tool_use", "status":{"type":"available", "tool_uses":{"toolu_01":{"type":"evaluated", "outcome":"not_flagged"}}}}]
        })).unwrap();
        let actual = BEDROCK_ANTHROPIC_MESSAGES_CONFIG
            .transform_anthropic_messages_response("model", response.clone())
            .unwrap();
        assert_eq!(actual, response);
    }
    #[rstest]
    #[case::foundation("anthropic.test-model-v1:0", "anthropic.test-model-v1%3A0")]
    #[case::profile_arn(
        "arn:aws:bedrock:us-east-1:123456789012:inference-profile/test",
        "arn%3Aaws%3Abedrock%3Aus-east-1%3A123456789012%3Ainference-profile%2Ftest"
    )]
    #[case::path_escape("../other?x=1#frag", "..%2Fother%3Fx%3D1%23frag")]
    fn invoke_model_id_is_one_encoded_url_segment(
        #[case] model: &str,
        #[case] expected_model: &str,
    ) {
        let expected = format!("https://bedrock.example/prefix/model/{expected_model}/invoke");
        let actual = BEDROCK_ANTHROPIC_MESSAGES_CONFIG
            .get_complete_url(Some("https://bedrock.example/prefix"), model, &|_| None)
            .unwrap();
        assert_eq!(actual, expected);
        let url = url::Url::parse(&actual).unwrap();
        assert!(url.query().is_none());
        assert!(url.fragment().is_none());
    }
    #[rstest]
    #[case::adaptive_entry(true)]
    #[case::entry_override_disables_adaptive(false)]
    fn test_messages_thinking_shape_follows_injected_provider_entry_flag(#[case] adaptive: bool) {
        use litellm_llms_types::formats::{
            chat_completions::ReasoningEffort,
            messages::{
                EffortLevel, Message, MessageContent, MessageRole, OutputConfig, ThinkingConfig,
                ThinkingDisplay,
            },
        };
        let request = MessagesRequest {
            model: "same-model".into(),
            messages: vec![Message {
                role: MessageRole::User,
                content: MessageContent::Text("hello".into()),
                extra: Default::default(),
            }],
            params: MessagesOptionalParams {
                max_tokens: Some(4096),
                reasoning_effort: Some(Recognized::Known(ReasoningEffort::Medium)),
                ..Default::default()
            },
        };
        let context = MessagesTransformContext::with_lookup(
            crate::base_llm::messages::context::MessagesModelCapabilities {
                supports_adaptive_thinking: adaptive,
                supports_output_config: adaptive,
                supports_reasoning: true,
                supports_legacy_thinking: true,
                ..Default::default()
            },
            false,
            &|_: &str| None,
        );
        let result = BEDROCK_ANTHROPIC_MESSAGES_CONFIG
            .transform_anthropic_messages_request(request, &context)
            .unwrap();
        assert_eq!(
            result.params.thinking,
            Some(Recognized::Known(if adaptive {
                ThinkingConfig::adaptive(Some(ThinkingDisplay::Summarized))
            } else {
                ThinkingConfig::enabled(2048)
            }))
        );
        assert_eq!(
            result.params.output_config,
            adaptive.then(|| Recognized::Known(OutputConfig {
                effort: Some(Recognized::Known(EffortLevel::Medium)),
                ..Default::default()
            }))
        );
    }
    #[rstest]
    #[case::delta_metrics(json!({"type":"message_delta", "amazon-bedrock-invocationMetrics":{"inputTokenCount":10,"outputTokenCount":5}}), json!({"type":"message_delta","usage":{"input_tokens":10,"output_tokens":5}}))]
    #[case::cache_fields_survive(json!({"type":"message_stop", "usage":{"cache_read_input_tokens":9821,"cache_creation_input_tokens":0}, "amazon-bedrock-invocationMetrics":{"inputTokenCount":10174,"outputTokenCount":500}}), json!({"type":"message_stop","usage":{"input_tokens":10174,"output_tokens":500,"cache_read_input_tokens":9821,"cache_creation_input_tokens":0}}))]
    #[case::metrics_cache_fields(json!({"type":"message_stop", "amazon-bedrock-invocationMetrics":{"inputTokenCount":10174,"outputTokenCount":500,"cacheReadInputTokenCount":9821,"cacheWriteInputTokenCount":42}}), json!({"type":"message_stop","usage":{"input_tokens":10174,"output_tokens":500,"cache_read_input_tokens":9821,"cache_creation_input_tokens":42}}))]
    #[case::chunk_counts_win(json!({"type":"message_stop", "usage":{"input_tokens":7,"output_tokens":11,"cache_read_input_tokens":3}, "amazon-bedrock-invocationMetrics":{"inputTokenCount":999,"outputTokenCount":999}}), json!({"type":"message_stop","usage":{"input_tokens":7,"output_tokens":11,"cache_read_input_tokens":3}}))]
    fn test_chunk_parser_transforms_invocation_metrics_without_losing_native_usage(
        #[case] input: Value,
        #[case] expected: Value,
    ) {
        assert_eq!(with_invocation_usage(input), expected);
    }

    #[rstest]
    #[case::truncated(false)]
    #[case::completed(true)]
    #[tokio::test]
    async fn test_bedrock_binary_stream_reports_incomplete_tool_use(#[case] completed: bool) {
        use litellm_llms_types::formats::messages::streaming::{
            MessagesContentBlock, MessagesContentBlockDelta, MessagesStreamMessage,
        };
        use litellm_llms_types::formats::messages::{
            ContentBlockPayload, ContentBlockType, MessageType,
        };
        let start = MessagesStreamEvent::MessageStart {
            message: Box::new(MessagesStreamMessage {
                id: "msg_fixture".into(),
                message_type: MessageType::Message,
                role: MessageRole::Assistant,
                model: "test-model".into(),
                content: vec![],
                stop_reason: None,
                stop_sequence: None,
                usage: MessagesUsage {
                    input_tokens: Some(Nullable::Value(3)),
                    output_tokens: Some(Nullable::Value(1)),
                    ..Default::default()
                },
                safeguard_results: None,
                extra: Map::new(),
            }),
            extra: Map::new(),
        };
        let block = MessagesStreamEvent::ContentBlockStart {
            index: 0,
            content_block: Box::new(MessagesContentBlock {
                block_type: ContentBlockType::ToolUse,
                tool_use_id: None,
                cache_control: None,
                payload: ContentBlockPayload {
                    id: Some(Nullable::Value("tooluse_1".into())),
                    name: Some(Nullable::Value("write".into())),
                    input: Some(Recognized::Known(Map::new())),
                    ..Default::default()
                },
            }),
            extra: Map::new(),
        };
        let delta = MessagesStreamEvent::ContentBlockDelta {
            index: 0,
            delta: MessagesContentBlockDelta::InputJsonDelta {
                partial_json: "{\"path\":\"partial".into(),
                extra: Map::new(),
            },
            extra: Map::new(),
        };
        let initial = vec![start, block, delta];
        let events: Vec<_> = initial
            .iter()
            .cloned()
            .chain(completed.then(|| MessagesStreamEvent::ContentBlockStop {
                index: 0,
                extra: Map::new(),
            }))
            .chain(completed.then(|| message_delta(json!({"output_tokens":2}))))
            .chain(completed.then(|| message_stop(None)))
            .collect();
        let wire: Vec<u8> = events
            .iter()
            .flat_map(|event| aws_frame(&serde_json::to_value(event).unwrap()))
            .collect();
        let bytes: ByteStream = futures_util::stream::iter(
            wire.chunks(7)
                .map(|chunk| Ok(Bytes::copy_from_slice(chunk)))
                .collect::<Vec<_>>(),
        )
        .boxed();
        let decoded = bedrock_anthropic_messages_event_stream(bytes)
            .try_collect::<Vec<_>>()
            .await
            .unwrap();
        assert_eq!(&decoded[..3], initial.as_slice());
        let sse = decoded
            .iter()
            .map(|event| encode_anthropic_sse(event).unwrap())
            .collect::<Vec<_>>();
        if completed {
            assert_eq!(decoded.len(), 6);
            assert!(matches!(
                decoded.last(),
                Some(MessagesStreamEvent::MessageStop { .. })
            ));
            assert!(
                !decoded
                    .iter()
                    .any(|event| matches!(event, MessagesStreamEvent::Error { .. }))
            );
            assert!(sse.last().unwrap().starts_with(b"event: message_stop\n"));
        } else {
            assert_eq!(decoded.len(), 4);
            let Some(MessagesStreamEvent::Error { error, .. }) = decoded.last() else {
                panic!("expected Anthropic error event")
            };
            assert_eq!(error.error_type, "api_error");
            assert!(sse.last().unwrap().starts_with(b"event: error\n"));
        }
    }
    #[rstest]
    #[case::absent(
        true,
        None,
        Some(32000),
        Some(ThinkingConfig::adaptive(None)),
        Some(EffortLevel::Low)
    )]
    #[case::disabled(
        true,
        Some(ThinkingConfig::Disabled(Default::default())),
        Some(32000),
        Some(ThinkingConfig::adaptive(None)),
        Some(EffortLevel::Low)
    )]
    #[case::existing_adaptive(
        true,
        Some(ThinkingConfig::adaptive(None)),
        Some(32000),
        Some(ThinkingConfig::adaptive(None)),
        None
    )]
    #[case::explicit_zero(
        true,
        Some(ThinkingConfig::enabled(0)),
        Some(32000),
        Some(ThinkingConfig::adaptive(None)),
        Some(EffortLevel::Low)
    )]
    #[case::medium_boundary(
        true,
        Some(ThinkingConfig::enabled(2048)),
        Some(32000),
        Some(ThinkingConfig::adaptive(None)),
        Some(EffortLevel::Medium)
    )]
    #[case::high_boundary(
        true,
        Some(ThinkingConfig::enabled(4096)),
        Some(32000),
        Some(ThinkingConfig::adaptive(None)),
        Some(EffortLevel::High)
    )]
    #[case::xhigh_boundary(
        true,
        Some(ThinkingConfig::enabled(8192)),
        Some(32000),
        Some(ThinkingConfig::adaptive(None)),
        Some(EffortLevel::Xhigh)
    )]
    #[case::nonadaptive(false, None, Some(32000), Some(ThinkingConfig::enabled(1024)), None)]
    #[case::enabled_nonadaptive(
        false,
        Some(ThinkingConfig::enabled(8000)),
        Some(32000),
        Some(ThinkingConfig::enabled(8000)),
        None
    )]
    #[case::no_room(true, None, Some(1024), None, None)]
    #[case::omitted_max(
        true,
        None,
        None,
        Some(ThinkingConfig::adaptive(None)),
        Some(EffortLevel::Low)
    )]
    fn test_bedrock_clear_thinking_injects_typed_adaptive_or_legacy_config(
        native_request: MessagesRequest,
        #[case] adaptive: bool,
        #[case] thinking: Option<ThinkingConfig>,
        #[case] max_tokens: Option<u64>,
        #[case] expected_thinking: Option<ThinkingConfig>,
        #[case] expected_effort: Option<EffortLevel>,
    ) {
        let request = MessagesRequest {
            params: MessagesOptionalParams {
                max_tokens,
                thinking: thinking.map(Recognized::Known),
                context_management: Some(Recognized::Known(ContextManagement {
                    edits: Some(vec![Recognized::Known(ContextEdit::ClearThinking {
                        extra: Map::new(),
                    })]),
                    extra: Map::new(),
                })),
                ..native_request.params
            },
            ..native_request
        };
        let context = MessagesTransformContext::with_lookup(
            crate::base_llm::messages::context::MessagesModelCapabilities {
                supports_reasoning: true,
                supports_adaptive_thinking: adaptive,
                supports_output_config: adaptive,
                ..Default::default()
            },
            false,
            &|_: &str| None,
        );
        let result = with_bedrock_clear_thinking(request, &context);
        assert_eq!(
            result.params.thinking,
            expected_thinking.map(Recognized::Known)
        );
        assert_eq!(
            result.params.output_config,
            expected_effort.map(|effort| Recognized::Known(OutputConfig {
                effort: Some(Recognized::Known(effort)),
                ..Default::default()
            }))
        );
    }

    #[rstest]
    #[case::zero(0, EffortLevel::Low)]
    #[case::below_medium(2047, EffortLevel::Low)]
    #[case::medium(2048, EffortLevel::Medium)]
    #[case::below_high(4095, EffortLevel::Medium)]
    #[case::high(4096, EffortLevel::High)]
    #[case::below_xhigh(8191, EffortLevel::High)]
    #[case::xhigh(8192, EffortLevel::Xhigh)]
    #[case::above_xhigh(16384, EffortLevel::Xhigh)]
    fn test_bedrock_clear_thinking_budget_boundaries_and_existing_effort(
        native_request: MessagesRequest,
        #[case] budget: u64,
        #[case] effort: EffortLevel,
    ) {
        let request = MessagesRequest {
            params: MessagesOptionalParams {
                thinking: Some(Recognized::Known(ThinkingConfig::enabled(budget))),
                context_management: Some(Recognized::Known(ContextManagement {
                    edits: Some(vec![Recognized::Known(ContextEdit::ClearThinking {
                        extra: Map::new(),
                    })]),
                    extra: Map::new(),
                })),
                ..native_request.params
            },
            ..native_request
        };
        let context = MessagesTransformContext::with_lookup(
            crate::base_llm::messages::context::MessagesModelCapabilities {
                supports_adaptive_thinking: true,
                supports_output_config: true,
                ..Default::default()
            },
            false,
            &|_: &str| None,
        );
        let output = OutputConfig {
            effort: Some(Recognized::Known(EffortLevel::Max)),
            extra: [("other".into(), json!("keep"))].into_iter().collect(),
            ..Default::default()
        };
        let preserved = with_bedrock_clear_thinking(
            MessagesRequest {
                params: MessagesOptionalParams {
                    output_config: Some(Recognized::Known(output.clone())),
                    ..request.params.clone()
                },
                ..request.clone()
            },
            &context,
        );
        assert_eq!(
            preserved.params.output_config,
            Some(Recognized::Known(output))
        );
        let derived = with_bedrock_clear_thinking(request, &context);
        assert_eq!(
            derived
                .params
                .output_config
                .and_then(|config| config.known().cloned())
                .unwrap()
                .effort,
            Some(Recognized::Known(effort))
        );
    }

    #[rstest]
    #[case::xhigh_clamped(
        litellm_llms_types::formats::chat_completions::ReasoningEffort::Xhigh,
        Some(EffortLevel::Max),
        Some(EffortLevel::Max)
    )]
    #[case::max_kept(
        litellm_llms_types::formats::chat_completions::ReasoningEffort::Max,
        Some(EffortLevel::Max),
        Some(EffortLevel::Max)
    )]
    #[case::high_kept(
        litellm_llms_types::formats::chat_completions::ReasoningEffort::High,
        Some(EffortLevel::Max),
        Some(EffortLevel::High)
    )]
    #[case::xhigh_supported(
        litellm_llms_types::formats::chat_completions::ReasoningEffort::Xhigh,
        Some(EffortLevel::Xhigh),
        Some(EffortLevel::Xhigh)
    )]
    #[case::xhigh_rejected(
        litellm_llms_types::formats::chat_completions::ReasoningEffort::Xhigh,
        None,
        None
    )]
    fn test_bedrock_invoke_messages_clamps_reasoning_effort_before_validation(
        native_request: MessagesRequest,
        #[case] effort: litellm_llms_types::formats::chat_completions::ReasoningEffort,
        #[case] ceiling: Option<EffortLevel>,
        #[case] expected: Option<EffortLevel>,
    ) {
        let request = MessagesRequest {
            params: MessagesOptionalParams {
                max_tokens: Some(1024),
                reasoning_effort: Some(Recognized::Known(effort)),
                ..native_request.params
            },
            ..native_request
        };
        let context = MessagesTransformContext::with_lookup(
            crate::base_llm::messages::context::MessagesModelCapabilities {
                supports_adaptive_thinking: true,
                supports_output_config: true,
                effort_ceiling: ceiling,
                effort_tiers: crate::base_llm::messages::context::SupportedEffortTiers {
                    xhigh: ceiling == Some(EffortLevel::Xhigh),
                    ..Default::default()
                },
                ..Default::default()
            },
            false,
            &|_: &str| None,
        );
        let result = BEDROCK_ANTHROPIC_MESSAGES_CONFIG
            .transform_anthropic_messages_request(request, &context);
        let Some(expected) = expected else {
            assert!(matches!(result, Err(Error::InvalidRequest(_))));
            return;
        };
        let result = result.unwrap();
        assert!(matches!(
            result.params.thinking,
            Some(Recognized::Known(ThinkingConfig::Adaptive(_)))
        ));
        assert_eq!(
            result.params.output_config.unwrap().known().unwrap().effort,
            Some(Recognized::Known(expected))
        );
        assert_eq!(result.params.max_tokens, Some(1024));
        assert!(result.params.reasoning_effort.is_none());
    }

    #[rstest]
    #[case::call_region(
        Some("eu-west-1"),
        None,
        None,
        "https://bedrock-runtime.eu-west-1.amazonaws.com",
        "anthropic.test-model-v1%3A0",
        "eu-west-1"
    )]
    #[case::model_region(
        None,
        None,
        None,
        "https://bedrock-runtime.us-west-2.amazonaws.com",
        "anthropic.test-model-v1%3A0",
        "us-west-2"
    )]
    #[case::model_override(
        Some("eu-west-1"),
        Some("profile/target"),
        Some("https://proxy.invalid/"),
        "https://proxy.invalid",
        "profile%2Ftarget",
        "eu-west-1"
    )]
    #[case::blank_api_base(
        None,
        None,
        Some("  "),
        "https://bedrock-runtime.us-west-2.amazonaws.com",
        "anthropic.test-model-v1%3A0",
        "us-west-2"
    )]
    #[case::trimmed_api_base(
        None,
        None,
        Some("  https://proxy.invalid/  "),
        "https://proxy.invalid",
        "anthropic.test-model-v1%3A0",
        "us-west-2"
    )]
    fn test_bedrock_connection_options_keep_url_and_signature_consistent(
        #[case] region: Option<&str>,
        #[case] model_id: Option<&str>,
        #[case] api_base: Option<&str>,
        #[case] endpoint: &str,
        #[case] target: &str,
        #[case] expected_region: &str,
    ) {
        let connection = BedrockMessagesConnection {
            region: region.map(str::to_string),
            model_id: model_id.map(str::to_string),
            api_base: api_base.map(str::to_string),
            ..Default::default()
        };
        let model = "bedrock/us-west-2/anthropic.test-model-v1:0";
        let model = model.strip_prefix("bedrock/").unwrap();
        for stream in [false, true] {
            let url = BEDROCK_ANTHROPIC_MESSAGES_CONFIG
                .get_complete_url_with_connection(model, &connection, stream, &|_| None)
                .unwrap();
            assert_eq!(
                url,
                format!(
                    "{endpoint}/model/{target}/{}",
                    if stream {
                        INVOKE_STREAM_PATH
                    } else {
                        INVOKE_PATH
                    }
                )
            );
        }
        let environment = BEDROCK_ANTHROPIC_MESSAGES_CONFIG
            .validate_environment_with_connection(vec![], None, model, &connection, &|_| None)
            .unwrap();
        let AuthScheme::AwsSigV4 { region, .. } = environment.auth else {
            panic!("AWS signing expected")
        };
        assert_eq!(region, expected_region);
    }
    #[rstest]
    #[case::empty(OutputConfig::default())]
    #[case::effort(OutputConfig { effort: Some(Recognized::Known(EffortLevel::High)), ..Default::default() })]
    #[case::format(serde_json::from_value(json!({"format":{"type":"text"}})).unwrap())]
    fn test_bedrock_messages_mid_conversation_output_config_beta(
        native_request: MessagesRequest,
        #[case] output_config: OutputConfig,
        #[values(false, true)] nested_output_config: bool,
        #[values(false, true)] explicit_beta: bool,
    ) {
        use litellm_llms_types::formats::messages::Message;
        let middle = Message {
            role: MessageRole::System,
            content: MessageContent::Blocks(vec![]),
            extra: [(
                "output_config".into(),
                serde_json::to_value(output_config).unwrap(),
            )]
            .into_iter()
            .collect(),
        };
        let messages: Vec<_> = native_request
            .messages
            .iter()
            .cloned()
            .chain(nested_output_config.then_some(middle))
            .chain([Message {
                role: MessageRole::User,
                content: MessageContent::Text("Reply with OK".into()),
                extra: Map::new(),
            }])
            .collect();
        let request = MessagesRequest {
            messages: messages.clone(),
            params: MessagesOptionalParams {
                max_tokens: Some(1024),
                output_config: Some(Recognized::Known(OutputConfig {
                    effort: Some(Recognized::Known(EffortLevel::High)),
                    ..Default::default()
                })),
                ..native_request.params
            },
            ..native_request
        };
        let context = native_context(
            crate::base_llm::messages::context::MessagesModelCapabilities {
                supports_mid_conversation_system: true,
                supports_output_config: true,
                ..Default::default()
            },
        );
        let result = BEDROCK_ANTHROPIC_MESSAGES_CONFIG
            .transform_anthropic_messages_request(request, &context)
            .unwrap();
        let beta = "mid-conversation-output-config-2026-07-01";
        let wire = BEDROCK_ANTHROPIC_MESSAGES_CONFIG
            .prepare_wire_request(
                &result,
                explicit_beta
                    .then(|| ("anthropic-beta".into(), beta.into()))
                    .into_iter()
                    .collect(),
            )
            .unwrap();
        assert_eq!(result.messages, messages);
        assert_eq!(wire.body["output_config"], json!({"effort":"high"}));
        assert_eq!(
            wire.body
                .get("anthropic_beta")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
                .filter(|value| value.as_str() == Some(beta))
                .count(),
            usize::from(nested_output_config || explicit_beta)
        );
    }

    #[rstest]
    #[case::absent(None)]
    #[case::addition(Some(litellm_llms_types::formats::messages::ContentBlockType::ToolAddition))]
    #[case::removal(Some(litellm_llms_types::formats::messages::ContentBlockType::ToolRemoval))]
    fn test_bedrock_messages_tool_changes_beta(
        native_request: MessagesRequest,
        #[case] action: Option<litellm_llms_types::formats::messages::ContentBlockType>,
        #[values(false, true)] explicit_beta: bool,
        #[values(false, true)] leading: bool,
    ) {
        use litellm_llms_types::formats::messages::{
            ContentBlockPayload, ContentBlockType, Message,
        };
        let content = match action.clone() {
            Some(action) => MessageContent::Blocks(vec![ContentBlock {
                block_type: Some(Nullable::Value(action)),
                payload: ContentBlockPayload {
                    tool: Some(Recognized::Known(Box::new(ContentBlock {
                        block_type: Some(Nullable::Value(ContentBlockType::ToolReference)),
                        payload: ContentBlockPayload {
                            name: Some(Nullable::Value("mcp__test__ping".into())),
                            ..Default::default()
                        },
                        ..Default::default()
                    }))),
                    ..Default::default()
                },
                ..Default::default()
            }]),
            None => MessageContent::Text("Answer briefly".into()),
        };
        let system = Message {
            role: MessageRole::System,
            content,
            extra: Map::new(),
        };
        let messages = if leading {
            vec![system, native_request.messages[0].clone()]
        } else {
            vec![native_request.messages[0].clone(), system]
        };
        let expected = if leading {
            native_request.messages.clone()
        } else {
            messages.clone()
        };
        let request = MessagesRequest {
            messages,
            ..native_request
        };
        let context = native_context(
            crate::base_llm::messages::context::MessagesModelCapabilities {
                supports_mid_conversation_system: true,
                ..Default::default()
            },
        );
        let result = BEDROCK_ANTHROPIC_MESSAGES_CONFIG
            .transform_anthropic_messages_request(request, &context)
            .unwrap();
        let beta = "mid-conversation-tool-changes-2026-07-01";
        let wire = BEDROCK_ANTHROPIC_MESSAGES_CONFIG
            .prepare_wire_request(
                &result,
                explicit_beta
                    .then(|| ("anthropic-beta".into(), beta.into()))
                    .into_iter()
                    .collect(),
            )
            .unwrap();
        assert_eq!(result.messages, expected);
        assert_eq!(
            wire.body
                .get("anthropic_beta")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
                .filter(|value| value.as_str() == Some(beta))
                .count(),
            usize::from((!leading && action.is_some()) || explicit_beta)
        );
    }

    #[rstest]
    #[case::absent(None)]
    #[case::summarized(Some(litellm_llms_types::formats::messages::ThinkingDisplay::Summarized))]
    #[case::omitted(Some(litellm_llms_types::formats::messages::ThinkingDisplay::Omitted))]
    #[case::updates(Some(litellm_llms_types::formats::messages::ThinkingDisplay::Updates))]
    fn test_bedrock_messages_thinking_display_updates_beta(
        native_request: MessagesRequest,
        #[case] display: Option<litellm_llms_types::formats::messages::ThinkingDisplay>,
        #[values(false, true)] explicit_beta: bool,
    ) {
        let request = MessagesRequest {
            params: MessagesOptionalParams {
                max_tokens: Some(512),
                thinking: Some(Recognized::Known(ThinkingConfig::adaptive(display))),
                ..native_request.params
            },
            ..native_request
        };
        let context = native_context(
            crate::base_llm::messages::context::MessagesModelCapabilities {
                supports_adaptive_thinking: true,
                ..Default::default()
            },
        );
        let result = BEDROCK_ANTHROPIC_MESSAGES_CONFIG
            .transform_anthropic_messages_request(request, &context)
            .unwrap();
        let beta = "thinking-display-updates-2026-08-18";
        let wire = BEDROCK_ANTHROPIC_MESSAGES_CONFIG
            .prepare_wire_request(
                &result,
                explicit_beta
                    .then(|| ("anthropic-beta".into(), beta.into()))
                    .into_iter()
                    .collect(),
            )
            .unwrap();
        assert_eq!(
            result.params.thinking,
            Some(Recognized::Known(ThinkingConfig::adaptive(display)))
        );
        assert_eq!(
            wire.body
                .get("anthropic_beta")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
                .filter(|value| value.as_str() == Some(beta))
                .count(),
            usize::from(
                display == Some(litellm_llms_types::formats::messages::ThinkingDisplay::Updates)
                    || explicit_beta
            )
        );
    }

    #[rstest]
    #[case::support(true, true)]
    #[case::unsupported(false, false)]
    fn test_bedrock_messages_tool_search_uses_injected_model_capability(
        native_request: MessagesRequest,
        #[case] support: bool,
        #[case] expected: bool,
    ) {
        let tools = vec![Recognized::Known(MessagesTool::Builtin(
            BuiltinMessagesTool::ToolSearchRegex(ToolDefinition {
                name: Some(Recognized::Known("tool_search_tool_regex".into())),
                ..Default::default()
            }),
        ))];
        let request = MessagesRequest {
            params: MessagesOptionalParams {
                tools: Some(tools.clone()),
                ..native_request.params
            },
            ..native_request
        };
        let context = native_context(
            crate::base_llm::messages::context::MessagesModelCapabilities {
                supports_tool_search: support,
                ..Default::default()
            },
        );
        let result = BEDROCK_ANTHROPIC_MESSAGES_CONFIG
            .transform_anthropic_messages_request(request, &context)
            .unwrap();
        let wire = BEDROCK_ANTHROPIC_MESSAGES_CONFIG
            .prepare_wire_request(&result, vec![])
            .unwrap();
        assert_eq!(result.params.tools, Some(tools));
        assert_eq!(
            wire.body
                .get("anthropic_beta")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
                .filter(|value| value.as_str() == Some("tool-search-tool-2025-10-19"))
                .count(),
            usize::from(expected)
        );
    }
    #[rstest]
    #[case::stop_cache(3, 8, None, Some((1562, 32392)), false)]
    #[case::stop_cache_with_metrics(10174, 500, None, Some((0, 9821)), true)]
    #[case::start_cache_without_stop_cache(1804, 272, Some((0, 22167)), None, false)]
    #[tokio::test]
    async fn test_bedrock_sse_wrapper_preserves_start_and_delta_cache_usage(
        #[case] input: u64,
        #[case] output: u64,
        #[case] start_cache: Option<(u64, u64)>,
        #[case] stop_cache: Option<(u64, u64)>,
        #[case] metrics: bool,
    ) {
        let cache_fields = |cache: Option<(u64, u64)>| {
            cache.into_iter().flat_map(|(creation, read)| {
                [
                    ("cache_creation_input_tokens".to_string(), json!(creation)),
                    ("cache_read_input_tokens".to_string(), json!(read)),
                ]
            })
        };
        let start_usage: Map<String, Value> = [
            ("input_tokens".into(), json!(input)),
            ("output_tokens".into(), json!(1)),
        ]
        .into_iter()
        .chain(cache_fields(start_cache))
        .collect();
        let stop_usage: Map<String, Value> = (!metrics)
            .then(|| ("input_tokens".into(), json!(input)))
            .into_iter()
            .chain(cache_fields(stop_cache))
            .collect();
        let stop: Map<String, Value> = [
            ("type".into(), json!("message_stop")),
            ("usage".into(), Value::Object(stop_usage)),
        ]
        .into_iter()
        .chain(metrics.then(|| {
            (
                "amazon-bedrock-invocationMetrics".into(),
                json!({"inputTokenCount":input,"outputTokenCount":output}),
            )
        }))
        .collect();
        let events = [
            serde_json::to_value(message_start(Value::Object(start_usage.clone()))).unwrap(),
            serde_json::to_value(message_delta(json!({"output_tokens":output}))).unwrap(),
            Value::Object(stop),
        ];
        let wire: Vec<_> = events.iter().flat_map(aws_frame).collect();
        let bytes: ByteStream = futures_util::stream::iter(
            wire.chunks(3)
                .map(|part| Ok(Bytes::copy_from_slice(part)))
                .collect::<Vec<_>>(),
        )
        .boxed();
        let decoded = bedrock_anthropic_messages_event_stream(bytes)
            .try_collect::<Vec<_>>()
            .await
            .unwrap();
        let sse: Vec<Value> = decoded
            .iter()
            .map(|event| {
                let wire =
                    String::from_utf8(encode_anthropic_sse(event).unwrap().to_vec()).unwrap();
                serde_json::from_str(wire.split_once("data: ").unwrap().1.trim()).unwrap()
            })
            .collect();
        assert_eq!(sse.len(), 3);
        assert_eq!(sse[0]["message"]["usage"], Value::Object(start_usage));
        let expected_cache = stop_cache.or(start_cache);
        let delta_usage: Map<String, Value> = [
            ("input_tokens".into(), json!(input)),
            ("output_tokens".into(), json!(output)),
        ]
        .into_iter()
        .chain(cache_fields(expected_cache))
        .collect();
        assert_eq!(sse[1]["usage"], Value::Object(delta_usage));
        assert_eq!(sse[2]["type"], json!("message_stop"));
    }

    #[rstest]
    #[tokio::test]
    async fn test_bedrock_messages_stream_decoder_keeps_safeguard_results() {
        let results = json!([{"type":"dangerous_tool_use","status":{"type":"available","tool_uses":{"toolu_01":{"type":"evaluated","outcome":"not_flagged"}}}}]);
        let events = [
            json!({"type":"message_start","message":{"id":"msg_01","type":"message","role":"assistant","model":"test-model","content":[],"stop_reason":null,"usage":{"input_tokens":3,"output_tokens":0},"safeguard_results":results}}),
            json!({"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null,"safeguard_results":results},"usage":{"output_tokens":1},"amazon-bedrock-invocationMetrics":{"inputTokenCount":3,"outputTokenCount":1}}),
            json!({"type":"message_stop"}),
        ];
        let wire: Vec<_> = events.iter().flat_map(aws_frame).collect();
        let bytes: ByteStream = futures_util::stream::iter(
            wire.chunks(5)
                .map(|part| Ok(Bytes::copy_from_slice(part)))
                .collect::<Vec<_>>(),
        )
        .boxed();
        let decoded = bedrock_anthropic_messages_event_stream(bytes)
            .try_collect::<Vec<_>>()
            .await
            .unwrap();
        let start = serde_json::to_value(&decoded[0]).unwrap();
        let delta = serde_json::to_value(&decoded[1]).unwrap();
        assert_eq!(start["message"]["safeguard_results"], results);
        assert_eq!(delta["delta"]["safeguard_results"], results);
        assert_eq!(delta["usage"], json!({"input_tokens":3,"output_tokens":1}));
    }
    #[rstest]
    #[case::wrapped(None, json!({"defer_loading":true}), true)]
    #[case::explicit(Some(false), json!({"defer_loading":true}), false)]
    fn test_bedrock_adapter_normalizes_custom_tools_before_wire_projection(
        native_request: MessagesRequest,
        #[case] explicit: Option<bool>,
        #[case] custom: Value,
        #[case] expected_defer: bool,
    ) {
        let schema = JsonSchema::Object(Box::new(JsonSchemaObject {
            schema_type: Some(Recognized::Known(JsonSchemaType::Name("custom".into()))),
            ..Default::default()
        }));
        let request = MessagesRequest {
            params: MessagesOptionalParams {
                tools: Some(vec![
                    Recognized::Known(MessagesTool::Builtin(BuiltinMessagesTool::Custom(
                        ToolDefinition {
                            name: None,
                            input_schema: Some(Recognized::Known(schema)),
                            defer_loading: explicit.map(Recognized::Known),
                            extra: [("custom".into(), custom)].into_iter().collect(),
                            ..Default::default()
                        },
                    ))),
                    Recognized::Known(MessagesTool::Builtin(BuiltinMessagesTool::ToolSearchRegex(
                        ToolDefinition::default(),
                    ))),
                ]),
                ..native_request.params
            },
            ..native_request
        };
        let request = BEDROCK_ANTHROPIC_MESSAGES_CONFIG
            .transform_anthropic_messages_request(
                request,
                &native_context(
                    crate::base_llm::messages::context::MessagesModelCapabilities {
                        supports_tool_search: true,
                        ..Default::default()
                    },
                ),
            )
            .unwrap();
        let wire = BEDROCK_ANTHROPIC_MESSAGES_CONFIG
            .prepare_wire_request(&request, vec![])
            .unwrap();
        assert_eq!(wire.body["tools"][0]["name"], "litellm_unnamed_tool_0");
        assert_eq!(wire.body["tools"][0]["type"], "custom");
        assert_eq!(
            wire.body["tools"][0]["input_schema"],
            json!({"type":"object"})
        );
        assert_eq!(wire.body["tools"][0]["defer_loading"], expected_defer);
        assert!(wire.body["tools"][0].get("custom").is_none());
        assert_eq!(
            wire.body["anthropic_beta"],
            json!(["tool-search-tool-2025-10-19"])
        );
    }

    #[rstest]
    fn test_bedrock_legacy_thinking_preserves_updates_in_adapter(
        native_request: MessagesRequest,
        #[values(false, true)] clear_thinking: bool,
    ) {
        use litellm_llms_types::formats::messages::{EnabledThinking, ThinkingDisplay};
        let request = MessagesRequest {
            params: MessagesOptionalParams {
                thinking: Some(Recognized::Known(ThinkingConfig::Enabled(
                    EnabledThinking {
                        budget_tokens: Some(Recognized::Known(2048)),
                        display: Some(Recognized::Known(ThinkingDisplay::Updates)),
                        ..Default::default()
                    },
                ))),
                context_management: clear_thinking.then(|| {
                    Recognized::Known(ContextManagement {
                        edits: Some(vec![Recognized::Known(ContextEdit::ClearThinking {
                            extra: Map::new(),
                        })]),
                        extra: Map::new(),
                    })
                }),
                ..native_request.params
            },
            ..native_request
        };
        let context = native_context(
            crate::base_llm::messages::context::MessagesModelCapabilities {
                supports_adaptive_thinking: true,
                supports_output_config: true,
                ..Default::default()
            },
        );
        let result = BEDROCK_ANTHROPIC_MESSAGES_CONFIG
            .transform_anthropic_messages_request(request, &context)
            .unwrap();
        assert_eq!(
            result.params.thinking,
            Some(Recognized::Known(ThinkingConfig::adaptive(Some(
                ThinkingDisplay::Updates
            ))))
        );
        let wire = BEDROCK_ANTHROPIC_MESSAGES_CONFIG
            .prepare_wire_request(&result, vec![])
            .unwrap();
        let betas = wire.body["anthropic_beta"].as_array().unwrap();
        assert_eq!(
            betas
                .iter()
                .filter(|beta| *beta == "thinking-display-updates-2026-08-18")
                .count(),
            1
        );
    }

    #[rstest]
    fn test_bedrock_stream_yields_start_before_upstream_is_polled_again() {
        use futures_util::FutureExt;
        let start = json!({"type":"message_start", "message":{"id":"msg", "type":"message", "role":"assistant", "model":"model", "content":[], "stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":1,"output_tokens":0}}});
        let upstream = futures_util::stream::iter([Ok(Bytes::from(aws_frame(&start)))])
            .chain(futures_util::stream::pending());
        let mut events = bedrock_anthropic_messages_event_stream(Box::pin(upstream));
        let Some(Some(Ok(actual))) = events.next().now_or_never() else {
            panic!("complete message_start must be available before the pending next frame")
        };
        assert_eq!(actual, event(start));
    }

    #[rstest]
    fn test_bedrock_output_config_effort_follows_injected_capability(
        native_request: MessagesRequest,
        #[values(false, true)] supports_output_config: bool,
        #[values(false, true)] drop_params: bool,
    ) {
        let config = OutputConfig {
            effort: Some(Recognized::Known(EffortLevel::High)),
            ..Default::default()
        };
        let input = MessagesRequest {
            params: MessagesOptionalParams {
                output_config: Some(Recognized::Known(config.clone())),
                ..native_request.params
            },
            ..native_request
        };
        let context = MessagesTransformContext::with_lookup(
            crate::base_llm::messages::context::MessagesModelCapabilities {
                supports_output_config,
                ..Default::default()
            },
            drop_params,
            &|_: &str| None,
        );
        let output = BEDROCK_ANTHROPIC_MESSAGES_CONFIG
            .transform_anthropic_messages_request(input.clone(), &context)
            .unwrap();
        assert_eq!(
            output.params.output_config,
            supports_output_config.then_some(Recognized::Known(config))
        );
        assert_eq!(output.params.max_tokens, input.params.max_tokens);
        assert_eq!(output.messages, input.messages);
        let wire = BEDROCK_ANTHROPIC_MESSAGES_CONFIG
            .prepare_wire_request(&output, vec![])
            .unwrap();
        assert_eq!(
            wire.body.get("output_config"),
            supports_output_config
                .then(|| json!({"effort":"high"}))
                .as_ref()
        );
    }
    #[rstest]
    fn test_bedrock_allows_converted_websearch_function_tool(native_request: MessagesRequest) {
        let tools = vec![Recognized::Known(MessagesTool::Custom(CustomTool {
            definition: ToolDefinition {
                name: Some(Recognized::Known("litellm_web_search".into())),
                description: Some(Recognized::Known("Search the web".into())),
                input_schema: Some(Recognized::Known(JsonSchema::Object(Box::new(
                    JsonSchemaObject {
                        schema_type: Some(Recognized::Known(JsonSchemaType::Name("object".into()))),
                        ..Default::default()
                    },
                )))),
                ..Default::default()
            },
        }))];
        let request = MessagesRequest {
            params: MessagesOptionalParams {
                tools: Some(tools.clone()),
                ..native_request.params
            },
            ..native_request
        };
        let request = BEDROCK_ANTHROPIC_MESSAGES_CONFIG
            .transform_anthropic_messages_request(request, &MessagesTransformContext::default())
            .unwrap();
        assert_eq!(request.params.tools, Some(tools.clone()));
        let wire = BEDROCK_ANTHROPIC_MESSAGES_CONFIG
            .prepare_wire_request(&request, vec![])
            .unwrap();
        assert_eq!(wire.body["tools"], serde_json::to_value(tools).unwrap());
    }
}
