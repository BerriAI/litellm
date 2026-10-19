//! Python's `VertexGeminiConfig` for `/chat/completions`, shared with Google AI Studio's
//! `GoogleAIStudioGeminiConfig`: `get_supported_openai_params` and `map_openai_params` on the
//! way in, `transform_response` and `_calculate_usage` on the way out.

use std::collections::BTreeMap;

use litellm_auth::VertexParams;
use litellm_auth_gcp::{
    VertexConfig, get_vertex_ai_location, get_vertex_ai_project,
    get_vertex_ai_project_from_credentials,
};
use litellm_core_utils::core_helpers::{finish_reason_for, unix_now};
use litellm_http::request::has_header;
use litellm_llms_types::{
    formats::chat_completions::{
        ChatCompletionThinkingBlock, ChatCompletionToolCallChunk,
        ChatCompletionToolCallFunctionChunk, ChatCompletionsChoice, ChatCompletionsChoiceMessage,
        ChatCompletionsResponse, ChatCompletionsUsage, ChatMessage, CompletionTokensDetails,
        PromptTokensDetails,
    },
    providers::vertex_ai::{DEFAULT_LOCATION, GENERATE_CONTENT, STREAM_GENERATE_CONTENT},
};
use serde_json::{Map, Value, json};

use crate::{
    Error,
    base_llm::{
        auth::{AuthScheme, Headers, ValidatedEnvironment},
        chat::{
            params::{
                ChatRequestContext, capability, model_field, ordered, sort_params, truthy,
                unsupported_value,
            },
            streaming::{ChatStream, StreamShape},
            transformation::{
                BaseConfig, ProviderChatRequestData, ProviderChatResponseData, Unsupported,
            },
        },
    },
    vertex_ai::{
        common_utils::{get_vertex_base_url, graft_custom_vertex_base, with_query},
        gemini::{
            handler::gemini_chat_stream,
            schema::{
                build_vertex_schema, remove_additional_properties, remove_strict_from_schema,
            },
            transformation::{
                THOUGHT_SIGNATURE_SEPARATOR, is_cached_message, remote_media,
                transform_request_body, transform_system_message,
            },
        },
    },
};

/// Vertex AI call configuration a host passes down: consumed for auth and the URL, never
/// serialized into the body.
pub const CONFIG_PARAMS: &[&str] = &[
    "vertex_credentials",
    "vertex_project",
    "vertex_location",
    "vertex_ai_credentials",
    "vertex_ai_project",
    "vertex_ai_location",
];

/// The env var a Vertex AI base URL falls back to.
pub const VERTEXAI_API_BASE_ENV: &str = "VERTEXAI_API_BASE";

/// `SUPPORTED_REASONING_EFFORTS`.
const REASONING_EFFORTS: &[&str] = &["minimal", "low", "medium", "high", "none", "disable"];

/// `OPENAI_TO_GEMINI_TTS_VOICES`.
const TTS_VOICES: &[(&str, &str)] = &[
    ("alloy", "Kore"),
    ("ash", "Iapetus"),
    ("ballad", "Algieba"),
    ("cedar", "Achird"),
    ("coral", "Sulafat"),
    ("echo", "Charon"),
    ("fable", "Umbriel"),
    ("marin", "Despina"),
    ("onyx", "Orus"),
    ("sage", "Vindemiatrix"),
    ("shimmer", "Achernar"),
    ("verse", "Puck"),
];

/// `get_flagged_finish_reasons`: a first candidate stopped for one of these answers nothing.
const FLAGGED_FINISH_REASONS: &[&str] = &[
    "SAFETY",
    "RECITATION",
    "BLOCKLIST",
    "PROHIBITED_CONTENT",
    "SPII",
    "IMAGE_SAFETY",
    "IMAGE_PROHIBITED_CONTENT",
];

/// `_search_tool_keys`.
const SEARCH_TOOL_KEYS: &[&str] = &[
    "googleSearch",
    "googleSearchRetrieval",
    "enterpriseWebSearch",
    "url_context",
    "google_search",
    "google_search_retrieval",
    "enterprise_web_search",
    "urlContext",
];

/// The models whose penalties Gemini rejects although they predate Gemini 3.
const NO_PENALTY_MODELS: &[&str] = &["gemini-2.5-pro-preview-06-05"];

/// `VertexAIPartnerModels.is_vertex_partner_model` prefixes.
const PARTNER_MODEL_PREFIXES: &[&str] = &[
    "meta/",
    "deepseek-ai",
    "mistral",
    "codestral",
    "jamba",
    "claude",
    "qwen",
    "openai/gpt-oss-",
    "minimaxai/",
    "moonshotai/",
    "zai-org/",
    "google/gemma-",
];

/// Which Google API serves the Gemini model.
#[derive(Clone, Copy, Debug, PartialEq, Eq, strum::IntoStaticStr)]
pub enum GeminiSurface {
    /// Google AI Studio, `gemini/<model>`, authenticated with an API key.
    #[strum(serialize = "gemini")]
    AiStudio,
    /// Vertex AI, `vertex_ai/<model>`, authenticated with a Google access token.
    #[strum(serialize = "vertex_ai")]
    Vertex,
}

impl GeminiSurface {
    pub fn provider(self) -> &'static str {
        self.into()
    }

    /// The cost map keys Python's `supports_*` helpers try for the model.
    pub fn candidates(self, model: &str) -> Vec<String> {
        match self {
            Self::AiStudio => vec![format!("gemini/{model}"), model.to_owned()],
            Self::Vertex => vec![model.to_owned(), format!("vertex_ai/{model}")],
        }
    }
}

/// `_is_gemini_3_or_newer`: anything named after Gemini that is not a 1.x, 2.x, `exp` or
/// unversioned `pro`/`flash` model, a fine-tuned endpoint or a Gemma model.
pub fn is_gemini_3_or_newer(model: &str) -> bool {
    let name = model.rsplit('/').next().unwrap_or(model).to_lowercase();
    let fine_tuned = name.chars().all(|c| c.is_ascii_digit())
        || (model.starts_with("gemini/") && !name.starts_with("gemini-"));
    if name.is_empty() || fine_tuned || name.starts_with("gemma-") {
        return false;
    }
    !is_pre_gemini_3(&name)
}

/// `^gemini-(?:[12](?:\.\d+)?|exp|(?:pro|flash)(?!-(?:lite-)?latest$))(?:-|$)`.
fn is_pre_gemini_3(name: &str) -> bool {
    let Some(rest) = name.strip_prefix("gemini-") else {
        return false;
    };
    let boundary = |tail: &str| tail.is_empty() || tail.starts_with('-');
    if let Some(after) = rest.strip_prefix(['1', '2']) {
        if boundary(after) {
            return true;
        }
        if let Some(fraction) = after.strip_prefix('.') {
            let digits = fraction.chars().take_while(char::is_ascii_digit).count();
            if digits > 0 && boundary(&fraction[digits..]) {
                return true;
            }
        }
    }
    if rest.strip_prefix("exp").is_some_and(boundary) {
        return true;
    }
    ["pro", "flash"].iter().any(|word| {
        rest.strip_prefix(word)
            .is_some_and(|after| boundary(after) && after != "-latest" && after != "-lite-latest")
    })
}

/// `supports_response_json_schema`: Gemini 2.0 and newer take a JSON Schema as is.
pub fn supports_response_json_schema(model: &str) -> bool {
    let lowered = model.to_lowercase();
    lowered.match_indices("gemini-").any(|(at, _)| {
        let tail = &lowered[at + "gemini-".len()..];
        let digits = tail.chars().take_while(char::is_ascii_digit).count();
        let version = &tail[..digits];
        let separated = tail[digits..].starts_with(['.', '-']);
        separated
            && match version.len() {
                0 => false,
                1 => version != "0" && version != "1",
                _ => !version.starts_with('0'),
            }
    })
}

/// `get_vertex_ai_model_route(model) == GEMINI`: the models Vertex AI serves through
/// `generateContent`, as opposed to its partner, Gemma and Model Garden routes.
pub fn is_vertex_gemini_model(model: &str) -> bool {
    !model.contains("agent_engine/")
        && !PARTNER_MODEL_PREFIXES
            .iter()
            .any(|prefix| model.starts_with(prefix))
        && !model.to_lowercase().contains("bge")
        && !model.contains("gemma/")
        && !model.contains("openai")
        && !model.starts_with("xai/")
        && model.contains("gemini")
}

/// `_supports_penalty_parameters`.
fn supports_penalties(model: &str) -> bool {
    !is_gemini_3_or_newer(model) && !NO_PENALTY_MODELS.contains(&model)
}

/// `get_supported_openai_params` of either config. The reasoning params depend on the cost
/// map, which is only read when the request carries one.
pub fn supported_params(
    surface: GeminiSurface,
    model: &str,
    context: &ChatRequestContext<'_>,
    openai: &Map<String, Value>,
) -> Result<Vec<&'static str>, Error> {
    let mut params: Vec<&'static str> = match surface {
        GeminiSurface::Vertex => vec![
            "temperature",
            "top_p",
            "top_k",
            "max_tokens",
            "max_completion_tokens",
            "stream",
            "tools",
            "functions",
            "tool_choice",
            "response_format",
            "n",
            "stop",
            "extra_headers",
            "seed",
            "logprobs",
            "top_logprobs",
            "modalities",
            "audio",
            "parallel_tool_calls",
            "web_search_options",
            "include_server_side_tool_invocations",
            "service_tier",
        ],
        GeminiSurface::AiStudio => vec![
            "temperature",
            "top_p",
            "max_tokens",
            "max_completion_tokens",
            "stream",
            "tools",
            "tool_choice",
            "functions",
            "response_format",
            "n",
            "stop",
            "logprobs",
            "frequency_penalty",
            "presence_penalty",
            "seed",
            "modalities",
            "parallel_tool_calls",
            "web_search_options",
            "include_server_side_tool_invocations",
            "service_tier",
        ],
    };
    if surface == GeminiSurface::Vertex && supports_penalties(model) {
        params.extend(["frequency_penalty", "presence_penalty"]);
    }
    let asks_reasoning = openai.contains_key("reasoning_effort") || openai.contains_key("thinking");
    if asks_reasoning
        && (is_gemini_3_or_newer(model)
            || capability(context, &surface.candidates(model), "supports_reasoning")? == Some(true))
    {
        params.extend(["reasoning_effort", "thinking"]);
    }
    if surface == GeminiSurface::AiStudio && model.contains("tts") {
        params.push("audio");
    }
    Ok(params)
}

fn add_tools(params: &mut Map<String, Value>, tools: Vec<Value>) {
    match params.get_mut("tools") {
        Some(Value::Array(existing)) => existing.extend(tools),
        _ => {
            params.insert("tools".into(), Value::Array(tools));
        }
    }
}

/// `get_tool_value`: the tool's config under its camelCase name or its snake_case one.
fn tool_value(tool: &Map<String, Value>, name: &str) -> Option<Value> {
    let snake: String = name
        .chars()
        .flat_map(|c| match c.is_uppercase() {
            true => vec!['_', c.to_ascii_lowercase()],
            false => vec![c],
        })
        .collect::<String>()
        .trim_start_matches('_')
        .to_owned();
    tool.get(name)
        .filter(|value| !value.is_null())
        .or_else(|| tool.get(&snake).filter(|value| !value.is_null()))
        .cloned()
}

/// `_transform_computer_use_config`.
fn computer_use_config(config: &Value) -> Value {
    let mut mapped = Map::new();
    match config.get("environment").and_then(Value::as_str) {
        Some("browser") => {
            mapped.insert("environment".into(), json!("ENVIRONMENT_BROWSER"));
        }
        Some("unspecified") => {
            mapped.insert("environment".into(), json!("ENVIRONMENT_UNSPECIFIED"));
        }
        Some(value @ ("ENVIRONMENT_BROWSER" | "ENVIRONMENT_UNSPECIFIED")) => {
            mapped.insert("environment".into(), json!(value));
        }
        _ => {}
    }
    if let Some(excluded) = config
        .get("excluded_predefined_functions")
        .or_else(|| config.get("excludedPredefinedFunctions"))
    {
        mapped.insert("excludedPredefinedFunctions".into(), excluded.clone());
    }
    Value::Object(mapped)
}

/// `_map_function`: function declarations grouped in one tool, each built-in tool in its own.
fn map_function(
    tools: &Value,
    include_server_side_tool_invocations: bool,
) -> Result<Vec<Value>, Error> {
    let mut tools = tools.clone();
    remove_additional_properties(&mut tools);
    remove_strict_from_schema(&mut tools);
    let mut declarations: Vec<Value> = Vec::new();
    let mut google_search = None;
    let mut google_search_retrieval = None;
    let mut enterprise_web_search = None;
    let mut url_context = None;
    let mut code_execution = None;
    let mut google_maps = None;
    let mut computer_use = None;
    for tool in tools.as_array().into_iter().flatten() {
        let Some(original) = tool.as_object() else {
            continue;
        };
        let function = match original.get("function") {
            Some(Value::Object(function)) => {
                let mut function = function.clone();
                if let Some(parameters @ Value::Object(_)) = function.remove("parameters") {
                    function.insert("parameters".into(), build_vertex_schema(parameters, false));
                } else if let Some(parameters) = original
                    .get("function")
                    .and_then(|function| function.get("parameters"))
                {
                    function.insert("parameters".into(), parameters.clone());
                }
                Some(function)
            }
            _ if original.contains_key("name") => Some(original.clone()),
            _ => None,
        };
        let tool: Map<String, Value> = match original.get("type").and_then(Value::as_str) {
            Some("computer_use") => {
                let config: Map<String, Value> = original
                    .iter()
                    .filter(|(key, _)| key.as_str() != "type")
                    .map(|(key, value)| (key.clone(), value.clone()))
                    .collect();
                Map::from_iter([("computerUse".to_owned(), Value::Object(config))])
            }
            Some("web_search" | "web_search_preview") => {
                Map::from_iter([("googleSearch".to_owned(), json!({}))])
            }
            _ => original
                .iter()
                .filter(|(key, _)| key.as_str() != "type")
                .map(|(key, value)| (key.clone(), value.clone()))
                .collect(),
        };
        let name = (tool.len() == 1)
            .then(|| tool.keys().next().map(String::as_str))
            .flatten();
        match name {
            Some("codeExecution" | "code_execution") => {
                code_execution = tool_value(&tool, "codeExecution");
            }
            Some(name @ ("googleSearch" | "google_search")) => {
                google_search = tool_value(&tool, name);
            }
            Some(name @ ("googleSearchRetrieval" | "google_search_retrieval")) => {
                google_search_retrieval = tool_value(&tool, name);
            }
            Some(name @ ("enterpriseWebSearch" | "enterprise_web_search")) => {
                enterprise_web_search = tool_value(&tool, name);
            }
            Some(name @ ("url_context" | "urlContext")) => {
                url_context = tool_value(&tool, name);
            }
            Some("googleMaps" | "google_maps") => {
                if let Some(Value::Object(config)) = tool_value(&tool, "googleMaps") {
                    google_maps = Some(Value::Object(
                        config
                            .into_iter()
                            .filter(|(key, _)| {
                                !matches!(key.as_str(), "latitude" | "longitude" | "languageCode")
                            })
                            .collect(),
                    ));
                }
            }
            Some("computerUse" | "computer_use") => {
                computer_use = Some(match tool_value(&tool, "computerUse") {
                    Some(config) => computer_use_config(&config),
                    None => json!({}),
                });
            }
            _ => {
                if let Some(function) = function {
                    let name = function.get("name").cloned().ok_or_else(|| {
                        unsupported_value("Invalid tool: a function declaration needs a name")
                    })?;
                    let mut declaration = json!({"name": name});
                    if let Some(description) = function.get("description").filter(|d| !d.is_null())
                    {
                        declaration["description"] = description.clone();
                    }
                    match function.get("parameters") {
                        Some(Value::String(text)) if text.is_empty() => {
                            declaration["parameters"] = json!({"type": "object"});
                        }
                        Some(Value::Null) | None => {}
                        Some(parameters) => declaration["parameters"] = parameters.clone(),
                    }
                    declarations.push(declaration);
                }
            }
        }
    }
    let has_search = [
        &google_search,
        &google_search_retrieval,
        &enterprise_web_search,
        &url_context,
    ]
    .iter()
    .any(|tool| tool.is_some());
    if !declarations.is_empty() && has_search && !include_server_side_tool_invocations {
        google_search = None;
        google_search_retrieval = None;
        enterprise_web_search = None;
        url_context = None;
    }
    let mut mapped = Vec::new();
    if !declarations.is_empty() {
        mapped.push(json!({"function_declarations": declarations}));
    }
    for (key, value) in [
        ("googleSearch", google_search),
        ("googleSearchRetrieval", google_search_retrieval),
        ("enterpriseWebSearch", enterprise_web_search),
        ("code_execution", code_execution),
        ("url_context", url_context),
        ("googleMaps", google_maps),
        ("computerUse", computer_use),
    ] {
        if let Some(value) = value {
            mapped.push(json!({key: value}));
        }
    }
    Ok(mapped)
}

/// `map_tool_choice_values`.
fn map_tool_choice(value: &Value) -> Result<Value, Error> {
    let mode = |mode: &str| json!({"functionCallingConfig": {"mode": mode}});
    match value {
        Value::String(choice) if choice == "none" => Ok(mode("NONE")),
        Value::String(choice) if choice == "required" => Ok(mode("ANY")),
        Value::String(choice) if choice == "auto" => Ok(mode("AUTO")),
        Value::Object(choice) => {
            let name = choice
                .get("function")
                .and_then(|function| function.get("name"))
                .cloned()
                .unwrap_or_else(|| json!(""));
            Ok(json!({"functionCallingConfig": {"mode": "ANY", "allowed_function_names": [name]}}))
        }
        other => Err(unsupported_value(format!(
            "VertexAI doesn't support tool_choice={}. Supported tool_choice values=['auto', 'required', json object]. To drop it from the call, set `litellm.drop_params = True.",
            other
                .as_str()
                .map_or_else(|| other.to_string(), str::to_owned)
        ))),
    }
}

/// `apply_response_schema_transformation`.
fn map_response_format(value: &Value, model: &str, params: &mut Map<String, Value>) {
    let mut value = value.clone();
    remove_strict_from_schema(&mut value);
    let json_schema = supports_response_json_schema(model);
    if !json_schema {
        remove_additional_properties(&mut value);
    }
    match value.get("type").and_then(Value::as_str) {
        Some("json_object") => {
            params.insert("response_mime_type".into(), json!("application/json"));
        }
        Some("text") => {
            params.insert("response_mime_type".into(), json!("text/plain"));
        }
        _ => {}
    }
    let schema = if let Some(schema) = value.get("response_schema") {
        params.insert("response_mime_type".into(), json!("application/json"));
        Some(schema.clone())
    } else if value.get("type").and_then(Value::as_str) == Some("json_schema")
        && let Some(schema) = value.get("json_schema").and_then(|s| s.get("schema"))
    {
        params.insert("response_mime_type".into(), json!("application/json"));
        Some(schema.clone())
    } else {
        None
    };
    if let Some(schema) = schema.filter(|schema| schema.as_object().is_some_and(|s| !s.is_empty()))
    {
        match json_schema {
            true => params.insert("response_json_schema".into(), schema),
            false => params.insert("response_schema".into(), build_vertex_schema(schema, true)),
        };
    }
}

/// `_map_audio_params`.
fn map_audio(value: &Value) -> Result<Value, Error> {
    if let Some(format) = value.get("format").filter(|format| !format.is_null())
        && format.as_str() != Some("pcm16")
    {
        return Err(unsupported_value(format!(
            "Unsupported audio format for Gemini TTS models: {}. Gemini TTS models only support 'pcm16' format as they return audio data in L16 PCM format. Please set audio format to 'pcm16'.",
            format
                .as_str()
                .map_or_else(|| format.to_string(), str::to_owned)
        )));
    }
    let mut speech = Map::new();
    if let Some(voice) = value.get("voice") {
        let name = match voice.as_str() {
            Some(voice) => {
                let lowered = voice.to_lowercase();
                json!(
                    TTS_VOICES
                        .iter()
                        .find(|(openai, _)| *openai == lowered)
                        .map_or(voice, |(_, gemini)| *gemini)
                )
            }
            None => voice.clone(),
        };
        speech.insert(
            "voiceConfig".into(),
            json!({"prebuiltVoiceConfig": {"voiceName": name}}),
        );
    }
    if let Some(language) = value.get("language_code") {
        speech.insert("languageCode".into(), language.clone());
    }
    Ok(Value::Object(speech))
}

/// `map_response_modalities`.
fn map_modalities(value: &[Value]) -> Value {
    Value::Array(
        value
            .iter()
            .map(|modality| {
                json!(match modality.as_str() {
                    Some("text") => "TEXT",
                    Some("image") => "IMAGE",
                    Some("audio") => "AUDIO",
                    Some("video") => "VIDEO",
                    _ => "MODALITY_UNSPECIFIED",
                })
            })
            .collect(),
    )
}

struct ReasoningModel<'a> {
    model: &'a str,
    surface: GeminiSurface,
    context: &'a ChatRequestContext<'a>,
}

impl ReasoningModel<'_> {
    /// `_supports_minimal_thinking_level`.
    fn supports_minimal(&self) -> bool {
        is_gemini_3_or_newer(self.model)
            && self.model.to_lowercase().contains("flash")
            && model_field(
                self.context,
                &self.surface.candidates(self.model),
                "supports_minimal_reasoning_effort",
            )
            .and_then(|value| value.as_bool())
                != Some(false)
    }

    /// `_map_reasoning_effort`: a thinking level on Gemini 3 and newer, a token budget before.
    fn reasoning_effort(&self, value: &Value) -> Result<Option<Value>, Error> {
        let effort = match value {
            Value::String(effort) => effort.as_str(),
            Value::Object(object) => match object.get("effort").and_then(Value::as_str) {
                Some(effort) => effort,
                None => return Ok(None),
            },
            _ => return Ok(None),
        };
        if !REASONING_EFFORTS.contains(&effort) {
            return Err(unsupported_value(format!(
                "Invalid `reasoning_effort`: '{effort}'. Must be one of: {}. To drop this param, set `litellm.drop_params = True` or pass in `(.., drop_params=True)` in the request - https://docs.litellm.ai/docs/completion/drop_params",
                REASONING_EFFORTS
                    .iter()
                    .map(|effort| format!("'{effort}'"))
                    .collect::<Vec<_>>()
                    .join(", ")
            )));
        }
        let thoughts = true;
        if is_gemini_3_or_newer(self.model) {
            let floor = match self.supports_minimal() {
                true => "minimal",
                false => "low",
            };
            return Ok(Some(match effort {
                "minimal" => json!({"thinkingLevel": floor, "includeThoughts": thoughts}),
                "disable" | "none" => json!({"thinkingLevel": floor, "includeThoughts": false}),
                level => json!({"thinkingLevel": level, "includeThoughts": thoughts}),
            }));
        }
        let lowered = self.model.to_lowercase();
        Ok(Some(match effort {
            "minimal" => {
                let budget = if lowered.contains("gemini-2.5-flash-lite") {
                    512
                } else if lowered.contains("gemini-2.5-pro") {
                    128
                } else if lowered.contains("gemini-2.5-flash") {
                    1
                } else {
                    128
                };
                json!({"thinkingBudget": budget, "includeThoughts": thoughts})
            }
            "low" => json!({"thinkingBudget": 1024, "includeThoughts": thoughts}),
            "medium" => json!({"thinkingBudget": 2048, "includeThoughts": thoughts}),
            "high" => json!({"thinkingBudget": 4096, "includeThoughts": thoughts}),
            _ => json!({"thinkingBudget": 0, "includeThoughts": false}),
        }))
    }

    /// `_map_thinking_param`.
    fn thinking(&self, value: &Value) -> Value {
        let enabled = value.get("type").and_then(Value::as_str) == Some("enabled");
        let budget = value
            .get("budget_tokens")
            .filter(|budget| !budget.is_null());
        let zero = budget.and_then(Value::as_f64) == Some(0.0);
        let mut config = Map::new();
        if is_gemini_3_or_newer(self.model) {
            config.insert("includeThoughts".into(), json!(enabled && !zero));
        } else {
            if enabled && !zero {
                config.insert("includeThoughts".into(), json!(true));
            }
            if let Some(budget) = budget.filter(|budget| budget.is_i64() || budget.is_u64()) {
                config.insert("thinkingBudget".into(), budget.clone());
            }
        }
        Value::Object(config)
    }
}

/// `_validate_thinking_config_conflicts`.
fn check_thinking_conflict(params: &Map<String, Value>, param: &str) -> Result<(), Error> {
    if params
        .get("thinkingConfig")
        .is_some_and(|config| config.get("thinkingLevel").is_some())
    {
        return Err(unsupported_value(format!(
            "Cannot specify both `{param}` (which maps to `thinking_budget`) and `thinking_level` in the same request. For Gemini 3 models, use `thinking_level` instead."
        )));
    }
    Ok(())
}

/// `map_openai_params`, over the OpenAI params in Python's mapping order.
pub fn map_openai_params(
    openai: &Map<String, Value>,
    surface: GeminiSurface,
    model: &str,
    context: &ChatRequestContext<'_>,
) -> Result<Map<String, Value>, Error> {
    let reasoning = ReasoningModel {
        model,
        surface,
        context,
    };
    let mut params = Map::new();
    let server_side = openai.get("include_server_side_tool_invocations") == Some(&json!(true));
    if server_side {
        params.insert("include_server_side_tool_invocations".into(), json!(true));
    }
    for (name, value) in ordered(openai) {
        match (name, value) {
            ("temperature" | "top_p" | "seed", value) => {
                params.insert(name.into(), value.clone());
            }
            ("n", value) => {
                params.insert("candidate_count".into(), value.clone());
            }
            ("audio", value @ Value::Object(_)) => {
                params.insert("speechConfig".into(), map_audio(value)?);
            }
            ("stop", Value::String(stop)) => {
                params.insert("stop_sequences".into(), json!([stop]));
            }
            ("stop", stop @ Value::Array(_)) => {
                params.insert("stop_sequences".into(), stop.clone());
            }
            ("max_tokens" | "max_completion_tokens", value) => {
                params.insert("max_output_tokens".into(), value.clone());
            }
            ("response_format", value @ Value::Object(_)) => {
                map_response_format(value, model, &mut params);
            }
            ("frequency_penalty" | "presence_penalty", value) => {
                if supports_penalties(model) {
                    params.insert(name.into(), value.clone());
                }
            }
            ("logprobs", value) => {
                params.insert("responseLogprobs".into(), value.clone());
            }
            ("top_logprobs", value) => {
                params.insert("logprobs".into(), value.clone());
            }
            ("tools" | "functions", tools @ Value::Array(list)) if !list.is_empty() => {
                let mapped = map_function(tools, server_side)?;
                add_tools(&mut params, mapped);
            }
            ("tool_choice", choice @ (Value::String(_) | Value::Object(_))) => {
                params.insert("tool_choice".into(), map_tool_choice(choice)?);
            }
            ("parallel_tool_calls", value) => {
                let declared = openai
                    .get("tools")
                    .or_else(|| openai.get("functions"))
                    .and_then(Value::as_array)
                    .map_or(0, Vec::len);
                if !(value == &json!(false) && declared > 1) {
                    params.insert(name.into(), value.clone());
                }
            }
            ("reasoning_effort", value) => {
                if let Some(config) = reasoning.reasoning_effort(value)? {
                    check_thinking_conflict(&params, "reasoning_effort")?;
                    params.insert("thinkingConfig".into(), config);
                }
            }
            ("thinking", value) => {
                check_thinking_conflict(&params, "thinking")?;
                params.insert("thinkingConfig".into(), reasoning.thinking(value));
            }
            ("modalities", Value::Array(modalities)) => {
                params.insert("responseModalities".into(), map_modalities(modalities));
            }
            ("web_search_options", Value::Object(_)) => {
                add_tools(&mut params, vec![json!({"googleSearch": {}})]);
            }
            ("service_tier", Value::String(tier)) => {
                let tier = match tier.to_lowercase() {
                    tier if tier == "auto" => "priority".to_owned(),
                    tier => tier,
                };
                params.insert("service_tier".into(), json!(tier));
            }
            _ => {}
        }
    }
    if params.contains_key("speechConfig") {
        match params.get_mut("responseModalities") {
            Some(Value::Array(modalities)) => {
                if !modalities.contains(&json!("AUDIO")) {
                    modalities.push(json!("AUDIO"));
                }
            }
            _ => {
                params.insert("responseModalities".into(), json!(["AUDIO"]));
            }
        }
    }
    drop_search_tools_mixed_with_functions(&mut params);
    Ok(params)
}

/// `_drop_search_tools_mixed_with_functions`: Vertex AI rejects search tools beside function
/// declarations unless the server side tool invocations are circulated.
fn drop_search_tools_mixed_with_functions(params: &mut Map<String, Value>) {
    if params
        .get("include_server_side_tool_invocations")
        .is_some_and(truthy)
    {
        return;
    }
    let Some(Value::Array(tools)) = params.get_mut("tools") else {
        return;
    };
    let is_search = |tool: &Value| SEARCH_TOOL_KEYS.iter().any(|key| tool.get(*key).is_some());
    let has_functions = tools
        .iter()
        .any(|tool| tool.get("function_declarations").is_some_and(truthy));
    if has_functions && tools.iter().any(is_search) {
        tools.retain(|tool| !is_search(tool));
    }
}

/// Builds the `generateContent` body for either surface.
pub fn build_request(
    surface: GeminiSurface,
    model: &str,
    messages: Vec<ChatMessage>,
    optional_params: Map<String, Value>,
    context: &ChatRequestContext<'_>,
) -> Result<ProviderChatRequestData, Error> {
    let candidates = surface.candidates(model);
    let sorted = sort_params(
        optional_params,
        CONFIG_PARAMS,
        context,
        |openai| supported_params(surface, model, context, openai),
        surface.provider(),
        model,
    )?;
    if sorted.openai.contains_key("functions") {
        return Err(Error::Unsupported(
            "legacy functions, answered as function_call",
        ));
    }
    let mut params = map_openai_params(&sorted.openai, surface, model, context)?;
    for name in &sorted.allowed {
        if let Some(value) = sorted.openai.get(name)
            && !params.contains_key(name)
        {
            params.insert(name.clone(), value.clone());
        }
    }
    params.extend(sorted.provider);

    let messages: Vec<Value> = messages
        .iter()
        .map(|message| {
            serde_json::to_value(message).map(|mut value| {
                if value.get("role").and_then(Value::as_str) == Some("developer") {
                    value["role"] = json!("system");
                }
                value
            })
        })
        .collect::<Result<_, _>>()
        .map_err(|error| Error::InvalidRequest(crate::ErrorDetail::invalid("messages", error)))?;
    if messages.iter().any(is_cached_message) {
        return Err(Error::Unsupported("Gemini context caching"));
    }
    let supports_system = model.contains("gemini/")
        || capability(context, &candidates, "supports_system_messages")? == Some(true);
    let (system_instruction, messages) = transform_system_message(supports_system, messages);
    if params.contains_key("response_schema")
        && capability(context, &candidates, "supports_response_schema")? != Some(true)
    {
        return Err(Error::Unsupported(
            "response_schema for a model without native response schema support",
        ));
    }
    let body = transform_request_body(&messages, system_instruction, model, params, surface)?;
    Ok(ProviderChatRequestData {
        body,
        stream_shape: StreamShape::default(),
        headers: Vec::new(),
    })
}

/// `_served_model_name`.
pub fn served_model_name(body: &Value) -> Option<String> {
    body.get("modelVersion")
        .and_then(Value::as_str)
        .filter(|version| !version.is_empty())
        .map(|version| version.split('@').next().unwrap_or(version).to_owned())
}

/// `_extract_candidate_metadata`: the grounding, URL context, safety and citation metadata.
pub fn candidate_metadata(candidate: &Value) -> [Vec<Value>; 4] {
    let listed = |key: &str| match candidate.get(key) {
        Some(Value::Array(items)) => items.clone(),
        Some(Value::Null) | None => Vec::new(),
        Some(value) => vec![value.clone()],
    };
    let safety = candidate
        .get("safetyRatings")
        .map(|ratings| vec![ratings.clone()])
        .unwrap_or_default();
    let citations = candidate
        .get("citationMetadata")
        .map(|citation| vec![citation.clone()])
        .unwrap_or_default();
    let url_context = candidate
        .get("urlContextMetadata")
        .map(|context| vec![context.clone()])
        .unwrap_or_default();
    [listed("groundingMetadata"), url_context, safety, citations]
}

/// `_convert_grounding_metadata_to_annotations`.
fn grounding_annotations(grounding: &[Value]) -> Vec<Value> {
    let mut annotations = Vec::new();
    for metadata in grounding {
        let chunks = metadata
            .get("groundingChunks")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        for support in metadata
            .get("groundingSupports")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
        {
            let segment = support.get("segment");
            let start = segment
                .and_then(|s| s.get("startIndex"))
                .filter(|v| !v.is_null());
            let end = segment
                .and_then(|s| s.get("endIndex"))
                .filter(|v| !v.is_null());
            let first = support
                .get("groundingChunkIndices")
                .and_then(Value::as_array)
                .and_then(|indices| indices.first())
                .and_then(Value::as_u64);
            let (Some(start), Some(end), Some(first)) = (start, end, first) else {
                continue;
            };
            let Some(web) = chunks
                .get(first as usize)
                .and_then(|chunk| chunk.get("web"))
            else {
                continue;
            };
            let field = |key: &str| web.get(key).cloned().unwrap_or_else(|| json!(""));
            annotations.push(json!({"type": "url_citation", "url_citation": {
                "start_index": start, "end_index": end, "url": field("uri"), "title": field("title")
            }}));
        }
    }
    annotations
}

/// `_extract_server_side_tool_invocations`: each server side `toolCall` with its
/// `toolResponse`.
fn server_side_tool_invocations(parts: &[Value]) -> Option<Vec<Value>> {
    let mut calls: Vec<(String, Map<String, Value>)> = Vec::new();
    let mut responses: Vec<(String, Map<String, Value>)> = Vec::new();
    let upsert = |entries: &mut Vec<(String, Map<String, Value>)>, id: String, entry| match entries
        .iter_mut()
        .find(|(seen, _)| *seen == id)
    {
        Some((_, existing)) => *existing = entry,
        None => entries.push((id, entry)),
    };
    for part in parts {
        let field = |holder: &Value, key: &str| holder.get(key).cloned().unwrap_or(Value::Null);
        let id_of = |holder: &Value| {
            holder
                .get("id")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_owned()
        };
        if let Some(call) = part.get("toolCall") {
            let mut entry = Map::new();
            entry.insert("tool_type".into(), field(call, "toolType"));
            entry.insert("id".into(), field(call, "id"));
            entry.insert("args".into(), field(call, "args"));
            if let Some(signature) = part.get("thoughtSignature").filter(|s| !s.is_null()) {
                entry.insert("thought_signature".into(), signature.clone());
            }
            upsert(&mut calls, id_of(call), entry);
        } else if let Some(response) = part.get("toolResponse") {
            let mut entry = Map::new();
            entry.insert("id".into(), field(response, "id"));
            entry.insert("tool_type".into(), field(response, "toolType"));
            entry.insert("response".into(), field(response, "response"));
            if let Some(signature) = part.get("thoughtSignature").filter(|s| !s.is_null()) {
                entry.insert("thought_signature".into(), signature.clone());
            }
            upsert(&mut responses, id_of(response), entry);
        }
    }
    let mut invocations = Vec::new();
    for (id, mut call) in calls {
        if let Some(at) = responses.iter().position(|(seen, _)| *seen == id) {
            let (_, response) = responses.remove(at);
            call.insert(
                "response".into(),
                response.get("response").cloned().unwrap_or(Value::Null),
            );
            if let Some(signature) = response.get("thought_signature") {
                call.insert("response_thought_signature".into(), signature.clone());
            }
        }
        invocations.push(Value::Object(call));
    }
    for (_, mut response) in responses {
        if let Some(signature) = response.get("thought_signature").cloned() {
            response.insert("response_thought_signature".into(), signature);
        }
        invocations.push(Value::Object(response));
    }
    (!invocations.is_empty()).then_some(invocations)
}

/// `_transform_logprobs`.
fn transform_logprobs(result: &Value) -> Option<Value> {
    let chosen = result.get("chosenCandidates")?.as_array()?;
    let top = result.get("topCandidates").and_then(Value::as_array);
    let field = |holder: &Value, key: &str| holder.get(key).cloned().unwrap_or(Value::Null);
    let content: Vec<Value> = chosen
        .iter()
        .enumerate()
        .map(|(index, candidate)| {
            let top_logprobs: Vec<Value> = top
                .and_then(|top| top.get(index))
                .and_then(|top| top.get("candidates"))
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
                .map(|option| json!({"token": field(option, "token"), "logprob": field(option, "logProbability")}))
                .collect();
            json!({"token": field(candidate, "token"), "logprob": field(candidate, "logProbability"), "top_logprobs": top_logprobs})
        })
        .collect();
    Some(json!({"content": content}))
}

/// What one candidate says, as `_process_candidates` reads it.
#[derive(Debug, Default)]
pub struct CandidateMessage {
    pub content: Option<String>,
    pub reasoning_content: Option<String>,
    pub thinking_blocks: Option<Vec<ChatCompletionThinkingBlock>>,
    pub tool_calls: Vec<ChatCompletionToolCallChunk>,
    pub provider_specific_fields: Option<Map<String, Value>>,
    pub annotations: Option<Vec<Value>>,
    pub logprobs: Option<Value>,
    /// Gemini's own `finishReason`.
    pub finish_reason: Option<String>,
}

impl CandidateMessage {
    /// `_check_finish_reason`.
    pub fn openai_finish_reason(&self) -> String {
        match (&self.finish_reason, self.tool_calls.is_empty()) {
            (_, false) => "tool_calls".to_owned(),
            (Some(reason), true) => finish_reason_for(reason).to_owned(),
            (None, true) => "stop".to_owned(),
        }
    }
}

fn is_audio_data_url(text: &str) -> bool {
    text.starts_with("data:audio")
        && text
            .split_once("data:")
            .and_then(|(_, rest)| rest.split_once(";base64,"))
            .is_some_and(|(media_type, _)| media_type.starts_with("audio/"))
}

/// One candidate's message. `next_tool_index` counts tool calls across a stream's chunks.
pub fn candidate_message(candidate: &Value, next_tool_index: &mut i64) -> CandidateMessage {
    let mut message = CandidateMessage {
        finish_reason: candidate
            .get("finishReason")
            .and_then(Value::as_str)
            .map(str::to_owned),
        logprobs: candidate.get("logprobsResult").and_then(transform_logprobs),
        ..CandidateMessage::default()
    };
    let parts = candidate
        .get("content")
        .filter(|content| truthy(content))
        .and_then(|content| content.get("parts"))
        .and_then(Value::as_array);
    let Some(parts) = parts else {
        return message;
    };
    let mut content: Option<String> = None;
    let mut reasoning: Option<String> = None;
    let mut thinking_blocks = Vec::new();
    let mut signatures = Vec::new();
    for part in parts {
        let text = match (part.get("text"), part.get("inlineData")) {
            (Some(Value::String(text)), _) if is_audio_data_url(text) => String::new(),
            (Some(Value::String(text)), _) => text.clone(),
            (Some(_), _) => String::new(),
            (None, Some(inline)) => {
                let mime = inline
                    .get("mimeType")
                    .and_then(Value::as_str)
                    .unwrap_or_default();
                let data = inline
                    .get("data")
                    .and_then(Value::as_str)
                    .unwrap_or_default();
                match mime.starts_with("audio/") || mime.starts_with("image/") {
                    true => String::new(),
                    false => format!("data:{mime};base64,{data}"),
                }
            }
            (None, None) => String::new(),
        };
        let thought = part.get("thought") == Some(&json!(true));
        if !text.is_empty() {
            let target = match thought {
                true => &mut reasoning,
                false => &mut content,
            };
            target.get_or_insert_with(String::new).push_str(&text);
        }
        let signature = part.get("thoughtSignature").filter(|s| !s.is_null());
        if thought {
            thinking_blocks.push(ChatCompletionThinkingBlock::Thinking {
                thinking: Some(
                    part.get("text")
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                        .to_owned(),
                ),
                signature: signature.and_then(Value::as_str).map(str::to_owned),
                cache_control: None,
                index: None,
            });
        }
        if let Some(signature) = signature {
            signatures.push(signature.clone());
        }
        if let Some(call) = part.get("functionCall") {
            let arguments = litellm_python_compat::json::dumps_unicode(
                &litellm_python_compat::json::from_json(
                    call.get("args").cloned().unwrap_or(Value::Null),
                ),
            )
            .unwrap_or_default();
            let mut id = call
                .get("id")
                .and_then(Value::as_str)
                .filter(|id| !id.is_empty())
                .map_or_else(
                    || format!("call_{}", &uuid::Uuid::new_v4().simple().to_string()[..28]),
                    str::to_owned,
                );
            let signature = signature.and_then(Value::as_str);
            if let Some(signature) = signature {
                id = format!("{id}{THOUGHT_SIGNATURE_SEPARATOR}{signature}");
            }
            message.tool_calls.push(ChatCompletionToolCallChunk {
                id: Some(id),
                tool_type: "function".into(),
                function: ChatCompletionToolCallFunctionChunk {
                    name: call.get("name").and_then(Value::as_str).map(str::to_owned),
                    arguments,
                    provider_specific_fields: None,
                },
                index: *next_tool_index,
                provider_specific_fields: signature.map(|signature| {
                    Map::from_iter([("thought_signature".to_owned(), json!(signature))])
                }),
            });
            *next_tool_index += 1;
        }
    }
    message.content = content;
    message.reasoning_content = reasoning;
    message.thinking_blocks = Some(thinking_blocks);
    let grounding = &candidate_metadata(candidate)[0];
    if !grounding.is_empty() {
        let annotations = grounding_annotations(grounding);
        message.annotations = (!annotations.is_empty()).then_some(annotations);
    }
    let mut fields = Map::new();
    if !signatures.is_empty() {
        fields.insert("thought_signatures".into(), Value::Array(signatures));
    }
    if let Some(invocations) = server_side_tool_invocations(parts) {
        fields.insert(
            "server_side_tool_invocations".into(),
            Value::Array(invocations),
        );
    }
    message.provider_specific_fields = (!fields.is_empty()).then_some(fields);
    message
}

fn usage_count(usage: &Value, key: &str) -> u64 {
    usage.get(key).and_then(Value::as_u64).unwrap_or(0)
}

fn modality_counts(usage: &Value, key: &str) -> Option<BTreeMap<&'static str, u64>> {
    let details = usage.get(key)?.as_array()?;
    let mut counts = BTreeMap::new();
    for detail in details {
        let modality = detail
            .get("modality")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_uppercase();
        let tokens = detail
            .get("tokenCount")
            .or_else(|| detail.get("token_count"))
            .and_then(Value::as_u64)
            .unwrap_or(0);
        let slot = match modality.as_str() {
            "TEXT" | "DOCUMENT" => "text",
            "AUDIO" => "audio",
            "IMAGE" => "image",
            "VIDEO" => "video",
            _ => continue,
        };
        *counts.entry(slot).or_insert(0) += tokens;
    }
    Some(counts)
}

/// `calculate_grounding_requests(...).has_billable_grounding()` over every candidate.
fn has_search_grounding(body: &Value) -> bool {
    let items: Vec<Value> = body
        .get("candidates")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .flat_map(|candidate| candidate_metadata(candidate)[0].clone())
        .collect();
    items.iter().any(|item| {
        let queries = item
            .get("webSearchQueries")
            .and_then(Value::as_array)
            .is_some_and(|queries| {
                queries
                    .iter()
                    .any(|query| query.as_str().is_some_and(|q| !q.is_empty()))
            });
        let maps = item
            .get("groundingChunks")
            .and_then(Value::as_array)
            .is_some_and(|chunks| chunks.iter().any(|chunk| chunk.get("maps").is_some()))
            || item.get("googleMapsWidgetContextToken").is_some_and(truthy);
        queries || maps
    })
}

/// `_calculate_usage`: cached tokens are part of `promptTokenCount`, thinking tokens are added
/// to the candidates when Gemini counted them apart, and tool use prompt tokens are billed
/// unless a search grounding (billed per query) produced them.
pub fn calculate_usage(body: &Value) -> Result<ChatCompletionsUsage, Error> {
    let usage = body.get("usageMetadata").ok_or_else(|| {
        Error::InvalidResponse(
            format!("usageMetadata not found in completion_response. Got={body}").into(),
        )
    })?;
    let cached = usage.get("cachedContentTokenCount").and_then(Value::as_u64);
    let candidates = usage_count(usage, "candidatesTokenCount");
    let completion_details = modality_counts(usage, "candidatesTokensDetails");
    let explicit_text = completion_details
        .as_ref()
        .and_then(|counts| counts.get("text").copied());
    let other_completion: u64 = completion_details
        .as_ref()
        .map(|counts| {
            ["image", "audio", "video"]
                .iter()
                .filter_map(|slot| counts.get(slot))
                .sum()
        })
        .unwrap_or(0);
    let completion_text = explicit_text
        .or_else(|| (candidates > 0).then(|| candidates.saturating_sub(other_completion)));
    let reasoning = usage.get("thoughtsTokenCount").and_then(Value::as_u64);

    let prompt_details = modality_counts(usage, "promptTokensDetails");
    let cache_details = modality_counts(usage, "cacheTokensDetails");
    let mut prompt_text = prompt_details
        .as_ref()
        .and_then(|counts| counts.get("text").copied());
    match (
        cache_details
            .as_ref()
            .and_then(|counts| counts.get("text").copied()),
        cached,
    ) {
        (Some(cached_text), _) => prompt_text = prompt_text.map(|t| t.saturating_sub(cached_text)),
        (None, Some(cached)) if cache_details.is_none() => {
            prompt_text = prompt_text.map(|t| t.saturating_sub(cached));
        }
        _ => {}
    }
    let tool_use = usage_count(usage, "toolUsePromptTokenCount");
    let billable_tool_use = match has_search_grounding(body) {
        true => 0,
        false => tool_use,
    };
    let prompt = usage_count(usage, "promptTokenCount");
    let total = usage_count(usage, "totalTokenCount");
    let inclusive = prompt + candidates + tool_use == total;
    let completion = match (inclusive, reasoning) {
        (false, Some(reasoning)) if reasoning > 0 => candidates + reasoning,
        _ => candidates,
    };
    let completion_tokens_details =
        (completion_text.is_some() || reasoning.is_some()).then_some(CompletionTokensDetails {
            reasoning_tokens: reasoning,
            text_tokens: completion_text,
        });
    Ok(ChatCompletionsUsage {
        prompt_tokens: prompt + billable_tool_use,
        completion_tokens: completion,
        total_tokens: total,
        prompt_tokens_details: PromptTokensDetails {
            cached_tokens: cached.unwrap_or(0),
            text_tokens: prompt_text.map_or(0, |text| text + billable_tool_use),
            ..PromptTokensDetails::default()
        },
        completion_tokens_details,
        cache_read_input_tokens: cached,
        ..ChatCompletionsUsage::default()
    })
}

/// The answer Python gives when the prompt or the first candidate was blocked.
fn blocked_response(model: String, body: &Value) -> ChatCompletionsResponse {
    let usage = body.get("usageMetadata").cloned().unwrap_or(Value::Null);
    let prompt = usage_count(&usage, "promptTokenCount");
    let completion = usage_count(&usage, "candidatesTokenCount");
    ChatCompletionsResponse {
        created: unix_now(),
        model,
        choices: vec![ChatCompletionsChoice {
            index: 0,
            message: ChatCompletionsChoiceMessage {
                role: "assistant".into(),
                ..ChatCompletionsChoiceMessage::default()
            },
            finish_reason: "content_filter".into(),
            logprobs: None,
        }],
        usage: ChatCompletionsUsage {
            prompt_tokens: prompt,
            completion_tokens: completion,
            total_tokens: usage_count(&usage, "totalTokenCount"),
            ..ChatCompletionsUsage::default()
        },
        extra: Map::new(),
    }
}

/// Whether the prompt itself was blocked.
pub fn prompt_blocked(body: &Value) -> bool {
    body.get("promptFeedback")
        .is_some_and(|feedback| feedback.get("blockReason").is_some())
}

/// `_transform_google_generate_content_to_openai_model_response`.
pub fn transform_generate_content_response(
    model: &str,
    body: Value,
) -> Result<ChatCompletionsResponse, Error> {
    if !body.is_object() {
        return Err(Error::InvalidResponse(
            "Gemini response is not an object".into(),
        ));
    }
    let model = served_model_name(&body).unwrap_or_else(|| model.to_owned());
    if prompt_blocked(&body) {
        return Ok(blocked_response(model, &body));
    }
    let candidates = body
        .get("candidates")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    if candidates
        .first()
        .and_then(|first| first.get("finishReason"))
        .and_then(Value::as_str)
        .is_some_and(|reason| FLAGGED_FINISH_REASONS.contains(&reason))
    {
        return Ok(blocked_response(model, &body));
    }
    let mut metadata: [Vec<Value>; 4] = Default::default();
    let mut choices = Vec::new();
    let mut tool_index = 0;
    for (position, candidate) in candidates.iter().enumerate() {
        for (all, found) in metadata.iter_mut().zip(candidate_metadata(candidate)) {
            all.extend(found);
        }
        if candidate.get("content").is_none() && candidate.get("finishReason").is_none() {
            continue;
        }
        let message = candidate_message(candidate, &mut tool_index);
        let finish_reason = message.openai_finish_reason();
        choices.push(ChatCompletionsChoice {
            index: candidate
                .get("index")
                .and_then(Value::as_u64)
                .unwrap_or(position as u64),
            message: ChatCompletionsChoiceMessage {
                role: "assistant".into(),
                content: message.content,
                tool_calls: (!message.tool_calls.is_empty()).then_some(message.tool_calls),
                reasoning_content: message.reasoning_content,
                thinking_blocks: message.thinking_blocks,
                provider_specific_fields: message.provider_specific_fields,
                annotations: message.annotations,
            },
            finish_reason,
            logprobs: message.logprobs,
        });
    }
    let usage = calculate_usage(&body)?;
    let [grounding, url_context, safety, citations] = metadata;
    // Python names the answer by Gemini's `responseId` when there is one; the host mints an id
    // otherwise.
    let response_id = body
        .get("responseId")
        .and_then(Value::as_str)
        .filter(|id| !id.is_empty())
        .map(|id| ("id".to_owned(), Value::from(id)));
    let extra = Map::from_iter(response_id.into_iter().chain([
        (
            "vertex_ai_grounding_metadata".to_owned(),
            Value::Array(grounding),
        ),
        (
            "vertex_ai_url_context_metadata".to_owned(),
            Value::Array(url_context),
        ),
        ("vertex_ai_safety_results".to_owned(), Value::Array(safety)),
        (
            "vertex_ai_citation_metadata".to_owned(),
            Value::Array(citations),
        ),
    ]));
    Ok(ChatCompletionsResponse {
        created: unix_now(),
        model,
        choices,
        usage,
        extra,
    })
}

/// The same call as a stream: `streamGenerateContent` with `alt=sse`.
pub fn streaming_url(url: &str) -> String {
    let method = format!(":{GENERATE_CONTENT}");
    let url = match url.rfind(&method) {
        Some(at) => format!(
            "{}:{STREAM_GENERATE_CONTENT}{}",
            &url[..at],
            &url[at + method.len()..]
        ),
        None => url.to_owned(),
    };
    with_query(&url, "alt=sse")
}

/// `VertexGeminiConfig`: Gemini on Vertex AI, addressed to a project and location and
/// authenticated with a Google access token.
pub struct VertexGeminiConfig;

pub const VERTEX_GEMINI_CHAT_COMPLETIONS_CONFIG: VertexGeminiConfig = VertexGeminiConfig;

/// `get_model_for_vertex_ai_url`: a `gemini/<id>` model calls the id.
fn model_for_vertex_url(model: &str) -> &str {
    match model.contains("gemini/") {
        true => model.rsplit('/').next().unwrap_or(model),
        false => model,
    }
}

fn vertex_config(optional_params: &Map<String, Value>) -> Result<VertexConfig, Error> {
    Ok(VertexConfig::from_sourced_optional_params(
        optional_params,
        &BTreeMap::new(),
    )?)
}

impl BaseConfig for VertexGeminiConfig {
    fn secret_names(&self) -> Vec<&'static str> {
        litellm_auth_gcp::secret_names()
            .iter()
            .copied()
            .chain([VERTEXAI_API_BASE_ENV])
            .collect()
    }

    /// `get_vertex_url` with `_check_custom_proxy`.
    fn get_complete_url(
        &self,
        api_base: Option<&str>,
        model: &str,
        optional_params: &Map<String, Value>,
        env_lookup: &dyn Fn(&str) -> Option<String>,
    ) -> Result<String, Error> {
        let config = vertex_config(optional_params)?;
        let project = get_vertex_ai_project(&config, env_lookup)
            .or_else(|| get_vertex_ai_project_from_credentials(&config, env_lookup))
            .ok_or_else(|| Error::Auth(VertexParams::PROJECT.missing("Vertex AI")))?;
        let location = get_vertex_ai_location(&config, env_lookup)
            .unwrap_or_else(|| DEFAULT_LOCATION.to_owned());
        let model = model_for_vertex_url(model);
        let target = match model.chars().all(|c| c.is_ascii_digit()) {
            true => format!("endpoints/{model}"),
            false => format!("publishers/google/models/{model}"),
        };
        let default = format!(
            "{}/v1/projects/{project}/locations/{location}/{target}:{GENERATE_CONTENT}",
            get_vertex_base_url(&location)?
        );
        let api_base = api_base
            .map(str::to_owned)
            .or_else(|| env_lookup(VERTEXAI_API_BASE_ENV))
            .filter(|base| !base.trim().is_empty());
        Ok(graft_custom_vertex_base(
            api_base.as_deref(),
            &default,
            GENERATE_CONTENT,
        ))
    }

    fn transform_request(
        &self,
        model: &str,
        messages: Vec<ChatMessage>,
        optional_params: Map<String, Value>,
        context: &ChatRequestContext<'_>,
    ) -> Result<ProviderChatRequestData, Error> {
        build_request(
            GeminiSurface::Vertex,
            model,
            messages,
            optional_params,
            context,
        )
    }

    fn transform_response(
        &self,
        model: &str,
        response: ProviderChatResponseData,
        _shape: &StreamShape,
    ) -> Result<ChatCompletionsResponse, Error> {
        transform_generate_content_response(model, response.body)
    }

    fn remote_media(&self, messages: &[ChatMessage]) -> Vec<String> {
        remote_media(messages, GeminiSurface::Vertex)
    }

    fn model_response_iterator(&self, _shape: StreamShape) -> Option<ChatStream> {
        Some(gemini_chat_stream())
    }

    fn streaming_request(&self, url: String, body: Value) -> Result<(String, Value), Error> {
        Ok((streaming_url(&url), body))
    }

    /// A forwarded bearer is sent as is; otherwise a Google access token for the configured
    /// credentials is acquired when the request is sent. Vertex AI ignores `api_key`.
    fn validate_environment(
        &self,
        headers: Headers,
        _api_key: Option<&str>,
        _model: &str,
        optional_params: &Map<String, Value>,
        _env_lookup: &dyn Fn(&str) -> Option<String>,
    ) -> Result<ValidatedEnvironment, Error> {
        if has_header(&headers, "authorization") {
            return Ok(ValidatedEnvironment {
                headers,
                auth: AuthScheme::Forwarded,
            });
        }
        Ok(ValidatedEnvironment {
            headers,
            auth: AuthScheme::GcpAccessToken {
                config: Box::new(vertex_config(optional_params)?),
            },
        })
    }

    fn config_params(&self) -> &'static [&'static str] {
        CONFIG_PARAMS
    }

    /// Every request is translated; a param or message Python refuses is refused here too.
    fn unsupported_reason(
        &self,
        _messages: &[ChatMessage],
        _optional_params: &Map<String, Value>,
    ) -> Option<Unsupported> {
        None
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use rstest::rstest;

    use super::*;
    use crate::base_llm::model_info::ModelInfo;

    struct CostMap(HashMap<String, Map<String, Value>>);

    impl ModelInfo for CostMap {
        fn entry(&self, model: &str) -> Option<std::borrow::Cow<'_, Map<String, Value>>> {
            self.0.get(model).map(std::borrow::Cow::Borrowed)
        }

        fn declares(&self, key: &str) -> bool {
            self.0.values().any(|entry| entry.contains_key(key))
        }
    }

    fn cost_map() -> CostMap {
        let entry = json!({
            "supports_system_messages": true,
            "supports_reasoning": true,
            "supports_response_schema": true
        });
        let Value::Object(entry) = entry else {
            unreachable!()
        };
        CostMap(HashMap::from([
            ("gemini-2.5-flash".to_owned(), entry.clone()),
            ("gemini/gemini-2.5-flash".to_owned(), entry),
        ]))
    }

    fn object(value: Value) -> Map<String, Value> {
        let Value::Object(map) = value else {
            panic!("expected an object");
        };
        map
    }

    fn request(
        surface: GeminiSurface,
        model: &str,
        messages: Value,
        params: Value,
        drop_params: bool,
    ) -> Result<Value, Error> {
        let map = cost_map();
        let context = ChatRequestContext {
            drop_params: Some(drop_params),
            model_info: Some(&map),
        };
        build_request(
            surface,
            model,
            serde_json::from_value(messages).unwrap(),
            object(params),
            &context,
        )
        .map(|request| request.body)
    }

    fn hi() -> Value {
        json!([{"role": "system", "content": "be brief"}, {"role": "user", "content": "hi"}])
    }

    #[rstest]
    #[case::gemini_1("gemini-1.5-pro", false)]
    #[case::gemini_2("gemini-2.0-flash-001", false)]
    #[case::gemini_25("vertex_ai/gemini-2.5-flash", false)]
    #[case::exp("gemini-exp-1206", false)]
    #[case::pro("gemini-pro", false)]
    #[case::flash_latest("gemini-flash-latest", true)]
    #[case::flash_lite_latest("gemini-flash-lite-latest", true)]
    #[case::gemini_3("gemini-3-pro-preview", true)]
    #[case::gemini_31("gemini-3.1-flash", true)]
    #[case::gemini_10("gemini-10-ultra", true)]
    #[case::fine_tuned("1234567890", false)]
    #[case::ai_studio_tuned("gemini/tunedModels/abc", false)]
    #[case::gemma("gemma-3-27b-it", false)]
    fn gemini_3_or_newer_follows_pythons_pattern(#[case] model: &str, #[case] expected: bool) {
        assert_eq!(is_gemini_3_or_newer(model), expected);
    }

    #[rstest]
    #[case::gemini_15("gemini-1.5-pro", false)]
    #[case::gemini_20("gemini-2.0-flash", true)]
    #[case::gemini_3("gemini-3-pro", true)]
    #[case::gemini_10("gemini-10.0-x", true)]
    #[case::pro("gemini-pro", false)]
    fn json_schema_support_starts_at_gemini_2(#[case] model: &str, #[case] expected: bool) {
        assert_eq!(supports_response_json_schema(model), expected);
    }

    #[rstest]
    #[case::gemini("gemini-2.5-flash", true)]
    #[case::tuned("gemini/123", true)]
    #[case::claude("claude-sonnet-4", false)]
    #[case::gemma("gemma/gemma-3", false)]
    #[case::openai("openai/gemini-like", false)]
    #[case::partner_gemma("google/gemma-3-gemini", false)]
    #[case::legacy("chat-bison", false)]
    fn vertex_routes_only_gemini_models_here(#[case] model: &str, #[case] expected: bool) {
        assert_eq!(is_vertex_gemini_model(model), expected);
    }

    #[rstest]
    fn builds_the_body_python_builds() {
        let body = request(
            GeminiSurface::Vertex,
            "gemini-2.5-flash",
            hi(),
            json!({"temperature": 0.3, "max_tokens": 50, "stop": "END", "n": 2, "seed": 7,
                "top_k": 4, "frequency_penalty": 0.1, "logprobs": true, "top_logprobs": 2}),
            false,
        )
        .unwrap();
        assert_eq!(
            body,
            json!({
                "contents": [{"role": "user", "parts": [{"text": "hi"}]}],
                "system_instruction": {"parts": [{"text": "be brief"}]},
                "generationConfig": {"temperature": 0.3, "candidate_count": 2, "stop_sequences": ["END"],
                    "max_output_tokens": 50, "frequency_penalty": 0.1, "seed": 7, "responseLogprobs": true,
                    "logprobs": 2, "top_k": 4}
            })
        );
    }

    #[rstest]
    #[case::refused(false, true)]
    #[case::dropped(true, false)]
    fn gemini_3_takes_no_penalties_on_vertex(#[case] drop_params: bool, #[case] refused: bool) {
        let result = request(
            GeminiSurface::Vertex,
            "gemini-3-pro-preview",
            hi(),
            json!({"presence_penalty": 0.5}),
            drop_params,
        );
        assert_eq!(result.is_err(), refused);
    }

    #[rstest]
    #[case::none(json!("none"), json!({"functionCallingConfig": {"mode": "NONE"}}))]
    #[case::required(json!("required"), json!({"functionCallingConfig": {"mode": "ANY"}}))]
    #[case::auto(json!("auto"), json!({"functionCallingConfig": {"mode": "AUTO"}}))]
    #[case::named(json!({"type": "function", "function": {"name": "f"}}), json!({"functionCallingConfig": {"mode": "ANY", "allowed_function_names": ["f"]}}))]
    fn tool_choice_becomes_a_tool_config(#[case] choice: Value, #[case] expected: Value) {
        let body = request(
            GeminiSurface::AiStudio,
            "gemini-2.5-flash",
            hi(),
            json!({"tools": [{"type": "function", "function": {"name": "f", "parameters": {"type": "object", "properties": {"x": {"type": "string"}}, "additionalProperties": false}, "strict": true}}], "tool_choice": choice}),
            false,
        )
        .unwrap();
        assert_eq!(body["toolConfig"], expected);
        assert_eq!(
            body["tools"],
            json!([{"function_declarations": [{"name": "f", "parameters": {"type": "object", "properties": {"x": {"type": "string"}}}}]}])
        );
    }

    #[rstest]
    fn an_unknown_tool_choice_is_refused_like_python() {
        assert!(matches!(
            request(
                GeminiSurface::Vertex,
                "gemini-2.5-flash",
                hi(),
                json!({"tools": [{"type": "function", "function": {"name": "f"}}], "tool_choice": "any"}),
                true,
            ),
            Err(Error::InvalidRequest(_))
        ));
    }

    #[rstest]
    #[case::functions_drop_search(json!([{"type": "function", "function": {"name": "f"}}, {"googleSearch": {}}]), false, json!([{"function_declarations": [{"name": "f"}]}]))]
    #[case::circulated_keeps_search(json!([{"type": "function", "function": {"name": "f"}}, {"googleSearch": {}}]), true, json!([{"function_declarations": [{"name": "f"}]}, {"googleSearch": {}}]))]
    #[case::built_ins(json!([{"code_execution": {}}, {"urlContext": {}}, {"type": "web_search"}]), false, json!([{"googleSearch": {}}, {"code_execution": {}}, {"url_context": {}}]))]
    #[case::computer_use(json!([{"type": "computer_use", "environment": "browser"}]), false, json!([{"computerUse": {"environment": "ENVIRONMENT_BROWSER"}}]))]
    fn built_in_tools_map_like_python(
        #[case] tools: Value,
        #[case] circulate: bool,
        #[case] expected: Value,
    ) {
        let mut params = json!({"tools": tools});
        if circulate {
            params["include_server_side_tool_invocations"] = json!(true);
        }
        let body = request(
            GeminiSurface::Vertex,
            "gemini-3-pro-preview",
            hi(),
            params,
            false,
        )
        .unwrap();
        assert_eq!(body["tools"], expected);
    }

    #[rstest]
    fn web_search_options_add_google_search() {
        let body = request(
            GeminiSurface::Vertex,
            "gemini-2.5-flash",
            hi(),
            json!({"web_search_options": {}}),
            false,
        )
        .unwrap();
        assert_eq!(body["tools"], json!([{"googleSearch": {}}]));
    }

    #[rstest]
    #[case::budget_low("gemini-2.5-flash", json!("low"), json!({"thinkingBudget": 1024, "includeThoughts": true}))]
    #[case::budget_minimal_flash("gemini-2.5-flash", json!("minimal"), json!({"thinkingBudget": 1, "includeThoughts": true}))]
    #[case::budget_disable("gemini-2.5-flash", json!("disable"), json!({"thinkingBudget": 0, "includeThoughts": false}))]
    #[case::level_high("gemini-3-pro-preview", json!("high"), json!({"thinkingLevel": "high", "includeThoughts": true}))]
    #[case::level_minimal_pro("gemini-3-pro-preview", json!("minimal"), json!({"thinkingLevel": "low", "includeThoughts": true}))]
    #[case::level_minimal_flash("gemini-3-flash-preview", json!("minimal"), json!({"thinkingLevel": "minimal", "includeThoughts": true}))]
    #[case::level_none("gemini-3-flash-preview", json!("none"), json!({"thinkingLevel": "minimal", "includeThoughts": false}))]
    #[case::summary_dict("gemini-2.5-flash", json!({"effort": "high", "summary": "auto"}), json!({"thinkingBudget": 4096, "includeThoughts": true}))]
    fn reasoning_effort_becomes_a_thinking_config(
        #[case] model: &str,
        #[case] effort: Value,
        #[case] expected: Value,
    ) {
        let body = request(
            GeminiSurface::Vertex,
            model,
            hi(),
            json!({"reasoning_effort": effort}),
            false,
        )
        .unwrap();
        assert_eq!(body["generationConfig"]["thinkingConfig"], expected);
    }

    #[rstest]
    fn an_invalid_reasoning_effort_is_refused() {
        assert!(matches!(
            request(
                GeminiSurface::Vertex,
                "gemini-2.5-flash",
                hi(),
                json!({"reasoning_effort": "extreme"}),
                true
            ),
            Err(Error::InvalidRequest(_))
        ));
    }

    #[rstest]
    #[case::budget("gemini-2.5-flash", json!({"type": "enabled", "budget_tokens": 2000}), json!({"includeThoughts": true, "thinkingBudget": 2000}))]
    #[case::disabled_budget_zero("gemini-2.5-flash", json!({"type": "enabled", "budget_tokens": 0}), json!({"thinkingBudget": 0}))]
    #[case::gemini_3("gemini-3-pro-preview", json!({"type": "enabled", "budget_tokens": 2000}), json!({"includeThoughts": true}))]
    #[case::gemini_3_disabled("gemini-3-pro-preview", json!({"type": "disabled"}), json!({"includeThoughts": false}))]
    fn thinking_becomes_a_thinking_config(
        #[case] model: &str,
        #[case] thinking: Value,
        #[case] expected: Value,
    ) {
        let body = request(
            GeminiSurface::Vertex,
            model,
            hi(),
            json!({"thinking": thinking}),
            false,
        )
        .unwrap();
        assert_eq!(body["generationConfig"]["thinkingConfig"], expected);
    }

    #[rstest]
    #[case::json_schema_on_gemini_2("gemini-2.5-flash",
        json!({"type": "json_schema", "json_schema": {"name": "x", "strict": true, "schema": {"type": "object", "properties": {"a": {"type": "string"}}, "additionalProperties": false}}}),
        json!({"response_mime_type": "application/json", "response_json_schema": {"type": "object", "properties": {"a": {"type": "string"}}, "additionalProperties": false}}))]
    #[case::json_object("gemini-2.5-flash", json!({"type": "json_object"}), json!({"response_mime_type": "application/json"}))]
    #[case::text("gemini-2.5-flash", json!({"type": "text"}), json!({"response_mime_type": "text/plain"}))]
    fn response_format_sets_the_output_type(
        #[case] model: &str,
        #[case] format: Value,
        #[case] expected: Value,
    ) {
        let body = request(
            GeminiSurface::Vertex,
            model,
            hi(),
            json!({"response_format": format}),
            false,
        )
        .unwrap();
        assert_eq!(body["generationConfig"], expected);
    }

    #[rstest]
    fn audio_maps_to_a_speech_config() {
        let body = request(
            GeminiSurface::Vertex,
            "gemini-2.5-flash-preview-tts",
            hi(),
            json!({"audio": {"voice": "alloy", "format": "pcm16"}, "modalities": ["text"]}),
            false,
        )
        .unwrap();
        assert_eq!(
            body["generationConfig"],
            json!({"speechConfig": {"voiceConfig": {"prebuiltVoiceConfig": {"voiceName": "Kore"}}}, "responseModalities": ["TEXT", "AUDIO"]})
        );
    }

    #[rstest]
    #[case::cache_control(json!([{"role": "user", "content": [{"type": "text", "text": "x", "cache_control": {"type": "ephemeral"}}]}]), json!({}))]
    #[case::functions(json!([{"role": "user", "content": "x"}]), json!({"functions": [{"name": "f"}]}))]
    fn what_python_answers_differently_is_left_to_it(
        #[case] messages: Value,
        #[case] params: Value,
    ) {
        assert!(matches!(
            request(
                GeminiSurface::Vertex,
                "gemini-2.5-flash",
                messages,
                params,
                false
            ),
            Err(Error::Unsupported(_))
        ));
    }

    #[rstest]
    #[case::unknown_model_merges_system("gemini-unknown", json!([{"role": "user", "parts": [{"text": "be brief"}, {"text": "hi"}]}]))]
    fn a_model_without_system_support_gets_the_system_text_as_user(
        #[case] model: &str,
        #[case] contents: Value,
    ) {
        let body = request(GeminiSurface::Vertex, model, hi(), json!({}), false).unwrap();
        assert_eq!(body["contents"], contents);
        assert!(body.get("system_instruction").is_none());
    }

    fn env(_: &str) -> Option<String> {
        None
    }

    #[rstest]
    #[case::regional(
        None,
        "gemini-2.5-flash",
        "https://europe-west4-aiplatform.googleapis.com/v1/projects/p/locations/europe-west4/publishers/google/models/gemini-2.5-flash:generateContent"
    )]
    #[case::tuned(
        None,
        "gemini/987",
        "https://europe-west4-aiplatform.googleapis.com/v1/projects/p/locations/europe-west4/endpoints/987:generateContent"
    )]
    #[case::custom_host(
        Some("http://127.0.0.1:9"),
        "gemini-2.5-flash",
        "http://127.0.0.1:9/v1/projects/p/locations/europe-west4/publishers/google/models/gemini-2.5-flash:generateContent"
    )]
    fn vertex_urls_name_the_project_and_location(
        #[case] api_base: Option<&str>,
        #[case] model: &str,
        #[case] expected: &str,
    ) {
        let params = object(json!({"vertex_project": "p", "vertex_location": "europe-west4"}));
        assert_eq!(
            VERTEX_GEMINI_CHAT_COMPLETIONS_CONFIG
                .get_complete_url(api_base, model, &params, &env)
                .unwrap(),
            expected
        );
    }

    #[rstest]
    #[case::default(
        "https://h/v1/projects/p/x:generateContent",
        "https://h/v1/projects/p/x:streamGenerateContent?alt=sse"
    )]
    #[case::custom_query(
        "https://h/x:generateContent?k=1",
        "https://h/x:streamGenerateContent?k=1&alt=sse"
    )]
    fn streaming_swaps_the_method_and_asks_for_sse(#[case] url: &str, #[case] expected: &str) {
        assert_eq!(streaming_url(url), expected);
    }

    #[rstest]
    #[case::forwarded(vec![("Authorization".to_owned(), "Bearer t".to_owned())], true)]
    #[case::token(Vec::new(), false)]
    fn vertex_takes_a_forwarded_bearer_or_a_google_token(
        #[case] headers: Headers,
        #[case] forwarded: bool,
    ) {
        let validated = VERTEX_GEMINI_CHAT_COMPLETIONS_CONFIG
            .validate_environment(
                headers,
                Some("ignored"),
                "gemini-2.5-flash",
                &Map::new(),
                &env,
            )
            .unwrap();
        assert_eq!(matches!(validated.auth, AuthScheme::Forwarded), forwarded);
        assert_eq!(
            matches!(validated.auth, AuthScheme::GcpAccessToken { .. }),
            !forwarded
        );
    }

    fn response(body: Value) -> Value {
        serde_json::to_value(transform_generate_content_response("gemini-2.5-flash", body).unwrap())
            .unwrap()
    }

    #[rstest]
    #[case::named(json!("resp_1"), Some(json!("resp_1")))]
    #[case::unnamed(Value::Null, None)]
    fn the_answer_is_named_by_the_response_id(#[case] id: Value, #[case] expected: Option<Value>) {
        let answer = response(json!({
            "responseId": id,
            "candidates": [{"content": {"role": "model", "parts": [{"text": "Hi"}]}, "finishReason": "STOP"}],
            "usageMetadata": {"promptTokenCount": 1, "candidatesTokenCount": 1, "totalTokenCount": 2}
        }));
        assert_eq!(answer.get("id").cloned(), expected);
    }

    #[rstest]
    fn text_reasoning_and_usage_come_back_in_openai_shape() {
        let answer = response(json!({
            "candidates": [{"content": {"role": "model", "parts": [
                {"text": "thinking", "thought": true, "thoughtSignature": "sig"},
                {"text": "Hello"}
            ]}, "finishReason": "STOP"}],
            "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 5, "thoughtsTokenCount": 3,
                "totalTokenCount": 18, "cachedContentTokenCount": 4,
                "promptTokensDetails": [{"modality": "TEXT", "tokenCount": 10}]},
            "modelVersion": "gemini-2.5-flash@001"
        }));
        assert_eq!(answer["model"], "gemini-2.5-flash");
        assert_eq!(
            answer["choices"],
            json!([{"index": 0, "finish_reason": "stop", "message": {
                "role": "assistant", "content": "Hello", "reasoning_content": "thinking",
                "thinking_blocks": [{"type": "thinking", "thinking": "thinking", "signature": "sig"}],
                "provider_specific_fields": {"thought_signatures": ["sig"]}
            }}])
        );
        assert_eq!(
            answer["usage"],
            json!({"prompt_tokens": 10, "completion_tokens": 8, "total_tokens": 18,
                "prompt_tokens_details": {"cached_tokens": 4, "cache_creation_tokens": 0, "text_tokens": 6},
                "completion_tokens_details": {"reasoning_tokens": 3, "text_tokens": 5},
                "cache_read_input_tokens": 4})
        );
    }

    #[rstest]
    fn function_calls_become_tool_calls_with_their_signature() {
        let answer = response(json!({
            "candidates": [{"content": {"parts": [
                {"functionCall": {"name": "get_weather", "args": {"city": "Zürich"}, "id": "fc_1"}, "thoughtSignature": "s"}
            ]}, "finishReason": "STOP"}],
            "usageMetadata": {"promptTokenCount": 1, "candidatesTokenCount": 1, "totalTokenCount": 2}
        }));
        assert_eq!(answer["choices"][0]["finish_reason"], "tool_calls");
        assert_eq!(
            answer["choices"][0]["message"]["tool_calls"],
            json!([{"id": "fc_1__thought__s", "type": "function", "index": 0,
                "function": {"name": "get_weather", "arguments": "{\"city\": \"Zürich\"}"},
                "provider_specific_fields": {"thought_signature": "s"}}])
        );
    }

    #[rstest]
    #[case::prompt_blocked(json!({"promptFeedback": {"blockReason": "SAFETY"}, "usageMetadata": {"promptTokenCount": 3, "totalTokenCount": 3}}))]
    #[case::candidate_flagged(json!({"candidates": [{"finishReason": "SAFETY"}], "usageMetadata": {"promptTokenCount": 3, "totalTokenCount": 3}}))]
    fn a_blocked_answer_is_a_content_filter(#[case] body: Value) {
        let answer = response(body);
        assert_eq!(answer["choices"][0]["finish_reason"], "content_filter");
        assert_eq!(answer["choices"][0]["message"]["content"], Value::Null);
        assert_eq!(answer["usage"]["prompt_tokens"], 3);
    }

    #[rstest]
    #[case::max_tokens("MAX_TOKENS", "length")]
    #[case::malformed("MALFORMED_FUNCTION_CALL", "stop")]
    #[case::language("LANGUAGE", "content_filter")]
    fn finish_reasons_map_like_python(#[case] reason: &str, #[case] expected: &str) {
        let answer = response(json!({
            "candidates": [{"content": {"parts": [{"text": "x"}]}, "finishReason": reason}],
            "usageMetadata": {"promptTokenCount": 1, "candidatesTokenCount": 1, "totalTokenCount": 2}
        }));
        assert_eq!(answer["choices"][0]["finish_reason"], expected);
    }

    #[rstest]
    fn grounding_becomes_annotations_and_metadata() {
        let grounding = json!({
            "webSearchQueries": ["weather paris"],
            "groundingChunks": [{"web": {"uri": "https://w", "title": "W"}}],
            "groundingSupports": [{"segment": {"startIndex": 0, "endIndex": 5}, "groundingChunkIndices": [0]}]
        });
        let answer = response(json!({
            "candidates": [{"content": {"parts": [{"text": "Sunny"}]}, "finishReason": "STOP", "groundingMetadata": grounding}],
            "usageMetadata": {"promptTokenCount": 4, "candidatesTokenCount": 1, "toolUsePromptTokenCount": 7, "totalTokenCount": 12}
        }));
        assert_eq!(
            answer["choices"][0]["message"]["annotations"],
            json!([{"type": "url_citation", "url_citation": {"start_index": 0, "end_index": 5, "url": "https://w", "title": "W"}}])
        );
        assert_eq!(answer["vertex_ai_grounding_metadata"], json!([grounding]));
        assert_eq!(answer["usage"]["prompt_tokens"], 4);
    }

    #[rstest]
    #[case::billed_tool_use(json!({"promptTokenCount": 4, "candidatesTokenCount": 1, "toolUsePromptTokenCount": 7, "totalTokenCount": 12,
        "promptTokensDetails": [{"modality": "TEXT", "tokenCount": 4}]}), 11, 11)]
    #[case::explicit_cache(json!({"promptTokenCount": 10, "candidatesTokenCount": 1, "totalTokenCount": 11, "cachedContentTokenCount": 6,
        "promptTokensDetails": [{"modality": "TEXT", "tokenCount": 8}, {"modality": "IMAGE", "tokenCount": 2}],
        "cacheTokensDetails": [{"modality": "TEXT", "tokenCount": 5}]}), 10, 3)]
    fn usage_counts_prompt_tokens_like_python(
        #[case] usage: Value,
        #[case] prompt: u64,
        #[case] text: u64,
    ) {
        let usage = calculate_usage(&json!({"usageMetadata": usage})).unwrap();
        assert_eq!(usage.prompt_tokens, prompt);
        assert_eq!(usage.prompt_tokens_details.text_tokens, text);
    }

    #[rstest]
    fn a_response_without_usage_is_invalid() {
        assert!(matches!(
            transform_generate_content_response("m", json!({"candidates": []})),
            Err(Error::InvalidResponse(_))
        ));
    }
}
