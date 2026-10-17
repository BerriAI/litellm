//! Python's `VertexAIAnthropicConfig` for `/chat/completions`: Claude on Vertex AI, reached
//! through `VertexAIPartnerModels.completion`. The request is `AnthropicConfig`'s, built for the
//! `vertex_ai` host, then addressed to the publisher URL: no `model` in the body, Vertex's
//! `anthropic_version`, a sanitized `output_config`, and the betas in `anthropic_beta`.

use std::collections::BTreeMap;

use litellm_auth::VertexParams;
use litellm_auth_gcp::{
    VertexConfig, get_vertex_ai_location, get_vertex_ai_project,
    get_vertex_ai_project_from_credentials,
};
use litellm_llms_types::{
    formats::chat_completions::{ChatCompletionsResponse, ChatMessage},
    providers::{
        anthropic::{AnthropicBeta, BETA_HEADER, BetaProvider, BetaSet, DEFAULT_HEADERS},
        vertex_ai::{
            ANTHROPIC_VERSION, DEFAULT_LOCATION, GLOBAL_API_BASE, RAW_PREDICT, STREAM_RAW_PREDICT,
        },
    },
};
use serde_json::{Map, Value, json};

use crate::{
    Error,
    anthropic::{
        beta_headers::BetaPolicy,
        chat::{
            handler::ModelResponseIterator,
            params::{AnthropicHost, ModelCapabilities},
            transformation::{build_messages_request, transform_messages_response},
        },
    },
    base_llm::{
        auth::{AuthScheme, Headers, ValidatedEnvironment},
        chat::{
            media::{data_url, remote_urls},
            params::ChatRequestContext,
            streaming::{ChatStream, StreamShape},
            transformation::{
                BaseConfig, ProviderChatRequestData, ProviderChatResponseData, STREAM_PARAM,
                Unsupported,
            },
        },
        messages::streaming::event_stream as messages_event_stream,
    },
    vertex_ai::{
        common_utils::with_query,
        gemini::vertex_and_google_ai_studio_gemini::{CONFIG_PARAMS, VERTEXAI_API_BASE_ENV},
        messages::transformation::complete_vertex_anthropic_url,
    },
};

/// `PartnerModelPrefixes`: the ids `is_vertex_partner_model` sends to the partner handler.
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

/// `should_use_openai_handler`: the partners served through Vertex's OpenAI-compatible API.
const OPENAI_LIKE_PARTNERS: &[&str] = &[
    "llama",
    "deepseek-ai",
    "qwen",
    "openai/gpt-oss-",
    "minimaxai/",
    "moonshotai/",
    "zai-org/",
    "google/gemma-",
];

/// The tool type prefix `is_web_search_tool_used` looks for.
const WEB_SEARCH_TOOL_PREFIX: &str = "web_search";

/// `get_vertex_ai_model_route(model) == PARTNER_MODELS` and the partner handler picking its
/// Claude branch: a partner model that is neither OpenAI-compatible, Mistral, nor AI21.
pub fn is_vertex_claude_model(model: &str) -> bool {
    !model.contains("agent_engine/")
        && PARTNER_MODEL_PREFIXES
            .iter()
            .any(|prefix| model.starts_with(prefix))
        && !OPENAI_LIKE_PARTNERS
            .iter()
            .any(|partner| model.contains(partner))
        && !model.contains("mistral")
        && !model.contains("codestral")
        && !model.contains("jamba")
        && model.contains("claude")
}

/// `VertexAIAnthropicConfig`: Claude on Vertex AI, addressed to a project and location and
/// authenticated with a Google access token.
pub struct VertexAiAnthropicConfig;

pub const VERTEX_ANTHROPIC_CHAT_COMPLETIONS_CONFIG: VertexAiAnthropicConfig =
    VertexAiAnthropicConfig;

fn vertex_config(optional_params: &Map<String, Value>) -> Result<VertexConfig, Error> {
    Ok(VertexConfig::from_sourced_optional_params(
        optional_params,
        &BTreeMap::new(),
    )?)
}

/// Whether `url` is on a host `get_vertex_base_url` builds. Any other host is a custom
/// `api_base`, which `_check_custom_proxy` asks for `alt=sse` when streaming.
fn is_vertex_host(url: &str) -> bool {
    let rest = url.split_once("://").map_or(url, |(_, rest)| rest);
    let host = rest.split(['/', '?', '#']).next().unwrap_or(rest);
    let global = GLOBAL_API_BASE
        .split_once("://")
        .map_or(GLOBAL_API_BASE, |(_, host)| host);
    host == global
        || host.ends_with("-aiplatform.googleapis.com")
        || (host.starts_with("aiplatform.") && host.ends_with(".rep.googleapis.com"))
}

/// The same call made as a stream: `create_vertex_url` with `stream=True`, which names
/// `streamRawPredict`, plus `alt=sse` on a custom `api_base`.
pub fn streaming_url(url: &str) -> String {
    let method = format!(":{RAW_PREDICT}");
    let (head, query) = url
        .split_once('?')
        .map_or((url, None), |(head, query)| (head, Some(query)));
    let head = match head.strip_suffix(&method) {
        Some(stem) => format!("{stem}:{STREAM_RAW_PREDICT}"),
        None => head.to_owned(),
    };
    let swapped = match query {
        Some(query) => format!("{head}?{query}"),
        None => head,
    };
    match is_vertex_host(&swapped) {
        true => swapped,
        false => with_query(&swapped, "alt=sse"),
    }
}

/// The betas a caller named in the request's `extra_headers`, as a comma separated
/// `anthropic-beta` string or a list.
fn extra_header_betas(optional_params: &Map<String, Value>) -> Vec<String> {
    let value = optional_params
        .get("extra_headers")
        .and_then(|headers| headers.get(BETA_HEADER));
    match value {
        Some(Value::String(text)) => text
            .split(',')
            .map(str::trim)
            .filter(|beta| !beta.is_empty())
            .map(str::to_owned)
            .collect(),
        Some(Value::Array(items)) => items
            .iter()
            .filter_map(Value::as_str)
            .map(str::to_owned)
            .collect(),
        _ => Vec::new(),
    }
}

/// `_add_context_management_beta_headers`: `compact_20260112` edits need the compaction beta,
/// any other edit the context management one.
fn context_management_betas(context_management: Option<&Value>) -> Vec<AnthropicBeta> {
    let Some(Value::Object(context_management)) = context_management else {
        return Vec::new();
    };
    let edits = context_management
        .get("edits")
        .and_then(Value::as_array)
        .map(Vec::as_slice)
        .unwrap_or_default();
    let compact =
        |edit: &Value| edit.get("type").and_then(Value::as_str) == Some("compact_20260112");
    [
        edits
            .iter()
            .any(compact)
            .then_some(AnthropicBeta::Compact20260112),
        edits
            .iter()
            .any(|edit| !compact(edit))
            .then_some(AnthropicBeta::ContextManagement20250627),
    ]
    .into_iter()
    .flatten()
    .collect()
}

/// `filter_and_transform_beta_headers` for Vertex AI: the betas it accepts, under its names.
fn accepted_on_vertex(betas: &BetaSet) -> BetaSet {
    betas
        .iter()
        .filter_map(|beta| beta.on(BetaProvider::VertexAi))
        .collect()
}

/// `sanitize_vertex_anthropic_output_params`: `output_format` folded into
/// `output_config.format`, `effort` kept only for models that accept it, and an emptied or
/// malformed `output_config` left out.
fn sanitize_output_params(
    body: &mut Map<String, Value>,
    caps: &ModelCapabilities<'_>,
) -> Result<(), Error> {
    let output_format = body
        .remove("output_format")
        .filter(|value| !value.is_null());
    let output_config = match body.get("output_config") {
        Some(Value::Object(config)) => Some(config.clone()),
        _ => None,
    };
    let output_config = match (output_config, output_format) {
        (config, Some(format)) => {
            let mut config = config.unwrap_or_default();
            config.insert("format".into(), format);
            Some(config)
        }
        (config, None) => config,
    };
    let Some(mut output_config) = output_config else {
        body.remove("output_config");
        return Ok(());
    };
    if output_config.contains_key("effort") && !caps.supports_effort()? {
        output_config.remove("effort");
    }
    match output_config.is_empty() {
        true => {
            body.remove("output_config");
        }
        false => {
            body.insert("output_config".into(), Value::Object(output_config));
        }
    }
    Ok(())
}

impl VertexAiAnthropicConfig {
    fn build_request(
        &self,
        model: &str,
        messages: Vec<ChatMessage>,
        mut optional_params: Map<String, Value>,
        context: &ChatRequestContext<'_>,
    ) -> Result<ProviderChatRequestData, Error> {
        let caller_betas = extra_header_betas(&optional_params);
        // `VertexAIPartnerModels.completion` adds the version after the mapped params.
        optional_params.insert("anthropic_version".into(), json!(ANTHROPIC_VERSION));
        let built = build_messages_request(
            model,
            messages,
            optional_params,
            context,
            AnthropicHost::VertexAi,
            CONFIG_PARAMS,
        )?;
        let Value::Object(mut body) = built.body else {
            return Err(Error::InvalidRequest(
                "the Messages request is not an object".into(),
            ));
        };
        body.remove("model");
        body.remove("extra_headers");
        let caps = ModelCapabilities::on(AnthropicHost::VertexAi, model, context);
        sanitize_output_params(&mut body, &caps)?;

        let betas: BetaSet = context_management_betas(body.get("context_management"))
            .into_iter()
            .chain(
                caller_betas
                    .iter()
                    .map(|beta| beta.parse().unwrap_or_else(|never| match never {})),
            )
            .collect();
        let web_search = body
            .get("tools")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .any(|tool| {
                tool.get("type")
                    .and_then(Value::as_str)
                    .is_some_and(|kind| kind.starts_with(WEB_SEARCH_TOOL_PREFIX))
            });
        // `get_anthropic_headers` sends Vertex only the web search beta; the request's own
        // betas replace it when there are any.
        let header_betas = match (betas.is_empty(), web_search) {
            (false, _) => accepted_on_vertex(&betas),
            (true, true) => {
                accepted_on_vertex(&[AnthropicBeta::WebSearch20250305].into_iter().collect())
            }
            (true, false) => BetaSet::default(),
        };
        let body_betas = accepted_on_vertex(&betas);
        if !body_betas.is_empty() {
            body.insert(
                "anthropic_beta".into(),
                Value::Array(body_betas.iter().map(|beta| json!(beta.as_str())).collect()),
            );
        }
        Ok(ProviderChatRequestData {
            body: Value::Object(body),
            stream_shape: built.stream_shape,
            headers: match header_betas.is_empty() {
                true => Vec::new(),
                false => vec![(BETA_HEADER.to_owned(), header_betas.to_string())],
            },
        })
    }
}

impl BaseConfig for VertexAiAnthropicConfig {
    fn secret_names(&self) -> Vec<&'static str> {
        litellm_auth_gcp::secret_names()
            .iter()
            .copied()
            .chain([VERTEXAI_API_BASE_ENV])
            .collect()
    }

    /// `get_complete_vertex_url` for the Claude partner, with `VERTEXAI_API_BASE` as the
    /// fallback base `completion` reads.
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
        let api_base = api_base
            .map(str::to_owned)
            .or_else(|| env_lookup(VERTEXAI_API_BASE_ENV))
            .filter(|base| !base.trim().is_empty());
        complete_vertex_anthropic_url(api_base.as_deref(), &project, &location, model, false)
    }

    fn transform_request(
        &self,
        model: &str,
        messages: Vec<ChatMessage>,
        optional_params: Map<String, Value>,
        context: &ChatRequestContext<'_>,
    ) -> Result<ProviderChatRequestData, Error> {
        self.build_request(model, messages, optional_params, context)
    }

    /// `transform_response` names the requested model, not the one Vertex reports.
    fn transform_response(
        &self,
        model: &str,
        response: ProviderChatResponseData,
        shape: &StreamShape,
    ) -> Result<ChatCompletionsResponse, Error> {
        let mut body = response.body;
        if let Value::Object(fields) = &mut body {
            fields.insert("model".into(), json!(model));
        }
        transform_messages_response(body, shape)
    }

    /// `inline_remote_image_urls`: Vertex takes no image by URL, so every remote image is
    /// downloaded.
    fn remote_media(&self, messages: &[ChatMessage]) -> Vec<String> {
        remote_urls(messages, &["user"], false, |url| {
            (url.starts_with("http://") || url.starts_with("https://")) && data_url(url).is_none()
        })
    }

    /// Python's partner handler sets `Authorization: Bearer <token>` over whatever was
    /// forwarded, so the Google token is always the credential. A forwarded `anthropic-beta`
    /// keeps only what Vertex accepts.
    fn validate_environment(
        &self,
        headers: Headers,
        _api_key: Option<&str>,
        _model: &str,
        optional_params: &Map<String, Value>,
        _env_lookup: &dyn Fn(&str) -> Option<String>,
    ) -> Result<ValidatedEnvironment, Error> {
        Ok(ValidatedEnvironment {
            headers: BetaPolicy::Filter(BetaProvider::VertexAi).apply(headers),
            auth: AuthScheme::GcpAccessToken {
                config: Box::new(vertex_config(optional_params)?),
            },
        })
    }

    fn model_response_iterator(&self, shape: StreamShape) -> Option<ChatStream> {
        Some(ChatStream::new(
            messages_event_stream,
            ModelResponseIterator::new(shape),
        ))
    }

    /// `streamRawPredict` with `stream: true` in the body, after the other params.
    fn streaming_request(&self, url: String, mut body: Value) -> Result<(String, Value), Error> {
        if let Value::Object(fields) = &mut body {
            fields.insert(STREAM_PARAM.into(), Value::Bool(true));
        }
        Ok((streaming_url(&url), body))
    }

    fn default_headers(&self) -> &'static [(&'static str, &'static str)] {
        DEFAULT_HEADERS
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

    const SONNET: &str = "claude-sonnet-4-5@20250929";
    const OPUS: &str = "claude-opus-4-7";

    fn cost_map() -> CostMap {
        let entry = |value: Value| value.as_object().cloned().unwrap();
        CostMap(HashMap::from([
            (
                format!("vertex_ai/{SONNET}"),
                entry(json!({"max_output_tokens": 64000, "supports_reasoning": true})),
            ),
            (
                "claude-sonnet-4-5".to_owned(),
                entry(
                    json!({"max_output_tokens": 64000, "supports_native_structured_output": true}),
                ),
            ),
            (
                format!("vertex_ai/{OPUS}"),
                entry(json!({"supports_reasoning": true, "supports_adaptive_thinking": true})),
            ),
            (
                OPUS.to_owned(),
                entry(json!({"supports_output_config": true, "supports_speed": true})),
            ),
            (
                "vertex_ai/claude-haiku-4-5@20251001".to_owned(),
                entry(json!({"supports_reasoning": true})),
            ),
        ]))
    }

    fn object(value: Value) -> Map<String, Value> {
        value.as_object().cloned().expect("an object")
    }

    fn request_with(
        model: &str,
        messages: Value,
        params: Value,
        drop_params: bool,
    ) -> Result<ProviderChatRequestData, Error> {
        let info = cost_map();
        let context = ChatRequestContext {
            drop_params: Some(drop_params),
            model_info: Some(&info),
        };
        VERTEX_ANTHROPIC_CHAT_COMPLETIONS_CONFIG.transform_request(
            model,
            serde_json::from_value(messages).unwrap(),
            object(params),
            &context,
        )
    }

    fn body(model: &str, params: Value) -> Value {
        request_with(model, hi(), params, false)
            .expect("request transforms")
            .body
    }

    fn hi() -> Value {
        json!([{"role": "system", "content": "be terse"}, {"role": "user", "content": "hi"}])
    }

    fn no_env(_: &str) -> Option<String> {
        None
    }

    #[rstest]
    #[case::versioned("claude-sonnet-4-5@20250929", true)]
    #[case::bare("claude-opus-4-7", true)]
    #[case::gemini("gemini-2.5-pro", false)]
    #[case::llama("meta/llama-3.1-405b-instruct-maas", false)]
    #[case::mistral("mistral-large@2407", false)]
    #[case::not_a_partner_prefix("my-claude", false)]
    fn only_claude_partner_models_route_here(#[case] model: &str, #[case] expected: bool) {
        assert_eq!(is_vertex_claude_model(model), expected);
    }

    #[rstest]
    fn builds_the_body_python_builds() {
        let body = body(
            SONNET,
            json!({
                "max_tokens": 128,
                "temperature": 0.2,
                "stop": "END",
                "vertex_project": "p",
                "vertex_location": "us-east5",
                "vertex_credentials": "{}"
            }),
        );
        assert_eq!(
            body,
            json!({
                "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
                "max_tokens": 128,
                "temperature": 0.2,
                "stop_sequences": ["END"],
                "anthropic_version": "vertex-2023-10-16",
                "system": [{"type": "text", "text": "be terse"}],
            })
        );
    }

    #[rstest]
    fn max_tokens_defaults_from_the_bare_entry_like_python() {
        // `get_max_tokens` reads the bare id, which the cost map has no entry for.
        assert_eq!(body(SONNET, json!({}))["max_tokens"], json!(4096));
    }

    #[rstest]
    fn reasoning_effort_becomes_a_thinking_budget() {
        let body = body(SONNET, json!({"reasoning_effort": "low"}));
        assert_eq!(
            body["thinking"],
            json!({"type": "enabled", "budget_tokens": 1024})
        );
        assert_eq!(body["max_tokens"], json!(1024 + 4096));
    }

    #[rstest]
    fn an_adaptive_model_gets_adaptive_thinking_and_its_effort() {
        let body = body(OPUS, json!({"reasoning_effort": "high", "max_tokens": 64}));
        assert_eq!(
            body["thinking"],
            json!({"type": "adaptive", "display": "summarized"})
        );
        assert_eq!(body["output_config"], json!({"effort": "high"}));
    }

    #[rstest]
    fn effort_is_dropped_for_a_model_that_takes_none() {
        let body = body(
            "claude-haiku-4-5@20251001",
            json!({"output_config": {"effort": "low"}, "max_tokens": 64}),
        );
        assert!(body.get("output_config").is_none(), "{body}");
    }

    #[rstest]
    fn native_structured_output_is_read_from_the_vertex_entry_only() {
        // The bare Anthropic entry supports it; the Vertex one does not say, so Python uses
        // the forced JSON tool.
        let request = request_with(
            SONNET,
            hi(),
            json!({"response_format": {"type": "json_schema", "json_schema": {"name": "x", "schema": {"type": "object", "properties": {"a": {"type": "string"}}}}}, "max_tokens": 64}),
            false,
        )
        .unwrap();
        assert!(request.body.get("output_config").is_none());
        assert!(request.body.get("output_format").is_none());
        assert_eq!(
            request.body["tool_choice"],
            json!({"name": "json_tool_call", "type": "tool"})
        );
        assert!(request.stream_shape.json_mode);
    }

    #[rstest]
    #[case::service_tier(json!({"service_tier": "auto"}))]
    #[case::speed(json!({"speed": "fast"}))]
    fn anthropic_api_only_params_are_refused_or_dropped(#[case] params: Value) {
        assert!(matches!(
            request_with(OPUS, hi(), params.clone(), false),
            Err(Error::InvalidRequest(_))
        ));
        let body = request_with(OPUS, hi(), params, true).unwrap().body;
        assert!(body.get("service_tier").is_none() && body.get("speed").is_none());
    }

    #[rstest]
    fn context_management_names_its_betas_in_the_body_and_the_header() {
        let request = request_with(
            SONNET,
            hi(),
            json!({
                "context_management": {"edits": [{"type": "clear_tool_uses_20250919"}, {"type": "compact_20260112"}]},
                "extra_headers": {"anthropic-beta": "advanced-tool-use-2025-11-20, oauth-2025-04-20"},
                "max_tokens": 64
            }),
            false,
        )
        .unwrap();
        assert_eq!(
            request.body["anthropic_beta"],
            json!([
                "compact-2026-01-12",
                "context-management-2025-06-27",
                "tool-search-tool-2025-10-19"
            ])
        );
        assert_eq!(
            request.headers,
            [(
                "anthropic-beta".to_owned(),
                "compact-2026-01-12,context-management-2025-06-27,tool-search-tool-2025-10-19"
                    .to_owned()
            )]
        );
        assert!(request.body.get("extra_headers").is_none());
    }

    #[rstest]
    fn web_search_alone_sends_only_its_header() {
        let request = request_with(
            SONNET,
            hi(),
            json!({"web_search_options": {}, "max_tokens": 64}),
            false,
        )
        .unwrap();
        assert!(request.body.get("anthropic_beta").is_none());
        assert_eq!(
            request.headers,
            [(
                "anthropic-beta".to_owned(),
                "web-search-2025-03-05".to_owned()
            )]
        );
    }

    #[rstest]
    fn without_betas_no_header_or_body_field() {
        let request = request_with(SONNET, hi(), json!({"max_tokens": 8}), false).unwrap();
        assert!(request.headers.is_empty());
        assert!(request.body.get("anthropic_beta").is_none());
    }

    #[rstest]
    fn tools_round_trip_into_anthropic_shape() {
        let body = request_with(
            SONNET,
            json!([
                {"role": "user", "content": "weather?"},
                {"role": "assistant", "content": null, "tool_calls": [{"id": "toolu_1", "type": "function", "function": {"name": "get_weather", "arguments": "{\"city\": \"Paris\"}"}}]},
                {"role": "tool", "tool_call_id": "toolu_1", "content": "sunny"}
            ]),
            json!({"tools": [{"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}}], "max_tokens": 64}),
            false,
        )
        .unwrap()
        .body;
        assert_eq!(
            body["tools"],
            json!([{"name": "get_weather", "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}}, "type": "custom"}])
        );
        assert_eq!(
            body["messages"][1]["content"][0],
            json!({"type": "tool_use", "id": "toolu_1", "name": "get_weather", "input": {"city": "Paris"}})
        );
        assert_eq!(body["messages"][2]["content"][0]["type"], "tool_result");
    }

    fn url(api_base: Option<&str>, location: &str) -> String {
        VERTEX_ANTHROPIC_CHAT_COMPLETIONS_CONFIG
            .get_complete_url(
                api_base,
                SONNET,
                &object(json!({"vertex_project": "proj", "vertex_location": location})),
                &no_env,
            )
            .unwrap()
    }

    const PATH: &str = "/v1/projects/proj/locations/{location}/publishers/anthropic/models/claude-sonnet-4-5@20250929:rawPredict";

    #[rstest]
    #[case::regional(None, "us-east5", "https://us-east5-aiplatform.googleapis.com")]
    #[case::global(None, "global", "https://aiplatform.googleapis.com")]
    #[case::custom_host(Some("http://127.0.0.1:9/"), "us-east5", "http://127.0.0.1:9")]
    fn the_url_names_the_project_location_and_model(
        #[case] api_base: Option<&str>,
        #[case] location: &str,
        #[case] host: &str,
    ) {
        assert_eq!(
            url(api_base, location),
            format!("{host}{}", PATH.replace("{location}", location))
        );
    }

    #[rstest]
    fn the_location_defaults_to_us_central1() {
        let url = VERTEX_ANTHROPIC_CHAT_COMPLETIONS_CONFIG
            .get_complete_url(
                None,
                SONNET,
                &object(json!({"vertex_project": "proj"})),
                &no_env,
            )
            .unwrap();
        assert!(
            url.starts_with("https://us-central1-aiplatform.googleapis.com/v1/projects/proj/locations/us-central1/"),
            "{url}"
        );
    }

    #[rstest]
    #[case::regional(
        "https://us-east5-aiplatform.googleapis.com/v1/x/models/m:rawPredict",
        "https://us-east5-aiplatform.googleapis.com/v1/x/models/m:streamRawPredict"
    )]
    #[case::global(
        "https://aiplatform.googleapis.com/v1/x/models/m:rawPredict",
        "https://aiplatform.googleapis.com/v1/x/models/m:streamRawPredict"
    )]
    #[case::custom_base(
        "http://127.0.0.1:9/v1/x/models/m:rawPredict",
        "http://127.0.0.1:9/v1/x/models/m:streamRawPredict?alt=sse"
    )]
    #[case::custom_path(
        "https://proxy.example/route:rawPredict",
        "https://proxy.example/route:streamRawPredict?alt=sse"
    )]
    fn a_stream_calls_stream_raw_predict(#[case] url: &str, #[case] expected: &str) {
        let (streamed, body) = VERTEX_ANTHROPIC_CHAT_COMPLETIONS_CONFIG
            .streaming_request(url.to_owned(), json!({"max_tokens": 1}))
            .unwrap();
        assert_eq!(streamed, expected);
        assert_eq!(body, json!({"max_tokens": 1, "stream": true}));
    }

    #[rstest]
    fn the_google_token_replaces_a_forwarded_bearer_and_betas_are_filtered() {
        let validated = VERTEX_ANTHROPIC_CHAT_COMPLETIONS_CONFIG
            .validate_environment(
                vec![
                    ("Authorization".to_owned(), "Bearer caller".to_owned()),
                    (
                        "anthropic-beta".to_owned(),
                        "oauth-2025-04-20,context-management-2025-06-27".to_owned(),
                    ),
                ],
                None,
                SONNET,
                &Map::new(),
                &no_env,
            )
            .unwrap();
        assert!(matches!(validated.auth, AuthScheme::GcpAccessToken { .. }));
        assert!(validated.headers.contains(&(
            "anthropic-beta".to_owned(),
            "context-management-2025-06-27".to_owned()
        )));
    }

    #[rstest]
    fn the_response_names_the_requested_model() {
        let response = VERTEX_ANTHROPIC_CHAT_COMPLETIONS_CONFIG
            .transform_response(
                SONNET,
                ProviderChatResponseData {
                    body: json!({
                        "id": "msg_1", "type": "message", "role": "assistant",
                        "content": [
                            {"type": "thinking", "thinking": "hmm", "signature": "sig"},
                            {"type": "text", "text": "hello"}
                        ],
                        "stop_reason": "end_turn",
                        "usage": {"input_tokens": 11, "output_tokens": 4, "cache_read_input_tokens": 2}
                    }),
                },
                &StreamShape::default(),
            )
            .unwrap();
        assert_eq!(response.model, SONNET);
        let message = &response.choices[0].message;
        assert_eq!(message.content.as_deref(), Some("hello"));
        assert_eq!(message.reasoning_content.as_deref(), Some("hmm"));
        assert_eq!(response.choices[0].finish_reason, "stop");
        let usage = &response.usage;
        assert_eq!((usage.prompt_tokens, usage.completion_tokens), (13, 4));
    }

    #[rstest]
    fn every_remote_image_is_downloaded() {
        let messages: Vec<ChatMessage> =
            serde_json::from_value(json!([{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": "https://example.com/a.png"}},
                {"type": "image_url", "image_url": {"url": "http://example.com/b.png"}},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}
            ]}]))
            .unwrap();
        assert_eq!(
            VERTEX_ANTHROPIC_CHAT_COMPLETIONS_CONFIG.remote_media(&messages),
            ["https://example.com/a.png", "http://example.com/b.png"]
        );
    }
}
