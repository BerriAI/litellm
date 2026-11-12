use litellm_core_utils::get_llm_provider_logic::LlmProviders;
use litellm_llms::ErrorDetail;
use litellm_llms::tencent::messages::transformation::TENCENT_MESSAGES_CONFIG;
use litellm_llms::{
    anthropic::messages::transformation::ANTHROPIC_MESSAGES_CONFIG,
    azure_ai::messages::transformation::AZURE_ANTHROPIC_MESSAGES_CONFIG,
    base_llm::messages::transformation::BaseMessagesConfig,
    bedrock::messages::invoke_transformations::anthropic_claude_transformation::BEDROCK_ANTHROPIC_MESSAGES_CONFIG,
    deepseek::messages::transformation::DEEPSEEK_ANTHROPIC_MESSAGES_CONFIG,
    edenai::messages::transformation::EDENAI_MESSAGES_CONFIG,
    minimax::messages::transformation::MINIMAX_MESSAGES_CONFIG,
    vertex_ai::messages::transformation::VERTEX_ANTHROPIC_MESSAGES_CONFIG,
};
use litellm_llms::{
    bedrock::claude_platform::messages::transformation::BEDROCK_CLAUDE_PLATFORM_MESSAGES_CONFIG,
    bedrock_mantle::messages::transformation::BEDROCK_MANTLE_MESSAGES_CONFIG,
};
use serde_json::{Map, Value};

use super::Error;

#[derive(Clone, Copy)]
pub(crate) struct MessagesProvider {
    provider: LlmProviders,
    config: &'static dyn BaseMessagesConfig,
}

impl MessagesProvider {
    pub(crate) fn as_str(self) -> &'static str {
        self.provider.into()
    }

    pub(crate) fn config(self) -> &'static dyn BaseMessagesConfig {
        self.config
    }

    /// Python's `MESSAGES_SERVICE_TIER_PROVIDERS`: everyone else has `service_tier` dropped.
    pub(crate) fn accepts_service_tier(self) -> bool {
        matches!(
            self.provider,
            LlmProviders::Anthropic | LlmProviders::AzureAi | LlmProviders::BedrockMantle
        )
    }
}

/// Python's `get_provider_anthropic_messages_config`: Vertex AI serves only its Claude
/// partner models on this route.
pub(crate) fn messages_provider(provider: LlmProviders, model: &str) -> Option<MessagesProvider> {
    use litellm_llms::openrouter::messages::transformation::OPENROUTER_MESSAGES_CONFIG;

    let config: &'static dyn BaseMessagesConfig = match provider {
        LlmProviders::Tencent => &TENCENT_MESSAGES_CONFIG,
        LlmProviders::Anthropic => &ANTHROPIC_MESSAGES_CONFIG,
        LlmProviders::AzureAi => &AZURE_ANTHROPIC_MESSAGES_CONFIG,
        LlmProviders::Bedrock if has_route_prefix(model, "claude_platform/") => {
            &BEDROCK_CLAUDE_PLATFORM_MESSAGES_CONFIG
        }
        LlmProviders::Bedrock if has_route_prefix(model, "converse/") => return None,
        LlmProviders::Bedrock if has_route_prefix(model, "mantle/") => {
            &BEDROCK_MANTLE_MESSAGES_CONFIG
        }
        LlmProviders::Bedrock if model.contains("claude") => &BEDROCK_ANTHROPIC_MESSAGES_CONFIG,
        LlmProviders::Bedrock => return None,
        LlmProviders::BedrockMantle => &BEDROCK_MANTLE_MESSAGES_CONFIG,
        LlmProviders::Deepseek => &DEEPSEEK_ANTHROPIC_MESSAGES_CONFIG,
        LlmProviders::Edenai => &EDENAI_MESSAGES_CONFIG,
        LlmProviders::Minimax => &MINIMAX_MESSAGES_CONFIG,
        LlmProviders::VertexAi if model.to_ascii_lowercase().contains("claude") => {
            &VERTEX_ANTHROPIC_MESSAGES_CONFIG
        }
        LlmProviders::Openrouter => &OPENROUTER_MESSAGES_CONFIG,
        _ => return None,
    };
    Some(MessagesProvider { provider, config })
}

fn has_route_prefix(model: &str, prefix: &str) -> bool {
    model.starts_with(prefix) || model.contains(&format!("/{prefix}"))
}

pub(super) fn string_headers(
    extra_headers: Option<Map<String, Value>>,
) -> Result<Vec<(String, String)>, Error> {
    extra_headers
        .unwrap_or_default()
        .into_iter()
        .map(|(name, value)| match value {
            Value::String(value) => Ok((name, value)),
            value => Err(Error::invalid_request(ErrorDetail::invalid(
                "extra_headers",
                NonStringHeader {
                    actual: python_type_name(&value),
                    name,
                },
            ))),
        })
        .collect()
}

/// A header value the caller set to something other than a string, named by its Python type
/// since the caller wrote it in Python.
#[derive(Debug)]
struct NonStringHeader {
    name: String,
    actual: &'static str,
}

impl std::fmt::Display for NonStringHeader {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "header {} must be a string, got {}",
            self.name, self.actual
        )
    }
}

impl std::error::Error for NonStringHeader {}

fn python_type_name(value: &Value) -> &'static str {
    match value {
        Value::Null => "NoneType",
        Value::Bool(_) => "bool",
        Value::Number(number) if number.is_f64() => "float",
        Value::Number(_) => "int",
        Value::String(_) => "str",
        Value::Array(_) => "list",
        Value::Object(_) => "dict",
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use rstest::rstest;

    use super::{messages_provider, string_headers};
    use litellm_core_utils::get_llm_provider_logic::LlmProviders;

    #[rstest]
    #[case::tencent(LlmProviders::Tencent, "tencent")]
    #[case::anthropic(LlmProviders::Anthropic, "anthropic")]
    #[case::azure_ai(LlmProviders::AzureAi, "azure_ai")]
    #[case::bedrock(LlmProviders::Bedrock, "bedrock")]
    #[case::deepseek(LlmProviders::Deepseek, "deepseek")]
    #[case::edenai(LlmProviders::Edenai, "edenai")]
    #[case::minimax(LlmProviders::Minimax, "minimax")]
    #[case::vertex_ai(LlmProviders::VertexAi, "vertex_ai")]
    #[case::openrouter(LlmProviders::Openrouter, "openrouter")]
    fn provider_keeps_its_python_name(#[case] provider: LlmProviders, #[case] name: &str) {
        let provider = messages_provider(provider, "claude-sonnet-4-5").unwrap();
        assert_eq!(provider.as_str(), name);
    }

    #[rstest]
    #[case::openai(LlmProviders::Openai, "gpt-5")]
    #[case::vertex_gemini(LlmProviders::VertexAi, "gemini-2.5-pro")]
    fn provider_without_a_messages_config_is_rejected(
        #[case] provider: LlmProviders,
        #[case] model: &str,
    ) {
        assert!(messages_provider(provider, model).is_none());
    }

    #[rstest]
    #[case::invoke_claude("anthropic.claude-sonnet-4-5-20250929-v1:0", Some("bedrock"))]
    #[case::invoke_prefix("invoke/anthropic.claude-3-5-sonnet", Some("bedrock"))]
    #[case::cross_region_claude("us.anthropic.claude-opus-4-5", Some("bedrock"))]
    #[case::claude_arn(
        "arn:aws:bedrock:us-east-1:123456789012:inference-profile/us.anthropic.claude-sonnet-4-5",
        Some("bedrock")
    )]
    #[case::converse_claude("converse/anthropic.claude-sonnet-4-5", None)]
    #[case::nested_converse("us-west-2/converse/anthropic.claude-sonnet-4-5", None)]
    #[case::non_claude("amazon.nova-pro-v1:0", None)]
    #[case::non_claude_invoke("invoke/meta.llama3-70b-instruct-v1:0", None)]
    #[case::capitalized_claude_is_not_claude("anthropic.Claude-x", None)]
    #[case::mantle("mantle/anthropic.claude-sonnet-5", Some("bedrock"))]
    #[case::claude_platform("claude_platform/claude-sonnet-4-5", Some("bedrock"))]
    #[case::bedrock_mantle_provider_prefix_is_not_the_mantle_route(
        "bedrock_mantle/openai.gpt-5",
        None
    )]
    fn bedrock_models_route_as_python_selects_their_messages_config(
        #[case] model: &str,
        #[case] expected: Option<&str>,
    ) {
        assert_eq!(
            messages_provider(LlmProviders::Bedrock, model).map(|provider| provider.as_str()),
            expected
        );
    }

    #[rstest]
    #[case::invoke("anthropic.claude-3", false, false)]
    #[case::mantle("mantle/anthropic.claude-sonnet-5", true, false)]
    #[case::claude_platform("claude_platform/claude-sonnet-4-5", false, true)]
    fn bedrock_route_prefixes_pick_their_config(
        #[case] model: &str,
        #[case] mantle: bool,
        #[case] claude_platform: bool,
    ) {
        use litellm_llms::{
            base_llm::messages::transformation::BaseMessagesConfig,
            bedrock::claude_platform::messages::transformation::BEDROCK_CLAUDE_PLATFORM_MESSAGES_CONFIG,
            bedrock_mantle::messages::transformation::BEDROCK_MANTLE_MESSAGES_CONFIG,
        };
        let params = litellm_router_types::LitellmParams::default();
        let region = |key: &str| {
            key.starts_with("AWS_REGION")
                .then(|| "us-east-1".to_owned())
        };
        let url = |config: &dyn BaseMessagesConfig| {
            format!(
                "{:?}",
                config.get_complete_url(None, model, &params, false, &region)
            )
        };
        let routed = url(messages_provider(LlmProviders::Bedrock, model)
            .unwrap()
            .config());
        assert_eq!(
            (
                routed == url(&BEDROCK_MANTLE_MESSAGES_CONFIG),
                routed == url(&BEDROCK_CLAUDE_PLATFORM_MESSAGES_CONFIG),
            ),
            (mantle, claude_platform)
        );
    }

    #[rstest]
    #[case::int(json!(12345), "int")]
    #[case::float(json!(1.5), "float")]
    #[case::bool(json!(true), "bool")]
    #[case::null(json!(null), "NoneType")]
    #[case::list(json!([1]), "list")]
    #[case::dict(json!({"a": 1}), "dict")]
    fn non_string_header_is_a_bad_request_naming_the_header_and_its_type(
        #[case] value: serde_json::Value,
        #[case] python_type: &str,
    ) {
        let headers = json!({"x-kept": "v", "x-count": value})
            .as_object()
            .unwrap()
            .clone();
        let error = string_headers(Some(headers)).expect_err("non-string header refused");
        assert!(error.is_request(), "{error:?}");
        assert_eq!(error.upstream_response(), None);
        let rejection = error.rejected_request().expect("a rejected request");
        assert_eq!(
            litellm_error::Report::new(rejection).to_string(),
            format!("invalid extra_headers: header x-count must be a string, got {python_type}")
        );
    }
}

/// Python's `_merge_messages_header_sources`: later sources win per header name, except
/// `anthropic-beta` (any casing), whose values from every source are unioned, deduplicated and
/// sorted into one lowercase `anthropic-beta` entry.
pub fn merge_messages_header_sources(
    sources: impl IntoIterator<Item = Map<String, Value>>,
) -> Map<String, Value> {
    let mut merged = Map::new();
    let mut betas = std::collections::BTreeSet::new();
    for (name, value) in sources.into_iter().flatten() {
        if !name.eq_ignore_ascii_case("anthropic-beta") {
            merged.insert(name, value);
            continue;
        }
        let values: Vec<&str> = match &value {
            Value::String(text) => vec![text.as_str()],
            Value::Array(entries) => entries.iter().filter_map(Value::as_str).collect(),
            _ => Vec::new(),
        };
        betas.extend(
            values
                .iter()
                .flat_map(|value| value.split(','))
                .map(str::trim)
                .filter(|beta| !beta.is_empty())
                .map(str::to_owned),
        );
    }
    if !betas.is_empty() {
        merged.insert(
            "anthropic-beta".to_owned(),
            Value::String(betas.into_iter().collect::<Vec<_>>().join(",")),
        );
    }
    merged
}

#[cfg(test)]
mod merge_tests {
    use serde_json::{Map, Value, json};

    use super::merge_messages_header_sources;

    fn map(value: Value) -> Map<String, Value> {
        value.as_object().cloned().unwrap()
    }

    #[test]
    fn later_sources_win_except_anthropic_beta_which_is_unioned_and_sorted() {
        let merged = merge_messages_header_sources([
            map(json!({"x-a": "1", "anthropic-beta": "b-2, a-1"})),
            map(json!({"x-a": "2", "Anthropic-Beta": ["c-3", 7, "a-1"]})),
            map(json!({"anthropic-beta": 5})),
        ]);
        assert_eq!(
            Value::Object(merged),
            json!({"x-a": "2", "anthropic-beta": "a-1,b-2,c-3"})
        );
    }

    #[test]
    fn no_beta_values_leave_no_beta_header() {
        let merged = merge_messages_header_sources([map(json!({"anthropic-beta": ""}))]);
        assert!(merged.is_empty());
    }
}
