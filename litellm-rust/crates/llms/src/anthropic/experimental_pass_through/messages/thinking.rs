//! Reasoning parameter translation for the Messages API: OpenAI-style `reasoning_effort`,
//! the 4.6+ adaptive interface and the legacy `budget_tokens` interface, reshaped to what
//! the target model accepts. Mirrors the thinking steps of Python's
//! `AnthropicMessagesConfig.transform_anthropic_messages_request`.

use litellm_core_utils::settings::Lookup;
use litellm_types::llms::anthropic_messages::anthropic_request::AnthropicMessagesRequest;
use serde_json::{Map, Value, json};

use crate::{
    anthropic::common_utils::AnthropicModelCapabilities, base_llm::chat::transformation::Error,
};

pub const ANTHROPIC_MIN_THINKING_BUDGET_TOKENS: u64 = 1024;

const EFFORT_NAMES: &str = "'minimal', 'low', 'medium', 'high', 'xhigh', 'max', 'none'";

/// The `budget_tokens` each `reasoning_effort` tier maps to. Every value honors the same
/// `DEFAULT_REASONING_EFFORT_<TIER>_THINKING_BUDGET` environment override Python reads.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ThinkingBudgets {
    pub minimal: u64,
    pub low: u64,
    pub medium: u64,
    pub high: u64,
    pub xhigh: u64,
    pub max: u64,
}

impl Default for ThinkingBudgets {
    fn default() -> Self {
        Self {
            minimal: 128,
            low: 1024,
            medium: 2048,
            high: 4096,
            xhigh: 8192,
            max: 16384,
        }
    }
}

impl ThinkingBudgets {
    pub fn from_lookup(env: &impl Lookup) -> Self {
        let defaults = Self::default();
        let tier = |name: &str, default: u64| {
            env.parsed::<u64>(&format!("DEFAULT_REASONING_EFFORT_{name}_THINKING_BUDGET"))
                .unwrap_or(default)
        };
        Self {
            minimal: tier("MINIMAL", defaults.minimal),
            low: tier("LOW", defaults.low),
            medium: tier("MEDIUM", defaults.medium),
            high: tier("HIGH", defaults.high),
            xhigh: tier("XHIGH", defaults.xhigh),
            max: tier("MAX", defaults.max),
        }
    }

    fn for_effort(&self, reasoning_effort: &str) -> Option<u64> {
        match reasoning_effort {
            "low" => Some(self.low),
            "medium" => Some(self.medium),
            "high" => Some(self.high),
            "xhigh" => Some(self.xhigh),
            "max" => Some(self.max),
            "minimal" => Some(self.minimal.max(ANTHROPIC_MIN_THINKING_BUDGET_TOKENS)),
            _ => None,
        }
    }

    /// Python's `_legacy_budget_to_effort`: the effort tier a legacy budget stands for.
    fn effort_for_budget(
        &self,
        budget_tokens: u64,
        capabilities: &AnthropicModelCapabilities,
    ) -> &'static str {
        if budget_tokens >= self.xhigh && capabilities.effort_tiers.xhigh {
            return "xhigh";
        }
        if budget_tokens >= self.high {
            return "high";
        }
        if budget_tokens >= self.medium {
            return "medium";
        }
        "low"
    }
}

/// Everything the thinking translation needs to know besides the request itself.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ThinkingContext {
    pub capabilities: AnthropicModelCapabilities,
    pub budgets: ThinkingBudgets,
}

fn bad_request(message: String) -> Error {
    Error::InvalidRequest(message)
}

fn thinking_type(thinking: Option<&Value>) -> Option<&str> {
    thinking?.get("type")?.as_str()
}

fn output_config_effort(output_config: Option<&Value>) -> Option<&str> {
    output_config?.get("effort")?.as_str()
}

fn enabled_thinking(budget_tokens: u64) -> Value {
    json!({"type": "enabled", "budget_tokens": budget_tokens})
}

/// Python's `AnthropicConfig._map_reasoning_effort`: `Ok(None)` for `none`, the adaptive
/// shape for adaptive models, else a legacy budget.
fn map_reasoning_effort(
    reasoning_effort: &str,
    context: &ThinkingContext,
) -> Result<Option<Value>, Error> {
    if reasoning_effort == "none" {
        return Ok(None);
    }
    if context.capabilities.supports_adaptive_thinking {
        return Ok(Some(json!({"type": "adaptive", "display": "summarized"})));
    }
    context
        .budgets
        .for_effort(reasoning_effort)
        .map(|budget| Some(enabled_thinking(budget)))
        .ok_or_else(|| {
            bad_request(format!(
                "Unmapped reasoning effort: {reasoning_effort:?}. Must be one of: {EFFORT_NAMES}."
            ))
        })
}

/// Cap a legacy `budget_tokens` below `max_tokens`; `None` when even the minimum budget
/// cannot fit and thinking should be dropped.
fn cap_thinking_budget_to_max_tokens(thinking: Value, max_tokens: Option<u64>) -> Option<Value> {
    let (Some(max_tokens), Some(budget)) = (
        max_tokens,
        thinking.get("budget_tokens").and_then(Value::as_u64),
    ) else {
        return Some(thinking);
    };
    if max_tokens <= ANTHROPIC_MIN_THINKING_BUDGET_TOKENS {
        return None;
    }
    if budget < max_tokens {
        return Some(thinking);
    }
    let thinking_type = thinking_type(Some(&thinking)).unwrap_or("enabled");
    Some(json!({"type": thinking_type, "budget_tokens": max_tokens - 1}))
}

fn reasoning_effort_to_output_config_effort(reasoning_effort: &str) -> Option<&'static str> {
    match reasoning_effort {
        "low" | "minimal" => Some("low"),
        "medium" => Some("medium"),
        "high" => Some("high"),
        "xhigh" => Some("xhigh"),
        "max" => Some("max"),
        _ => None,
    }
}

fn with_effort(output_config: Option<Value>, effort: &str, caller_wins: bool) -> Value {
    let mut config = match output_config {
        Some(Value::Object(config)) => config,
        _ => Map::new(),
    };
    if !caller_wins || !config.contains_key("effort") {
        config.insert("effort".to_string(), Value::String(effort.to_string()));
    }
    Value::Object(config)
}

/// `reasoning_effort` becomes native `thinking` (and `output_config.effort` on adaptive
/// models). Caller-supplied `thinking` / `output_config` win; `none` clears both.
fn translate_reasoning_effort(
    request: AnthropicMessagesRequest,
    context: &ThinkingContext,
) -> Result<AnthropicMessagesRequest, Error> {
    let Some(reasoning_effort) = request.reasoning_effort.clone() else {
        return Ok(request);
    };
    let request = AnthropicMessagesRequest {
        reasoning_effort: None,
        ..request
    };
    let Some(mapped) = map_reasoning_effort(&reasoning_effort, context)? else {
        return Ok(AnthropicMessagesRequest {
            thinking: None,
            output_config: None,
            ..request
        });
    };
    let Some(fitted) = cap_thinking_budget_to_max_tokens(mapped, request.max_tokens) else {
        return Ok(request);
    };
    let thinking = Some(request.thinking.clone().unwrap_or(fitted));
    if !context.capabilities.supports_adaptive_thinking {
        return Ok(AnthropicMessagesRequest {
            thinking,
            ..request
        });
    }
    let effort = reasoning_effort_to_output_config_effort(&reasoning_effort).ok_or_else(|| {
        bad_request(format!(
            "Invalid reasoning_effort: {reasoning_effort:?}. Must be one of: {EFFORT_NAMES}"
        ))
    })?;
    if let Some(rejection) = context
        .capabilities
        .effort_level_rejection(effort, &request.model)
    {
        return Err(bad_request(rejection));
    }
    Ok(AnthropicMessagesRequest {
        thinking,
        output_config: Some(with_effort(request.output_config.clone(), effort, true)),
        ..request
    })
}

/// Always-on-thinking models 400 on `thinking.type=disabled`; omit it instead.
fn drop_disabled_thinking(
    request: AnthropicMessagesRequest,
    context: &ThinkingContext,
) -> AnthropicMessagesRequest {
    if !context.capabilities.thinking_always_on
        || thinking_type(request.thinking.as_ref()) != Some("disabled")
    {
        return request;
    }
    AnthropicMessagesRequest {
        thinking: None,
        ..request
    }
}

/// Adaptive models that reject the legacy shape get `thinking.type=enabled` translated to
/// adaptive plus an `output_config.effort` bucketed from the budget.
fn translate_legacy_thinking_for_adaptive_model(
    request: AnthropicMessagesRequest,
    context: &ThinkingContext,
) -> AnthropicMessagesRequest {
    let capabilities = &context.capabilities;
    if !capabilities.supports_adaptive_thinking
        || capabilities.supports_legacy_thinking
        || thinking_type(request.thinking.as_ref()) != Some("enabled")
    {
        return request;
    }
    let budget = request
        .thinking
        .as_ref()
        .and_then(|thinking| thinking.get("budget_tokens"))
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let effort = context.budgets.effort_for_budget(budget, capabilities);
    AnthropicMessagesRequest {
        thinking: Some(json!({"type": "adaptive"})),
        output_config: Some(with_effort(request.output_config.clone(), effort, true)),
        ..request
    }
}

fn output_config_without_effort(output_config: Option<Value>) -> Option<Value> {
    let Some(Value::Object(config)) = output_config else {
        return output_config;
    };
    if !config.contains_key("effort") {
        return Some(Value::Object(config));
    }
    let residual: Map<String, Value> = config
        .into_iter()
        .filter(|(key, _)| key != "effort")
        .collect();
    (!residual.is_empty()).then_some(Value::Object(residual))
}

/// The 4.6+ adaptive interface (`thinking.type=adaptive`, `output_config.effort`) reshaped
/// for an older model: native effort is kept where the model takes it, otherwise the effort
/// becomes a capped legacy budget, or thinking is dropped for models without reasoning.
fn translate_adaptive_effort_for_non_adaptive_model(
    request: AnthropicMessagesRequest,
    context: &ThinkingContext,
) -> Result<AnthropicMessagesRequest, Error> {
    let capabilities = &context.capabilities;
    if capabilities.supports_adaptive_thinking {
        return Ok(request);
    }
    let effort = output_config_effort(request.output_config.as_ref()).map(str::to_string);
    let adaptive_thinking = thinking_type(request.thinking.as_ref()) == Some("adaptive");
    if effort.is_none() && !adaptive_thinking {
        return Ok(request);
    }
    let level_supported = effort.as_deref().is_none_or(|effort| {
        capabilities
            .effort_level_rejection(effort, &request.model)
            .is_none()
    });
    if capabilities.supports_effort_param() && (!adaptive_thinking || level_supported) {
        return Ok(AnthropicMessagesRequest {
            thinking: if adaptive_thinking {
                None
            } else {
                request.thinking.clone()
            },
            ..request
        });
    }
    let legacy = if capabilities.supports_reasoning {
        map_reasoning_effort(effort.as_deref().unwrap_or("medium"), context)?
    } else {
        None
    };
    let capped =
        legacy.and_then(|thinking| cap_thinking_budget_to_max_tokens(thinking, request.max_tokens));
    Ok(AnthropicMessagesRequest {
        thinking: capped,
        output_config: output_config_without_effort(request.output_config.clone()),
        ..request
    })
}

/// Anthropic only accepts `temperature=1` while extended thinking is on, so a pinned
/// temperature is dropped on non-adaptive models once thinking or effort is in play.
fn drop_incompatible_temperature_for_thinking(
    request: AnthropicMessagesRequest,
    context: &ThinkingContext,
) -> AnthropicMessagesRequest {
    if context.capabilities.supports_adaptive_thinking {
        return request;
    }
    let pinned = request
        .temperature
        .is_some_and(|temperature| temperature != 1.0);
    let thinking_enabled = thinking_type(request.thinking.as_ref()) == Some("enabled");
    let effort_enabled = output_config_effort(request.output_config.as_ref()).is_some();
    if !pinned || !(thinking_enabled || effort_enabled) {
        return request;
    }
    AnthropicMessagesRequest {
        temperature: None,
        ..request
    }
}

/// The full reasoning translation, in Python's order.
pub fn translate_thinking(
    request: AnthropicMessagesRequest,
    context: &ThinkingContext,
) -> Result<AnthropicMessagesRequest, Error> {
    let request = translate_reasoning_effort(request, context)?;
    let request = drop_disabled_thinking(request, context);
    let request = translate_legacy_thinking_for_adaptive_model(request, context);
    let request = translate_adaptive_effort_for_non_adaptive_model(request, context)?;
    Ok(drop_incompatible_temperature_for_thinking(request, context))
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;
    use crate::anthropic::common_utils::SupportedEffortTiers;

    fn request(fields: Value) -> AnthropicMessagesRequest {
        let mut body =
            json!({"model": "claude", "messages": [{"role": "user", "content": "Hello"}]});
        body.as_object_mut()
            .unwrap()
            .extend(fields.as_object().unwrap().clone());
        serde_json::from_value(body).unwrap()
    }

    fn body(request: &AnthropicMessagesRequest) -> Value {
        serde_json::to_value(request).unwrap()
    }

    fn context(capabilities: AnthropicModelCapabilities) -> ThinkingContext {
        ThinkingContext {
            capabilities,
            budgets: ThinkingBudgets::default(),
        }
    }

    fn haiku_4_5() -> AnthropicModelCapabilities {
        AnthropicModelCapabilities {
            supports_reasoning: true,
            ..Default::default()
        }
    }

    fn opus_4_5() -> AnthropicModelCapabilities {
        AnthropicModelCapabilities {
            supports_reasoning: true,
            supports_output_config: true,
            ..Default::default()
        }
    }

    fn sonnet_4_6() -> AnthropicModelCapabilities {
        AnthropicModelCapabilities {
            supports_reasoning: true,
            supports_adaptive_thinking: true,
            supports_legacy_thinking: true,
            supports_output_config: true,
            effort_tiers: SupportedEffortTiers {
                max: true,
                ..Default::default()
            },
            ..Default::default()
        }
    }

    fn opus_4_7() -> AnthropicModelCapabilities {
        AnthropicModelCapabilities {
            supports_reasoning: true,
            supports_adaptive_thinking: true,
            supports_output_config: true,
            effort_tiers: SupportedEffortTiers {
                xhigh: true,
                max: true,
                ..Default::default()
            },
            ..Default::default()
        }
    }

    fn fable_5_1() -> AnthropicModelCapabilities {
        AnthropicModelCapabilities {
            thinking_always_on: true,
            ..opus_4_7()
        }
    }

    fn claude_code_payload(effort: &str, max_tokens: u64) -> Value {
        json!({"max_tokens": max_tokens, "thinking": {"type": "adaptive"}, "output_config": {"effort": effort}})
    }

    #[rstest]
    #[case("minimal", "low")]
    #[case("low", "low")]
    #[case("medium", "medium")]
    #[case("high", "high")]
    #[case("xhigh", "xhigh")]
    #[case("max", "max")]
    fn reasoning_effort_maps_to_output_config_for_adaptive_model(
        #[case] reasoning_effort: &str,
        #[case] expected_effort: &str,
    ) {
        let result = translate_thinking(
            request(json!({"max_tokens": 1024, "reasoning_effort": reasoning_effort})),
            &context(opus_4_7()),
        )
        .unwrap();
        let sent = body(&result);
        assert!(sent.get("reasoning_effort").is_none());
        assert_eq!(
            sent["thinking"],
            json!({"type": "adaptive", "display": "summarized"})
        );
        assert_eq!(sent["output_config"], json!({"effort": expected_effort}));
    }

    #[test]
    fn reasoning_effort_none_clears_thinking_and_output_config() {
        let result = translate_thinking(
            request(json!({
                "max_tokens": 1024,
                "reasoning_effort": "none",
                "thinking": {"type": "adaptive"},
                "output_config": {"effort": "high"}
            })),
            &context(opus_4_7()),
        )
        .unwrap();
        let sent = body(&result);
        assert!(sent.get("thinking").is_none());
        assert!(sent.get("output_config").is_none());
        assert!(sent.get("reasoning_effort").is_none());
    }

    #[test]
    fn reasoning_effort_on_non_adaptive_model_uses_a_capped_budget() {
        let result = translate_thinking(
            request(json!({"max_tokens": 8192, "reasoning_effort": "high"})),
            &context(opus_4_5()),
        )
        .unwrap();
        assert_eq!(
            result.thinking,
            Some(json!({"type": "enabled", "budget_tokens": 4096}))
        );
        assert!(result.output_config.is_none());

        let capped = translate_thinking(
            request(json!({"max_tokens": 3000, "reasoning_effort": "high"})),
            &context(haiku_4_5()),
        )
        .unwrap();
        assert_eq!(
            capped.thinking,
            Some(json!({"type": "enabled", "budget_tokens": 2999}))
        );

        let dropped = translate_thinking(
            request(json!({"max_tokens": 512, "reasoning_effort": "high"})),
            &context(haiku_4_5()),
        )
        .unwrap();
        assert!(dropped.thinking.is_none());
    }

    #[rstest]
    #[case("bogus")]
    #[case("ultra")]
    fn invalid_reasoning_effort_is_a_request_error(#[case] bad_effort: &str) {
        let error = translate_thinking(
            request(json!({"max_tokens": 1024, "reasoning_effort": bad_effort})),
            &context(opus_4_5()),
        )
        .expect_err("rejected");
        assert!(
            matches!(error, Error::InvalidRequest(message) if message.contains("Unmapped reasoning effort"))
        );
    }

    #[test]
    fn reasoning_effort_unsupported_tier_is_a_request_error_on_adaptive_models() {
        let error = translate_thinking(
            request(json!({"max_tokens": 1024, "reasoning_effort": "xhigh"})),
            &context(sonnet_4_6()),
        )
        .expect_err("rejected");
        assert!(
            matches!(error, Error::InvalidRequest(message) if message.contains("effort='xhigh'"))
        );
        let accepted = translate_thinking(
            request(json!({"max_tokens": 1024, "reasoning_effort": "max"})),
            &context(sonnet_4_6()),
        )
        .unwrap();
        assert_eq!(accepted.output_config, Some(json!({"effort": "max"})));
    }

    #[test]
    fn explicit_thinking_and_output_config_win_over_reasoning_effort() {
        let result = translate_thinking(
            request(json!({
                "max_tokens": 16000,
                "reasoning_effort": "low",
                "thinking": {"type": "enabled", "budget_tokens": 8000},
                "output_config": {"effort": "high"}
            })),
            &context(sonnet_4_6()),
        )
        .unwrap();
        assert_eq!(
            result.thinking,
            Some(json!({"type": "enabled", "budget_tokens": 8000}))
        );
        assert_eq!(result.output_config, Some(json!({"effort": "high"})));
    }

    #[test]
    fn legacy_thinking_is_preserved_verbatim_on_4_6() {
        let result = translate_thinking(
            request(json!({"max_tokens": 32000, "thinking": {"type": "enabled", "budget_tokens": 31999}})),
            &context(sonnet_4_6()),
        )
        .unwrap();
        assert_eq!(
            result.thinking,
            Some(json!({"type": "enabled", "budget_tokens": 31999}))
        );
        assert!(result.output_config.is_none());
    }

    #[rstest]
    #[case(1000, "low")]
    #[case(2048, "medium")]
    #[case(4096, "high")]
    #[case(24000, "xhigh")]
    fn legacy_thinking_translates_to_adaptive_buckets_on_4_7(
        #[case] budget_tokens: u64,
        #[case] expected_effort: &str,
    ) {
        let result = translate_thinking(
            request(json!({"max_tokens": 32000, "thinking": {"type": "enabled", "budget_tokens": budget_tokens}})),
            &context(opus_4_7()),
        )
        .unwrap();
        assert_eq!(result.thinking, Some(json!({"type": "adaptive"})));
        assert_eq!(
            result.output_config,
            Some(json!({"effort": expected_effort}))
        );
    }

    #[test]
    fn legacy_thinking_does_not_override_explicit_output_config() {
        let result = translate_thinking(
            request(json!({
                "max_tokens": 32000,
                "thinking": {"type": "enabled", "budget_tokens": 31999},
                "output_config": {"effort": "low", "format": {"type": "json_schema"}}
            })),
            &context(opus_4_7()),
        )
        .unwrap();
        assert_eq!(
            result.output_config,
            Some(json!({"effort": "low", "format": {"type": "json_schema"}}))
        );
    }

    #[rstest]
    #[case(fable_5_1(), true)]
    #[case(opus_4_7(), false)]
    fn disabled_thinking_is_omitted_only_for_always_on_models(
        #[case] capabilities: AnthropicModelCapabilities,
        #[case] expected_dropped: bool,
    ) {
        let result = translate_thinking(
            request(json!({"max_tokens": 64, "thinking": {"type": "disabled"}})),
            &context(capabilities),
        )
        .unwrap();
        assert_eq!(result.thinking.is_none(), expected_dropped);
    }

    #[test]
    fn adaptive_payload_is_translated_to_legacy_for_haiku_4_5() {
        let result = translate_thinking(
            request(claude_code_payload("high", 8192)),
            &context(haiku_4_5()),
        )
        .unwrap();
        assert_eq!(
            result.thinking,
            Some(json!({"type": "enabled", "budget_tokens": 4096}))
        );
        assert!(result.output_config.is_none());
    }

    #[test]
    fn adaptive_thinking_only_is_translated_or_dropped_by_reasoning_support() {
        let translated = translate_thinking(
            request(json!({"max_tokens": 8192, "thinking": {"type": "adaptive"}})),
            &context(haiku_4_5()),
        )
        .unwrap();
        assert_eq!(
            translated.thinking,
            Some(json!({"type": "enabled", "budget_tokens": 2048}))
        );

        let dropped = translate_thinking(
            request(claude_code_payload("medium", 8192)),
            &context(AnthropicModelCapabilities::default()),
        )
        .unwrap();
        assert!(dropped.thinking.is_none());
        assert!(dropped.output_config.is_none());
    }

    #[test]
    fn adaptive_payload_passes_through_untouched_on_4_6() {
        let result = translate_thinking(
            request(claude_code_payload("medium", 8192)),
            &context(sonnet_4_6()),
        )
        .unwrap();
        assert_eq!(result.thinking, Some(json!({"type": "adaptive"})));
        assert_eq!(result.output_config, Some(json!({"effort": "medium"})));
    }

    #[test]
    fn residual_output_config_survives_effort_translation() {
        let result = translate_thinking(
            request(json!({
                "max_tokens": 8192,
                "thinking": {"type": "adaptive"},
                "output_config": {"effort": "medium", "format": {"type": "json_schema"}}
            })),
            &context(haiku_4_5()),
        )
        .unwrap();
        assert_eq!(
            result.output_config,
            Some(json!({"format": {"type": "json_schema"}}))
        );
    }

    #[test]
    fn opus_4_5_keeps_supported_effort_and_drops_adaptive_thinking() {
        let result = translate_thinking(
            request(claude_code_payload("high", 8192)),
            &context(opus_4_5()),
        )
        .unwrap();
        assert!(result.thinking.is_none());
        assert_eq!(result.output_config, Some(json!({"effort": "high"})));
    }

    #[test]
    fn opus_4_5_unsupported_effort_with_adaptive_thinking_falls_back_to_legacy() {
        let result = translate_thinking(
            request(claude_code_payload("xhigh", 64000)),
            &context(opus_4_5()),
        )
        .unwrap();
        assert_eq!(
            result.thinking,
            Some(json!({"type": "enabled", "budget_tokens": 8192}))
        );
        assert!(result.output_config.is_none());
    }

    #[test]
    fn opus_4_5_effort_only_request_is_left_untouched() {
        let result = translate_thinking(
            request(json!({"max_tokens": 4096, "output_config": {"effort": "xhigh"}})),
            &context(opus_4_5()),
        )
        .unwrap();
        assert!(result.thinking.is_none());
        assert_eq!(result.output_config, Some(json!({"effort": "xhigh"})));
    }

    #[test]
    fn budget_is_capped_below_max_tokens_and_dropped_when_it_cannot_fit() {
        let capped = translate_thinking(
            request(claude_code_payload("high", 3000)),
            &context(haiku_4_5()),
        )
        .unwrap();
        assert_eq!(
            capped.thinking,
            Some(json!({"type": "enabled", "budget_tokens": 2999}))
        );

        let dropped = translate_thinking(
            request(claude_code_payload("medium", 512)),
            &context(haiku_4_5()),
        )
        .unwrap();
        assert!(dropped.thinking.is_none());
        assert!(dropped.output_config.is_none());
    }

    #[rstest]
    #[case(haiku_4_5(), claude_code_payload("medium", 8192), 0.0, true)]
    #[case(haiku_4_5(), claude_code_payload("medium", 8192), 1.0, false)]
    #[case(haiku_4_5(), claude_code_payload("medium", 512), 0.0, false)]
    #[case(opus_4_7(), claude_code_payload("medium", 8192), 0.0, false)]
    #[case(opus_4_5(), claude_code_payload("high", 8192), 0.0, true)]
    #[case(haiku_4_5(), json!({"max_tokens": 8192, "reasoning_effort": "high"}), 0.2, true)]
    #[case(haiku_4_5(), json!({"max_tokens": 8192}), 0.0, false)]
    fn pinned_temperature_is_dropped_only_when_thinking_survives_on_a_non_adaptive_model(
        #[case] capabilities: AnthropicModelCapabilities,
        #[case] payload: Value,
        #[case] temperature: f64,
        #[case] expected_dropped: bool,
    ) {
        let mut payload = payload;
        payload
            .as_object_mut()
            .unwrap()
            .insert("temperature".to_string(), json!(temperature));
        let result = translate_thinking(request(payload), &context(capabilities)).unwrap();
        assert_eq!(result.temperature.is_none(), expected_dropped);
    }

    #[test]
    fn budgets_honor_environment_overrides() {
        let env = |name: &str| {
            (name == "DEFAULT_REASONING_EFFORT_HIGH_THINKING_BUDGET").then(|| " 6000 ".to_string())
        };
        let budgets = ThinkingBudgets::from_lookup(&env);
        assert_eq!(budgets.high, 6000);
        assert_eq!(budgets.medium, ThinkingBudgets::default().medium);
    }
}
