use litellm_core_utils::settings::Lookup;
use litellm_types::llms::anthropic_messages::anthropic_request::AnthropicMessagesRequest;
use serde_json::{Map, Value, json};

use crate::{
    anthropic::common_utils::AnthropicModelCapabilities, base_llm::chat::transformation::Error,
};

pub const ANTHROPIC_MIN_THINKING_BUDGET_TOKENS: u64 = 1024;

const EFFORT_NAMES: &str = "'minimal', 'low', 'medium', 'high', 'xhigh', 'max', 'none'";

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
                "Unmapped reasoning effort: '{reasoning_effort}'. Must be one of: {EFFORT_NAMES}."
            ))
        })
}

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
    Some(enabled_thinking(max_tokens - 1))
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

fn with_default_effort(output_config: Option<Value>, effort: &str) -> Value {
    let mut config = match output_config {
        Some(Value::Object(config)) => config,
        _ => Map::new(),
    };
    if !config.contains_key("effort") {
        config.insert("effort".to_string(), Value::String(effort.to_string()));
    }
    Value::Object(config)
}

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
            "Invalid reasoning_effort: '{reasoning_effort}'. Must be one of: {EFFORT_NAMES}"
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
        output_config: Some(with_default_effort(request.output_config.clone(), effort)),
        ..request
    })
}

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
        output_config: Some(with_default_effort(request.output_config.clone(), effort)),
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
        map_reasoning_effort(
            effort
                .as_deref()
                .filter(|effort| !effort.is_empty())
                .unwrap_or("medium"),
            context,
        )?
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
    use rstest::{fixture, rstest};

    use super::*;
    use crate::anthropic::common_utils::SupportedEffortTiers;

    const EFFORT_CHOICES: &str = "'minimal', 'low', 'medium', 'high', 'xhigh', 'max', 'none'";

    fn request(fields: Value) -> AnthropicMessagesRequest {
        let mut body =
            json!({"model": "claude", "messages": [{"role": "user", "content": "Hello"}]});
        body.as_object_mut()
            .unwrap()
            .extend(fields.as_object().unwrap().clone());
        serde_json::from_value(body).unwrap()
    }

    fn context(capabilities: AnthropicModelCapabilities) -> ThinkingContext {
        ThinkingContext {
            capabilities,
            budgets: ThinkingBudgets::default(),
        }
    }

    fn translate(
        capabilities: AnthropicModelCapabilities,
        fields: Value,
    ) -> Result<AnthropicMessagesRequest, Error> {
        translate_thinking(request(fields), &context(capabilities))
    }

    fn overridden_budgets(overrides: &[(&str, &str)]) -> ThinkingBudgets {
        let env = |name: &str| {
            overrides
                .iter()
                .find(|(tier, _)| {
                    name == format!("DEFAULT_REASONING_EFFORT_{tier}_THINKING_BUDGET")
                })
                .map(|(_, value)| value.to_string())
        };
        ThinkingBudgets::from_lookup(&env)
    }

    fn claude_code_payload(effort: &str, max_tokens: u64) -> Value {
        json!({"max_tokens": max_tokens, "thinking": {"type": "adaptive"}, "output_config": {"effort": effort}})
    }

    fn with_temperature(fields: Value, temperature: f64) -> Value {
        let mut fields = fields;
        fields
            .as_object_mut()
            .unwrap()
            .insert("temperature".to_string(), json!(temperature));
        fields
    }

    #[fixture]
    fn haiku_3_5() -> AnthropicModelCapabilities {
        AnthropicModelCapabilities::default()
    }

    #[fixture]
    fn haiku_4_5() -> AnthropicModelCapabilities {
        AnthropicModelCapabilities {
            supports_reasoning: true,
            ..Default::default()
        }
    }

    #[fixture]
    fn opus_4_5() -> AnthropicModelCapabilities {
        AnthropicModelCapabilities {
            supports_reasoning: true,
            supports_output_config: true,
            ..Default::default()
        }
    }

    #[fixture]
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

    #[fixture]
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

    #[fixture]
    fn fable_5_1() -> AnthropicModelCapabilities {
        AnthropicModelCapabilities {
            thinking_always_on: true,
            ..opus_4_7()
        }
    }

    #[fixture]
    fn newfamily_6() -> AnthropicModelCapabilities {
        AnthropicModelCapabilities {
            supports_reasoning: true,
            supports_adaptive_thinking: true,
            ..Default::default()
        }
    }

    #[rstest]
    #[case::minimal_maps_to_low(opus_4_7(), "minimal", "low")]
    #[case::low(opus_4_7(), "low", "low")]
    #[case::medium(opus_4_7(), "medium", "medium")]
    #[case::high(opus_4_7(), "high", "high")]
    #[case::xhigh_with_xhigh_tier(opus_4_7(), "xhigh", "xhigh")]
    #[case::max(opus_4_7(), "max", "max")]
    #[case::minimal_maps_to_low_on_4_6(sonnet_4_6(), "minimal", "low")]
    #[case::low_on_4_6(sonnet_4_6(), "low", "low")]
    #[case::max_without_max_tier_is_allowed_on_adaptive_models(newfamily_6(), "max", "max")]
    fn reasoning_effort_on_adaptive_model_becomes_summarized_adaptive_thinking_and_effort(
        #[case] capabilities: AnthropicModelCapabilities,
        #[case] reasoning_effort: &str,
        #[case] expected_effort: &str,
    ) {
        assert_eq!(
            translate(
                capabilities,
                json!({"max_tokens": 1024, "reasoning_effort": reasoning_effort})
            ),
            Ok(request(json!({
                "max_tokens": 1024,
                "thinking": {"type": "adaptive", "display": "summarized"},
                "output_config": {"effort": expected_effort}
            })))
        );
    }

    #[rstest]
    #[case::adaptive_shape_is_not_dropped_for_small_max_tokens(
        opus_4_7(),
        json!({"max_tokens": 64, "reasoning_effort": "high"}),
        json!({"max_tokens": 64, "thinking": {"type": "adaptive", "display": "summarized"}, "output_config": {"effort": "high"}})
    )]
    #[case::caller_output_config_effort_wins(
        opus_4_7(),
        json!({"max_tokens": 1024, "reasoning_effort": "low", "output_config": {"effort": "max"}}),
        json!({"max_tokens": 1024, "thinking": {"type": "adaptive", "display": "summarized"}, "output_config": {"effort": "max"}})
    )]
    #[case::effort_merges_into_caller_output_config(
        opus_4_7(),
        json!({"max_tokens": 1024, "reasoning_effort": "high", "output_config": {"format": {"type": "json_schema"}}}),
        json!({
            "max_tokens": 1024,
            "thinking": {"type": "adaptive", "display": "summarized"},
            "output_config": {"format": {"type": "json_schema"}, "effort": "high"}
        })
    )]
    #[case::non_object_output_config_is_replaced(
        opus_4_7(),
        json!({"max_tokens": 1024, "reasoning_effort": "high", "output_config": "bogus"}),
        json!({"max_tokens": 1024, "thinking": {"type": "adaptive", "display": "summarized"}, "output_config": {"effort": "high"}})
    )]
    #[case::caller_thinking_and_output_config_win(
        sonnet_4_6(),
        json!({
            "max_tokens": 16000,
            "reasoning_effort": "low",
            "thinking": {"type": "enabled", "budget_tokens": 8000},
            "output_config": {"effort": "high"}
        }),
        json!({
            "max_tokens": 16000,
            "thinking": {"type": "enabled", "budget_tokens": 8000},
            "output_config": {"effort": "high"}
        })
    )]
    #[case::caller_legacy_thinking_is_then_translated_while_reasoning_effort_level_stays(
        opus_4_7(),
        json!({"max_tokens": 16000, "reasoning_effort": "low", "thinking": {"type": "enabled", "budget_tokens": 8000}}),
        json!({"max_tokens": 16000, "thinking": {"type": "adaptive"}, "output_config": {"effort": "low"}})
    )]
    #[case::caller_disabled_thinking_is_kept_then_omitted_on_always_on_model(
        fable_5_1(),
        json!({"max_tokens": 1024, "reasoning_effort": "high", "thinking": {"type": "disabled"}}),
        json!({"max_tokens": 1024, "output_config": {"effort": "high"}})
    )]
    #[case::non_adaptive_model_gets_no_output_config(
        opus_4_5(),
        json!({"max_tokens": 8192, "reasoning_effort": "high"}),
        json!({"max_tokens": 8192, "thinking": {"type": "enabled", "budget_tokens": 4096}})
    )]
    #[case::caller_thinking_wins_on_non_adaptive_model(
        opus_4_5(),
        json!({"max_tokens": 16000, "reasoning_effort": "low", "thinking": {"type": "enabled", "budget_tokens": 8000}}),
        json!({"max_tokens": 16000, "thinking": {"type": "enabled", "budget_tokens": 8000}})
    )]
    #[case::caller_thinking_survives_when_mapped_budget_cannot_fit(
        opus_4_5(),
        json!({"max_tokens": 1024, "reasoning_effort": "low", "thinking": {"type": "enabled", "budget_tokens": 8000}}),
        json!({"max_tokens": 1024, "thinking": {"type": "enabled", "budget_tokens": 8000}})
    )]
    #[case::missing_max_tokens_leaves_budget_uncapped(
        haiku_4_5(),
        json!({"reasoning_effort": "high"}),
        json!({"thinking": {"type": "enabled", "budget_tokens": 4096}})
    )]
    #[case::budget_below_max_tokens_is_kept(
        haiku_4_5(),
        json!({"max_tokens": 4097, "reasoning_effort": "high"}),
        json!({"max_tokens": 4097, "thinking": {"type": "enabled", "budget_tokens": 4096}})
    )]
    #[case::budget_equal_to_max_tokens_is_capped(
        haiku_4_5(),
        json!({"max_tokens": 4096, "reasoning_effort": "high"}),
        json!({"max_tokens": 4096, "thinking": {"type": "enabled", "budget_tokens": 4095}})
    )]
    #[case::budget_above_max_tokens_is_capped(
        haiku_4_5(),
        json!({"max_tokens": 4000, "reasoning_effort": "xhigh"}),
        json!({"max_tokens": 4000, "thinking": {"type": "enabled", "budget_tokens": 3999}})
    )]
    #[case::max_tokens_just_above_min_budget_caps_to_min_budget(
        haiku_4_5(),
        json!({"max_tokens": 1025, "reasoning_effort": "xhigh"}),
        json!({"max_tokens": 1025, "thinking": {"type": "enabled", "budget_tokens": 1024}})
    )]
    #[case::max_tokens_at_min_budget_drops_thinking(
        haiku_4_5(),
        json!({"max_tokens": 1024, "reasoning_effort": "xhigh"}),
        json!({"max_tokens": 1024})
    )]
    #[case::pinned_temperature_is_dropped_after_thinking_is_synthesized(
        haiku_4_5(),
        json!({"max_tokens": 8192, "reasoning_effort": "low", "temperature": 0}),
        json!({"max_tokens": 8192, "thinking": {"type": "enabled", "budget_tokens": 1024}})
    )]
    fn reasoning_effort_is_translated(
        #[case] capabilities: AnthropicModelCapabilities,
        #[case] input: Value,
        #[case] expected: Value,
    ) {
        assert_eq!(translate(capabilities, input), Ok(request(expected)));
    }

    #[rstest]
    #[case::minimal_floors_at_min_budget("minimal", 1024)]
    #[case::low("low", 1024)]
    #[case::medium("medium", 2048)]
    #[case::high("high", 4096)]
    #[case::xhigh("xhigh", 8192)]
    #[case::max("max", 16384)]
    fn reasoning_effort_on_non_adaptive_model_uses_the_tier_budget(
        haiku_4_5: AnthropicModelCapabilities,
        #[case] reasoning_effort: &str,
        #[case] expected_budget: u64,
    ) {
        assert_eq!(
            translate(
                haiku_4_5,
                json!({"max_tokens": 32000, "reasoning_effort": reasoning_effort})
            ),
            Ok(request(json!({
                "max_tokens": 32000,
                "thinking": {"type": "enabled", "budget_tokens": expected_budget}
            })))
        );
    }

    #[rstest]
    #[case::adaptive_model(opus_4_7())]
    #[case::effort_capable_model(opus_4_5())]
    #[case::budget_model(haiku_4_5())]
    fn reasoning_effort_none_clears_thinking_and_output_config(
        #[case] capabilities: AnthropicModelCapabilities,
    ) {
        assert_eq!(
            translate(
                capabilities,
                json!({
                    "max_tokens": 1024,
                    "reasoning_effort": "none",
                    "thinking": {"type": "adaptive"},
                    "output_config": {"effort": "high"}
                })
            ),
            Ok(request(json!({"max_tokens": 1024})))
        );
    }

    #[rstest]
    #[case::bogus_on_budget_model(
        opus_4_5(),
        json!({"max_tokens": 1024, "reasoning_effort": "bogus"}),
        format!("Unmapped reasoning effort: 'bogus'. Must be one of: {EFFORT_CHOICES}.")
    )]
    #[case::disabled_on_budget_model(
        haiku_4_5(),
        json!({"max_tokens": 1024, "reasoning_effort": "disabled"}),
        format!("Unmapped reasoning effort: 'disabled'. Must be one of: {EFFORT_CHOICES}.")
    )]
    #[case::empty_on_budget_model(
        haiku_4_5(),
        json!({"max_tokens": 1024, "reasoning_effort": ""}),
        format!("Unmapped reasoning effort: ''. Must be one of: {EFFORT_CHOICES}.")
    )]
    #[case::invalid_on_adaptive_model(
        opus_4_7(),
        json!({"max_tokens": 1024, "reasoning_effort": "invalid"}),
        format!("Invalid reasoning_effort: 'invalid'. Must be one of: {EFFORT_CHOICES}")
    )]
    #[case::disabled_on_adaptive_model(
        opus_4_7(),
        json!({"max_tokens": 1024, "reasoning_effort": "disabled"}),
        format!("Invalid reasoning_effort: 'disabled'. Must be one of: {EFFORT_CHOICES}")
    )]
    #[case::empty_on_adaptive_model(
        opus_4_7(),
        json!({"max_tokens": 1024, "reasoning_effort": ""}),
        format!("Invalid reasoning_effort: ''. Must be one of: {EFFORT_CHOICES}")
    )]
    #[case::xhigh_without_xhigh_tier_on_4_6(
        sonnet_4_6(),
        json!({"max_tokens": 1024, "reasoning_effort": "xhigh"}),
        "effort='xhigh' is not supported by this model. Got model: claude".to_string()
    )]
    #[case::xhigh_without_xhigh_tier_on_unmapped_adaptive_model(
        newfamily_6(),
        json!({"max_tokens": 1024, "reasoning_effort": "xhigh"}),
        "effort='xhigh' is not supported by this model. Got model: claude".to_string()
    )]
    #[case::unrecognized_adaptive_effort_on_budget_model(
        haiku_4_5(),
        claude_code_payload("turbo", 8192),
        format!("Unmapped reasoning effort: 'turbo'. Must be one of: {EFFORT_CHOICES}.")
    )]
    fn unsupported_effort_is_a_request_error(
        #[case] capabilities: AnthropicModelCapabilities,
        #[case] input: Value,
        #[case] expected_message: String,
    ) {
        assert_eq!(
            translate(capabilities, input),
            Err(Error::InvalidRequest(expected_message))
        );
    }

    #[rstest]
    #[case::omitted_on_always_on_model(fable_5_1(), json!({"type": "disabled"}), None)]
    #[case::kept_on_adaptive_model(opus_4_7(), json!({"type": "disabled"}), Some(json!({"type": "disabled"})))]
    #[case::kept_on_budget_model(haiku_4_5(), json!({"type": "disabled"}), Some(json!({"type": "disabled"})))]
    #[case::adaptive_kept_on_always_on_model(
        fable_5_1(),
        json!({"type": "adaptive"}),
        Some(json!({"type": "adaptive"}))
    )]
    fn disabled_thinking_is_omitted_only_for_always_on_models(
        #[case] capabilities: AnthropicModelCapabilities,
        #[case] thinking: Value,
        #[case] expected_thinking: Option<Value>,
    ) {
        let expected = match expected_thinking {
            Some(thinking) => json!({"max_tokens": 64, "thinking": thinking}),
            None => json!({"max_tokens": 64}),
        };
        assert_eq!(
            translate(
                capabilities,
                json!({"max_tokens": 64, "thinking": thinking})
            ),
            Ok(request(expected))
        );
    }

    #[rstest]
    #[case::far_above_xhigh_budget(opus_4_7(), json!(16384), "xhigh")]
    #[case::at_xhigh_budget(opus_4_7(), json!(8192), "xhigh")]
    #[case::below_xhigh_budget(opus_4_7(), json!(8191), "high")]
    #[case::xhigh_budget_without_xhigh_tier(newfamily_6(), json!(8192), "high")]
    #[case::large_budget_without_xhigh_tier(newfamily_6(), json!(31999), "high")]
    #[case::at_high_budget(opus_4_7(), json!(4096), "high")]
    #[case::below_high_budget(opus_4_7(), json!(4095), "medium")]
    #[case::at_medium_budget(opus_4_7(), json!(2048), "medium")]
    #[case::below_medium_budget(opus_4_7(), json!(2047), "low")]
    #[case::tiny_budget(opus_4_7(), json!(1), "low")]
    #[case::missing_budget(opus_4_7(), Value::Null, "low")]
    #[case::always_on_model(fable_5_1(), json!(24000), "xhigh")]
    fn legacy_thinking_is_bucketed_into_adaptive_effort_on_adaptive_only_models(
        #[case] capabilities: AnthropicModelCapabilities,
        #[case] budget_tokens: Value,
        #[case] expected_effort: &str,
    ) {
        let thinking = match budget_tokens {
            Value::Null => json!({"type": "enabled"}),
            budget_tokens => json!({"type": "enabled", "budget_tokens": budget_tokens}),
        };
        assert_eq!(
            translate(
                capabilities,
                json!({"max_tokens": 1024, "thinking": thinking})
            ),
            Ok(request(json!({
                "max_tokens": 1024,
                "thinking": {"type": "adaptive"},
                "output_config": {"effort": expected_effort}
            })))
        );
    }

    #[rstest]
    #[case::verbatim_on_model_accepting_legacy_thinking(
        sonnet_4_6(),
        json!({"max_tokens": 1024, "thinking": {"type": "enabled", "budget_tokens": 31999}}),
        json!({"max_tokens": 1024, "thinking": {"type": "enabled", "budget_tokens": 31999}})
    )]
    #[case::verbatim_with_explicit_output_config_on_model_accepting_legacy_thinking(
        sonnet_4_6(),
        json!({"max_tokens": 1024, "thinking": {"type": "enabled", "budget_tokens": 31999}, "output_config": {"effort": "low"}}),
        json!({"max_tokens": 1024, "thinking": {"type": "enabled", "budget_tokens": 31999}, "output_config": {"effort": "low"}})
    )]
    #[case::verbatim_on_non_adaptive_model(
        opus_4_5(),
        json!({"max_tokens": 1024, "thinking": {"type": "enabled", "budget_tokens": 31999}}),
        json!({"max_tokens": 1024, "thinking": {"type": "enabled", "budget_tokens": 31999}})
    )]
    #[case::caller_output_config_effort_wins(
        opus_4_7(),
        json!({
            "max_tokens": 32000,
            "thinking": {"type": "enabled", "budget_tokens": 31999},
            "output_config": {"effort": "low", "format": {"type": "json_schema"}}
        }),
        json!({
            "max_tokens": 32000,
            "thinking": {"type": "adaptive"},
            "output_config": {"effort": "low", "format": {"type": "json_schema"}}
        })
    )]
    #[case::effort_merges_into_caller_output_config(
        opus_4_7(),
        json!({
            "max_tokens": 32000,
            "thinking": {"type": "enabled", "budget_tokens": 4096},
            "output_config": {"format": {"type": "json_schema"}}
        }),
        json!({
            "max_tokens": 32000,
            "thinking": {"type": "adaptive"},
            "output_config": {"effort": "high", "format": {"type": "json_schema"}}
        })
    )]
    #[case::adaptive_thinking_is_left_alone(
        opus_4_7(),
        json!({"max_tokens": 8192, "thinking": {"type": "adaptive", "display": "summarized"}}),
        json!({"max_tokens": 8192, "thinking": {"type": "adaptive", "display": "summarized"}})
    )]
    fn legacy_thinking_on_adaptive_capable_models(
        #[case] capabilities: AnthropicModelCapabilities,
        #[case] input: Value,
        #[case] expected: Value,
    ) {
        assert_eq!(translate(capabilities, input), Ok(request(expected)));
    }

    #[rstest]
    #[case::bare_adaptive_becomes_medium_budget_on_budget_model(
        haiku_4_5(),
        json!({"max_tokens": 8192, "thinking": {"type": "adaptive"}}),
        json!({"max_tokens": 8192, "thinking": {"type": "enabled", "budget_tokens": 2048}})
    )]
    #[case::medium_effort_becomes_medium_budget_on_budget_model(
        haiku_4_5(),
        claude_code_payload("medium", 8192),
        json!({"max_tokens": 8192, "thinking": {"type": "enabled", "budget_tokens": 2048}})
    )]
    #[case::empty_effort_becomes_medium_budget_on_budget_model(
        haiku_4_5(),
        claude_code_payload("", 8192),
        json!({"max_tokens": 8192, "thinking": {"type": "enabled", "budget_tokens": 2048}})
    )]
    #[case::high_effort_becomes_high_budget_on_budget_model(
        haiku_4_5(),
        claude_code_payload("high", 8192),
        json!({"max_tokens": 8192, "thinking": {"type": "enabled", "budget_tokens": 4096}})
    )]
    #[case::effort_only_becomes_budget_on_budget_model(
        haiku_4_5(),
        json!({"max_tokens": 8192, "output_config": {"effort": "high"}}),
        json!({"max_tokens": 8192, "thinking": {"type": "enabled", "budget_tokens": 4096}})
    )]
    #[case::effort_replaces_caller_legacy_budget_on_budget_model(
        haiku_4_5(),
        json!({"max_tokens": 8192, "thinking": {"type": "enabled", "budget_tokens": 3000}, "output_config": {"effort": "high"}}),
        json!({"max_tokens": 8192, "thinking": {"type": "enabled", "budget_tokens": 4096}})
    )]
    #[case::residual_output_config_survives_effort_translation(
        haiku_4_5(),
        json!({
            "max_tokens": 8192,
            "thinking": {"type": "adaptive"},
            "output_config": {"effort": "medium", "format": {"type": "json_schema"}}
        }),
        json!({
            "max_tokens": 8192,
            "thinking": {"type": "enabled", "budget_tokens": 2048},
            "output_config": {"format": {"type": "json_schema"}}
        })
    )]
    #[case::effortless_output_config_is_kept(
        haiku_4_5(),
        json!({"max_tokens": 8192, "thinking": {"type": "adaptive"}, "output_config": {"format": {"type": "json_schema"}}}),
        json!({
            "max_tokens": 8192,
            "thinking": {"type": "enabled", "budget_tokens": 2048},
            "output_config": {"format": {"type": "json_schema"}}
        })
    )]
    #[case::empty_output_config_is_kept(
        haiku_4_5(),
        json!({"max_tokens": 8192, "thinking": {"type": "adaptive"}, "output_config": {}}),
        json!({"max_tokens": 8192, "thinking": {"type": "enabled", "budget_tokens": 2048}, "output_config": {}})
    )]
    #[case::missing_max_tokens_leaves_budget_uncapped(
        haiku_4_5(),
        json!({"thinking": {"type": "adaptive"}}),
        json!({"thinking": {"type": "enabled", "budget_tokens": 2048}})
    )]
    #[case::budget_is_capped_below_max_tokens(
        haiku_4_5(),
        claude_code_payload("high", 3000),
        json!({"max_tokens": 3000, "thinking": {"type": "enabled", "budget_tokens": 2999}})
    )]
    #[case::max_tokens_just_above_min_budget_caps_to_min_budget(
        haiku_4_5(),
        claude_code_payload("medium", 1025),
        json!({"max_tokens": 1025, "thinking": {"type": "enabled", "budget_tokens": 1024}})
    )]
    #[case::max_tokens_at_min_budget_drops_thinking_and_effort(
        haiku_4_5(),
        claude_code_payload("medium", 1024),
        json!({"max_tokens": 1024})
    )]
    #[case::max_tokens_below_min_budget_drops_thinking_and_effort(
        haiku_4_5(),
        claude_code_payload("medium", 512),
        json!({"max_tokens": 512})
    )]
    #[case::bare_adaptive_is_dropped_on_non_reasoning_model(
        haiku_3_5(),
        json!({"max_tokens": 8192, "thinking": {"type": "adaptive"}}),
        json!({"max_tokens": 8192})
    )]
    #[case::adaptive_and_effort_are_dropped_on_non_reasoning_model(
        haiku_3_5(),
        claude_code_payload("medium", 8192),
        json!({"max_tokens": 8192})
    )]
    #[case::effort_only_is_dropped_on_non_reasoning_model(
        haiku_3_5(),
        json!({"max_tokens": 8192, "output_config": {"effort": "high", "format": {"type": "json_schema"}}}),
        json!({"max_tokens": 8192, "output_config": {"format": {"type": "json_schema"}}})
    )]
    #[case::supported_effort_is_kept_and_adaptive_thinking_dropped_on_effort_model(
        opus_4_5(),
        claude_code_payload("medium", 8192),
        json!({"max_tokens": 8192, "output_config": {"effort": "medium"}})
    )]
    #[case::bare_adaptive_is_dropped_on_effort_model(
        opus_4_5(),
        json!({"max_tokens": 8192, "thinking": {"type": "adaptive"}}),
        json!({"max_tokens": 8192})
    )]
    #[case::effort_only_is_left_alone_on_effort_model(
        opus_4_5(),
        json!({"max_tokens": 8192, "output_config": {"effort": "high"}}),
        json!({"max_tokens": 8192, "output_config": {"effort": "high"}})
    )]
    #[case::unsupported_effort_only_is_left_for_provider_normalization(
        opus_4_5(),
        json!({"max_tokens": 4096, "output_config": {"effort": "xhigh"}}),
        json!({"max_tokens": 4096, "output_config": {"effort": "xhigh"}})
    )]
    #[case::legacy_thinking_is_kept_beside_native_effort_on_effort_model(
        opus_4_5(),
        json!({"max_tokens": 8192, "thinking": {"type": "enabled", "budget_tokens": 4096}, "output_config": {"effort": "high"}}),
        json!({"max_tokens": 8192, "thinking": {"type": "enabled", "budget_tokens": 4096}, "output_config": {"effort": "high"}})
    )]
    #[case::unsupported_xhigh_with_adaptive_thinking_falls_back_to_budget(
        opus_4_5(),
        claude_code_payload("xhigh", 64000),
        json!({"max_tokens": 64000, "thinking": {"type": "enabled", "budget_tokens": 8192}})
    )]
    #[case::unsupported_max_with_adaptive_thinking_falls_back_to_budget(
        opus_4_5(),
        claude_code_payload("max", 64000),
        json!({"max_tokens": 64000, "thinking": {"type": "enabled", "budget_tokens": 16384}})
    )]
    #[case::bare_adaptive_is_native_on_4_6(
        sonnet_4_6(),
        json!({"max_tokens": 8192, "thinking": {"type": "adaptive"}}),
        json!({"max_tokens": 8192, "thinking": {"type": "adaptive"}})
    )]
    #[case::adaptive_payload_is_native_on_4_6(
        sonnet_4_6(),
        claude_code_payload("high", 8192),
        claude_code_payload("high", 8192)
    )]
    #[case::request_without_adaptive_interface_is_left_alone(
        haiku_4_5(),
        json!({"max_tokens": 1024}),
        json!({"max_tokens": 1024})
    )]
    fn adaptive_interface_is_reshaped_for_non_adaptive_models(
        #[case] capabilities: AnthropicModelCapabilities,
        #[case] input: Value,
        #[case] expected: Value,
    ) {
        assert_eq!(translate(capabilities, input), Ok(request(expected)));
    }

    #[rstest]
    #[case::adaptive_downgraded_to_enabled_thinking(
        haiku_4_5(),
        claude_code_payload("medium", 8192),
        0.0,
        json!({"max_tokens": 8192, "thinking": {"type": "enabled", "budget_tokens": 2048}})
    )]
    #[case::bare_adaptive_downgraded_to_enabled_thinking(
        haiku_4_5(),
        json!({"max_tokens": 8192, "thinking": {"type": "adaptive"}}),
        0.0,
        json!({"max_tokens": 8192, "thinking": {"type": "enabled", "budget_tokens": 2048}})
    )]
    #[case::reasoning_effort_synthesized_enabled_thinking(
        haiku_4_5(),
        json!({"max_tokens": 8192, "reasoning_effort": "high"}),
        0.2,
        json!({"max_tokens": 8192, "thinking": {"type": "enabled", "budget_tokens": 4096}})
    )]
    #[case::above_one_with_enabled_thinking(
        haiku_4_5(),
        json!({"max_tokens": 8192, "thinking": {"type": "enabled", "budget_tokens": 2048}}),
        1.5,
        json!({"max_tokens": 8192, "thinking": {"type": "enabled", "budget_tokens": 2048}})
    )]
    #[case::native_effort_kept_on_effort_model(
        opus_4_5(),
        claude_code_payload("medium", 8192),
        0.0,
        json!({"max_tokens": 8192, "output_config": {"effort": "medium"}})
    )]
    #[case::effort_only_on_effort_model(
        opus_4_5(),
        json!({"max_tokens": 8192, "output_config": {"effort": "high"}}),
        0.0,
        json!({"max_tokens": 8192, "output_config": {"effort": "high"}})
    )]
    fn pinned_temperature_is_dropped_when_thinking_or_effort_survives_on_non_adaptive_model(
        #[case] capabilities: AnthropicModelCapabilities,
        #[case] input: Value,
        #[case] temperature: f64,
        #[case] expected: Value,
    ) {
        assert_eq!(
            translate(capabilities, with_temperature(input, temperature)),
            Ok(request(expected))
        );
    }

    #[rstest]
    #[case::temperature_one_with_enabled_thinking(
        haiku_4_5(),
        claude_code_payload("medium", 8192),
        1.0,
        json!({"max_tokens": 8192, "thinking": {"type": "enabled", "budget_tokens": 2048}})
    )]
    #[case::thinking_dropped_for_small_max_tokens(
        haiku_4_5(),
        claude_code_payload("medium", 512),
        0.0,
        json!({"max_tokens": 512})
    )]
    #[case::thinking_dropped_on_non_reasoning_model(
        haiku_3_5(),
        claude_code_payload("medium", 8192),
        0.0,
        json!({"max_tokens": 8192})
    )]
    #[case::disabled_thinking(
        haiku_4_5(),
        json!({"max_tokens": 8192, "thinking": {"type": "disabled"}}),
        0.0,
        json!({"max_tokens": 8192, "thinking": {"type": "disabled"}})
    )]
    #[case::no_thinking(haiku_4_5(), json!({"max_tokens": 8192}), 0.0, json!({"max_tokens": 8192}))]
    #[case::output_config_without_effort(
        haiku_4_5(),
        json!({"max_tokens": 8192, "output_config": {"format": {"type": "json_schema"}}}),
        0.0,
        json!({"max_tokens": 8192, "output_config": {"format": {"type": "json_schema"}}})
    )]
    #[case::adaptive_model(
        opus_4_7(),
        claude_code_payload("medium", 8192),
        0.0,
        claude_code_payload("medium", 8192)
    )]
    #[case::legacy_thinking_on_adaptive_model(
        sonnet_4_6(),
        json!({"max_tokens": 8192, "thinking": {"type": "enabled", "budget_tokens": 4096}}),
        0.0,
        json!({"max_tokens": 8192, "thinking": {"type": "enabled", "budget_tokens": 4096}})
    )]
    fn temperature_is_kept(
        #[case] capabilities: AnthropicModelCapabilities,
        #[case] input: Value,
        #[case] temperature: f64,
        #[case] expected: Value,
    ) {
        assert_eq!(
            translate(capabilities, with_temperature(input, temperature)),
            Ok(request(with_temperature(expected, temperature)))
        );
    }

    #[rstest]
    #[case::minimal("MINIMAL", ThinkingBudgets { minimal: 5000, ..ThinkingBudgets::default() })]
    #[case::low("LOW", ThinkingBudgets { low: 5000, ..ThinkingBudgets::default() })]
    #[case::medium("MEDIUM", ThinkingBudgets { medium: 5000, ..ThinkingBudgets::default() })]
    #[case::high("HIGH", ThinkingBudgets { high: 5000, ..ThinkingBudgets::default() })]
    #[case::xhigh("XHIGH", ThinkingBudgets { xhigh: 5000, ..ThinkingBudgets::default() })]
    #[case::max("MAX", ThinkingBudgets { max: 5000, ..ThinkingBudgets::default() })]
    fn each_tier_budget_reads_only_its_own_environment_override(
        #[case] tier: &str,
        #[case] expected: ThinkingBudgets,
    ) {
        assert_eq!(overridden_budgets(&[(tier, "5000")]), expected);
    }

    #[rstest]
    #[case::whitespace_is_trimmed(" 6000 ", 6000)]
    #[case::unparseable_value_keeps_default("lots", 4096)]
    fn environment_override_parsing(#[case] raw: &str, #[case] expected_high: u64) {
        assert_eq!(
            overridden_budgets(&[("HIGH", raw)]),
            ThinkingBudgets {
                high: expected_high,
                ..ThinkingBudgets::default()
            }
        );
    }

    #[rstest]
    #[case::reasoning_effort_uses_overridden_budget(
        &[("HIGH", "6000")],
        haiku_4_5(),
        json!({"max_tokens": 32000, "reasoning_effort": "high"}),
        json!({"max_tokens": 32000, "thinking": {"type": "enabled", "budget_tokens": 6000}})
    )]
    #[case::minimal_override_below_min_budget_is_floored(
        &[("MINIMAL", "512")],
        haiku_4_5(),
        json!({"max_tokens": 32000, "reasoning_effort": "minimal"}),
        json!({"max_tokens": 32000, "thinking": {"type": "enabled", "budget_tokens": 1024}})
    )]
    #[case::minimal_override_above_min_budget_is_used(
        &[("MINIMAL", "2000")],
        haiku_4_5(),
        json!({"max_tokens": 32000, "reasoning_effort": "minimal"}),
        json!({"max_tokens": 32000, "thinking": {"type": "enabled", "budget_tokens": 2000}})
    )]
    #[case::adaptive_fallback_uses_overridden_medium_budget(
        &[("MEDIUM", "3000")],
        haiku_4_5(),
        json!({"max_tokens": 32000, "thinking": {"type": "adaptive"}}),
        json!({"max_tokens": 32000, "thinking": {"type": "enabled", "budget_tokens": 3000}})
    )]
    #[case::legacy_bucket_below_overridden_high_budget(
        &[("HIGH", "6000")],
        opus_4_7(),
        json!({"max_tokens": 32000, "thinking": {"type": "enabled", "budget_tokens": 5999}}),
        json!({"max_tokens": 32000, "thinking": {"type": "adaptive"}, "output_config": {"effort": "medium"}})
    )]
    #[case::legacy_bucket_at_overridden_high_budget(
        &[("HIGH", "6000")],
        opus_4_7(),
        json!({"max_tokens": 32000, "thinking": {"type": "enabled", "budget_tokens": 6000}}),
        json!({"max_tokens": 32000, "thinking": {"type": "adaptive"}, "output_config": {"effort": "high"}})
    )]
    #[case::legacy_bucket_below_overridden_xhigh_budget(
        &[("XHIGH", "20000")],
        opus_4_7(),
        json!({"max_tokens": 32000, "thinking": {"type": "enabled", "budget_tokens": 19999}}),
        json!({"max_tokens": 32000, "thinking": {"type": "adaptive"}, "output_config": {"effort": "high"}})
    )]
    #[case::legacy_bucket_at_overridden_medium_budget(
        &[("MEDIUM", "3000")],
        opus_4_7(),
        json!({"max_tokens": 32000, "thinking": {"type": "enabled", "budget_tokens": 3000}}),
        json!({"max_tokens": 32000, "thinking": {"type": "adaptive"}, "output_config": {"effort": "medium"}})
    )]
    #[case::legacy_bucket_below_overridden_medium_budget(
        &[("MEDIUM", "3000")],
        opus_4_7(),
        json!({"max_tokens": 32000, "thinking": {"type": "enabled", "budget_tokens": 2999}}),
        json!({"max_tokens": 32000, "thinking": {"type": "adaptive"}, "output_config": {"effort": "low"}})
    )]
    fn translation_honors_budget_overrides(
        #[case] overrides: &[(&str, &str)],
        #[case] capabilities: AnthropicModelCapabilities,
        #[case] input: Value,
        #[case] expected: Value,
    ) {
        let context = ThinkingContext {
            capabilities,
            budgets: overridden_budgets(overrides),
        };
        assert_eq!(
            translate_thinking(request(input), &context),
            Ok(request(expected))
        );
    }
}
