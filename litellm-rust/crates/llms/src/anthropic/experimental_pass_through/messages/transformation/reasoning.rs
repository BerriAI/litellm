use serde::Deserialize;
use serde_json::{Value, json};

#[derive(Clone, Debug, Default, Deserialize)]
pub struct MessagesFeatures {
    pub adaptive: bool,
    pub always_on: bool,
    pub legacy: bool,
    pub reasoning: bool,
    pub effort: bool,
    pub xhigh: bool,
    pub max: bool,
    pub speed: bool,
    pub auto_summary: bool,
    pub prompt_cache_supported: bool,
    pub prompt_cache_enabled: bool,
    pub prompt_cache_ttl: Option<String>,
}

fn thinking_budget(effort: &str) -> Result<u64, String> {
    let (name, default) = match effort {
        "minimal" => ("DEFAULT_REASONING_EFFORT_MINIMAL_THINKING_BUDGET", 128),
        "low" => ("DEFAULT_REASONING_EFFORT_LOW_THINKING_BUDGET", 1024),
        "medium" => ("DEFAULT_REASONING_EFFORT_MEDIUM_THINKING_BUDGET", 2048),
        "high" => ("DEFAULT_REASONING_EFFORT_HIGH_THINKING_BUDGET", 4096),
        "xhigh" => ("DEFAULT_REASONING_EFFORT_XHIGH_THINKING_BUDGET", 8192),
        "max" => ("DEFAULT_REASONING_EFFORT_MAX_THINKING_BUDGET", 16384),
        _ => {
            return Err(format!(
                "Unmapped reasoning effort: {effort:?}. Must be one of: 'minimal', 'low', 'medium', 'high', 'xhigh', 'max', 'none'."
            ));
        }
    };
    Ok(std::env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default)
        .max(if effort == "minimal" { 1024 } else { 0 }))
}

fn fitted_thinking(
    effort: &str,
    max_tokens: Option<u64>,
    adaptive: bool,
) -> Result<Option<Value>, String> {
    if adaptive {
        return thinking_budget(effort)
            .map(|_| Some(json!({"type":"adaptive","display":"summarized"})));
    }
    let budget = thinking_budget(effort)?;
    Ok(match max_tokens {
        Some(limit) if limit <= 1024 => None,
        Some(limit) => Some(json!({"type":"enabled","budget_tokens":budget.min(limit - 1)})),
        None => Some(json!({"type":"enabled","budget_tokens":budget})),
    })
}

fn valid_effort(effort: &str, capabilities: &MessagesFeatures) -> bool {
    match effort {
        "max" => capabilities.adaptive || capabilities.max,
        "xhigh" => capabilities.xhigh,
        _ => true,
    }
}

pub fn normalize_reasoning(body: Value, capabilities: &MessagesFeatures) -> Result<Value, String> {
    let Value::Object(mut fields) = body else {
        return Ok(body);
    };
    let max_tokens = fields.get("max_tokens").and_then(Value::as_u64);
    if let Some(effort) = fields
        .remove("reasoning_effort")
        .and_then(|value| value.as_str().map(str::to_owned))
    {
        if effort == "none" {
            fields.remove("thinking");
            fields.remove("output_config");
        } else if let Some(thinking) = fitted_thinking(&effort, max_tokens, capabilities.adaptive)?
        {
            fields.entry("thinking").or_insert(thinking);
            if capabilities.adaptive {
                let mapped = if effort == "minimal" {
                    "low"
                } else {
                    effort.as_str()
                };
                if !valid_effort(mapped, capabilities) {
                    return Err(format!("effort='{mapped}' is not supported by this model"));
                }
                let output = fields.entry("output_config").or_insert_with(|| json!({}));
                if let Some(config) = output.as_object_mut() {
                    config.entry("effort").or_insert(json!(mapped));
                }
            }
        }
    }
    if capabilities.always_on
        && fields
            .get("thinking")
            .and_then(|value| value.get("type"))
            .and_then(Value::as_str)
            == Some("disabled")
    {
        fields.remove("thinking");
    }
    if capabilities.adaptive
        && !capabilities.legacy
        && fields
            .get("thinking")
            .and_then(|value| value.get("type"))
            .and_then(Value::as_str)
            == Some("enabled")
    {
        let budget = fields
            .get("thinking")
            .and_then(|value| value.get("budget_tokens"))
            .and_then(Value::as_u64)
            .unwrap_or(0);
        let effort = if budget >= 8192 && capabilities.xhigh {
            "xhigh"
        } else if budget >= 4096 {
            "high"
        } else if budget >= 2048 {
            "medium"
        } else {
            "low"
        };
        fields.insert("thinking".into(), json!({"type":"adaptive"}));
        let output = fields.entry("output_config").or_insert_with(|| json!({}));
        if let Some(config) = output.as_object_mut() {
            config.entry("effort").or_insert(json!(effort));
        }
    }
    if !capabilities.adaptive {
        let adaptive = fields
            .get("thinking")
            .and_then(|value| value.get("type"))
            .and_then(Value::as_str)
            == Some("adaptive");
        let effort = fields
            .get("output_config")
            .and_then(|value| value.get("effort"))
            .and_then(Value::as_str)
            .map(str::to_owned);
        if adaptive || effort.is_some() {
            if capabilities.effort
                && (!adaptive
                    || effort
                        .as_deref()
                        .is_none_or(|value| valid_effort(value, capabilities)))
            {
                if adaptive {
                    fields.remove("thinking");
                }
            } else {
                let legacy = if capabilities.reasoning {
                    fitted_thinking(effort.as_deref().unwrap_or("medium"), max_tokens, false)?
                } else {
                    None
                };
                match legacy {
                    Some(thinking) => {
                        fields.insert("thinking".into(), thinking);
                    }
                    None => {
                        fields.remove("thinking");
                    }
                }
                if let Some(config) = fields
                    .get_mut("output_config")
                    .and_then(Value::as_object_mut)
                {
                    config.remove("effort");
                    if config.is_empty() {
                        fields.remove("output_config");
                    }
                }
            }
        }
        let thinking = fields
            .get("thinking")
            .and_then(|value| value.get("type"))
            .and_then(Value::as_str)
            == Some("enabled");
        let effort = fields
            .get("output_config")
            .and_then(|value| value.get("effort"))
            .is_some();
        if (thinking || effort) && fields.get("temperature").is_some_and(|value| value != 1) {
            fields.remove("temperature");
        }
    }
    if capabilities.auto_summary {
        if let Some(thinking) = fields.get_mut("thinking").and_then(Value::as_object_mut) {
            if thinking.get("type").and_then(Value::as_str) != Some("disabled") {
                thinking.insert("display".into(), json!("summarized"));
            }
        }
    }
    Ok(Value::Object(fields))
}
