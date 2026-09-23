use std::collections::BTreeSet;

use serde_json::{Map, Value};

use crate::base_llm::{
    anthropic_messages::transformation::BaseAnthropicMessagesConfig, chat::transformation::Error,
};

const ANTHROPIC_API_KEY_ENV: &str = "ANTHROPIC_API_KEY";
const ANTHROPIC_API_BASE_ENV: &str = "ANTHROPIC_API_BASE";
const DEFAULT_ANTHROPIC_API_BASE: &str = "https://api.anthropic.com";
const MESSAGES_PATH_SUFFIX: &str = "/v1/messages";

pub struct AnthropicMessagesConfig;

pub const ANTHROPIC_MESSAGES_CONFIG: AnthropicMessagesConfig = AnthropicMessagesConfig;

mod prompt_cache;
mod reasoning;

pub use prompt_cache::inject_cache_control;
pub use reasoning::{MessagesFeatures, normalize_reasoning};

pub fn normalize_context_management(value: Value) -> Value {
    let Value::Array(entries) = value else {
        return match value {
            Value::Object(ref fields) if fields.contains_key("edits") => value,
            _ => Value::Null,
        };
    };
    let edits: Vec<Value> =
        entries
            .iter()
            .filter_map(|entry| {
                let Value::Object(fields) = entry else {
                    return None;
                };
                if fields.get("type").and_then(Value::as_str) != Some("compaction") {
                    return None;
                }
                let threshold = fields.get("compact_threshold").and_then(|value| {
                value.as_f64().or_else(|| value.as_bool().map(|value| i32::from(value) as f64))
            }).map(
                |threshold| serde_json::json!({"type":"input_tokens","value":threshold as i64}),
            );
                Some(Value::Object(
                    fields
                        .iter()
                        .filter(|(name, _)| !matches!(name.as_str(), "type" | "compact_threshold"))
                        .map(|(name, value)| (name.clone(), value.clone()))
                        .chain([("type".into(), Value::String("compact_20260112".into()))])
                        .chain(threshold.map(|value| ("trigger".into(), value)))
                        .collect(),
                ))
            })
            .collect();
    if edits.is_empty() {
        Value::Null
    } else {
        Value::Object(Map::from_iter([("edits".into(), Value::Array(edits))]))
    }
}

impl BaseAnthropicMessagesConfig for AnthropicMessagesConfig {
    fn get_complete_url(
        &self,
        api_base: Option<&str>,
        _model: &str,
        env_lookup: &dyn Fn(&str) -> Option<String>,
    ) -> Result<String, Error> {
        Ok(complete_anthropic_url(api_base, env_lookup))
    }

    fn resolve_api_key(
        &self,
        api_key: Option<&str>,
        env_lookup: &dyn Fn(&str) -> Option<String>,
    ) -> Result<String, Error> {
        resolve_anthropic_api_key(api_key, env_lookup).map_err(Error::from)
    }
}

pub fn anthropic_beta_headers(
    headers: Vec<(String, String)>,
    body: &Value,
) -> Vec<(String, String)> {
    let existing: BTreeSet<String> = headers
        .iter()
        .filter(|(name, _)| name.eq_ignore_ascii_case("anthropic-beta"))
        .flat_map(|(_, value)| value.split(','))
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_string)
        .collect();
    let messages = body.get("messages").and_then(Value::as_array);
    let tools = body.get("tools").and_then(Value::as_array);
    let edits = body
        .get("context_management")
        .and_then(|value| value.get("edits"))
        .and_then(Value::as_array);
    let compact_replay = messages.is_some_and(|messages| {
        messages.iter().any(|message| {
            message
                .get("content")
                .and_then(Value::as_array)
                .is_some_and(|content| {
                    content.iter().any(|block| {
                        block.get("type").and_then(Value::as_str) == Some("compaction")
                            && block
                                .get("signature")
                                .and_then(Value::as_str)
                                .is_some_and(|signature| !signature.is_empty())
                    })
                })
        })
    });
    let generated = [
        (
            body.get("compaction").is_some() || compact_replay,
            "compact-2026-09-04",
        ),
        (
            edits.is_some_and(|edits| {
                edits.iter().any(|edit| {
                    edit.get("type").and_then(Value::as_str) == Some("compact_20260112")
                })
            }),
            "compact-2026-01-12",
        ),
        (
            edits.is_some_and(|edits| {
                edits.iter().any(|edit| {
                    edit.get("type").and_then(Value::as_str) != Some("compact_20260112")
                })
            }),
            "context-management-2025-06-27",
        ),
        (
            body.get("output_format").is_some()
                || body
                    .get("output_config")
                    .and_then(|value| value.get("format"))
                    .is_some(),
            "structured-outputs-2025-11-13",
        ),
        (
            body.get("speed").and_then(Value::as_str) == Some("fast"),
            "fast-mode-2026-02-01",
        ),
        (
            messages.is_some_and(|messages| {
                messages
                    .iter()
                    .any(|message| message.get("output_config").is_some())
            }),
            "per-turn-control-2026-07-01",
        ),
        (
            tools.is_some_and(|tools| {
                tools.iter().any(|tool| {
                    tool.get("type").and_then(Value::as_str) == Some("advisor_20260301")
                })
            }),
            "advisor-tool-2026-03-01",
        ),
        (
            tools.is_some_and(|tools| {
                tools.iter().any(|tool| {
                    matches!(
                        tool.get("type").and_then(Value::as_str),
                        Some("tool_search_tool_regex_20251119" | "tool_search_tool_bm25_20251119")
                    )
                })
            }),
            "advanced-tool-use-2025-11-20",
        ),
    ];
    let betas: BTreeSet<String> = existing
        .into_iter()
        .chain(
            generated
                .into_iter()
                .filter_map(|(enabled, value)| enabled.then(|| value.to_string())),
        )
        .collect();
    if betas.is_empty() {
        return headers;
    }
    headers
        .into_iter()
        .filter(|(name, _)| !name.eq_ignore_ascii_case("anthropic-beta"))
        .chain([(
            "anthropic-beta".into(),
            betas.into_iter().collect::<Vec<_>>().join(","),
        )])
        .collect()
}

pub fn non_empty(value: Option<&str>) -> Option<&str> {
    value.map(str::trim).filter(|value| !value.is_empty())
}

pub fn resolve_anthropic_api_key(
    api_key: Option<&str>,
    env_lookup: &dyn Fn(&str) -> Option<String>,
) -> Result<String, litellm_auth::Error> {
    non_empty(api_key)
        .map(str::to_string)
        .or_else(|| env_lookup(ANTHROPIC_API_KEY_ENV).filter(|value| !value.trim().is_empty()))
        .ok_or(litellm_auth::Error::MissingApiKey {
            provider: "Anthropic",
            environment_variable: ANTHROPIC_API_KEY_ENV,
        })
}

pub fn complete_anthropic_url(
    api_base: Option<&str>,
    env_lookup: &dyn Fn(&str) -> Option<String>,
) -> String {
    let api_base = resolve_anthropic_api_base(api_base, env_lookup);

    let api_base = api_base.trim_end_matches('/');
    if api_base.ends_with(MESSAGES_PATH_SUFFIX) {
        return api_base.to_string();
    }
    format!("{api_base}{MESSAGES_PATH_SUFFIX}")
}

pub fn resolve_anthropic_api_base(
    api_base: Option<&str>,
    env_lookup: &dyn Fn(&str) -> Option<String>,
) -> String {
    non_empty(api_base)
        .map(str::to_string)
        .or_else(|| env_lookup(ANTHROPIC_API_BASE_ENV).filter(|value| !value.trim().is_empty()))
        .or_else(|| env_lookup("ANTHROPIC_BASE_URL").filter(|value| !value.trim().is_empty()))
        .unwrap_or_else(|| DEFAULT_ANTHROPIC_API_BASE.to_string())
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn openai_compaction_context_becomes_anthropic_edit() {
        assert_eq!(
            normalize_context_management(json!([
                {"type":"compaction","compact_threshold":200000.8,"pause_after_compaction":true},
                {"type":"other"}
            ])),
            json!({"edits":[{"type":"compact_20260112","trigger":{"type":"input_tokens","value":200000},"pause_after_compaction":true}]})
        );
    }

    #[test]
    fn unsupported_context_management_is_omitted() {
        assert_eq!(
            normalize_context_management(json!([{"type":"other"}])),
            Value::Null
        );
        assert_eq!(
            normalize_context_management(json!({"other":true})),
            Value::Null
        );
        assert_eq!(
            normalize_context_management(json!({"edits":[]})),
            json!({"edits":[]})
        );
    }

    #[test]
    fn reasoning_effort_uses_adaptive_thinking_and_keeps_caller_effort() {
        let capabilities = MessagesFeatures {
            adaptive: true,
            xhigh: true,
            ..Default::default()
        };
        assert_eq!(
            normalize_reasoning(
                json!({"max_tokens":8192,"reasoning_effort":"minimal"}),
                &capabilities
            )
            .unwrap(),
            json!({"max_tokens":8192,"thinking":{"type":"adaptive","display":"summarized"},"output_config":{"effort":"low"}})
        );
        assert_eq!(
            normalize_reasoning(json!({"max_tokens":8192,"reasoning_effort":"low","output_config":{"effort":"high"}}), &capabilities).unwrap(),
            json!({"max_tokens":8192,"thinking":{"type":"adaptive","display":"summarized"},"output_config":{"effort":"high"}})
        );
    }

    #[test]
    fn older_model_maps_adaptive_effort_to_capped_legacy_budget() {
        let capabilities = MessagesFeatures {
            reasoning: true,
            ..Default::default()
        };
        assert_eq!(
            normalize_reasoning(json!({"max_tokens":1500,"temperature":0.2,"thinking":{"type":"adaptive"},"output_config":{"effort":"high","format":{"type":"json_schema"}}}), &capabilities).unwrap(),
            json!({"max_tokens":1500,"thinking":{"type":"enabled","budget_tokens":1499},"output_config":{"format":{"type":"json_schema"}}})
        );
    }

    #[test]
    fn auto_summary_adds_display_to_enabled_thinking_only() {
        let capabilities = MessagesFeatures {
            auto_summary: true,
            ..Default::default()
        };
        assert_eq!(
            normalize_reasoning(
                json!({"thinking":{"type":"enabled","budget_tokens":1024}}),
                &capabilities
            )
            .unwrap()["thinking"],
            json!({"type":"enabled","budget_tokens":1024,"display":"summarized"})
        );
        assert_eq!(
            normalize_reasoning(json!({"thinking":{"type":"disabled"}}), &capabilities).unwrap()["thinking"],
            json!({"type":"disabled"})
        );
    }

    #[test]
    fn none_effort_clears_explicit_thinking_and_output_config() {
        assert_eq!(
            normalize_reasoning(
                json!({"reasoning_effort":"none","thinking":{"type":"adaptive"},"output_config":{"effort":"high"}}),
                &MessagesFeatures::default(),
            ).unwrap(),
            json!({})
        );
    }

    #[test]
    fn invalid_and_unsupported_reasoning_efforts_fail_before_send() {
        let adaptive = MessagesFeatures {
            adaptive: true,
            ..Default::default()
        };
        assert!(normalize_reasoning(json!({"reasoning_effort":"invalid"}), &adaptive).is_err());
        assert!(
            normalize_reasoning(json!({"reasoning_effort":"xhigh"}), &adaptive)
                .unwrap_err()
                .contains("not supported")
        );
    }

    #[test]
    fn always_on_model_drops_disabled_thinking_and_legacy_budget_becomes_adaptive() {
        let capabilities = MessagesFeatures {
            adaptive: true,
            always_on: true,
            xhigh: true,
            ..Default::default()
        };
        assert_eq!(
            normalize_reasoning(json!({"thinking":{"type":"disabled"}}), &capabilities).unwrap(),
            json!({})
        );
        assert_eq!(
            normalize_reasoning(
                json!({"thinking":{"type":"enabled","budget_tokens":8192}}),
                &capabilities
            )
            .unwrap(),
            json!({"thinking":{"type":"adaptive"},"output_config":{"effort":"xhigh"}})
        );
    }

    #[test]
    fn explicit_message_cache_injection_respects_existing_blocks_and_limit() {
        let body = json!({
            "messages": [
                {"role":"user","content":[{"type":"text","text":"first","cache_control":{"type":"ephemeral","ttl":"1h"}}]},
                {"role":"user","content":[{"type":"text","text":"second"}]},
                {"role":"user","content":[{"type":"text","text":"third"}]},
                {"role":"user","content":[{"type":"text","text":"fourth"}]},
                {"role":"user","content":[{"type":"text","text":"fifth"}]}
            ],
            "system":"instructions",
            "cache_control_injection_points":[
                {"location":"message","role":"system"},
                {"location":"message","role":"user"}
            ]
        });
        let injected = inject_cache_control(body.clone(), &MessagesFeatures::default());
        assert_eq!(
            injected["messages"][0]["content"][0]["cache_control"],
            json!({"type":"ephemeral","ttl":"1h"})
        );
        assert_eq!(
            injected["messages"][1]["content"][0]["cache_control"],
            json!({"type":"ephemeral"})
        );
        assert!(
            injected["messages"][4]["content"][0]
                .get("cache_control")
                .is_none()
        );
        assert_eq!(
            injected["system"][0]["cache_control"],
            json!({"type":"ephemeral"})
        );
        assert!(injected.get("cache_control_injection_points").is_none());
        assert!(body["system"].is_string());
    }

    #[test]
    fn default_prompt_cache_marks_system_and_last_turn_only_when_unmarked() {
        let capabilities = MessagesFeatures {
            prompt_cache_supported: true,
            prompt_cache_enabled: true,
            prompt_cache_ttl: Some("1h".into()),
            ..Default::default()
        };
        let body = json!({
            "system":"stable instructions",
            "messages":[
                {"role":"user","content":[{"type":"text","text":"first"}]},
                {"role":"user","content":[{"type":"text","text":"latest"}]}
            ]
        });
        let injected = inject_cache_control(body.clone(), &capabilities);
        assert_eq!(
            injected["system"][0]["cache_control"],
            json!({"type":"ephemeral","ttl":"1h"})
        );
        assert_eq!(
            injected["messages"][1]["content"][0]["cache_control"],
            json!({"type":"ephemeral","ttl":"1h"})
        );
        assert!(
            injected["messages"][0]["content"][0]
                .get("cache_control")
                .is_none()
        );
        let marked = json!({
            "system":"stable instructions",
            "messages":[{"role":"user","content":[{"type":"text","text":"latest","cache_control":{"type":"ephemeral"}}]}]
        });
        assert_eq!(inject_cache_control(marked.clone(), &capabilities), marked);
    }

    #[test]
    fn beta_headers_merge_existing_and_requested_features_without_duplicates() {
        let headers = anthropic_beta_headers(
            vec![(
                "Anthropic-Beta".into(),
                "fast-mode-2026-02-01,custom-beta".into(),
            )],
            &json!({
                "speed":"fast",
                "output_config":{"format":{"type":"json_schema"}},
                "messages":[{"role":"user","content":"hi","output_config":{"effort":"high"}}]
            }),
        );
        assert_eq!(headers.len(), 1);
        let betas = headers[0].1.split(',').collect::<Vec<_>>();
        assert_eq!(
            betas
                .iter()
                .filter(|value| **value == "fast-mode-2026-02-01")
                .count(),
            1
        );
        assert!(betas.contains(&"custom-beta"));
        assert!(betas.contains(&"structured-outputs-2025-11-13"));
        assert!(betas.contains(&"per-turn-control-2026-07-01"));
    }

    #[test]
    fn url_defaults_to_public_anthropic_endpoint() {
        assert_eq!(
            complete_anthropic_url(None, &|_| None),
            "https://api.anthropic.com/v1/messages"
        );
    }

    #[test]
    fn url_appends_messages_suffix_to_custom_base() {
        assert_eq!(
            complete_anthropic_url(Some("https://proxy.internal"), &|_| None),
            "https://proxy.internal/v1/messages"
        );
    }

    #[test]
    fn url_leaves_complete_messages_endpoint_untouched() {
        assert_eq!(
            complete_anthropic_url(Some("https://proxy.internal/v1/messages"), &|_| None),
            "https://proxy.internal/v1/messages"
        );
    }

    #[test]
    fn url_falls_back_to_env_base() {
        let with_env = |key: &str| {
            (key == ANTHROPIC_API_BASE_ENV).then(|| "https://env.anthropic".to_string())
        };
        assert_eq!(
            complete_anthropic_url(Some("  "), &with_env),
            "https://env.anthropic/v1/messages"
        );
    }

    #[test]
    fn api_key_prefers_param_then_env_then_errors() {
        assert_eq!(
            resolve_anthropic_api_key(Some("sk-param"), &|_| None).unwrap(),
            "sk-param"
        );
        let with_env = |key: &str| (key == ANTHROPIC_API_KEY_ENV).then(|| "sk-env".to_string());
        assert_eq!(
            resolve_anthropic_api_key(Some("  "), &with_env).unwrap(),
            "sk-env"
        );
        assert_eq!(
            resolve_anthropic_api_key(None, &|_| None)
                .expect_err("missing key")
                .to_string(),
            "Missing Anthropic API Key - Set `api_key` or the ANTHROPIC_API_KEY environment variable"
        );
    }

    #[test]
    fn auth_strategy_and_default_headers_match_anthropic() {
        assert_eq!(
            ANTHROPIC_MESSAGES_CONFIG.auth_strategy().header_name(),
            "x-api-key"
        );
        assert_eq!(
            ANTHROPIC_MESSAGES_CONFIG.default_headers(),
            &[
                ("anthropic-version", "2023-06-01"),
                ("content-type", "application/json"),
            ]
        );
    }
}
