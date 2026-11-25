//! What a logging callback hands the limiter, and the usage the limiter settles from it.
//!
//! The Python bridge projects a callback's kwargs and response objects by their class into the
//! shapes below. Which of them carries the usage, the total charged under the configured
//! `token_rate_limit_type`, the reconciled prompt, completion and cache-read split, and the
//! partial usage a failed stream recovered are all decided here, the way the Python limiter
//! decided them.

use std::collections::BTreeMap;

use litellm_python_compat::truthy::json_truthy;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::{
    limiter::{
        Candidate, FailureFacts, IoUsage, ModelFacts, PolicyFacts, PostCallFailureFacts, PtuFacts,
        SuccessFacts, TargetFacts,
    },
    ptu::{Ceiling, Usage, settlement_tokens},
    settings::TokenRateLimitType,
    state::RateLimitedModel,
};

/// `CallTypes.aresponses_websocket`, whose turns settle without a PTU charge.
pub const RESPONSES_WEBSOCKET_CALL_TYPE: &str = "_aresponses_websocket";
/// The one internal call origin that still consumes the caller's token limits.
const COMPACTION_ORIGIN: &str = "autorouter_compaction";

/// A usage object, by its Python class.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum UsageObject {
    /// litellm's `Usage`: `prompt_tokens`, `completion_tokens`, `total_tokens`, and
    /// `prompt_tokens_details.cached_tokens` as `cached_tokens`.
    Chat {
        #[serde(default)]
        members: Map<String, Value>,
    },
    /// `ResponseAPIUsage`: `input_tokens`, `output_tokens`, `total_tokens`, and
    /// `input_tokens_details.cached_tokens` as `cached_tokens`.
    Responses {
        #[serde(default)]
        members: Map<String, Value>,
    },
    /// A plain dict, with the usage object nested under its `usage` member when it holds one.
    Mapping {
        #[serde(default)]
        members: Map<String, Value>,
        #[serde(default)]
        nested: Option<Box<UsageObject>>,
    },
    /// Any other object.
    Other,
}

/// A callback's response, or its `combined_usage_object`, by its Python class.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum LoggedObject {
    #[default]
    Absent,
    /// A `Usage` or `ResponseAPIUsage` itself.
    Usage {
        usage: UsageObject,
    },
    /// A `ModelResponse`, `EmbeddingResponse`, `TextCompletionResponse` or Responses API object,
    /// and its `usage` member.
    Modeled {
        #[serde(default)]
        usage: Option<UsageObject>,
    },
    /// An `ExtractionResponse`, and the usage `get_usage_object` reads off it.
    Extraction {
        #[serde(default)]
        usage: Option<UsageObject>,
    },
    /// A `RerankResponse`, and its `meta`.
    Rerank {
        #[serde(default)]
        meta: Option<Map<String, Value>>,
    },
    /// A plain dict.
    Mapping {
        usage: UsageObject,
    },
    /// A Responses WebSocket session's events, and the usage they combine to.
    Events {
        #[serde(default)]
        usage: Option<UsageObject>,
    },
    Other,
}

/// A logging callback's call, as the limiter settles it.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct LoggedCall {
    #[serde(default)]
    pub call_id: Option<String>,
    #[serde(default)]
    pub call_type: Option<String>,
    /// The request metadata's `internal_call_origin`.
    #[serde(default)]
    pub origin: Value,
    /// The callback's `user`.
    #[serde(default)]
    pub user: Value,
    /// The standard logging payload's `metadata`.
    #[serde(default)]
    pub metadata: Map<String, Value>,
    #[serde(default)]
    pub response: LoggedObject,
    /// The callback's `combined_usage_object`.
    #[serde(default)]
    pub combined: LoggedObject,
    /// Whether a non-streamed round ended before a nested call reused the request's slot, so the
    /// slot is now that call's to release.
    #[serde(default)]
    pub ended_before_slot_reuse: bool,
}

/// A model the settlement may charge, as the router and the key's records resolve it.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct SettledModel {
    #[serde(default)]
    pub candidates: Vec<Candidate>,
    #[serde(default)]
    pub ptu_ceiling: Option<Ceiling>,
}

/// What only the proxy can resolve for a settlement: the router's model aliases, the key's own
/// model limits, the team's PTU share, and the live enforcement policy.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Resolved {
    #[serde(default)]
    pub served_model: Option<RateLimitedModel>,
    #[serde(default)]
    pub models: BTreeMap<String, SettledModel>,
    #[serde(default)]
    pub policy: PolicyFacts,
}

fn truthy(value: Option<&Value>) -> bool {
    value.is_some_and(json_truthy)
}

/// Python's `first or second`.
fn either<'a>(first: Option<&'a Value>, second: Option<&'a Value>) -> Option<&'a Value> {
    if truthy(first) { first } else { second }
}

/// A Python `int`, a `bool` among them; a count that is not one counts nothing.
fn count(value: Option<&Value>) -> i64 {
    match value {
        Some(Value::Bool(flag)) => i64::from(*flag),
        Some(Value::Number(number)) => number
            .as_i64()
            .or_else(|| number.as_u64().map(|_| i64::MAX))
            .unwrap_or(0),
        _ => 0,
    }
}

/// `f"{value}"` of a truthy scalar, else None.
fn text(value: Option<&Value>) -> Option<String> {
    if !truthy(value) {
        return None;
    }
    match value? {
        Value::String(text) => Some(text.clone()),
        Value::Number(number) => Some(number.to_string()),
        Value::Bool(flag) => Some(if *flag { "True" } else { "False" }.to_owned()),
        _ => None,
    }
}

/// `_get_total_tokens_from_usage`: the tokens a usage charges under the rate limit type, cache
/// reads left out of input and total. Only litellm's `Usage` and plain dicts report them.
pub fn charged_tokens(usage: Option<&UsageObject>, kind: TokenRateLimitType) -> i64 {
    let (charged, cached) = counted_and_cached(usage, kind);
    if cached > 0 {
        charged.saturating_sub(cached).max(0)
    } else {
        charged
    }
}

/// The cache-read prompt tokens `charged_tokens` leaves out of a usage's input or total.
pub fn cached_tokens_left_out(usage: Option<&UsageObject>, kind: TokenRateLimitType) -> i64 {
    let (charged, cached) = counted_and_cached(usage, kind);
    cached.clamp(0, charged.max(0))
}

fn counted_and_cached(usage: Option<&UsageObject>, kind: TokenRateLimitType) -> (i64, i64) {
    let (members, cached) = match usage {
        Some(UsageObject::Chat { members }) => (members, count(members.get("cached_tokens"))),
        Some(UsageObject::Mapping { members, .. }) => {
            let cached = match members.get("prompt_tokens_details") {
                Some(Value::Object(details)) => count(details.get("cached_tokens")),
                _ => 0,
            };
            (members, cached)
        }
        _ => return (0, 0),
    };
    match kind {
        TokenRateLimitType::Output => (count(members.get("completion_tokens")), 0),
        TokenRateLimitType::Input => (count(members.get("prompt_tokens")), cached),
        TokenRateLimitType::Total => (count(members.get("total_tokens")), cached),
    }
}

/// `_aggregate_only_total_tokens`: the total of a usage that reports no input/output split, which
/// a pass-through target pricing its own multi-model call charges under every rate limit type.
pub fn aggregate_only(usage: Option<&UsageObject>) -> i64 {
    let (prompt, completion, total) = match usage {
        Some(UsageObject::Chat { members }) => (
            members.get("prompt_tokens"),
            members.get("completion_tokens"),
            members.get("total_tokens"),
        ),
        Some(UsageObject::Responses { members }) => (
            members.get("input_tokens"),
            members.get("output_tokens"),
            members.get("total_tokens"),
        ),
        Some(UsageObject::Mapping { members, .. }) => (
            either(members.get("prompt_tokens"), members.get("input_tokens")),
            either(
                members.get("completion_tokens"),
                members.get("output_tokens"),
            ),
            members.get("total_tokens"),
        ),
        Some(UsageObject::Other) | None => return 0,
    };
    if truthy(prompt) || truthy(completion) {
        return 0;
    }
    count(total)
}

/// `resolve_reconciled_usage` of a usage object: its prompt, completion and cache-read counts,
/// cache reads still inside the prompt, or None when it reports neither prompt nor completion.
/// A dict in Anthropic's own shape reports `input_tokens` without its cache reads and cache
/// writes, so its prompt is the sum of the three.
pub fn reconciled(usage: Option<&UsageObject>) -> Option<Usage> {
    let (prompt_tokens, completion_tokens, cached_tokens) = match usage? {
        UsageObject::Chat { members } => (
            count(members.get("prompt_tokens")),
            count(members.get("completion_tokens")),
            count(members.get("cached_tokens")),
        ),
        UsageObject::Responses { members } => (
            count(members.get("input_tokens")),
            count(members.get("output_tokens")),
            count(members.get("cached_tokens")),
        ),
        UsageObject::Mapping { members, .. } => {
            let detailed = match either(
                members.get("prompt_tokens_details"),
                members.get("input_tokens_details"),
            ) {
                Some(Value::Object(details)) => details.get("cached_tokens"),
                _ => None,
            };
            let completion_tokens = count(either(
                members.get("completion_tokens"),
                members.get("output_tokens"),
            ));
            let cache_reads = count(members.get("cache_read_input_tokens"));
            let cache_writes = count(members.get("cache_creation_input_tokens"));
            let anthropic_shaped = !truthy(members.get("prompt_tokens"))
                && detailed.is_none()
                && (cache_reads != 0 || cache_writes != 0);
            if anthropic_shaped {
                (
                    count(members.get("input_tokens"))
                        .saturating_add(cache_writes)
                        .saturating_add(cache_reads),
                    completion_tokens,
                    cache_reads,
                )
            } else {
                (
                    count(either(
                        members.get("prompt_tokens"),
                        members.get("input_tokens"),
                    )),
                    completion_tokens,
                    count(either(detailed, members.get("cache_read_input_tokens"))),
                )
            }
        }
        UsageObject::Other => return None,
    };
    (prompt_tokens != 0 || completion_tokens != 0).then_some(Usage {
        prompt_tokens,
        completion_tokens,
        cached_tokens,
    })
}

/// `_response_usage` of a usage object: a dict's nested usage when it holds one.
fn source(usage: &UsageObject) -> Option<&UsageObject> {
    match usage {
        UsageObject::Mapping {
            nested: Some(nested),
            ..
        } => Some(nested),
        UsageObject::Other => None,
        usage => Some(usage),
    }
}

/// `_resolve_rerank_token_usage`: a rerank's billed tokens, from its `meta`.
fn rerank_usage(meta: &Map<String, Value>) -> Option<Usage> {
    if let Some(Value::Object(tokens)) = meta.get("tokens") {
        let input_tokens = count(tokens.get("input_tokens"));
        let output_tokens = count(tokens.get("output_tokens"));
        if input_tokens != 0 || output_tokens != 0 {
            return Some(Usage {
                prompt_tokens: input_tokens.max(0),
                completion_tokens: output_tokens.max(0),
                cached_tokens: 0,
            });
        }
    }
    if let Some(Value::Object(billed)) = meta.get("billed_units") {
        let total_tokens = count(billed.get("total_tokens"));
        if total_tokens != 0 {
            return Some(Usage {
                prompt_tokens: total_tokens.max(0),
                completion_tokens: 0,
                cached_tokens: 0,
            });
        }
    }
    None
}

impl LoggedObject {
    /// `_response_usage`: the usage object the response reports.
    fn usage(&self) -> Option<&UsageObject> {
        match self {
            Self::Usage { usage } | Self::Extraction { usage: Some(usage) } => Some(usage),
            Self::Mapping { usage } => source(usage),
            Self::Modeled { usage: Some(usage) } if !matches!(usage, UsageObject::Other) => {
                Some(usage)
            }
            _ => None,
        }
    }

    /// `resolve_reconciled_usage` of the response.
    fn reconciled(&self) -> Option<Usage> {
        if let Self::Rerank { meta: Some(meta) } = self
            && let Some(usage) = rerank_usage(meta)
        {
            return Some(usage);
        }
        reconciled(self.usage())
    }
}

impl LoggedCall {
    /// The `combined_usage_object` when it is litellm's `Usage`.
    fn combined_usage(&self) -> Option<&UsageObject> {
        match &self.combined {
            LoggedObject::Usage {
                usage: usage @ UsageObject::Chat { .. },
            } => Some(usage),
            _ => None,
        }
    }

    /// `_success_usage`: the usage a successful call charges. Responses LiteLLM does not model
    /// (a pass-through, a Responses WebSocket session) carry it in `combined_usage_object`.
    fn success_usage(&self) -> Option<&UsageObject> {
        match &self.response {
            LoggedObject::Modeled { usage } | LoggedObject::Extraction { usage } => usage.as_ref(),
            LoggedObject::Usage {
                usage: UsageObject::Responses { .. },
            } => None,
            LoggedObject::Events { usage } => self.combined_usage().or(usage.as_ref()),
            _ => self.combined_usage(),
        }
    }

    fn socket(&self) -> bool {
        self.call_type.as_deref() == Some(RESPONSES_WEBSOCKET_CALL_TYPE)
    }

    /// Background evaluations are exempt from the caller's token limits; foreground compaction is
    /// caller traffic and is charged.
    fn exempt(&self) -> bool {
        truthy(Some(&self.origin)) && self.origin.as_str() != Some(COMPACTION_ORIGIN)
    }

    fn targets(&self) -> TargetFacts {
        let metadata = &self.metadata;
        let member = |name: &str| text(metadata.get(name));
        TargetFacts {
            app_id: member("user_api_key_app_id"),
            app_group_id: member("user_api_key_app_group_id"),
            user_api_key_hash: member("user_api_key_hash"),
            user_api_key_user_id: member("user_api_key_user_id"),
            user_api_key_team_id: member("user_api_key_team_id"),
            user_api_key_org_id: member("user_api_key_org_id"),
            user_api_key_project_id: member("user_api_key_project_id"),
            end_user_id: text(either(
                Some(&self.user),
                metadata.get("user_api_key_end_user_id"),
            )),
            agent_id: member("agent_id"),
            session_id: text(either(metadata.get("session_id"), metadata.get("trace_id"))),
            customer_org_id: member("user_api_key_customer_org_id"),
            customer_team_id: member("user_api_key_customer_team_id"),
            customer_member_id: member("user_api_key_customer_member_id"),
        }
    }

    /// `f"{metadata.get('user_api_key_team_id')}"`, the owner of a key-owned per-team model scope.
    fn team_label(&self) -> String {
        match self.metadata.get("user_api_key_team_id") {
            None | Some(Value::Null) => "None".to_owned(),
            Some(Value::String(team)) => team.clone(),
            Some(Value::Bool(flag)) => if *flag { "True" } else { "False" }.to_owned(),
            Some(other) => other.to_string(),
        }
    }

    /// `recovered_partial_usage_tokens`: the usage a failed stream recovered before it broke,
    /// charged under the rate limit type, and its reconciled split.
    pub fn recovered(&self, kind: TokenRateLimitType) -> (i64, i64, Option<Usage>) {
        match self.combined_usage() {
            Some(usage @ UsageObject::Chat { members })
                if count(members.get("completion_tokens")) > 0 =>
            {
                (
                    charged_tokens(Some(usage), kind),
                    cached_tokens_left_out(Some(usage), kind),
                    reconciled(Some(usage)),
                )
            }
            _ => (0, 0, None),
        }
    }

    pub fn failure_facts(&self, kind: TokenRateLimitType) -> FailureFacts {
        let (tpm_actual, cached_tokens_left_out, partial_usage) = self.recovered(kind);
        FailureFacts {
            call_id: self.call_id.clone(),
            tpm_actual,
            cached_tokens_left_out,
            partial_usage,
        }
    }

    pub fn post_call_failure_facts(&self, kind: TokenRateLimitType) -> PostCallFailureFacts {
        let (tpm_actual, cached_tokens_left_out, partial_usage) = self.recovered(kind);
        PostCallFailureFacts {
            tpm_actual,
            cached_tokens_left_out,
            partial_usage,
        }
    }

    /// The facts a successful call settles with.
    pub fn success_facts(
        &self,
        resolved: &Resolved,
        kind: TokenRateLimitType,
        keep_slot: bool,
    ) -> SuccessFacts {
        let usage = self.success_usage();
        let total_tokens = match charged_tokens(usage, kind) {
            0 => aggregate_only(usage),
            charged => charged,
        };
        let settled = match usage {
            Some(usage) => reconciled(source(usage)),
            None => self.response.reconciled(),
        };
        let socket = self.socket();
        let ptu_usage = if usage.is_none() || socket {
            self.response.reconciled()
        } else {
            settled
        };
        let ptu_total = if socket { 0 } else { total_tokens };
        let targets = self.targets();
        let models = resolved
            .models
            .iter()
            .map(|(requested, model)| {
                let ptu = match (&model.ptu_ceiling, &targets.user_api_key_team_id) {
                    (Some(ceiling), Some(team)) => Some(PtuFacts {
                        value: format!("{team}:{}", ceiling.model_group),
                        actual_tokens: settlement_tokens(
                            Some(ceiling),
                            ptu_usage.as_ref(),
                            ptu_total,
                        ),
                    }),
                    _ => None,
                };
                (
                    requested.clone(),
                    ModelFacts {
                        candidates: model.candidates.clone(),
                        ptu,
                    },
                )
            })
            .collect();
        SuccessFacts {
            call_id: self.call_id.clone(),
            exempt: self.exempt(),
            total_tokens,
            cached_tokens_left_out: cached_tokens_left_out(usage, kind),
            team_label: self.team_label(),
            targets,
            served_model: resolved.served_model.clone(),
            models,
            policy: resolved.policy,
            usage: settled,
            io_usage: IoUsage {
                response: self.response.reconciled(),
                combined: self.combined.reconciled(),
                aggregate_total: match aggregate_only(self.response.usage()) {
                    0 => aggregate_only(self.combined.usage()),
                    total => total,
                },
            },
            keep_slot,
        }
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use serde_json::json;

    use super::*;

    fn members(value: Value) -> Map<String, Value> {
        match value {
            Value::Object(members) => members,
            _ => Map::new(),
        }
    }

    fn chat(value: Value) -> UsageObject {
        UsageObject::Chat {
            members: members(value),
        }
    }

    #[rstest]
    #[case::total_without_cache_reads(TokenRateLimitType::Total, 70)]
    #[case::input_without_cache_reads(TokenRateLimitType::Input, 20)]
    #[case::output(TokenRateLimitType::Output, 50)]
    fn a_chat_usage_charges_by_the_rate_limit_type(
        #[case] kind: TokenRateLimitType,
        #[case] expected: i64,
    ) {
        let usage = chat(json!({
            "prompt_tokens": 50, "completion_tokens": 50, "total_tokens": 100, "cached_tokens": 30,
        }));
        assert_eq!(charged_tokens(Some(&usage), kind), expected);
    }

    #[rstest]
    #[case::output_reads_no_cache(TokenRateLimitType::Output, 0)]
    #[case::input(TokenRateLimitType::Input, 30)]
    #[case::total(TokenRateLimitType::Total, 30)]
    fn the_cached_tokens_left_out_are_what_charging_subtracts(
        #[case] kind: TokenRateLimitType,
        #[case] expected: i64,
    ) {
        let usage = chat(json!({
            "prompt_tokens": 50, "completion_tokens": 50, "total_tokens": 100, "cached_tokens": 30,
        }));
        assert_eq!(cached_tokens_left_out(Some(&usage), kind), expected);
    }

    #[rstest]
    fn cache_reads_above_the_prompt_leave_out_only_the_prompt() {
        let usage = chat(json!({"prompt_tokens": 10, "cached_tokens": 30}));
        assert_eq!(
            cached_tokens_left_out(Some(&usage), TokenRateLimitType::Input),
            10
        );
    }

    #[test]
    fn a_responses_usage_charges_only_through_its_aggregate() {
        let split = UsageObject::Responses {
            members: members(json!({"input_tokens": 5, "output_tokens": 7, "total_tokens": 12})),
        };
        assert_eq!(charged_tokens(Some(&split), TokenRateLimitType::Total), 0);
        assert_eq!(aggregate_only(Some(&split)), 0);
        let aggregate = UsageObject::Mapping {
            members: members(json!({"total_tokens": 42})),
            nested: None,
        };
        assert_eq!(aggregate_only(Some(&aggregate)), 42);
    }

    #[test]
    fn a_dict_reconciles_through_its_nested_usage() {
        let response = LoggedObject::Mapping {
            usage: UsageObject::Mapping {
                members: members(json!({"usage": {}})),
                nested: Some(Box::new(UsageObject::Mapping {
                    members: members(json!({
                        "input_tokens": 9, "output_tokens": 4,
                        "input_tokens_details": {"cached_tokens": 3},
                    })),
                    nested: None,
                })),
            },
        };
        assert_eq!(
            response.reconciled(),
            Some(Usage {
                prompt_tokens: 9,
                completion_tokens: 4,
                cached_tokens: 3,
            })
        );
    }

    #[rstest]
    #[case::anthropic_shape_adds_cache_writes_and_reads_to_the_prompt(
        json!({
            "input_tokens": 40, "output_tokens": 25,
            "cache_creation_input_tokens": 60, "cache_read_input_tokens": 900,
        }),
        (1000, 25, 900, 100)
    )]
    #[case::anthropic_shape_with_cache_writes_only(
        json!({"input_tokens": 40, "output_tokens": 25, "cache_creation_input_tokens": 60}),
        (100, 25, 0, 100)
    )]
    #[case::prompt_tokens_already_hold_the_cache_reads(
        json!({"prompt_tokens": 100, "completion_tokens": 25, "cache_read_input_tokens": 30}),
        (100, 25, 30, 70)
    )]
    #[case::input_tokens_details_already_hold_the_cache_reads(
        json!({
            "input_tokens": 100, "output_tokens": 25,
            "input_tokens_details": {"cached_tokens": 30}, "cache_creation_input_tokens": 60,
        }),
        (100, 25, 30, 70)
    )]
    fn a_plain_dict_reconciles_its_billable_input(
        #[case] usage: Value,
        #[case] expected: (i64, i64, i64, i64),
    ) {
        let resolved = reconciled(Some(&UsageObject::Mapping {
            members: members(usage),
            nested: None,
        }));
        assert_eq!(
            resolved.map(|usage| (
                usage.prompt_tokens,
                usage.completion_tokens,
                usage.cached_tokens,
                usage.billable_input_tokens(),
            )),
            Some(expected)
        );
    }

    #[test]
    fn a_rerank_reconciles_its_billed_units() {
        let response = LoggedObject::Rerank {
            meta: Some(members(json!({"billed_units": {"total_tokens": 11}}))),
        };
        assert_eq!(
            response.reconciled(),
            Some(Usage {
                prompt_tokens: 11,
                completion_tokens: 0,
                cached_tokens: 0,
            })
        );
    }

    #[test]
    fn a_failed_stream_recovers_its_partial_usage_only_once_it_produced_output() {
        let partial = |completion: i64| LoggedCall {
            combined: LoggedObject::Usage {
                usage: chat(json!({
                    "prompt_tokens": 10, "completion_tokens": completion, "total_tokens": 10 + completion,
                })),
            },
            ..LoggedCall::default()
        };
        assert_eq!(
            partial(0).recovered(TokenRateLimitType::Total),
            (0, 0, None)
        );
        assert_eq!(
            partial(5).recovered(TokenRateLimitType::Total),
            (
                15,
                0,
                Some(Usage {
                    prompt_tokens: 10,
                    completion_tokens: 5,
                    cached_tokens: 0,
                })
            )
        );
    }

    #[test]
    fn an_unmodeled_response_charges_its_combined_usage() {
        let call = LoggedCall {
            call_type: Some(RESPONSES_WEBSOCKET_CALL_TYPE.to_owned()),
            origin: json!("autorouter_compaction"),
            metadata: members(json!({"user_api_key_team_id": "team-a", "session_id": 7})),
            response: LoggedObject::Other,
            combined: LoggedObject::Usage {
                usage: chat(json!({"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7})),
            },
            ..LoggedCall::default()
        };
        let facts = call.success_facts(&Resolved::default(), TokenRateLimitType::Total, true);
        assert_eq!(facts.total_tokens, 7);
        assert!(!facts.exempt);
        assert_eq!(facts.team_label, "team-a");
        assert_eq!(facts.targets.session_id.as_deref(), Some("7"));
        assert_eq!(
            facts.io_usage.combined,
            Some(Usage {
                prompt_tokens: 3,
                completion_tokens: 4,
                cached_tokens: 0,
            })
        );
    }
}
