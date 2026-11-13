use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ModelPreference {
    pub model: Option<String>,
    pub alternates: Option<Vec<String>>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct KeyIdentity {
    pub key_hash: Option<String>,
    pub api_key: Option<String>,
    pub key_alias: Option<String>,
    pub key_name: Option<String>,
    pub user_id: Option<String>,
    pub user_email: Option<String>,
    pub team_id: Option<String>,
    pub team_alias: Option<String>,
    pub org_id: Option<String>,
    pub org_alias: Option<String>,
    pub project_id: Option<String>,
    pub project_alias: Option<String>,
    pub end_user_id: Option<String>,
    pub agent_id: Option<String>,
    pub invoked_agent_id: Option<String>,
    pub team_priority: Option<String>,
    pub key_priority: Option<String>,
    pub model_preferences: Option<Vec<ModelPreference>>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct EstimatedOutputTokens {
    pub key_per_model: Option<i64>,
    pub key_default: Option<i64>,
    pub team_per_model: Option<i64>,
    pub team_default: Option<i64>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct KeyLimits {
    pub rpm_limit: Option<i64>,
    pub tpm_limit: Option<i64>,
    pub max_parallel_requests: Option<i64>,
    pub budget_throttle_pct: Option<f64>,
    pub user_rpm_limit: Option<i64>,
    pub user_tpm_limit: Option<i64>,
    pub team_rpm_limit: Option<i64>,
    pub team_tpm_limit: Option<i64>,
    pub team_member_rpm_limit: Option<i64>,
    pub team_member_tpm_limit: Option<i64>,
    pub end_user_rpm_limit: Option<i64>,
    pub end_user_tpm_limit: Option<i64>,
    pub organization_rpm_limit: Option<i64>,
    pub organization_tpm_limit: Option<i64>,
    pub rpm_limit_type: Option<String>,
    pub tpm_limit_type: Option<String>,
    pub estimated_output_tokens: Option<EstimatedOutputTokens>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct BudgetPlanCounter {
    pub counter_key: Option<String>,
    pub max_budget: Option<f64>,
    pub fallback_spend: Option<f64>,
    pub entity_type: Option<String>,
    pub entity_id: Option<String>,
    pub spend_log_entity_id: Option<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct BudgetPlan {
    pub counters: Option<Vec<BudgetPlanCounter>>,
    pub reservation_cost: Option<f64>,
    pub input_cost: Option<f64>,
    pub input_tokens: Option<i64>,
    pub throttle_counter: Option<String>,
    pub fail_closed: Option<bool>,
    pub status_code: Option<u16>,
    pub reservation_id: Option<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ReservationEntry {
    pub counter_key: Option<String>,
    pub entity_type: Option<String>,
    pub entity_id: Option<String>,
    pub reserved_cost: Option<f64>,
    pub applied_adjustment: Option<f64>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct BudgetReservation {
    pub reserved_cost: Option<f64>,
    pub entries: Option<Vec<ReservationEntry>>,
    pub finalized: Option<bool>,
    pub input_cost: Option<f64>,
    pub input_tokens: Option<i64>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct RateLimitStatus {
    pub code: Option<String>,
    pub current_limit: Option<i64>,
    pub limit_remaining: Option<i64>,
    pub rate_limit_type: Option<String>,
    pub descriptor_key: Option<String>,
    pub descriptor_value: Option<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct RateLimitStatuses {
    pub overall_code: Option<String>,
    pub statuses: Option<Vec<RateLimitStatus>>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct OutputCap {
    pub explicit: Option<i64>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Moment {
    pub year: i32,
    pub month: u8,
    pub day: u8,
    pub hour: u8,
    pub minute: u8,
    pub second: u8,
    pub microsecond: u32,
    pub utc_offset_seconds: Option<i32>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct AttemptTiming {
    pub start: Option<Moment>,
    pub end: Option<Moment>,
    pub api_call_start: Option<Moment>,
    pub completion_start: Option<Moment>,
    pub completion_start_present: Option<bool>,
    pub queue_seconds: Option<f64>,
    pub overhead_ms: Option<f64>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct UsageRecord {
    pub prompt_tokens: Option<i64>,
    pub completion_tokens: Option<i64>,
    pub total_tokens: Option<i64>,
    pub cached_tokens: Option<i64>,
    pub prompt_audio_tokens: Option<i64>,
    pub cache_write_tokens: Option<i64>,
    pub cache_creation_tokens: Option<i64>,
    pub reasoning_tokens: Option<i64>,
    pub completion_audio_tokens: Option<i64>,
    pub cache_read_input_tokens: Option<i64>,
    pub cache_creation_input_tokens: Option<i64>,
    pub duration_seconds: Option<f64>,
    pub output_image_count: Option<i64>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Usage {
    pub response: Option<UsageRecord>,
    pub streamed: Option<UsageRecord>,
    pub combined: Option<UsageRecord>,
    pub upstream: Option<UsageRecord>,
    pub socket_session: Option<UsageRecord>,
    pub logged: Option<UsageRecord>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Outcome {
    Success,
    Failure,
    FallbackSuccess,
    FallbackFailure,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ZeroCost {
    pub reason: Option<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Cost {
    pub response_cost: Option<f64>,
    pub zero_cost: Option<ZeroCost>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ModelInfo {
    pub id: Option<String>,
    pub tpm: Option<i64>,
    pub rpm: Option<i64>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Deployment {
    pub model_info: Option<ModelInfo>,
    pub provider: Option<String>,
    pub tpm: Option<i64>,
    pub rpm: Option<i64>,
    pub original_model_group: Option<String>,
    pub params_present: Option<bool>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct GuardrailEntry {
    pub modes: Option<Vec<String>>,
    pub duration: Option<f64>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct RateLimitState {
    pub key_model_group: Option<String>,
    pub key_remaining_requests: Option<f64>,
    pub key_remaining_tokens: Option<f64>,
    pub limiter_headers: Option<BTreeMap<String, f64>>,
    pub provider_remaining_requests: Option<f64>,
    pub provider_remaining_tokens: Option<f64>,
    pub hidden_params_present: Option<bool>,
    pub additional_headers_present: Option<bool>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct LoggedRequest {
    pub client_ip: Option<String>,
    pub user_agent: Option<String>,
    pub service_tier: Option<String>,
    pub stream: Option<bool>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct McpToolCall {
    pub name: Option<String>,
    pub server_name: Option<String>,
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use serde_json::{Value, json};

    use super::*;
    use crate::field::{Field, TypedField, typed};

    const RECORDS: &str = include_str!("../records.json");

    fn round_trip<F: TypedField>(sample: &Value) -> Value
    where
        F::Value: Serialize,
    {
        let decoded: F::Value = serde_json::from_value(sample.clone())
            .unwrap_or_else(|error| panic!("{:?} sample does not decode: {error}", F::FIELD));
        serde_json::to_value(decoded).unwrap()
    }

    fn decoded(field: Field, sample: &Value) -> Value {
        match field {
            Field::KeyIdentity => round_trip::<typed::KeyIdentity>(sample),
            Field::Deployment => round_trip::<typed::Deployment>(sample),
            Field::RequestTags => round_trip::<typed::RequestTags>(sample),
            Field::GuardrailInfo => round_trip::<typed::GuardrailInfo>(sample),
            Field::Outcome => round_trip::<typed::Outcome>(sample),
            Field::Usage => round_trip::<typed::Usage>(sample),
            Field::Cost => round_trip::<typed::Cost>(sample),
            Field::KeyLimits => round_trip::<typed::KeyLimits>(sample),
            Field::BudgetPlan => round_trip::<typed::BudgetPlan>(sample),
            Field::BudgetReservation => round_trip::<typed::BudgetReservation>(sample),
            Field::RateLimitStatuses => round_trip::<typed::RateLimitStatuses>(sample),
            Field::OutputCap => round_trip::<typed::OutputCap>(sample),
            Field::AttemptTiming => round_trip::<typed::AttemptTiming>(sample),
            Field::RateLimitState => round_trip::<typed::RateLimitState>(sample),
            Field::LoggedRequest => round_trip::<typed::LoggedRequest>(sample),
            Field::McpToolCall => round_trip::<typed::McpToolCall>(sample),
            Field::LogId => round_trip::<typed::LogId>(sample),
            Field::ColdStorageObjectKey => round_trip::<typed::ColdStorageObjectKey>(sample),
            other => panic!("{other:?} has no typed record sample"),
        }
    }

    #[rstest]
    fn every_python_record_sample_decodes_to_its_serde_type_and_back_unchanged() {
        let samples: Value = serde_json::from_str(RECORDS).unwrap();
        let samples = samples.as_object().unwrap();
        for (name, sample) in samples {
            let field: Field = serde_json::from_value(json!(name)).unwrap();
            assert_eq!(&decoded(field, sample), sample, "{name}");
        }
        let typed: Vec<&str> = samples.keys().map(String::as_str).collect();
        assert_eq!(typed.len(), 18);
    }

    #[rstest]
    fn a_member_python_did_not_send_decodes_as_absent() {
        let identity: KeyIdentity = serde_json::from_value(json!({"team_id": "t"})).unwrap();
        assert_eq!(identity.team_id.as_deref(), Some("t"));
        assert_eq!(identity.key_alias, None);
    }

    #[rstest]
    fn a_member_the_record_does_not_know_is_rejected() {
        assert!(serde_json::from_value::<KeyIdentity>(json!({"team": "t"})).is_err());
    }

    #[rstest]
    #[case::success("success", Outcome::Success)]
    #[case::fallback("fallback_success", Outcome::FallbackSuccess)]
    #[case::fallback_failure("fallback_failure", Outcome::FallbackFailure)]
    fn the_outcome_keeps_the_router_event_spellings(#[case] name: &str, #[case] outcome: Outcome) {
        assert_eq!(
            serde_json::from_value::<Outcome>(json!(name)).unwrap(),
            outcome
        );
    }
}
