use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::scalar::Scalar;

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct KeyIdentity {
    pub rejected: Option<Vec<String>>,
    pub key_hash: Option<Scalar>,
    pub api_key: Option<Scalar>,
    pub key_alias: Option<Scalar>,
    pub key_name: Option<Scalar>,
    pub user_id: Option<Scalar>,
    pub user_email: Option<Scalar>,
    pub team_id: Option<Scalar>,
    pub team_alias: Option<Scalar>,
    pub org_id: Option<Scalar>,
    pub org_alias: Option<Scalar>,
    pub project_id: Option<Scalar>,
    pub project_alias: Option<Scalar>,
    pub end_user_id: Option<Scalar>,
    pub agent_id: Option<Scalar>,
    pub invoked_agent_id: Option<Scalar>,
    pub team_priority: Option<Scalar>,
    pub key_priority: Option<Scalar>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MetadataKind {
    Mapping,
    Missing,
    Null,
    Other,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct LoggedKeyIdentity {
    pub rejected: Option<Vec<String>>,
    pub metadata_kind: Option<MetadataKind>,
    pub missing_keys: Option<Vec<String>>,
    pub key_hash: Option<Scalar>,
    pub key_alias: Option<Scalar>,
    pub key_spend: Option<Scalar>,
    pub key_max_budget: Option<Scalar>,
    pub key_budget_reset_at: Option<Scalar>,
    pub user_id: Option<Scalar>,
    pub user_email: Option<Scalar>,
    pub user_spend: Option<Scalar>,
    pub user_max_budget: Option<Scalar>,
    pub team_id: Option<Scalar>,
    pub team_alias: Option<Scalar>,
    pub team_spend: Option<Scalar>,
    pub team_max_budget: Option<Scalar>,
    pub org_id: Option<Scalar>,
    pub org_alias: Option<Scalar>,
    pub project_id: Option<Scalar>,
    pub project_alias: Option<Scalar>,
    pub end_user_id: Option<Scalar>,
    pub request_route: Option<Scalar>,
    pub auth_metadata_priority: Option<Scalar>,
    pub agent_id: Option<Scalar>,
    pub request_user: Option<Scalar>,
    pub cost_tracking_end_user_id: Option<Scalar>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct EstimatedOutputTokens {
    pub rejected: Option<Vec<String>>,
    pub key_per_model: Option<i64>,
    pub key_default: Option<i64>,
    pub team_per_model: Option<i64>,
    pub team_default: Option<i64>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct AgentPolicy {
    pub rejected: Option<Vec<String>>,
    pub rpm_limit: Option<Scalar>,
    pub tpm_limit: Option<Scalar>,
    pub session_rpm_limit: Option<Scalar>,
    pub session_tpm_limit: Option<Scalar>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct KeyLimits {
    pub rejected: Option<Vec<String>>,
    pub rpm_limit: Option<Scalar>,
    pub tpm_limit: Option<Scalar>,
    pub max_parallel_requests: Option<Scalar>,
    pub budget_throttle_pct: Option<Scalar>,
    pub user_rpm_limit: Option<Scalar>,
    pub user_tpm_limit: Option<Scalar>,
    pub team_rpm_limit: Option<Scalar>,
    pub team_tpm_limit: Option<Scalar>,
    pub team_member_rpm_limit: Option<Scalar>,
    pub team_member_tpm_limit: Option<Scalar>,
    pub end_user_rpm_limit: Option<Scalar>,
    pub end_user_tpm_limit: Option<Scalar>,
    pub organization_rpm_limit: Option<Scalar>,
    pub organization_tpm_limit: Option<Scalar>,
    pub rpm_limit_type: Option<String>,
    pub tpm_limit_type: Option<String>,
    pub estimated_output_tokens: Option<EstimatedOutputTokens>,
    pub managed_agent_policy: Option<AgentPolicy>,
    pub invoked_agent_policy: Option<AgentPolicy>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct BudgetPlanCounter {
    pub counter_key: Option<String>,
    #[serde(with = "crate::scalar::float")]
    pub max_budget: Option<f64>,
    #[serde(with = "crate::scalar::float")]
    pub fallback_spend: Option<f64>,
    pub entity_type: Option<String>,
    pub entity_id: Option<String>,
    pub spend_log_entity_id: Option<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct BudgetPlan {
    pub rejected: Option<Vec<String>>,
    pub counters: Option<Vec<BudgetPlanCounter>>,
    #[serde(with = "crate::scalar::float")]
    pub reservation_cost: Option<f64>,
    #[serde(with = "crate::scalar::float")]
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
    pub rejected: Option<Vec<String>>,
    pub counter_key: Option<String>,
    pub entity_type: Option<String>,
    pub entity_id: Option<String>,
    #[serde(with = "crate::scalar::float")]
    pub reserved_cost: Option<f64>,
    #[serde(with = "crate::scalar::float")]
    pub applied_adjustment: Option<f64>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct BudgetReservation {
    pub rejected: Option<Vec<String>>,
    #[serde(with = "crate::scalar::float")]
    pub reserved_cost: Option<f64>,
    pub entries: Option<Vec<Option<ReservationEntry>>>,
    pub entries_is_list: Option<bool>,
    pub finalized: Option<bool>,
    pub callback_bound: Option<bool>,
    #[serde(with = "crate::scalar::float")]
    pub input_cost: Option<f64>,
    pub input_tokens: Option<i64>,
    pub not_a_mapping: Option<Scalar>,
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
    pub rejected: Option<Vec<String>>,
    pub overall_code: Option<String>,
    pub statuses: Option<Vec<RateLimitStatus>>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct LimiterRequest {
    pub rejected: Option<Vec<String>>,
    pub session_id: Option<String>,
    pub session_id_error_type: Option<String>,
    pub session_id_error_text: Option<String>,
    pub tags: Option<Vec<String>>,
    pub resolved_agent_id: Option<Scalar>,
    pub resolved_agent_id_error_type: Option<String>,
    pub resolved_agent_id_error_text: Option<String>,
    pub fallback_on_saturation: Option<bool>,
    pub fallback_on_saturation_invalid: Option<String>,
    pub fallback_walk_armed: Option<bool>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct OutputCap {
    pub rejected: Option<Vec<String>>,
    pub explicit: Option<Scalar>,
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
    pub rejected: Option<Vec<String>>,
    pub start: Option<Moment>,
    pub end: Option<Moment>,
    pub api_call_start: Option<Moment>,
    pub completion_start: Option<Moment>,
    pub completion_start_present: Option<bool>,
    pub queue_seconds: Option<Scalar>,
    pub overhead_ms: Option<Scalar>,
    pub logged_start: Option<Moment>,
    pub logged_end: Option<Moment>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct UsageRecord {
    pub rejected: Option<Vec<String>>,
    pub prompt_tokens: Option<Scalar>,
    pub completion_tokens: Option<Scalar>,
    pub total_tokens: Option<Scalar>,
    pub cached_tokens: Option<Scalar>,
    pub prompt_audio_tokens: Option<Scalar>,
    pub cache_write_tokens: Option<Scalar>,
    pub cache_creation_tokens: Option<Scalar>,
    pub reasoning_tokens: Option<Scalar>,
    pub completion_audio_tokens: Option<Scalar>,
    pub cache_read_input_tokens: Option<Scalar>,
    pub cache_creation_input_tokens: Option<Scalar>,
    pub duration_seconds: Option<Scalar>,
    pub output_image_count: Option<Scalar>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Usage {
    pub rejected: Option<Vec<String>>,
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

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ZeroCost {
    pub rejected: Option<Vec<String>>,
    pub reason: Option<Scalar>,
    pub reason_present: Option<bool>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Cost {
    pub rejected: Option<Vec<String>>,
    pub response_cost: Option<Scalar>,
    pub zero_cost: Option<ZeroCost>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ModelInfo {
    pub rejected: Option<Vec<String>>,
    pub id: Option<Scalar>,
    pub tpm: Option<Scalar>,
    pub rpm: Option<Scalar>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Deployment {
    pub rejected: Option<Vec<String>>,
    pub model_info: Option<ModelInfo>,
    pub provider: Option<Scalar>,
    pub tpm: Option<Scalar>,
    pub rpm: Option<Scalar>,
    pub original_model_group: Option<String>,
    pub params_present: Option<bool>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct GuardrailEntry {
    pub rejected: Option<Vec<String>>,
    pub mapping: Option<bool>,
    pub modes: Option<Vec<Option<Scalar>>>,
    pub duration: Option<Scalar>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct RateLimitState {
    pub rejected: Option<Vec<String>>,
    pub key_model_group: Option<Scalar>,
    pub key_remaining_requests: Option<Scalar>,
    pub key_remaining_tokens: Option<Scalar>,
    pub limiter_headers: Option<BTreeMap<String, Scalar>>,
    pub provider_remaining_requests: Option<Scalar>,
    pub provider_remaining_tokens: Option<Scalar>,
    pub hidden_params_present: Option<bool>,
    pub additional_headers_present: Option<bool>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct LoggedModelGroup {
    pub rejected: Option<Vec<String>>,
    pub value: Option<Scalar>,
    pub error_type: Option<String>,
    pub error_text: Option<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct InputSequenceLength {
    pub rejected: Option<Vec<String>>,
    pub bucket: Option<String>,
    pub error_type: Option<String>,
    pub error_text: Option<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct LoggedRequest {
    pub rejected: Option<Vec<String>>,
    pub client_ip: Option<Scalar>,
    pub user_agent: Option<Scalar>,
    pub service_tier: Option<String>,
    pub stream: Option<Scalar>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MetadataLabel {
    pub name: String,
    pub value: String,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct McpToolCall {
    pub rejected: Option<Vec<String>>,
    pub name: Option<Scalar>,
    pub server_name: Option<Scalar>,
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
            Field::LoggedKeyIdentity => round_trip::<typed::LoggedKeyIdentity>(sample),
            Field::LoggedModelGroup => round_trip::<typed::LoggedModelGroup>(sample),
            Field::MetadataLabels => round_trip::<typed::MetadataLabels>(sample),
            Field::InputSequenceLength => round_trip::<typed::InputSequenceLength>(sample),
            Field::LimiterRequest => round_trip::<typed::LimiterRequest>(sample),
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
        assert_eq!(typed.len(), 23);
    }

    #[rstest]
    fn a_member_python_did_not_send_decodes_as_absent() {
        let identity: KeyIdentity = serde_json::from_value(json!({"team_id": "t"})).unwrap();
        assert_eq!(identity.team_id, Some(Scalar::Text("t".to_owned())));
        assert_eq!(identity.key_alias, None);
    }

    #[rstest]
    fn a_member_the_record_does_not_know_is_rejected() {
        assert!(serde_json::from_value::<KeyIdentity>(json!({"team": "t"})).is_err());
    }

    #[rstest]
    fn an_infinite_reservation_cost_and_a_non_mapping_entry_survive_a_write_back() {
        let sent = json!({
            "reserved_cost": litellm_host::raw_json::encode("Infinity"),
            "entries": [{"counter_key": "k", "reserved_cost": 0.5}, null],
            "callback_bound": false,
        });
        let reservation: BudgetReservation = serde_json::from_value(sent).unwrap();
        assert_eq!(reservation.reserved_cost, Some(f64::INFINITY));
        assert_eq!(reservation.entries.as_ref().map(Vec::len), Some(2));
        assert_eq!(reservation.callback_bound, Some(false));
        let written = serde_json::to_value(&reservation).unwrap();
        assert_eq!(
            written["reserved_cost"],
            json!(litellm_host::raw_json::encode("Infinity"))
        );
        assert_eq!(
            serde_json::from_value::<BudgetReservation>(written).unwrap(),
            reservation
        );
    }

    #[rstest]
    fn reservation_entries_python_held_as_a_tuple_say_they_were_not_a_list() {
        let reservation: BudgetReservation = serde_json::from_value(json!({
            "entries": [{"counter_key": "k"}, null],
            "entries_is_list": false,
        }))
        .unwrap();
        assert_eq!(reservation.entries_is_list, Some(false));
        assert_eq!(
            reservation.entries,
            Some(vec![
                Some(ReservationEntry {
                    counter_key: Some("k".to_owned()),
                    ..ReservationEntry::default()
                }),
                None
            ])
        );
    }

    #[rstest]
    fn a_reservation_python_held_as_another_type_is_not_an_empty_one() {
        let held: BudgetReservation =
            serde_json::from_value(json!({"not_a_mapping": "reserved"})).unwrap();
        let empty: BudgetReservation = serde_json::from_value(json!({})).unwrap();
        assert_eq!(
            held.not_a_mapping.as_ref().map(Scalar::type_name),
            Some("str")
        );
        assert_ne!(held, empty);
    }

    #[rstest]
    fn identity_members_carry_a_stringified_value_apart_from_text() {
        let identity: LoggedKeyIdentity = serde_json::from_value(json!({
            "team_id": {"text": "['t1']", "truthy": true, "hashable": false, "integer": false, "type_name": "list", "hash_error": "unhashable type: 'list'"},
            "team_alias": "['t1']",
        }))
        .unwrap();
        assert_eq!(
            identity.team_id,
            Some(Scalar::Stringified {
                text: "['t1']".to_owned(),
                truthy: true,
                hashable: false,
                integer: false,
                type_name: "list".to_owned(),
                hash_error: Some("unhashable type: 'list'".to_owned()),
            })
        );
        assert_ne!(identity.team_id, identity.team_alias);
    }

    #[rstest]
    #[case::i64(json!({"explicit": 64}), Scalar::Int(64))]
    #[case::past_i64(json!({"explicit": litellm_host::raw_json::encode("1180591620717411303424")}), Scalar::Int(1 << 70))]
    fn an_output_cap_carries_an_int_past_i64(#[case] sent: Value, #[case] explicit: Scalar) {
        let cap: OutputCap = serde_json::from_value(sent).unwrap();
        assert_eq!(cap.explicit, Some(explicit));
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
