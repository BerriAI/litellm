use litellm_callbacks::{KeyIdentity, LoggedModelGroup, Scalar, Usage, UsageRecord};
use litellm_callbacks_parallel_request_limiter::{Policy, PolicyFacts};
use litellm_python_compat::{json::loaded_equals, repr::float_repr};
use serde::{Deserialize, Serialize};
use serde_json::{Number, Value};

use crate::error::CapacityError;

pub const MODEL_SATURATION_CHECK: &str = "model_saturation_check";
pub const PRIORITY_MODEL: &str = "priority_model";
pub const PRIORITY_CLASS_SHARE_AT_SATURATION: &str = "priority_class_share_saturated";
pub const SATURATION_FALLBACK_DESCRIPTOR_KEY: &str = "model_saturation_fallback";
pub const DEFAULT_CLASS: &str = "\u{1e}default";

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct RateLimit {
    pub requests_per_unit: Option<i64>,
    pub tokens_per_unit: Option<i64>,
    pub window_size: Option<i64>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Descriptor {
    pub key: String,
    pub value: String,
    pub rate_limit: Option<RateLimit>,
}

impl Descriptor {
    pub fn tokens_per_unit(&self) -> Option<i64> {
        self.rate_limit.as_ref()?.tokens_per_unit
    }

    pub fn counter(&self, kind: CounterKind) -> Counter {
        Counter {
            key: self.key.clone(),
            value: self.value.clone(),
            kind,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CounterKind {
    Requests,
    Tokens,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Counter {
    pub key: String,
    pub value: String,
    pub kind: CounterKind,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Increment {
    pub requests: i64,
    pub tokens: i64,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Window {
    pub counter: Counter,
    pub start: String,
    pub backend: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Status {
    pub code: String,
    pub current_limit: Number,
    pub limit_remaining: Option<Number>,
    pub rate_limit_type: String,
    pub descriptor_key: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub descriptor_value: Option<String>,
}

impl Status {
    pub fn over_limit(&self) -> bool {
        self.code == "OVER_LIMIT"
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LimiterResponse {
    pub overall_code: String,
    pub statuses: Vec<Status>,
    #[serde(default)]
    pub reservation_windows: Vec<Window>,
}

impl LimiterResponse {
    pub fn over_limit(&self) -> bool {
        self.overall_code == "OVER_LIMIT"
    }

    pub fn window_of(&self, counter: &Counter) -> Option<Window> {
        self.reservation_windows
            .iter()
            .find(|window| &window.counter == counter)
            .cloned()
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case")]
pub enum ReservationOp {
    Increment {
        counter: Counter,
        amount: i64,
        ttl: Option<i64>,
    },
    Guarded {
        counter: Counter,
        amount: i64,
        ttl: Option<i64>,
        expected_window_start: String,
        backend: String,
    },
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct GroupInfo {
    pub tpm: Option<i64>,
    pub rpm: Option<i64>,
    pub input_cost_per_token: Option<f64>,
    pub output_cost_per_token: Option<f64>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Charge {
    Reported(i64),
    Estimated(i64),
}

impl Charge {
    pub fn against(self, reserved: i64) -> i64 {
        match self {
            Self::Reported(tokens) => tokens,
            Self::Estimated(_) => reserved,
        }
    }

    pub fn tokens(self) -> i64 {
        match self {
            Self::Reported(tokens) | Self::Estimated(tokens) => tokens,
        }
    }

    pub fn map(self, f: impl FnOnce(i64) -> i64) -> Self {
        match self {
            Self::Reported(tokens) => Self::Reported(f(tokens)),
            Self::Estimated(tokens) => Self::Estimated(f(tokens)),
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TokenRateLimitType {
    Input,
    Output,
    #[default]
    Total,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Settings {
    pub priority_reservation: Option<Vec<(Value, Value)>>,
    pub default_priority: f64,
    pub saturation_threshold: f64,
    pub class_split: Vec<ClassShareEntity>,
    pub saturation_check_cache_ttl: Value,
    pub window_size: i64,
    pub rate_limit_type: TokenRateLimitType,
    #[serde(default)]
    pub licensed: bool,
    #[serde(default)]
    pub policy: PolicyFacts,
}

impl Settings {
    pub fn policy(&self) -> Policy {
        self.policy.into()
    }

    pub fn reserves<'a>(
        &self,
        priority: Option<&'a Priority>,
    ) -> Result<Option<&'a Value>, CapacityError> {
        let (Some(priority), Some(reservation)) = (
            priority.filter(|priority| !priority.value.is_null()),
            &self.priority_reservation,
        ) else {
            return Ok(None);
        };
        if let Some(error) = &priority.hash_error {
            return Err(CapacityError::Type(error.clone()));
        }
        Ok(reservation
            .iter()
            .any(|(name, _)| loaded_equals(name, &priority.value))
            .then_some(&priority.value))
    }

    pub fn reserves_name<'a>(&self, priority: Option<&'a str>) -> Option<&'a str> {
        let reservation = self.priority_reservation.as_ref()?;
        let priority = priority?;
        reservation
            .iter()
            .any(|(name, _)| name.as_str() == Some(priority))
            .then_some(priority)
    }
}

/// The entity types that each take their own slice of a priority class's share of a model.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ClassShareEntity {
    Team,
    Key,
    CustomerOrganization,
    CustomerMember,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct KeyGaps {
    pub team_priority: Option<Value>,
    pub key_priority: Option<Value>,
    pub customer_org_alias: Option<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct KeyInputs {
    pub identity: Option<KeyIdentity>,
    pub gaps: KeyGaps,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Priority {
    pub value: Value,
    pub hash_error: Option<String>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct KeyFacts {
    pub team_id: Option<Scalar>,
    pub team_alias: Option<String>,
    pub api_key: Option<String>,
    pub key_alias: Option<String>,
    pub customer_org_id: Option<String>,
    pub customer_org_alias: Option<String>,
    pub customer_member_id: Option<String>,
    pub priority: Option<Priority>,
}

pub fn scalar_value(scalar: &Scalar) -> Value {
    serde_json::to_value(scalar).unwrap_or(Value::Null)
}

pub fn python_str(scalar: &Scalar) -> String {
    match scalar {
        Scalar::Bool(true) => "True".into(),
        Scalar::Bool(false) => "False".into(),
        Scalar::Int(number) => number.to_string(),
        Scalar::Float(number) => float_repr(*number),
        Scalar::Text(text) | Scalar::Stringified { text, .. } => text.clone(),
    }
}

fn held(scalar: &Scalar, object: Option<Value>) -> Priority {
    let value = match scalar {
        Scalar::Stringified {
            hashable: false, ..
        } => object.unwrap_or_else(|| Value::String(python_str(scalar))),
        Scalar::Stringified { text, .. } => Value::String(text.clone()),
        scalar => scalar_value(scalar),
    };
    Priority {
        value,
        hash_error: scalar.hash_error().map(str::to_owned),
    }
}

fn text(scalar: Option<&Scalar>) -> Option<String> {
    match scalar {
        Some(Scalar::Text(text)) => Some(text.clone()),
        _ => None,
    }
}

impl KeyFacts {
    pub fn new(identity: Option<&KeyIdentity>, gaps: KeyGaps) -> Self {
        let identity = identity.cloned().unwrap_or_default();
        let priority = match (&identity.team_priority, &identity.key_priority) {
            (Some(team), _) => Some(held(team, gaps.team_priority)),
            (None, Some(key)) => Some(held(key, gaps.key_priority)),
            (None, None) => None,
        };
        Self {
            team_id: identity.team_id.clone(),
            team_alias: text(identity.team_alias.as_ref()),
            api_key: identity.api_key.as_ref().map(python_str),
            key_alias: text(identity.key_alias.as_ref()),
            customer_org_id: identity.customer_org_id.clone(),
            customer_org_alias: gaps.customer_org_alias,
            customer_member_id: identity.customer_member_id.clone(),
            priority,
        }
    }

    pub fn from_inputs(inputs: KeyInputs) -> Self {
        Self::new(inputs.identity.as_ref(), inputs.gaps)
    }

    pub fn priority_value(&self) -> Option<Value> {
        self.priority
            .as_ref()
            .map(|priority| priority.value.clone())
    }

    pub fn team_text(&self) -> Option<String> {
        self.team_id.as_ref().map(python_str)
    }

    pub fn team_str(&self) -> Option<&str> {
        match &self.team_id {
            Some(Scalar::Text(text)) => Some(text),
            _ => None,
        }
    }
}

/// Who holds a per-model limit that borrows idle capacity: a team, a key or a customer organization.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BorrowerKind {
    Team,
    Key,
    CustomerOrg,
}

pub const BORROWER_KINDS: [BorrowerKind; 3] = [
    BorrowerKind::Team,
    BorrowerKind::Key,
    BorrowerKind::CustomerOrg,
];

impl BorrowerKind {
    pub fn name(self) -> &'static str {
        match self {
            Self::Team => "team",
            Self::Key => "key",
            Self::CustomerOrg => "customer_org",
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Team => "Team",
            Self::Key => "Key",
            Self::CustomerOrg => "Customer organization",
        }
    }

    pub fn index(self) -> usize {
        match self {
            Self::Team => 0,
            Self::Key => 1,
            Self::CustomerOrg => 2,
        }
    }

    pub fn at_saturation(self) -> String {
        format!("model_per_{}_saturated", self.name())
    }

    pub fn max_borrow_multiple(self) -> String {
        format!("model_per_{}_max_borrow_multiple", self.name())
    }

    /// The borrower a refusal reason names, from either of its descriptor keys.
    pub fn by_reason(reason: &str) -> Option<Self> {
        BORROWER_KINDS
            .into_iter()
            .find(|kind| reason == kind.at_saturation() || reason == kind.max_borrow_multiple())
    }
}

/// One borrower's per-model limit on a model the dynamic limiter covers, as the host resolved it.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BorrowingLimits {
    pub kind: BorrowerKind,
    pub entity_id: String,
    pub alias: Option<String>,
    pub group: String,
    pub tpm: Option<i64>,
    pub rpm: Option<i64>,
    pub max_borrow_multiple: Option<f64>,
}

impl BorrowingLimits {
    pub fn share(&self, rate_limit_type: &str) -> Option<i64> {
        if rate_limit_type == "requests" {
            self.rpm
        } else {
            self.tpm
        }
    }

    pub fn ceiling(&self) -> Option<(Option<i64>, Option<i64>)> {
        let multiple = self.max_borrow_multiple?;
        Some((
            self.tpm.map(|tpm| (tpm as f64 * multiple) as i64),
            self.rpm.map(|rpm| (rpm as f64 * multiple) as i64),
        ))
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TeamShare {
    pub fraction: f64,
    pub active_teams_in_class: i64,
    pub active_classes: i64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LoggedGroup {
    pub text: String,
    pub truthy: bool,
    pub keyed: bool,
    pub hash_error: Option<String>,
}

impl LoggedGroup {
    pub fn read(record: Option<&LoggedModelGroup>) -> Result<Option<Self>, String> {
        let Some(record) = record else {
            return Ok(None);
        };
        if record.error_type.is_some() {
            return Err(record.error_text.clone().unwrap_or_default());
        }
        Ok(record.value.as_ref().map(|value| Self {
            text: python_str(value),
            truthy: value.truthy(),
            keyed: matches!(value, Scalar::Text(_)),
            hash_error: value.hash_error().map(str::to_owned),
        }))
    }

    pub fn served(text: String) -> Self {
        Self {
            truthy: !text.is_empty(),
            keyed: true,
            hash_error: None,
            text,
        }
    }

    pub fn key(&self) -> Result<Option<&str>, CapacityError> {
        if let Some(error) = &self.hash_error {
            return Err(CapacityError::Type(error.clone()));
        }
        Ok(self.keyed.then_some(self.text.as_str()))
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct UsageTokens {
    pub prompt_tokens: i64,
    pub completion_tokens: i64,
    pub total_tokens: i64,
    pub cached_tokens: i64,
}

fn count(value: Option<&Scalar>) -> i64 {
    match value {
        Some(Scalar::Int(number)) => {
            i64::try_from(*number).unwrap_or(if *number < 0 { i64::MIN } else { i64::MAX })
        }
        Some(Scalar::Float(number)) => number.trunc() as i64,
        Some(Scalar::Bool(flag)) => i64::from(*flag),
        Some(Scalar::Text(_) | Scalar::Stringified { .. }) | None => 0,
    }
}

fn modeled(record: Option<&UsageRecord>) -> Option<&UsageRecord> {
    record.filter(|record| {
        record.prompt_tokens.is_some()
            && record.completion_tokens.is_some()
            && record.total_tokens.is_some()
    })
}

impl UsageTokens {
    fn of(record: &UsageRecord) -> Self {
        Self {
            prompt_tokens: count(record.prompt_tokens.as_ref()),
            completion_tokens: count(record.completion_tokens.as_ref()),
            total_tokens: count(record.total_tokens.as_ref()),
            cached_tokens: count(record.cached_tokens.as_ref()),
        }
    }

    pub fn delivered(usage: Option<&Usage>) -> Option<Self> {
        let usage = usage?;
        let reply = usage.streamed.as_ref().or(usage.response.as_ref());
        modeled(reply)
            .or_else(|| modeled(usage.combined.as_ref()))
            .or(usage.socket_session.as_ref())
            .or(usage.logged.as_ref())
            .map(Self::of)
    }

    pub fn failed(usage: Option<&Usage>) -> Option<Self> {
        let usage = usage?;
        modeled(usage.response.as_ref())
            .or_else(|| modeled(usage.combined.as_ref()))
            .or(usage.socket_session.as_ref())
            .map(Self::of)
    }

    pub fn settles_at_estimate(
        usage: Option<&Self>,
        rate_limit_type: TokenRateLimitType,
        embedding: bool,
    ) -> bool {
        let reported_nothing = Self::settled(usage, TokenRateLimitType::Total) == 0
            && Self::settled(usage, TokenRateLimitType::Input) == 0;
        embedding && reported_nothing && rate_limit_type != TokenRateLimitType::Output
    }

    pub fn charged(
        usage: Option<&Self>,
        rate_limit_type: TokenRateLimitType,
        embedding: bool,
    ) -> i64 {
        match Self::settled(usage, rate_limit_type) {
            0 if embedding && rate_limit_type == TokenRateLimitType::Total => {
                Self::settled(usage, TokenRateLimitType::Input)
            }
            tokens => tokens,
        }
    }

    pub fn settled(usage: Option<&Self>, rate_limit_type: TokenRateLimitType) -> i64 {
        usage.map_or(0, |usage| match rate_limit_type {
            TokenRateLimitType::Output => usage.completion_tokens,
            TokenRateLimitType::Input => usage
                .prompt_tokens
                .saturating_sub(usage.cached_tokens)
                .max(0),
            TokenRateLimitType::Total => usage
                .total_tokens
                .saturating_sub(usage.cached_tokens)
                .max(0),
        })
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use serde_json::json;

    use super::*;

    #[rstest]
    #[case::no_usage(None, TokenRateLimitType::Total, true, true)]
    #[case::zero_usage(Some((0, 0, 0)), TokenRateLimitType::Input, true, true)]
    #[case::reported_tokens(Some((7, 0, 7)), TokenRateLimitType::Total, true, false)]
    #[case::prompt_without_total(Some((7, 0, 0)), TokenRateLimitType::Total, true, false)]
    #[case::output_limit(None, TokenRateLimitType::Output, true, false)]
    #[case::chat(None, TokenRateLimitType::Total, false, false)]
    fn only_an_embedding_reporting_no_tokens_settles_at_its_estimate_under_a_total_or_input_limit(
        #[case] usage: Option<(i64, i64, i64)>,
        #[case] rate_limit_type: TokenRateLimitType,
        #[case] embedding: bool,
        #[case] expected: bool,
    ) {
        let usage = usage.map(
            |(prompt_tokens, completion_tokens, total_tokens)| UsageTokens {
                prompt_tokens,
                completion_tokens,
                total_tokens,
                cached_tokens: 0,
            },
        );
        assert_eq!(
            UsageTokens::settles_at_estimate(usage.as_ref(), rate_limit_type, embedding),
            expected
        );
    }

    fn identity(members: Value) -> KeyIdentity {
        serde_json::from_value(members).unwrap()
    }

    fn priority(value: Value, hash_error: Option<&str>) -> Option<Priority> {
        Some(Priority {
            value,
            hash_error: hash_error.map(str::to_owned),
        })
    }

    #[rstest]
    #[case::team_first(json!({"team_priority": "gold", "key_priority": "x"}), KeyGaps::default(), priority(json!("gold"), None))]
    #[case::key_when_team_absent(json!({"key_priority": 7}), KeyGaps::default(), priority(json!(7), None))]
    #[case::unhashable_team_wins_over_key(json!({"team_priority": {"text": "[1]", "truthy": true, "hashable": false, "integer": false, "type_name": "list", "hash_error": "unhashable type: 'list'"}, "key_priority": "x"}), KeyGaps { team_priority: Some(json!([1])), ..KeyGaps::default() }, priority(json!([1]), Some("unhashable type: 'list'")))]
    #[case::unhashable_key_keeps_its_object(json!({"key_priority": {"text": "{'a': 1}", "truthy": true, "hashable": false, "integer": false, "type_name": "dict", "hash_error": "unhashable type: 'dict'"}}), KeyGaps { key_priority: Some(json!({"a": 1})), ..KeyGaps::default() }, priority(json!({"a": 1}), Some("unhashable type: 'dict'")))]
    #[case::unhashable_without_its_object_is_its_text(json!({"key_priority": {"text": "{1, 2}", "truthy": true, "hashable": false, "integer": false, "type_name": "set", "hash_error": "unhashable type: 'set'"}}), KeyGaps::default(), priority(json!("{1, 2}"), Some("unhashable type: 'set'")))]
    #[case::tuple_holding_a_list_names_the_list(json!({"key_priority": {"text": "(1, [2])", "truthy": true, "hashable": false, "integer": false, "type_name": "tuple", "hash_error": "unhashable type: 'list'"}}), KeyGaps { key_priority: Some(json!([1, [2]])), ..KeyGaps::default() }, priority(json!([1, [2]]), Some("unhashable type: 'list'")))]
    #[case::hashable_stringified_is_its_text(json!({"key_priority": {"text": "(1, 2)", "truthy": true, "hashable": true, "integer": false, "type_name": "tuple", "hash_error": null}}), KeyGaps::default(), priority(json!("(1, 2)"), None))]
    #[case::none(json!({}), KeyGaps::default(), None)]
    fn the_team_priority_wins_and_keeps_the_hash_error_python_raises(
        #[case] members: Value,
        #[case] gaps: KeyGaps,
        #[case] expected: Option<Priority>,
    ) {
        let key = KeyFacts::new(Some(&identity(members)), gaps);
        assert_eq!(key.priority, expected);
    }

    #[rstest]
    #[case::text(json!({"team_id": "t1"}), Some("t1"), Some("t1"))]
    #[case::int(json!({"team_id": 5}), Some("5"), None)]
    #[case::flag(json!({"team_id": true}), Some("True"), None)]
    #[case::float(json!({"team_id": 1.0}), Some("1.0"), None)]
    #[case::stringified(json!({"team_id": {"text": "['a']", "truthy": true, "hashable": false, "integer": false, "type_name": "list", "hash_error": "unhashable type: 'list'"}}), Some("['a']"), None)]
    #[case::absent(json!({}), None, None)]
    fn a_team_id_formats_the_way_python_formats_it(
        #[case] members: Value,
        #[case] text: Option<&str>,
        #[case] exact: Option<&str>,
    ) {
        let key = KeyFacts::new(Some(&identity(members)), KeyGaps::default());
        assert_eq!(key.team_text().as_deref(), text);
        assert_eq!(key.team_str(), exact);
    }

    #[rstest]
    #[case::prompt_without_total(tokens(7, 0, 0), TokenRateLimitType::Total, 7)]
    #[case::reported_total(tokens(7, 0, 9), TokenRateLimitType::Total, 9)]
    #[case::input_limit(tokens(7, 0, 0), TokenRateLimitType::Input, 7)]
    #[case::nothing_reported(tokens(0, 0, 0), TokenRateLimitType::Total, 0)]
    fn an_embedding_reporting_prompt_tokens_without_a_total_charges_its_prompt_tokens(
        #[case] usage: Option<UsageTokens>,
        #[case] rate_limit_type: TokenRateLimitType,
        #[case] expected: i64,
    ) {
        assert_eq!(
            UsageTokens::charged(usage.as_ref(), rate_limit_type, true),
            expected
        );
    }

    fn usage(members: Value) -> Usage {
        serde_json::from_value(members).unwrap()
    }

    fn tokens(prompt: i64, completion: i64, total: i64) -> Option<UsageTokens> {
        Some(UsageTokens {
            prompt_tokens: prompt,
            completion_tokens: completion,
            total_tokens: total,
            cached_tokens: 0,
        })
    }

    #[rstest]
    #[case::reply_before_logged(json!({"response": {"prompt_tokens": 7, "completion_tokens": 5, "total_tokens": 12}, "logged": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}), tokens(7, 5, 12))]
    #[case::streamed_before_reply(json!({"streamed": {"prompt_tokens": 70, "completion_tokens": 50, "total_tokens": 120}, "response": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}), tokens(70, 50, 120))]
    #[case::unmodeled_reply_falls_to_combined(json!({"response": {"total_tokens": 9}, "combined": {"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7}}), tokens(3, 4, 7))]
    #[case::logged_counts_take_numbers_only(json!({"logged": {"prompt_tokens": "ten", "completion_tokens": 2.9, "total_tokens": true}}), tokens(0, 2, 1))]
    #[case::logged_stringified_count_is_zero(json!({"logged": {"prompt_tokens": {"text": "[10]", "truthy": true, "hashable": false, "integer": false, "type_name": "list", "hash_error": "unhashable type: 'list'"}, "completion_tokens": 1, "total_tokens": 1}}), tokens(0, 1, 1))]
    #[case::nothing(json!({}), None)]
    fn a_delivered_reply_settles_the_usage_python_reads_first(
        #[case] members: Value,
        #[case] expected: Option<UsageTokens>,
    ) {
        assert_eq!(UsageTokens::delivered(Some(&usage(members))), expected);
    }

    #[rstest]
    #[case::input(TokenRateLimitType::Input, 20)]
    #[case::total(TokenRateLimitType::Total, 25)]
    #[case::output(TokenRateLimitType::Output, 5)]
    fn cache_read_tokens_are_left_out_of_the_tokens_a_response_settles(
        #[case] rate_limit_type: TokenRateLimitType,
        #[case] expected: i64,
    ) {
        let members = json!({"response": {"prompt_tokens": 50, "completion_tokens": 5, "total_tokens": 55, "cached_tokens": 30}});
        let used = UsageTokens::delivered(Some(&usage(members)));
        assert_eq!(
            UsageTokens::settled(used.as_ref(), rate_limit_type),
            expected
        );
    }

    #[rstest]
    fn cache_reads_above_the_prompt_settle_nothing() {
        let members = json!({"response": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15, "cached_tokens": 30}});
        let used = UsageTokens::delivered(Some(&usage(members)));
        assert_eq!(
            UsageTokens::settled(used.as_ref(), TokenRateLimitType::Input),
            0
        );
        assert_eq!(
            UsageTokens::settled(used.as_ref(), TokenRateLimitType::Total),
            0
        );
    }

    #[rstest]
    fn a_failed_attempt_settles_the_usage_its_response_or_combined_usage_recorded() {
        let members = json!({
            "response": {"prompt_tokens": 7, "completion_tokens": 5, "total_tokens": 12},
            "logged": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        });
        assert_eq!(UsageTokens::failed(Some(&usage(members))), tokens(7, 5, 12));
        let logged =
            json!({"logged": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}});
        assert_eq!(UsageTokens::failed(Some(&usage(logged))), None);
        let combined =
            json!({"combined": {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5}});
        assert_eq!(UsageTokens::failed(Some(&usage(combined))), tokens(2, 3, 5));
    }

    #[rstest]
    #[case::team_share("model_per_team_saturated", Some(BorrowerKind::Team))]
    #[case::key_ceiling("model_per_key_max_borrow_multiple", Some(BorrowerKind::Key))]
    #[case::org_share("model_per_customer_org_saturated", Some(BorrowerKind::CustomerOrg))]
    #[case::class_share(PRIORITY_CLASS_SHARE_AT_SATURATION, None)]
    fn a_refusal_reason_names_the_borrower_whose_share_refused(
        #[case] reason: &str,
        #[case] expected: Option<BorrowerKind>,
    ) {
        assert_eq!(BorrowerKind::by_reason(reason), expected);
    }
}
