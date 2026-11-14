use litellm_callbacks::{
    AttemptTiming, CallbackError, Cost, Deployment, GuardrailEntry, HookInput, InputSequenceLength,
    LoggedKeyIdentity, LoggedModelGroup, LoggedRequest, McpToolCall, MetadataKind, MetadataLabel,
    Moment as Stamp, Outcome, RateLimitState, Scalar, TypedField, UsageRecord, typed,
};
use serde::Deserialize;
use serde_json::{Map, Number, Value};
use sha2::{Digest, Sha256};

use crate::{
    emit::{failed, raised},
    labels::Settings,
    pyfmt::{NUMBER_KEY, float_repr, stringified},
};

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq)]
pub struct Moment {
    pub us: i64,
    pub aware: bool,
}

#[derive(Clone, Debug, Default, Deserialize, PartialEq)]
pub struct Residual {
    pub settings: Settings,
    #[serde(default)]
    pub emit_input_sequence_length_label: bool,
    #[serde(default)]
    pub payload_labels: Map<String, Value>,
    #[serde(default)]
    pub now: Option<Moment>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct KeyFacts {
    pub hashed: Value,
    pub model_group: Value,
    pub remaining_requests: Value,
    pub remaining_tokens: Value,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct AttemptFacts {
    pub model: Value,
    pub identity: Option<LoggedKeyIdentity>,
    pub usage: UsageRecord,
    pub cost: Cost,
    pub tags: Option<Vec<Option<Scalar>>>,
    pub guardrails: Option<Vec<GuardrailEntry>>,
    pub deployment: Deployment,
    pub timing: AttemptTiming,
    pub rate_limits: RateLimitState,
    pub request: LoggedRequest,
    pub mcp: Option<McpToolCall>,
    pub model_group: Option<LoggedModelGroup>,
    pub metadata_labels: Vec<MetadataLabel>,
    pub input_sequence_length: Option<InputSequenceLength>,
    pub stream: bool,
    pub cache_hit: Option<bool>,
    pub residual: Residual,
}

fn tagged(text: String) -> Value {
    Value::Object(Map::from_iter([(
        NUMBER_KEY.to_owned(),
        Value::String(text),
    )]))
}

pub fn py(value: Option<&Scalar>) -> Value {
    match value {
        None => Value::Null,
        Some(Scalar::Bool(flag)) => Value::Bool(*flag),
        Some(Scalar::Int(number)) => i64::try_from(*number)
            .map(Value::from)
            .or_else(|_| u64::try_from(*number).map(Value::from))
            .unwrap_or_else(|_| tagged(number.to_string())),
        Some(Scalar::Float(number)) => {
            Number::from_f64(*number).map_or_else(|| tagged(float_repr(*number)), Value::Number)
        }
        Some(Scalar::Text(text)) => Value::String(text.clone()),
        Some(Scalar::Stringified {
            text,
            integer: true,
            ..
        }) => tagged(text.clone()),
        Some(Scalar::Stringified {
            text,
            truthy,
            type_name,
            ..
        }) => stringified(text, *truthy, type_name),
    }
}

pub fn rejected(record: Option<&Vec<String>>, member: &str) -> bool {
    record.is_some_and(|names| names.iter().any(|name| name == member))
}

fn days_from_civil(year: i32, month: u8, day: u8) -> i64 {
    let year = i64::from(year) - i64::from(month <= 2);
    let era = year.div_euclid(400);
    let year_of_era = year - era * 400;
    let month = i64::from(month);
    let shifted = if month > 2 { month - 3 } else { month + 9 };
    let day_of_year = (153 * shifted + 2) / 5 + i64::from(day) - 1;
    let day_of_era = year_of_era * 365 + year_of_era / 4 - year_of_era / 100 + day_of_year;
    era * 146_097 + day_of_era - 719_468
}

pub fn moment(stamp: Option<&Stamp>) -> Option<Moment> {
    let stamp = stamp?;
    let seconds = days_from_civil(stamp.year, stamp.month, stamp.day) * 86_400
        + i64::from(stamp.hour) * 3_600
        + i64::from(stamp.minute) * 60
        + i64::from(stamp.second)
        - i64::from(stamp.utc_offset_seconds.unwrap_or(0));
    Some(Moment {
        us: seconds * 1_000_000 + i64::from(stamp.microsecond),
        aware: stamp.utc_offset_seconds.is_some(),
    })
}

fn read<F: TypedField>(input: &HookInput) -> Result<Option<F::Value>, CallbackError> {
    match input.fields.raw(F::FIELD)? {
        None | Some(Value::Null) => Ok(None),
        Some(value) => F::Value::deserialize(value)
            .map(Some)
            .map_err(|error| failed(format!("prometheus attempt: {:?} {error}", F::FIELD))),
    }
}

impl AttemptFacts {
    pub fn read(input: &HookInput) -> Result<Option<(Outcome, Self)>, CallbackError> {
        let (Some(outcome), Some(residual)) = (
            read::<typed::Outcome>(input)?,
            input
                .fields
                .raw(litellm_callbacks::Field::DeprecatedKwargs)?,
        ) else {
            return Ok(None);
        };
        if residual.is_null() {
            return Ok(None);
        }
        let residual = Residual::deserialize(residual)
            .map_err(|error| failed(format!("prometheus attempt: {error}")))?;
        let call = input.call.as_ref();
        let facts = Self {
            model: input
                .fields
                .raw(litellm_callbacks::Field::Model)?
                .cloned()
                .unwrap_or_else(|| Value::from("")),
            identity: read::<typed::LoggedKeyIdentity>(input)?,
            usage: read::<typed::Usage>(input)?
                .and_then(|usage| usage.logged)
                .unwrap_or_default(),
            cost: read::<typed::Cost>(input)?.unwrap_or_default(),
            tags: read::<typed::RequestTags>(input)?,
            guardrails: read::<typed::GuardrailInfo>(input)?,
            deployment: read::<typed::Deployment>(input)?.unwrap_or_default(),
            timing: read::<typed::AttemptTiming>(input)?.unwrap_or_default(),
            rate_limits: read::<typed::RateLimitState>(input)?.unwrap_or_default(),
            request: read::<typed::LoggedRequest>(input)?.unwrap_or_default(),
            mcp: read::<typed::McpToolCall>(input)?,
            model_group: read::<typed::LoggedModelGroup>(input)?,
            metadata_labels: read::<typed::MetadataLabels>(input)?.unwrap_or_default(),
            input_sequence_length: read::<typed::InputSequenceLength>(input)?,
            stream: call.is_some_and(|call| call.kind.stream),
            cache_hit: call.and_then(|call| call.cache_hit),
            residual,
        };
        Ok(Some((outcome, facts)))
    }

    pub const fn settings(&self) -> &Settings {
        &self.residual.settings
    }

    pub fn completion_tokens(&self) -> Value {
        py(self.usage.completion_tokens.as_ref())
    }

    pub fn custom_metadata_labels(&self) -> Vec<(String, String)> {
        self.metadata_labels
            .iter()
            .map(|label| (label.name.clone(), label.value.clone()))
            .collect()
    }

    pub fn input_sequence_length(&self) -> Result<Option<String>, CallbackError> {
        match &self.input_sequence_length {
            Some(InputSequenceLength {
                error_type: Some(error_type),
                error_text,
                ..
            }) => Err(raised(
                error_type,
                error_text.as_deref().unwrap_or_default(),
            )),
            Some(length) => Ok(length.bucket.clone()),
            None => Ok(None),
        }
    }

    pub fn latency_window(&self) -> (Option<Moment>, Option<Moment>) {
        let timing = &self.timing;
        let end = if rejected(timing.rejected.as_ref(), "logged_end") {
            None
        } else {
            moment(timing.logged_end.as_ref()).or(self.residual.now)
        };
        (moment(timing.logged_start.as_ref()), end)
    }

    pub fn end_user_id(&self) -> Value {
        let settings = self.settings();
        if settings.disable_end_user_cost_tracking || !settings.end_user_prometheus_only {
            return Value::Null;
        }
        py(self
            .identity
            .as_ref()
            .and_then(|identity| identity.cost_tracking_end_user_id.as_ref()))
    }

    pub fn key(&self) -> Result<Option<KeyFacts>, CallbackError> {
        if let Some(LoggedModelGroup {
            error_type: Some(error_type),
            error_text,
            ..
        }) = &self.model_group
        {
            return Err(raised(
                error_type,
                error_text.as_deref().unwrap_or_default(),
            ));
        }
        let Some(identity) = self.identity.as_ref() else {
            return Ok(None);
        };
        if identity.metadata_kind != Some(MetadataKind::Mapping) {
            return Ok(None);
        }
        let hashed = match identity.key_hash.as_ref() {
            Some(Scalar::Text(key)) if key.starts_with("sk-") => {
                Value::from(format!("{:x}", Sha256::digest(key.as_bytes())))
            }
            other => py(other),
        };
        let state = &self.rate_limits;
        let model_group = self.model_group.as_ref().map_or_else(
            || state.key_model_group.as_ref(),
            |group| group.value.as_ref(),
        );
        Ok(Some(KeyFacts {
            hashed,
            model_group: py(model_group),
            remaining_requests: py(state.key_remaining_requests.as_ref()),
            remaining_tokens: py(state.key_remaining_tokens.as_ref()),
        }))
    }

    pub fn tag_values(&self) -> Result<Vec<Value>, CallbackError> {
        self.tags
            .as_ref()
            .map(|tags| tags.iter().map(|tag| py(tag.as_ref())).collect())
            .ok_or_else(|| failed("TypeError: 'NoneType' object is not iterable"))
    }

    pub fn limiter_header(&self, name: &str) -> Option<&Scalar> {
        self.rate_limits.limiter_headers.as_ref()?.get(name)
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use serde_json::json;

    use super::*;

    fn stringified_scalar(text: &str, truthy: bool, integer: bool, type_name: &str) -> Scalar {
        Scalar::Stringified {
            text: text.to_owned(),
            truthy,
            hashable: true,
            integer,
            type_name: type_name.to_owned(),
            hash_error: None,
        }
    }

    #[rstest]
    #[case::small_int(Scalar::Int(-3), json!(-3))]
    #[case::unsigned(Scalar::Int(i128::from(u64::MAX)), json!(u64::MAX))]
    #[case::big_int(Scalar::Int(10_i128.pow(20)), json!({"$num": "100000000000000000000"}))]
    #[case::float(Scalar::Float(1.5), json!(1.5))]
    #[case::nan(Scalar::Float(f64::NAN), json!({"$num": "nan"}))]
    #[case::negative_infinity(Scalar::Float(f64::NEG_INFINITY), json!({"$num": "-inf"}))]
    #[case::flag(Scalar::Bool(true), json!(true))]
    #[case::text(Scalar::Text("x".to_owned()), json!("x"))]
    #[case::stringified(stringified_scalar("['a']", true, false, "list"), json!({"$py": "['a']", "$type": "list"}))]
    #[case::huge_int(stringified_scalar(&"1".repeat(50), true, true, "int"), json!({"$num": "1".repeat(50)}))]
    #[case::falsy_stringified(stringified_scalar("[]", false, false, "list"), json!({"$py": "[]", "$truthy": false, "$type": "list"}))]
    #[case::object(stringified_scalar("<obj>", true, false, "Opaque"), json!({"$py": "<obj>", "$type": "Opaque"}))]
    fn a_scalar_becomes_the_value_python_formats(#[case] scalar: Scalar, #[case] expected: Value) {
        assert_eq!(py(Some(&scalar)), expected);
    }

    #[rstest]
    #[case::naive_epoch((1970, 1, 1), (0, 0), None, 0, false)]
    #[case::naive((2026, 11, 13), (9, 0), None, 1_794_560_400_000_007, false)]
    #[case::aware_shifted((2026, 11, 13), (14, 0), Some(18_000), 1_794_560_400_000_007, true)]
    #[case::leap_day((2024, 2, 29), (0, 0), Some(0), 1_709_164_800_000_007, true)]
    #[case::before_epoch((1969, 12, 31), (23, 59), None, -60_000_000 + 7, false)]
    fn a_moment_is_microseconds_since_the_epoch_python_subtracts(
        #[case] (year, month, day): (i32, u8, u8),
        #[case] (hour, minute): (u8, u8),
        #[case] offset: Option<i32>,
        #[case] us: i64,
        #[case] aware: bool,
    ) {
        let stamp = Stamp {
            year,
            month,
            day,
            hour,
            minute,
            second: 0,
            microsecond: if us == 0 { 0 } else { 7 },
            utc_offset_seconds: offset,
        };
        assert_eq!(moment(Some(&stamp)), Some(Moment { us, aware }));
    }
}
