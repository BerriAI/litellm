use litellm_redis_store::rate_limits::Scope;
use serde::{Deserialize, Deserializer, Serialize};
use serde_json::Value;

pub const PROJECT_ITPM: &str = "model_per_project_itpm";
pub const PROJECT_OTPM: &str = "model_per_project_otpm";
pub const PTU_TEAM: &str = "model_per_team_ptu";

fn int_text(text: &str) -> Option<i64> {
    let trimmed = text.trim();
    let digits = trimmed.strip_prefix(['+', '-']).unwrap_or(trimmed);
    let valid = !digits.is_empty()
        && !digits.starts_with('_')
        && !digits.ends_with('_')
        && !digits.contains("__")
        && digits.chars().all(|c| c.is_ascii_digit() || c == '_');
    if !valid {
        return None;
    }
    trimmed.replace('_', "").parse().ok()
}

fn truncated(negative: bool, digits: &str, exponent: i64) -> i64 {
    let whole = if exponent >= 0 {
        let padding = usize::try_from(exponent).unwrap_or(usize::MAX);
        if digits.trim_start_matches('0').len().saturating_add(padding) > 19 {
            return if negative { i64::MIN } else { i64::MAX };
        }
        format!("{digits}{}", "0".repeat(padding))
    } else {
        let cut = usize::try_from(exponent.unsigned_abs()).unwrap_or(usize::MAX);
        digits[..digits.len().saturating_sub(cut)].to_owned()
    };
    let magnitude = whole.trim_start_matches('0');
    let parsed = if magnitude.is_empty() {
        0
    } else {
        magnitude.parse::<i64>().unwrap_or(i64::MAX)
    };
    if negative { -parsed } else { parsed }
}

fn decimal_int(text: &str) -> Result<i64, (&'static str, String)> {
    let negative = text.starts_with('-');
    let unsigned = text.trim_start_matches(['+', '-']);
    match unsigned {
        "NaN" | "sNaN" => return Err((NOT_A_NUMBER, "cannot convert NaN to integer".to_owned())),
        "Infinity" => {
            return Err((OVERFLOW, "cannot convert Infinity to integer".to_owned()));
        }
        _ => {}
    }
    let (mantissa, exponent) = match unsigned.split_once(['E', 'e']) {
        Some((mantissa, exponent)) => (mantissa, exponent.parse::<i64>().unwrap_or(0)),
        None => (unsigned, 0),
    };
    let (whole, fraction) = mantissa.split_once('.').unwrap_or((mantissa, ""));
    let shift = i64::try_from(fraction.len()).unwrap_or(i64::MAX);
    Ok(truncated(
        negative,
        &format!("{whole}{fraction}"),
        exponent.saturating_sub(shift),
    ))
}

fn fraction_int(text: &str) -> Option<i64> {
    let (numerator, denominator) = text.split_once('/').unwrap_or((text, "1"));
    let numerator: i128 = numerator.parse().ok()?;
    let denominator: i128 = denominator.parse().ok()?;
    let quotient = numerator.checked_div(denominator)?;
    Some(i64::try_from(quotient).unwrap_or(if quotient < 0 { i64::MIN } else { i64::MAX }))
}

pub(crate) const NOT_A_NUMBER: &str = "ValueError";
pub(crate) const OVERFLOW: &str = "OverflowError";
pub(crate) const INVALID_INT_LITERAL: &str = "invalid literal for int() with base 10: ";
pub(crate) const INT_ARGUMENT: &str =
    "int() argument must be a string, a bytes-like object or a real number, not ";

fn int_value<E: serde::de::Error>(value: Value) -> Result<i64, E> {
    match value {
        Value::Number(number) => Ok(number
            .as_i64()
            .unwrap_or_else(|| number.as_f64().map_or(0, |float| float.trunc() as i64))),
        Value::Bool(flag) => Ok(i64::from(flag)),
        Value::String(text) => {
            int_text(&text).ok_or_else(|| E::custom(format!("{INVALID_INT_LITERAL}'{text}'")))
        }
        Value::Object(_) if let Some(found) = crate::python::stringified(&value) => {
            stringified_int(found.type_name.unwrap_or("object"), found.text)
        }
        other => Err(E::custom(format!(
            "{INT_ARGUMENT}'{}'",
            crate::python::type_name(&other)
        ))),
    }
}

fn stringified_int<E: serde::de::Error>(type_name: &str, text: &str) -> Result<i64, E> {
    match type_name {
        "Decimal" => decimal_int(text)
            .map_err(|(exception, message)| E::custom(format!("{exception}: {message}"))),
        "Fraction" => {
            fraction_int(text).ok_or_else(|| E::custom(format!("{INT_ARGUMENT}'{type_name}'")))
        }
        _ => Err(E::custom(format!("{INT_ARGUMENT}'{type_name}'"))),
    }
}

fn python_int<'de, D: Deserializer<'de>>(deserializer: D) -> Result<Option<i64>, D::Error> {
    Option::<Value>::deserialize(deserializer)?
        .filter(|value| !value.is_null())
        .map(int_value)
        .transpose()
}

fn truthy_int<'de, D: Deserializer<'de>>(deserializer: D) -> Result<Option<i64>, D::Error> {
    let value = Option::<Value>::deserialize(deserializer)?;
    let truthy = match &value {
        None | Some(Value::Null) => false,
        Some(Value::Bool(flag)) => *flag,
        Some(Value::Number(number)) => number.as_f64().is_some_and(|float| float != 0.0),
        Some(Value::String(text)) => !text.is_empty(),
        Some(Value::Array(items)) => !items.is_empty(),
        Some(Value::Object(fields)) => !fields.is_empty(),
    };
    value.filter(|_| truthy).map(int_value).transpose()
}

pub(crate) fn python_text(value: &Value) -> Option<String> {
    if let Some(found) = crate::python::stringified(value) {
        return Some(found.text.to_owned());
    }
    Some(match value {
        Value::String(text) => text.clone(),
        Value::Null => "None".to_owned(),
        Value::Bool(true) => "True".to_owned(),
        Value::Bool(false) => "False".to_owned(),
        Value::Number(number) => match number.as_f64() {
            Some(float)
                if !number.is_i64()
                    && !number.is_u64()
                    && float.fract() == 0.0
                    && float.abs() < 1e16 =>
            {
                format!("{float:.1}")
            }
            _ => number.to_string(),
        },
        Value::Array(_) | Value::Object(_) => return None,
    })
}

fn python_str<'de, D: Deserializer<'de>>(deserializer: D) -> Result<String, D::Error> {
    let value = Value::deserialize(deserializer)?;
    python_text(&value).ok_or_else(|| {
        serde::de::Error::custom(format!("a descriptor value must be a scalar, not {value}"))
    })
}

fn declared_int<'de, D: Deserializer<'de>>(
    deserializer: D,
) -> Result<Option<Option<i64>>, D::Error> {
    python_int(deserializer).map(Some)
}

fn declared_truthy_int<'de, D: Deserializer<'de>>(
    deserializer: D,
) -> Result<Option<Option<i64>>, D::Error> {
    truthy_int(deserializer).map(Some)
}

#[derive(Deserialize)]
struct DeclaredLimits {
    #[serde(default, deserialize_with = "declared_int")]
    requests_per_unit: Option<Option<i64>>,
    #[serde(default, deserialize_with = "declared_int")]
    tokens_per_unit: Option<Option<i64>>,
    #[serde(default, deserialize_with = "declared_int")]
    max_parallel_requests: Option<Option<i64>>,
    #[serde(default, deserialize_with = "declared_truthy_int")]
    window_size: Option<Option<i64>>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Declared {
    pub requests_per_unit: bool,
    pub tokens_per_unit: bool,
    pub max_parallel_requests: bool,
    pub window_size: bool,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Deserialize)]
#[serde(from = "DeclaredLimits")]
pub struct Limits {
    pub requests_per_unit: Option<i64>,
    pub tokens_per_unit: Option<i64>,
    pub max_parallel_requests: Option<i64>,
    pub window_size: Option<i64>,
    pub declared: Declared,
}

impl From<DeclaredLimits> for Limits {
    fn from(wire: DeclaredLimits) -> Self {
        Self {
            requests_per_unit: wire.requests_per_unit.flatten(),
            tokens_per_unit: wire.tokens_per_unit.flatten(),
            max_parallel_requests: wire.max_parallel_requests.flatten(),
            window_size: wire.window_size.flatten(),
            declared: Declared {
                requests_per_unit: wire.requests_per_unit.is_some(),
                tokens_per_unit: wire.tokens_per_unit.is_some(),
                max_parallel_requests: wire.max_parallel_requests.is_some(),
                window_size: wire.window_size.is_some(),
            },
        }
    }
}

impl Serialize for Limits {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeMap;
        let members = [
            (
                "requests_per_unit",
                self.requests_per_unit,
                self.declared.requests_per_unit,
            ),
            (
                "tokens_per_unit",
                self.tokens_per_unit,
                self.declared.tokens_per_unit,
            ),
            (
                "max_parallel_requests",
                self.max_parallel_requests,
                self.declared.max_parallel_requests,
            ),
            ("window_size", self.window_size, self.declared.window_size),
        ];
        let shown: Vec<_> = members
            .into_iter()
            .filter(|(_, value, declared)| *declared || value.is_some())
            .collect();
        let mut map = serializer.serialize_map(Some(shown.len()))?;
        for (name, value, _) in shown {
            map.serialize_entry(name, &value)?;
        }
        map.end()
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Descriptor {
    pub key: String,
    #[serde(deserialize_with = "python_str")]
    pub value: String,
    #[serde(default)]
    pub rate_limit: Option<Limits>,
}

impl Descriptor {
    pub fn limits(&self) -> Limits {
        self.rate_limit.clone().unwrap_or_default()
    }

    pub fn scope(&self) -> Scope {
        Scope {
            descriptor: self.key.clone(),
            value: self.value.clone(),
        }
    }

    pub fn window_size(&self, default: i64) -> i64 {
        self.limits().window_size.unwrap_or(default)
    }

    pub fn is_project_io(&self) -> bool {
        self.key == PROJECT_ITPM || self.key == PROJECT_OTPM
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use serde_json::json;

    use super::*;

    #[rstest]
    fn a_descriptor_reads_the_python_typed_dict_with_float_and_missing_limits() {
        let descriptor: Descriptor = serde_json::from_value(json!({
            "key": "api_key", "value": "h",
            "rate_limit": {"requests_per_unit": 5.9, "tokens_per_unit": null, "window_size": 0}
        }))
        .unwrap();
        assert_eq!(descriptor.limits().requests_per_unit, Some(5));
        assert_eq!(descriptor.limits().tokens_per_unit, None);
        assert_eq!(descriptor.window_size(60), 60);
    }

    #[rstest]
    fn a_descriptor_keeps_the_limit_keys_python_wrote() {
        let written = json!({"key": "user", "value": "u", "rate_limit": {
            "requests_per_unit": null, "tokens_per_unit": 3, "window_size": 60}});
        let descriptor: Descriptor = serde_json::from_value(written.clone()).unwrap();
        assert_eq!(serde_json::to_value(&descriptor).unwrap(), written);
    }

    #[rstest]
    fn a_descriptor_without_limits_has_none() {
        let descriptor: Descriptor =
            serde_json::from_value(json!({"key": "tag", "value": "t", "rate_limit": null}))
                .unwrap();
        assert_eq!(descriptor.limits(), Limits::default());
    }

    #[rstest]
    #[case(json!("10"), Some(10))]
    #[case(json!(" -1_000 "), Some(-1000))]
    #[case(json!(true), Some(1))]
    #[case(json!(7.9), Some(7))]
    #[case(json!(null), None)]
    fn a_limit_converts_like_python_int(#[case] limit: Value, #[case] expected: Option<i64>) {
        let descriptor: Descriptor = serde_json::from_value(json!({
            "key": "team", "value": "t", "rate_limit": {"requests_per_unit": limit}
        }))
        .unwrap();
        assert_eq!(descriptor.limits().requests_per_unit, expected);
    }

    #[rstest]
    #[case(json!("1.5"))]
    #[case(json!(""))]
    #[case(json!("1__0"))]
    #[case(json!([1]))]
    fn a_limit_python_int_rejects_is_an_error(#[case] limit: Value) {
        let decoded = serde_json::from_value::<Descriptor>(json!({
            "key": "team", "value": "t", "rate_limit": {"tokens_per_unit": limit}
        }));
        assert!(decoded.is_err());
    }

    #[rstest]
    #[case(json!(0), 60)]
    #[case(json!(0.5), 0)]
    #[case(json!("0"), 0)]
    #[case(json!(""), 60)]
    #[case(json!("30"), 30)]
    fn a_window_size_falls_back_only_when_python_reads_it_as_false(
        #[case] window: Value,
        #[case] expected: i64,
    ) {
        let descriptor: Descriptor = serde_json::from_value(json!({
            "key": "team", "value": "t", "rate_limit": {"window_size": window}
        }))
        .unwrap();
        assert_eq!(descriptor.window_size(60), expected);
    }

    #[rstest]
    #[case(json!(123), "123")]
    #[case(json!(1.0), "1.0")]
    #[case(json!(2.5), "2.5")]
    #[case(json!(false), "False")]
    #[case(json!(null), "None")]
    fn a_descriptor_value_is_formatted_like_python_str(
        #[case] value: Value,
        #[case] expected: &str,
    ) {
        let descriptor: Descriptor =
            serde_json::from_value(json!({"key": "team", "value": value})).unwrap();
        assert_eq!(descriptor.value, expected);
    }
}
