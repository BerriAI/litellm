use serde::{Deserialize, Deserializer};
use serde_json::Value;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum TokenRateLimitType {
    Input,
    Output,
    #[default]
    Total,
}

impl TokenRateLimitType {
    pub fn from_setting(raw: &Value) -> Self {
        match raw.as_str() {
            Some("input") => Self::Input,
            Some("output") => Self::Output,
            _ => Self::Total,
        }
    }
}

fn lax_bool(raw: &Value) -> Result<Option<bool>, ()> {
    match raw {
        Value::Null => Ok(None),
        Value::Bool(flag) => Ok(Some(*flag)),
        Value::Number(number) => match number.as_f64() {
            Some(0.0) => Ok(Some(false)),
            Some(1.0) => Ok(Some(true)),
            _ => Err(()),
        },
        Value::String(text) => match text.to_lowercase().as_str() {
            "0" | "off" | "f" | "false" | "n" | "no" => Ok(Some(false)),
            "1" | "on" | "t" | "true" | "y" | "yes" => Ok(Some(true)),
            _ => Err(()),
        },
        Value::Array(_) | Value::Object(_) => Err(()),
    }
}

pub fn fail_closed(raw: &Value) -> bool {
    match lax_bool(raw) {
        Ok(flag) => flag == Some(true),
        Err(()) => {
            tracing::warn!(
                value = %raw,
                "general_settings.fail_closed_rate_limit_enforcement is not a boolean, treating it as disabled"
            );
            false
        }
    }
}

pub fn force_grouping(raw: &Value) -> bool {
    matches!(raw, Value::Bool(true))
}

pub fn fail_closed_setting<'de, D: Deserializer<'de>>(deserializer: D) -> Result<bool, D::Error> {
    Ok(fail_closed(&Value::deserialize(deserializer)?))
}

pub fn force_grouping_setting<'de, D: Deserializer<'de>>(
    deserializer: D,
) -> Result<bool, D::Error> {
    Ok(force_grouping(&Value::deserialize(deserializer)?))
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use serde_json::json;

    use super::*;

    #[rstest]
    #[case::true_flag(json!(true), true)]
    #[case::yes(json!("YES"), true)]
    #[case::one(json!(1), true)]
    #[case::float_one(json!(1.0), true)]
    #[case::off(json!("off"), false)]
    #[case::absent(json!(null), false)]
    #[case::padded(json!(" true"), false)]
    #[case::two(json!(2), false)]
    #[case::half(json!(0.5), false)]
    #[case::list(json!([true]), false)]
    fn fail_closed_reads_the_setting_like_pydantic_lax_bool(
        #[case] raw: Value,
        #[case] expected: bool,
    ) {
        assert_eq!(fail_closed(&raw), expected);
    }

    #[rstest]
    #[case::true_flag(json!(true), true)]
    #[case::truthy_text(json!("true"), false)]
    #[case::one(json!(1), false)]
    fn force_grouping_is_only_the_literal_true(#[case] raw: Value, #[case] expected: bool) {
        assert_eq!(force_grouping(&raw), expected);
    }

    #[rstest]
    #[case::input(json!("input"), TokenRateLimitType::Input)]
    #[case::output(json!("output"), TokenRateLimitType::Output)]
    #[case::unknown(json!("both"), TokenRateLimitType::Total)]
    #[case::absent(json!(null), TokenRateLimitType::Total)]
    fn the_token_rate_limit_type_defaults_to_total(
        #[case] raw: Value,
        #[case] expected: TokenRateLimitType,
    ) {
        assert_eq!(TokenRateLimitType::from_setting(&raw), expected);
    }
}
