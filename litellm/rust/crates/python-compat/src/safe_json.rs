//! LiteLLM's `safe_json_structure` and `safe_dumps` from
//! `litellm/litellm_core_utils/safe_json_dumps.py`, over plain [`Value`]s.
//!
//! The structure strips NUL characters (which Postgres text and jsonb reject) from every string
//! and key, drops keys that are not strings, replaces anything nested deeper than
//! [`DEFAULT_MAX_DEPTH`] with [`MAX_DEPTH_EXCEEDED`], and optionally turns non-finite floats into
//! `None` and clamps integers. Tuples and sets become lists; sets keep their literal order (see the
//! crate's known limits). Cycles cannot occur in a [`Value`], so the circular reference marker
//! never appears.

use num_bigint::BigInt;

use crate::{Error, Value, json, repr};

/// The marker written in place of a value nested past the depth limit.
pub const MAX_DEPTH_EXCEEDED: &str = "MaxDepthExceeded";

/// `DEFAULT_MAX_RECURSE_DEPTH` from `litellm.constants`.
pub const DEFAULT_MAX_DEPTH: usize = 100;

/// The keyword options of `safe_json_structure`.
#[derive(Clone, Copy, Debug, Default)]
pub struct SafeJsonOptions {
    /// `non_finite_as_null=True`: NaN and the infinities become `None`.
    pub non_finite_as_null: bool,
    /// `integer_limit=limit`: integers are clamped to `[-limit, limit]`.
    pub integer_limit: Option<i64>,
}

/// `safe_json_structure(value, non_finite_as_null=..., integer_limit=...)`.
pub fn safe_json_structure(value: &Value, options: SafeJsonOptions) -> Value {
    structure_at(value, 0, options)
}

/// `safe_dumps(value, non_finite_as_null=...)`: `json.dumps` of [`safe_json_structure`].
pub fn safe_dumps(value: &Value, non_finite_as_null: bool) -> Result<String, Error> {
    json::dumps(&safe_json_structure(
        value,
        SafeJsonOptions {
            non_finite_as_null,
            integer_limit: None,
        },
    ))
}

fn strip_nul(text: &str) -> String {
    text.replace('\0', "")
}

fn structure_at(value: &Value, depth: usize, options: SafeJsonOptions) -> Value {
    if depth > DEFAULT_MAX_DEPTH {
        return Value::Str(MAX_DEPTH_EXCEEDED.to_owned());
    }
    match value {
        Value::Str(content) => Value::Str(strip_nul(content)),
        Value::Int(number) => match options.integer_limit {
            Some(limit) => Value::Int(
                number
                    .clone()
                    .clamp(BigInt::from(-limit), BigInt::from(limit)),
            ),
            None => value.clone(),
        },
        Value::Float(number) if options.non_finite_as_null && !number.is_finite() => Value::None,
        Value::None | Value::Bool(_) | Value::Float(_) => value.clone(),
        Value::Dict(items) => Value::Dict(
            items
                .iter()
                .filter_map(|(key, item)| match key {
                    Value::Str(key) => Some((
                        Value::Str(strip_nul(key)),
                        structure_at(item, depth + 1, options),
                    )),
                    _ => None,
                })
                .collect(),
        ),
        Value::List(items) | Value::Tuple(items) | Value::Set(items) => Value::List(
            items
                .iter()
                .map(|item| structure_at(item, depth + 1, options))
                .collect(),
        ),
        other => Value::Str(strip_nul(&repr::to_str(other))),
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::{SafeJsonOptions, safe_dumps, safe_json_structure};
    use crate::{Value, json};

    fn nested(depth: usize) -> String {
        format!("{}1{}", "[".repeat(depth), "]".repeat(depth))
    }

    // Expected values are CPython json.dumps(safe_json_structure(json.loads(text), ...))
    #[rstest]
    #[case::nul_stripped(r#"{"a\u0000": "x\u0000y"}"#, false, r#"{"a": "xy"}"#)]
    #[case::non_finite_kept(r#"[NaN, 1.5]"#, false, r#"[NaN, 1.5]"#)]
    #[case::non_finite_as_null(r#"[NaN, Infinity, 1.5]"#, true, r#"[null, null, 1.5]"#)]
    fn dumps_what_safe_dumps_writes(
        #[case] text: &str,
        #[case] non_finite_as_null: bool,
        #[case] expected: &str,
    ) {
        let value = json::loads(text).expect("parses");
        assert_eq!(
            safe_dumps(&value, non_finite_as_null).expect("dumps"),
            expected
        );
    }

    #[rstest]
    #[case::at_the_limit(100, nested(100))]
    #[case::past_the_limit(101, format!("{}\"MaxDepthExceeded\"{}", "[".repeat(101), "]".repeat(101)))]
    fn caps_depth_at_one_hundred(#[case] depth: usize, #[case] expected: String) {
        let value = json::loads(&nested(depth)).expect("parses");
        assert_eq!(
            json::dumps_compact(&safe_json_structure(&value, SafeJsonOptions::default()))
                .expect("dumps"),
            expected
        );
    }

    #[rstest]
    #[case::non_string_keys_dropped(
        Value::Dict(vec![(Value::Int(1.into()), Value::None), (Value::Str("k".into()), Value::Bool(true))]),
        r#"{"k": true}"#
    )]
    #[case::integers_clamped(Value::Int((1_i64 << 60).into()), "9007199254740991")]
    fn applies_the_options(#[case] value: Value, #[case] expected: &str) {
        let options = SafeJsonOptions {
            non_finite_as_null: true,
            integer_limit: Some((1 << 53) - 1),
        };
        assert_eq!(
            json::dumps(&safe_json_structure(&value, options)).expect("dumps"),
            expected
        );
    }
}
