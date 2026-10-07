use litellm_auth::SecretValue;
use litellm_core_utils::settings::resolve_non_empty;
use serde_json::Value;

pub const EDENAI_API_BASE: &str = "https://api.edenai.run/v3";
pub const EDENAI_API_KEY_ENV: &str = "EDENAI_API_KEY";
pub const EDENAI_API_BASE_ENV: &str = "EDENAI_API_BASE";
pub const EDENAI_COST_FIELD: &str = "cost";

pub fn resolve_edenai_api_base(
    api_base: Option<&str>,
    env_lookup: &dyn Fn(&str) -> Option<String>,
) -> String {
    resolve_non_empty(api_base, env_lookup, &[EDENAI_API_BASE_ENV])
        .unwrap_or_else(|| EDENAI_API_BASE.to_string())
}

pub fn resolve_edenai_api_key(
    api_key: Option<&str>,
    env_lookup: &dyn Fn(&str) -> Option<String>,
) -> Option<SecretValue> {
    resolve_non_empty(api_key, env_lookup, &[EDENAI_API_KEY_ENV]).map(SecretValue::new)
}

pub fn missing_edenai_api_key() -> litellm_auth::Error {
    litellm_auth::Error::MissingApiKey {
        provider: "Eden AI",
        environment_variable: EDENAI_API_KEY_ENV,
    }
}

/// Eden's per-request spend, a JSON number in bodies and a decimal string in the
/// `x-edenai-cost` header. Anything else, including a non-finite value, is no report.
pub fn reported_cost(value: &Value) -> Option<f64> {
    let cost = match value {
        Value::Number(number) => number.as_f64(),
        Value::String(text) => text.trim().parse::<f64>().ok(),
        _ => None,
    }?;
    cost.is_finite().then_some(cost)
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use serde_json::json;

    use super::*;

    fn env(pairs: &'static [(&'static str, &'static str)]) -> impl Fn(&str) -> Option<String> {
        move |name| {
            pairs
                .iter()
                .find(|(key, _)| *key == name)
                .map(|(_, value)| value.to_string())
        }
    }

    #[rstest]
    #[case::default(None, &[], EDENAI_API_BASE)]
    #[case::env(None, &[(EDENAI_API_BASE_ENV, "https://eu.example/v3")], "https://eu.example/v3")]
    #[case::explicit_wins(
        Some("https://internal.example/v3"),
        &[(EDENAI_API_BASE_ENV, "https://eu.example/v3")],
        "https://internal.example/v3"
    )]
    #[case::blank_explicit_falls_through(Some("  "), &[], EDENAI_API_BASE)]
    #[case::blank_env_falls_through(None, &[(EDENAI_API_BASE_ENV, " ")], EDENAI_API_BASE)]
    fn api_base_prefers_explicit_then_env_then_default(
        #[case] api_base: Option<&str>,
        #[case] pairs: &'static [(&'static str, &'static str)],
        #[case] expected: &str,
    ) {
        assert_eq!(resolve_edenai_api_base(api_base, &env(pairs)), expected);
    }

    #[rstest]
    #[case::explicit(Some("sk-explicit"), &[(EDENAI_API_KEY_ENV, "sk-env")], Some("sk-explicit"))]
    #[case::env(None, &[(EDENAI_API_KEY_ENV, "sk-env")], Some("sk-env"))]
    #[case::blank_explicit_uses_env(Some(" "), &[(EDENAI_API_KEY_ENV, "sk-env")], Some("sk-env"))]
    #[case::absent(None, &[], None)]
    fn api_key_prefers_explicit_then_env(
        #[case] api_key: Option<&str>,
        #[case] pairs: &'static [(&'static str, &'static str)],
        #[case] expected: Option<&str>,
    ) {
        assert_eq!(
            resolve_edenai_api_key(api_key, &env(pairs))
                .as_ref()
                .map(SecretValue::expose),
            expected
        );
    }

    #[test]
    fn the_missing_key_error_names_the_env_variable() {
        assert!(
            missing_edenai_api_key()
                .to_string()
                .contains(EDENAI_API_KEY_ENV)
        );
    }

    #[rstest]
    #[case::number(json!(0.0042), Some(0.0042))]
    #[case::integer(json!(1), Some(1.0))]
    #[case::header_string(json!("0.00015"), Some(0.00015))]
    #[case::null(json!(null), None)]
    #[case::garbage_string(json!("free"), None)]
    #[case::non_finite_string(json!("inf"), None)]
    #[case::boolean(json!(true), None)]
    #[case::object(json!({"total": 1.0}), None)]
    fn reported_cost_reads_numbers_and_ignores_malformed_values(
        #[case] value: Value,
        #[case] expected: Option<f64>,
    ) {
        assert_eq!(reported_cost(&value), expected);
    }
}
