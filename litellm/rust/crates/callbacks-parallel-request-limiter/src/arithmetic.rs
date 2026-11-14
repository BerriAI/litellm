use serde::Deserialize;
use serde_json::{Map, Value, json};

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum ArithmeticError {
    #[error("cannot throttle a limit that is not a number")]
    NotANumber,
    #[error("cannot convert a non-finite throttled limit to an integer")]
    NotFinite,
}

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default)]
pub struct AuthFacts {
    pub api_key: Value,
    pub rpm_limit: Value,
    pub tpm_limit: Value,
    pub max_parallel_requests: Value,
    pub budget_throttle_pct: Option<f64>,
    pub user_id: Value,
    pub user_rpm_limit: Value,
    pub user_tpm_limit: Value,
    pub team_id: Value,
    pub team_rpm_limit: Value,
    pub team_tpm_limit: Value,
    pub team_member_rpm_limit: Value,
    pub team_member_tpm_limit: Value,
    pub end_user_id: Value,
    pub end_user_rpm_limit: Value,
    pub end_user_tpm_limit: Value,
    pub org_id: Value,
    pub organization_rpm_limit: Value,
    pub organization_tpm_limit: Value,
    pub rpm_limit_type: Option<String>,
    pub tpm_limit_type: Option<String>,
    pub model_has_failures: bool,
    pub window_size: Value,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct AuthDescriptors {
    pub leading: Vec<Value>,
    pub organization: Vec<Value>,
}

fn truthy(value: &Value) -> bool {
    if let Some(found) = crate::python::stringified(value) {
        return found.truthy;
    }
    match value {
        Value::Null => false,
        Value::Bool(flag) => *flag,
        Value::Number(number) => number.as_f64().is_some_and(|float| float != 0.0),
        Value::String(text) => !text.is_empty(),
        Value::Array(items) => !items.is_empty(),
        Value::Object(fields) => !fields.is_empty(),
    }
}

fn python_str(value: &Value) -> String {
    if let Some(found) = crate::python::stringified(value) {
        return found.text.to_owned();
    }
    match value {
        Value::Null => "None".to_owned(),
        Value::Bool(true) => "True".to_owned(),
        Value::Bool(false) => "False".to_owned(),
        Value::String(text) => text.clone(),
        other => other.to_string(),
    }
}

pub fn throttled_limit(limit: &Value, pct: Option<f64>) -> Result<Value, ArithmeticError> {
    let Some(pct) = pct else {
        return Ok(limit.clone());
    };
    let base = match limit {
        Value::Null => return Ok(Value::Null),
        Value::Bool(flag) => f64::from(u8::from(*flag)),
        Value::Number(number) => number.as_f64().ok_or(ArithmeticError::NotANumber)?,
        _ => return Err(ArithmeticError::NotANumber),
    };
    let scaled = (base * pct).floor();
    if !scaled.is_finite() {
        return Err(ArithmeticError::NotFinite);
    }
    Ok(json!((scaled as i64).max(1)))
}

fn enforces(limit_type: Option<&str>, model_has_failures: bool) -> bool {
    limit_type != Some("dynamic") || model_has_failures
}

fn enforced(limit: Value, limit_type: Option<&str>, model_has_failures: bool) -> Value {
    if limit.is_null() || enforces(limit_type, model_has_failures) {
        limit
    } else {
        Value::Null
    }
}

fn descriptor(key: &str, value: Value, rate_limit: Map<String, Value>) -> Value {
    json!({"key": key, "value": value, "rate_limit": rate_limit})
}

fn limits(requests: &Value, tokens: &Value, window_size: &Value) -> Map<String, Value> {
    let mut rate_limit = Map::new();
    rate_limit.insert("requests_per_unit".to_owned(), requests.clone());
    rate_limit.insert("tokens_per_unit".to_owned(), tokens.clone());
    rate_limit.insert("window_size".to_owned(), window_size.clone());
    rate_limit
}

fn either_set(requests: &Value, tokens: &Value) -> bool {
    !requests.is_null() || !tokens.is_null()
}

pub fn auth_descriptors(facts: &AuthFacts) -> Result<AuthDescriptors, ArithmeticError> {
    let mut leading = Vec::new();
    if truthy(&facts.api_key)
        && (either_set(&facts.rpm_limit, &facts.tpm_limit)
            || !facts.max_parallel_requests.is_null())
    {
        let requests = enforced(
            throttled_limit(&facts.rpm_limit, facts.budget_throttle_pct)?,
            facts.rpm_limit_type.as_deref(),
            facts.model_has_failures,
        );
        let tokens = enforced(
            throttled_limit(&facts.tpm_limit, facts.budget_throttle_pct)?,
            facts.tpm_limit_type.as_deref(),
            facts.model_has_failures,
        );
        let mut rate_limit = limits(&requests, &tokens, &facts.window_size);
        rate_limit.insert(
            "max_parallel_requests".to_owned(),
            facts.max_parallel_requests.clone(),
        );
        leading.push(descriptor("api_key", facts.api_key.clone(), rate_limit));
    }
    if truthy(&facts.user_id) && either_set(&facts.user_rpm_limit, &facts.user_tpm_limit) {
        leading.push(descriptor(
            "user",
            facts.user_id.clone(),
            limits(
                &facts.user_rpm_limit,
                &facts.user_tpm_limit,
                &facts.window_size,
            ),
        ));
    }
    if truthy(&facts.team_id) && either_set(&facts.team_rpm_limit, &facts.team_tpm_limit) {
        leading.push(descriptor(
            "team",
            facts.team_id.clone(),
            limits(
                &facts.team_rpm_limit,
                &facts.team_tpm_limit,
                &facts.window_size,
            ),
        ));
    }
    if truthy(&facts.user_id)
        && either_set(&facts.team_member_rpm_limit, &facts.team_member_tpm_limit)
    {
        leading.push(descriptor(
            "team_member",
            Value::String(format!(
                "{}:{}",
                python_str(&facts.team_id),
                python_str(&facts.user_id)
            )),
            limits(
                &facts.team_member_rpm_limit,
                &facts.team_member_tpm_limit,
                &facts.window_size,
            ),
        ));
    }
    if truthy(&facts.end_user_id)
        && either_set(&facts.end_user_rpm_limit, &facts.end_user_tpm_limit)
    {
        leading.push(descriptor(
            "end_user",
            facts.end_user_id.clone(),
            limits(
                &facts.end_user_rpm_limit,
                &facts.end_user_tpm_limit,
                &facts.window_size,
            ),
        ));
    }
    let mut organization = Vec::new();
    if !facts.org_id.is_null()
        && either_set(&facts.organization_rpm_limit, &facts.organization_tpm_limit)
    {
        organization.push(descriptor(
            "organization",
            facts.org_id.clone(),
            limits(
                &facts.organization_rpm_limit,
                &facts.organization_tpm_limit,
                &facts.window_size,
            ),
        ));
    }
    Ok(AuthDescriptors {
        leading,
        organization,
    })
}

#[derive(Clone, Copy, Debug, Deserialize)]
pub struct OutputCapFacts {
    #[serde(default)]
    pub min_configured_limit: Option<i64>,
    pub responses: bool,
    pub exempt: bool,
    #[serde(default)]
    pub configured_output_tokens: Option<i64>,
    pub fraction: f64,
    pub default_max_tokens_estimate: i64,
    pub responses_min_output_tokens: i64,
}

pub fn no_max_tokens_output_floor(
    min_configured_tpm_limit: Option<i64>,
    default_max_tokens_estimate: i64,
    fraction: f64,
) -> i64 {
    let baseline = (default_max_tokens_estimate as f64 * fraction).trunc() as i64;
    match min_configured_tpm_limit {
        None => baseline,
        Some(limit) => baseline.min(((limit as f64 * fraction).trunc() as i64).max(1)),
    }
}

pub fn output_cap(facts: &OutputCapFacts) -> Option<i64> {
    let floor = no_max_tokens_output_floor(
        facts.min_configured_limit,
        facts.default_max_tokens_estimate,
        facts.fraction,
    );
    let capped = if facts.responses {
        floor.max(facts.responses_min_output_tokens)
    } else {
        floor
    };
    let baseline = (facts.default_max_tokens_estimate as f64 * facts.fraction).trunc() as i64;
    if capped >= baseline || facts.exempt {
        return None;
    }
    Some(capped.max(facts.configured_output_tokens.unwrap_or(0)))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cap(
        min: Option<i64>,
        responses: bool,
        exempt: bool,
        configured: Option<i64>,
    ) -> Option<i64> {
        output_cap(&OutputCapFacts {
            min_configured_limit: min,
            responses,
            exempt,
            configured_output_tokens: configured,
            fraction: 0.25,
            default_max_tokens_estimate: 4096,
            responses_min_output_tokens: 16,
        })
    }

    #[test]
    fn the_floor_is_a_fraction_of_the_smallest_limit() {
        assert_eq!(no_max_tokens_output_floor(None, 4096, 0.25), 1024);
        assert_eq!(no_max_tokens_output_floor(Some(100), 4096, 0.25), 25);
        assert_eq!(no_max_tokens_output_floor(Some(1), 4096, 0.25), 1);
        assert_eq!(no_max_tokens_output_floor(Some(100_000), 4096, 0.25), 1024);
    }

    #[test]
    fn the_cap_applies_only_below_the_baseline() {
        assert_eq!(cap(Some(100), false, false, None), Some(25));
        assert_eq!(cap(Some(100), false, false, Some(60)), Some(60));
        assert_eq!(cap(Some(40), true, false, None), Some(16));
        assert_eq!(cap(Some(100), false, true, None), None);
        assert_eq!(cap(Some(100_000), false, false, None), None);
        assert_eq!(cap(None, false, false, None), None);
    }

    #[test]
    fn a_throttled_limit_keeps_a_trickle() {
        assert_eq!(throttled_limit(&json!(10), Some(0.01)), Ok(json!(1)));
        assert_eq!(throttled_limit(&json!(100), Some(0.5)), Ok(json!(50)));
        assert_eq!(throttled_limit(&json!(100), None), Ok(json!(100)));
        assert_eq!(throttled_limit(&Value::Null, Some(0.5)), Ok(Value::Null));
        assert_eq!(
            throttled_limit(&json!(100), Some(f64::INFINITY)),
            Err(ArithmeticError::NotFinite)
        );
    }

    #[test]
    fn auth_fields_become_python_shaped_descriptors() {
        let facts = AuthFacts {
            api_key: json!("hashed"),
            rpm_limit: json!(10),
            tpm_limit: json!(100),
            budget_throttle_pct: Some(0.5),
            user_id: json!("u"),
            team_member_tpm_limit: json!(7),
            org_id: json!(""),
            organization_rpm_limit: json!(3),
            tpm_limit_type: Some("dynamic".to_owned()),
            window_size: json!(60),
            ..AuthFacts::default()
        };
        let built = auth_descriptors(&facts).unwrap();
        assert_eq!(
            built.leading,
            vec![
                json!({"key": "api_key", "value": "hashed", "rate_limit": {
                    "requests_per_unit": 5, "tokens_per_unit": null,
                    "max_parallel_requests": null, "window_size": 60}}),
                json!({"key": "team_member", "value": "None:u", "rate_limit": {
                    "requests_per_unit": null, "tokens_per_unit": 7, "window_size": 60}}),
            ]
        );
        assert_eq!(
            built.organization,
            vec![json!({"key": "organization", "value": "", "rate_limit": {
                "requests_per_unit": 3, "tokens_per_unit": null, "window_size": 60}})]
        );
    }
}
