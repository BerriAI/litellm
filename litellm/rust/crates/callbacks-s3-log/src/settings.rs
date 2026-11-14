use std::time::Duration;

use litellm_auth_aws::AwsAuthConfig;
use litellm_secret_value::SecretValue;
use serde::Deserialize;
use serde_json::{Map, Value};

use crate::pyjson::{py_str, truthy};

pub const PARTITION_GRANULARITY_ENV_VAR: &str = "S3_PARTITION_GRANULARITY";
pub const LOG_PROMPTS_ONLY_ENV_VAR: &str = "S3_LOG_PROMPTS_ONLY";
const DEFAULT_MAX_QUEUE_SIZE: usize = 50_000;
const DEFAULT_MAX_RETRY_AGE_SECONDS: u64 = 3600;
const DEFAULT_REGION: &str = "us-west-2";

#[derive(Clone, Debug, PartialEq, Deserialize)]
pub struct Constants {
    pub batch_size: usize,
    pub max_concurrent_uploads: usize,
    pub max_adaptive_concurrency: usize,
    pub upload_attempt_timeout_seconds: f64,
    pub download_read_timeout_seconds: f64,
    pub download_connect_timeout_seconds: f64,
    pub web_identity_session_policy: Option<String>,
}

impl Default for Constants {
    fn default() -> Self {
        Self {
            batch_size: 512,
            max_concurrent_uploads: 16,
            max_adaptive_concurrency: 200,
            upload_attempt_timeout_seconds: 30.0,
            download_read_timeout_seconds: 600.0,
            download_connect_timeout_seconds: 5.0,
            web_identity_session_policy: None,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Granularity {
    #[default]
    Day,
    Hour,
}

impl Granularity {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Day => "day",
            Self::Hour => "hour",
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Settings {
    pub bucket_name: Option<String>,
    pub region_name: Option<Value>,
    pub endpoint_url: Option<String>,
    pub path: Option<String>,
    pub verify: Option<Value>,
    pub auth: AwsAuthConfig,
    pub use_team_prefix: bool,
    pub use_key_prefix: bool,
    pub strip_base64_files: bool,
    pub use_virtual_hosted_style: bool,
    pub log_prompts_only: Option<Value>,
    pub partition_granularity: Option<Value>,
    pub server_side_encryption: Option<String>,
    pub sse_kms_key_id: Option<String>,
    pub max_concurrent_uploads: usize,
    pub max_queue_size: usize,
    pub max_retry_age: Option<Duration>,
    pub drop_on_terminal_error: bool,
    pub adaptive_concurrency: bool,
    pub max_adaptive_concurrency: usize,
    pub batch_file_upload: bool,
    pub batch_size: usize,
    pub attempt_timeout: Duration,
    pub download_read_timeout: Duration,
    pub web_identity_session_policy: Option<String>,
}

pub type Environment = dyn Fn(&str) -> Option<String> + Send + Sync;

fn text(params: &Map<String, Value>, name: &str) -> Option<String> {
    params.get(name).filter(|value| truthy(value)).map(py_str)
}

fn secret(params: &Map<String, Value>, name: &str) -> Option<SecretValue> {
    text(params, name).map(SecretValue::new)
}

fn present(params: &Map<String, Value>, name: &str) -> Option<Value> {
    params.get(name).filter(|value| !value.is_null()).cloned()
}

fn is_blank(value: &Value) -> bool {
    matches!(value, Value::String(text) if text.is_empty())
}

pub fn lax_bool(value: &Value) -> Option<bool> {
    match value {
        Value::Bool(flag) => Some(*flag),
        Value::Number(number) => number
            .as_f64()
            .and_then(|float| {
                [(0.0, false), (1.0, true)]
                    .into_iter()
                    .find(|(bound, _)| *bound == float)
            })
            .map(|(_, flag)| flag),
        Value::String(text) => match text.trim().to_ascii_lowercase().as_str() {
            "0" | "off" | "f" | "false" | "n" | "no" => Some(false),
            "1" | "on" | "t" | "true" | "y" | "yes" => Some(true),
            _ => None,
        },
        _ => None,
    }
}

pub fn lax_int(value: &Value) -> Option<i128> {
    match value {
        Value::Bool(flag) => Some(i128::from(*flag)),
        Value::Number(number) => {
            if let Some(integer) = number.as_i64() {
                return Some(i128::from(integer));
            }
            if let Some(integer) = number.as_u64() {
                return Some(i128::from(integer));
            }
            let float = number.as_f64()?;
            (float.is_finite() && float.fract() == 0.0 && float.abs() <= i64::MAX as f64)
                .then_some(float as i128)
        }
        Value::String(text) => int_text(text.trim()),
        _ => None,
    }
}

fn int_text(text: &str) -> Option<i128> {
    let (whole, fraction) = text.split_once('.').unwrap_or((text, ""));
    if !fraction.chars().all(|character| character == '0') {
        return None;
    }
    if text.contains('.') && fraction.is_empty() {
        return None;
    }
    let (sign, digits) = match whole.strip_prefix('-') {
        Some(rest) => (-1, rest),
        None => (1, whole.strip_prefix('+').unwrap_or(whole)),
    };
    let well_formed = !digits.is_empty()
        && !digits.starts_with('_')
        && !digits.ends_with('_')
        && !digits.contains("__")
        && digits
            .chars()
            .all(|character| character.is_ascii_digit() || character == '_');
    if !well_formed {
        return None;
    }
    digits
        .replace('_', "")
        .parse::<i128>()
        .ok()
        .map(|number| sign * number)
}

fn strip_text(value: &Value) -> Value {
    match value {
        Value::String(text) => Value::String(text.trim().to_owned()),
        other => other.clone(),
    }
}

fn positive_int(
    setting: &str,
    configured: Option<&Value>,
    fallback: usize,
    reject_bool: bool,
) -> usize {
    let Some(configured) = configured.filter(|value| !value.is_null() && !is_blank(value)) else {
        return fallback;
    };
    if reject_bool && configured.is_boolean() {
        tracing::warn!(
            "s3 logging: {setting}={configured} is a boolean, not an integer, using {fallback}"
        );
        return fallback;
    }
    match lax_int(&strip_text(configured)) {
        None => {
            tracing::warn!(
                "s3 logging: {setting}={configured} is not an integer, using {fallback}"
            );
            fallback
        }
        Some(bound) if bound < 1 => {
            tracing::warn!(
                "s3 logging: {setting}={configured} must be at least 1, using {fallback}"
            );
            fallback
        }
        Some(bound) => usize::try_from(bound).unwrap_or(usize::MAX),
    }
}

fn retry_age(configured: &Value, fallback: Option<u64>) -> Option<u64> {
    if configured.is_null() || is_blank(configured) {
        return None;
    }
    if configured.is_boolean() {
        tracing::warn!(
            "s3 logging: s3_max_retry_age_seconds={configured} is a boolean, not an integer, falling back to {fallback:?}"
        );
        return fallback;
    }
    match lax_int(&strip_text(configured)) {
        None => {
            tracing::warn!(
                "s3 logging: s3_max_retry_age_seconds={configured} is not an integer, falling back to {fallback:?}"
            );
            fallback
        }
        Some(bound) if bound < 0 => {
            tracing::warn!(
                "s3 logging: s3_max_retry_age_seconds={configured} must be at least 0, falling back to {fallback:?}"
            );
            fallback
        }
        Some(0) => None,
        Some(bound) => Some(u64::try_from(bound).unwrap_or(u64::MAX)),
    }
}

fn flag(
    setting: &str,
    configured: Option<&Value>,
    unset: bool,
    invalid: bool,
    consequence: &str,
) -> bool {
    let Some(configured) = configured.filter(|value| !value.is_null() && !is_blank(value)) else {
        return unset;
    };
    lax_bool(&strip_text(configured)).unwrap_or_else(|| {
        tracing::warn!("s3 logging: {setting}={configured} is not a boolean, {consequence}");
        invalid
    })
}

fn sse_value(name: &str, value: Option<&Value>) -> Option<String> {
    match value {
        None => None,
        Some(Value::String(text)) => Some(text.clone()),
        Some(other) => {
            tracing::warn!(
                "s3 logging: ignoring {name} because it has invalid type {other}; expected a string"
            );
            None
        }
    }
}

pub fn resolve_sse(
    algorithm: Option<&Value>,
    key_id: Option<&Value>,
) -> (Option<String>, Option<String>) {
    let valid_sse = sse_value("s3_server_side_encryption", algorithm);
    let valid_key_id = sse_value("s3_sse_kms_key_id", key_id).filter(|key| !key.is_empty());
    let Some(algorithm) = valid_sse
        .filter(|algorithm| !algorithm.is_empty())
        .or_else(|| valid_key_id.as_ref().map(|_| "aws:kms".to_owned()))
    else {
        return (None, None);
    };
    if valid_key_id.is_some() && !algorithm.starts_with("aws:kms") {
        tracing::warn!(
            "s3 logging: ignoring s3_sse_kms_key_id because s3_server_side_encryption is {algorithm}; set it to aws:kms to encrypt with the KMS key"
        );
        return (Some(algorithm), None);
    }
    (Some(algorithm), valid_key_id)
}

impl Settings {
    pub fn from_params(params: &Map<String, Value>, constants: &Constants) -> Self {
        let truthy_param = |name: &str| params.get(name).is_some_and(truthy);
        let (server_side_encryption, sse_kms_key_id) = resolve_sse(
            params
                .get("s3_server_side_encryption")
                .filter(|value| truthy(value)),
            params
                .get("s3_sse_kms_key_id")
                .filter(|value| truthy(value)),
        );
        let max_concurrent_uploads = positive_int(
            "s3_max_concurrent_uploads",
            Some(
                params
                    .get("s3_max_concurrent_uploads")
                    .filter(|value| !value.is_null() && !is_blank(value))
                    .cloned()
                    .unwrap_or_else(|| Value::from(constants.max_concurrent_uploads)),
            )
            .as_ref(),
            constants.max_concurrent_uploads,
            false,
        );
        let max_queue_size = positive_int(
            "s3_max_queue_size",
            params.get("s3_max_queue_size"),
            DEFAULT_MAX_QUEUE_SIZE,
            true,
        );
        let max_retry_age = match params.get("s3_max_retry_age_seconds") {
            Some(value) if !value.is_null() && !is_blank(value) => {
                retry_age(value, Some(DEFAULT_MAX_RETRY_AGE_SECONDS))
            }
            _ => Some(DEFAULT_MAX_RETRY_AGE_SECONDS),
        };
        Self {
            bucket_name: text(params, "s3_bucket_name"),
            region_name: params
                .get("s3_region_name")
                .filter(|value| truthy(value))
                .cloned(),
            endpoint_url: text(params, "s3_endpoint_url"),
            path: text(params, "s3_path"),
            verify: present(params, "s3_verify"),
            auth: AwsAuthConfig {
                access_key_id: secret(params, "s3_aws_access_key_id"),
                secret_access_key: secret(params, "s3_aws_secret_access_key"),
                session_token: secret(params, "s3_aws_session_token"),
                region_name: text(params, "s3_region_name"),
                session_name: text(params, "s3_aws_session_name"),
                profile_name: text(params, "s3_aws_profile_name"),
                role_name: text(params, "s3_aws_role_name"),
                web_identity_token: secret(params, "s3_aws_web_identity_token"),
                sts_endpoint: text(params, "s3_aws_sts_endpoint"),
                external_id: None,
            },
            use_team_prefix: truthy_param("s3_use_team_prefix"),
            use_key_prefix: truthy_param("s3_use_key_prefix"),
            strip_base64_files: truthy_param("s3_strip_base64_files"),
            use_virtual_hosted_style: truthy_param("s3_use_virtual_hosted_style"),
            log_prompts_only: present(params, "s3_log_prompts_only"),
            partition_granularity: present(params, "s3_partition_granularity"),
            server_side_encryption,
            sse_kms_key_id,
            max_concurrent_uploads,
            max_queue_size,
            max_retry_age: max_retry_age.map(Duration::from_secs),
            drop_on_terminal_error: flag(
                "s3_drop_on_terminal_error",
                params.get("s3_drop_on_terminal_error"),
                true,
                true,
                "dropping terminal-failed uploads",
            ),
            adaptive_concurrency: flag(
                "s3_adaptive_concurrency",
                params.get("s3_adaptive_concurrency"),
                false,
                false,
                "keeping the fixed upload width",
            ),
            max_adaptive_concurrency: positive_int(
                "s3_max_adaptive_concurrency",
                params.get("s3_max_adaptive_concurrency"),
                constants.max_adaptive_concurrency,
                true,
            ),
            batch_file_upload: flag(
                "s3_batch_file_upload",
                params.get("s3_batch_file_upload"),
                false,
                false,
                "keeping per-request objects",
            ),
            batch_size: constants.batch_size,
            attempt_timeout: Duration::from_secs_f64(
                constants.upload_attempt_timeout_seconds.max(0.0),
            ),
            download_read_timeout: Duration::from_secs_f64(
                constants.download_read_timeout_seconds.max(0.0),
            ),
            web_identity_session_policy: constants.web_identity_session_policy.clone(),
        }
    }

    pub fn partition_granularity(&self, environment: &Environment) -> Granularity {
        let raw = match &self.partition_granularity {
            Some(value) => Some(value.clone()),
            None => environment(PARTITION_GRANULARITY_ENV_VAR).map(Value::String),
        };
        match raw {
            Some(Value::String(text)) if text == "hour" => Granularity::Hour,
            None => Granularity::Day,
            Some(Value::String(text)) if text.is_empty() || text == "day" => Granularity::Day,
            Some(other) => {
                tracing::warn!(
                    "s3 logging: s3_partition_granularity={other} is not one of day, hour, using day"
                );
                Granularity::Day
            }
        }
    }

    pub fn log_prompts_only(&self, environment: &Environment) -> bool {
        let raw = match &self.log_prompts_only {
            Some(value) => Some(value.clone()),
            None => environment(LOG_PROMPTS_ONLY_ENV_VAR).map(Value::String),
        };
        let Some(raw) = raw.filter(|value| !is_blank(value)) else {
            return false;
        };
        lax_bool(&strip_text(&raw)).unwrap_or_else(|| {
            tracing::warn!(
                "s3 logging: s3_log_prompts_only={raw} is not a boolean, logging prompts only"
            );
            true
        })
    }

    pub fn signing_region(
        &self,
        from_environment: impl FnOnce() -> Result<Option<String>, String>,
    ) -> Result<String, String> {
        let region = match &self.region_name {
            Some(Value::String(region)) => region.clone(),
            Some(other) => return Err(invalid_region(&py_str(other))),
            None => from_environment()?.unwrap_or_else(|| DEFAULT_REGION.to_owned()),
        };
        if valid_region(&region) {
            Ok(region)
        } else {
            Err(invalid_region(&region))
        }
    }

    pub fn url_region(&self) -> String {
        self.region_name
            .as_ref()
            .map_or_else(|| "None".to_owned(), py_str)
    }
}

fn invalid_region(region: &str) -> String {
    format!(
        "Invalid AWS region format: {region}. Region names must contain only lowercase letters, digits, and hyphens."
    )
}

fn valid_region(region: &str) -> bool {
    !region.is_empty()
        && region
            .bytes()
            .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'-')
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use serde_json::json;

    use super::*;

    fn settings(params: Value) -> Settings {
        Settings::from_params(params.as_object().unwrap(), &Constants::default())
    }

    fn no_env(_: &str) -> Option<String> {
        None
    }

    #[rstest]
    #[case(json!(true), Some(1))]
    #[case(json!(2.0), Some(2))]
    #[case(json!(2.5), None)]
    #[case(json!(" 4 "), Some(4))]
    #[case(json!("4.0"), Some(4))]
    #[case(json!("+4"), Some(4))]
    #[case(json!("1_000"), Some(1000))]
    #[case(json!("0x10"), None)]
    #[case(json!("1e3"), None)]
    #[case(json!("4.5"), None)]
    #[case(json!("007"), Some(7))]
    #[case(json!(1e20), None)]
    fn integers_parse_as_pydantic_parses_them(
        #[case] value: Value,
        #[case] expected: Option<i128>,
    ) {
        assert_eq!(lax_int(&strip_text(&value)), expected);
    }

    #[rstest]
    #[case(json!(0), Some(false))]
    #[case(json!(2), None)]
    #[case(json!(1.0), Some(true))]
    #[case(json!(0.5), None)]
    #[case(json!(" yes "), Some(true))]
    #[case(json!("TRUE"), Some(true))]
    #[case(json!("off"), Some(false))]
    #[case(json!("1.0"), None)]
    #[case(json!("2"), None)]
    fn booleans_parse_as_pydantic_parses_them(
        #[case] value: Value,
        #[case] expected: Option<bool>,
    ) {
        assert_eq!(lax_bool(&strip_text(&value)), expected);
    }

    #[rstest]
    fn defaults_match_the_python_constructor() {
        let settings = settings(json!({}));
        assert_eq!(settings.bucket_name, None);
        assert_eq!(settings.path, None);
        assert_eq!(settings.max_concurrent_uploads, 16);
        assert_eq!(settings.max_queue_size, 50_000);
        assert_eq!(settings.max_retry_age, Some(Duration::from_secs(3600)));
        assert!(settings.drop_on_terminal_error);
        assert!(!settings.adaptive_concurrency);
        assert_eq!(settings.max_adaptive_concurrency, 200);
        assert!(!settings.batch_file_upload);
        assert_eq!(settings.partition_granularity(&no_env), Granularity::Day);
        assert!(!settings.log_prompts_only(&no_env));
    }

    #[rstest]
    fn prefix_flags_take_python_truthiness() {
        let settings = settings(json!({"s3_use_team_prefix": "false", "s3_use_key_prefix": 0}));
        assert!(settings.use_team_prefix);
        assert!(!settings.use_key_prefix);
    }

    #[rstest]
    #[case(json!({"s3_max_retry_age_seconds": 0}), None)]
    #[case(json!({"s3_max_retry_age_seconds": "60"}), Some(60))]
    #[case(json!({"s3_max_retry_age_seconds": -1}), Some(3600))]
    #[case(json!({"s3_max_retry_age_seconds": true}), Some(3600))]
    #[case(json!({"s3_max_retry_age_seconds": ""}), Some(3600))]
    fn retry_age_parses_as_python(#[case] params: Value, #[case] seconds: Option<u64>) {
        assert_eq!(
            settings(params).max_retry_age,
            seconds.map(Duration::from_secs)
        );
    }

    #[rstest]
    #[case(json!({"s3_sse_kms_key_id": "k"}), Some("aws:kms"), Some("k"))]
    #[case(json!({"s3_server_side_encryption": "AES256", "s3_sse_kms_key_id": "k"}), Some("AES256"), None)]
    #[case(json!({"s3_server_side_encryption": 3}), None, None)]
    fn encryption_combines_as_python(
        #[case] params: Value,
        #[case] algorithm: Option<&str>,
        #[case] key: Option<&str>,
    ) {
        let settings = settings(params);
        assert_eq!(settings.server_side_encryption.as_deref(), algorithm);
        assert_eq!(settings.sse_kms_key_id.as_deref(), key);
    }

    #[rstest]
    #[case(json!({"s3_partition_granularity": "hour"}), None, Granularity::Hour)]
    #[case(json!({}), Some("hour"), Granularity::Hour)]
    #[case(json!({"s3_partition_granularity": "day"}), Some("hour"), Granularity::Day)]
    #[case(json!({"s3_partition_granularity": "minute"}), None, Granularity::Day)]
    fn granularity_reads_the_param_then_the_environment(
        #[case] params: Value,
        #[case] env: Option<&'static str>,
        #[case] expected: Granularity,
    ) {
        let environment = move |name: &str| {
            (name == PARTITION_GRANULARITY_ENV_VAR)
                .then_some(env)
                .flatten()
                .map(str::to_owned)
        };
        assert_eq!(
            settings(params).partition_granularity(&environment),
            expected
        );
    }

    #[rstest]
    #[case::default(json!({}), Ok(None), Ok("us-west-2"))]
    #[case::param(json!({"s3_region_name": "eu-west-1"}), Ok(Some("ap-south-1")), Ok("eu-west-1"))]
    #[case::bad_param(json!({"s3_region_name": "EU"}), Ok(None), Err(()))]
    #[case::environment(json!({}), Ok(Some("ap-south-1")), Ok("ap-south-1"))]
    #[case::bad_environment(json!({}), Ok(Some("Ap_South")), Err(()))]
    #[case::empty_environment(json!({}), Ok(Some("")), Err(()))]
    #[case::secret_failure(json!({}), Err(()), Err(()))]
    fn the_signing_region_falls_back_like_python(
        #[case] params: Value,
        #[case] environment: Result<Option<&str>, ()>,
        #[case] expected: Result<&str, ()>,
    ) {
        let from_environment = || {
            environment
                .map(|region| region.map(str::to_owned))
                .map_err(|()| "secret manager down".to_owned())
        };
        assert_eq!(
            settings(params)
                .signing_region(from_environment)
                .map_err(|_| ()),
            expected.map(str::to_owned)
        );
    }
}
