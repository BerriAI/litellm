//! `litellm_settings.s3_callback_params`, each key read the way `S3Logger` reads it.
//!
//! No key refuses a value. One Python does not read as the setting's type is kept as its `repr()`,
//! so the logger can name it in the warning it falls back with.

use litellm_python_compat::{BigInt, json::from_json, pydantic, repr};
use litellm_secret_value::SecretValue;
use serde::{Deserialize, Deserializer};
use serde_json::Value;

fn python(value: &Value) -> litellm_python_compat::Value {
    from_json(value.clone())
}

fn py_repr(value: &Value) -> String {
    repr::repr(&python(value))
}

fn py_str(value: &Value) -> String {
    repr::to_str(&python(value))
}

fn truthy(value: &Value) -> bool {
    litellm_python_compat::truthy::truthy(&python(value))
}

fn is_blank(value: &Value) -> bool {
    matches!(value, Value::String(text) if text.is_empty())
}

fn unset(value: &Value) -> bool {
    value.is_null() || is_blank(value)
}

fn strip_text(value: &Value) -> Value {
    match value {
        Value::String(text) => Value::String(text.trim().to_owned()),
        other => other.clone(),
    }
}

fn lax_bool(value: &Value) -> Option<bool> {
    pydantic::lax_bool(&python(&strip_text(value)))
}

fn lax_int(value: &Value) -> Option<i128> {
    pydantic::lax_int(&python(&strip_text(value))).map(|number| {
        i128::try_from(&number).unwrap_or(if number < BigInt::from(0) {
            i128::MIN
        } else {
            i128::MAX
        })
    })
}

/// A lenient setting as written: how Python reads it, `None` when it does not, and its `repr()`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Written<T> {
    pub repr: String,
    pub value: Option<T>,
}

/// An integer setting as written. Pydantic reads a bool as an integer, so whether it was one is
/// kept for the settings that refuse it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WrittenCount {
    pub repr: String,
    pub boolean: bool,
    pub value: Option<i128>,
}

/// A setting only text names, with the Python type of a value that is not text.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum WrittenText {
    Text(String),
    Other { type_name: String },
}

/// The region as written. One that is not text is kept the way `str()` prints it, since the
/// logger refuses it by that text.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum S3Region {
    Name(String),
    NotText(String),
}

impl S3Region {
    pub fn text(&self) -> &str {
        match self {
            Self::Name(text) | Self::NotText(text) => text,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum S3PartitionGranularity {
    #[default]
    Day,
    Hour,
}

impl S3PartitionGranularity {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Day => "day",
            Self::Hour => "hour",
        }
    }
}

/// `s3_log_prompts_only` as the params or `S3_LOG_PROMPTS_ONLY` write it: unset for a null, off
/// for an empty text.
pub fn log_prompts_only(written: &Value) -> Option<Written<bool>> {
    if written.is_null() {
        return None;
    }
    Some(Written {
        repr: py_repr(written),
        value: if is_blank(written) {
            Some(false)
        } else {
            lax_bool(written)
        },
    })
}

/// `s3_partition_granularity` as the params or `S3_PARTITION_GRANULARITY` write it. An empty text
/// is a day, and only `day` and `hour` are read.
pub fn partition_granularity(written: &Value) -> Option<Written<S3PartitionGranularity>> {
    if written.is_null() {
        return None;
    }
    Some(Written {
        repr: py_repr(written),
        value: match written.as_str() {
            Some("hour") => Some(S3PartitionGranularity::Hour),
            Some("" | "day") => Some(S3PartitionGranularity::Day),
            _ => None,
        },
    })
}

fn text<'de, D: Deserializer<'de>>(deserializer: D) -> Result<Option<String>, D::Error> {
    let written = Value::deserialize(deserializer)?;
    Ok(truthy(&written).then(|| py_str(&written)))
}

fn secret<'de, D: Deserializer<'de>>(deserializer: D) -> Result<Option<SecretValue>, D::Error> {
    Ok(text(deserializer)?.map(SecretValue::new))
}

fn region<'de, D: Deserializer<'de>>(deserializer: D) -> Result<Option<S3Region>, D::Error> {
    let written = Value::deserialize(deserializer)?;
    Ok(truthy(&written).then(|| match &written {
        Value::String(name) => S3Region::Name(name.clone()),
        other => S3Region::NotText(py_str(other)),
    }))
}

fn is_truthy<'de, D: Deserializer<'de>>(deserializer: D) -> Result<bool, D::Error> {
    Ok(truthy(&Value::deserialize(deserializer)?))
}

fn only_text<'de, D: Deserializer<'de>>(deserializer: D) -> Result<Option<WrittenText>, D::Error> {
    let written = Value::deserialize(deserializer)?;
    Ok(truthy(&written).then(|| match &written {
        Value::String(text) => WrittenText::Text(text.clone()),
        other => WrittenText::Other {
            type_name: python(other).type_name().to_owned(),
        },
    }))
}

fn flag<'de, D: Deserializer<'de>>(deserializer: D) -> Result<Option<Written<bool>>, D::Error> {
    let written = Value::deserialize(deserializer)?;
    Ok((!unset(&written)).then(|| Written {
        repr: py_repr(&written),
        value: lax_bool(&written),
    }))
}

fn count<'de, D: Deserializer<'de>>(deserializer: D) -> Result<Option<WrittenCount>, D::Error> {
    let written = Value::deserialize(deserializer)?;
    Ok((!unset(&written)).then(|| WrittenCount {
        repr: py_repr(&written),
        boolean: written.is_boolean(),
        value: lax_int(&written),
    }))
}

fn prompts_only<'de, D: Deserializer<'de>>(
    deserializer: D,
) -> Result<Option<Written<bool>>, D::Error> {
    Ok(log_prompts_only(&Value::deserialize(deserializer)?))
}

fn granularity<'de, D: Deserializer<'de>>(
    deserializer: D,
) -> Result<Option<Written<S3PartitionGranularity>>, D::Error> {
    Ok(partition_granularity(&Value::deserialize(deserializer)?))
}

/// Each field is its key in `s3_callback_params`. A key the logger does not read is ignored.
#[derive(Clone, Debug, Default, Deserialize, PartialEq)]
#[serde(default)]
pub struct S3CallbackParams {
    #[serde(deserialize_with = "text")]
    pub s3_bucket_name: Option<String>,
    #[serde(deserialize_with = "region")]
    pub s3_region_name: Option<S3Region>,
    #[serde(deserialize_with = "text")]
    pub s3_endpoint_url: Option<String>,
    #[serde(deserialize_with = "text")]
    pub s3_path: Option<String>,
    #[serde(deserialize_with = "secret")]
    pub s3_aws_access_key_id: Option<SecretValue>,
    #[serde(deserialize_with = "secret")]
    pub s3_aws_secret_access_key: Option<SecretValue>,
    #[serde(deserialize_with = "secret")]
    pub s3_aws_session_token: Option<SecretValue>,
    #[serde(deserialize_with = "text")]
    pub s3_aws_session_name: Option<String>,
    #[serde(deserialize_with = "text")]
    pub s3_aws_profile_name: Option<String>,
    #[serde(deserialize_with = "text")]
    pub s3_aws_role_name: Option<String>,
    #[serde(deserialize_with = "secret")]
    pub s3_aws_web_identity_token: Option<SecretValue>,
    #[serde(deserialize_with = "text")]
    pub s3_aws_sts_endpoint: Option<String>,
    #[serde(deserialize_with = "is_truthy")]
    pub s3_use_team_prefix: bool,
    #[serde(deserialize_with = "is_truthy")]
    pub s3_use_key_prefix: bool,
    #[serde(deserialize_with = "is_truthy")]
    pub s3_strip_base64_files: bool,
    #[serde(deserialize_with = "is_truthy")]
    pub s3_use_virtual_hosted_style: bool,
    #[serde(deserialize_with = "prompts_only")]
    pub s3_log_prompts_only: Option<Written<bool>>,
    #[serde(deserialize_with = "granularity")]
    pub s3_partition_granularity: Option<Written<S3PartitionGranularity>>,
    #[serde(deserialize_with = "only_text")]
    pub s3_server_side_encryption: Option<WrittenText>,
    #[serde(deserialize_with = "only_text")]
    pub s3_sse_kms_key_id: Option<WrittenText>,
    #[serde(deserialize_with = "count")]
    pub s3_max_concurrent_uploads: Option<WrittenCount>,
    #[serde(deserialize_with = "count")]
    pub s3_max_queue_size: Option<WrittenCount>,
    #[serde(deserialize_with = "count")]
    pub s3_max_adaptive_concurrency: Option<WrittenCount>,
    #[serde(deserialize_with = "count")]
    pub s3_max_retry_age_seconds: Option<WrittenCount>,
    #[serde(deserialize_with = "flag")]
    pub s3_drop_on_terminal_error: Option<Written<bool>>,
    #[serde(deserialize_with = "flag")]
    pub s3_adaptive_concurrency: Option<Written<bool>>,
    #[serde(deserialize_with = "flag")]
    pub s3_batch_file_upload: Option<Written<bool>>,
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use serde_json::json;

    use super::*;

    fn params(written: Value) -> S3CallbackParams {
        serde_json::from_value(written).unwrap()
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
        assert_eq!(lax_int(&value), expected);
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
        assert_eq!(lax_bool(&value), expected);
    }

    #[rstest]
    #[case(json!(["us-east-1"]), "['us-east-1']")]
    #[case(json!({"a": true}), "{'a': True}")]
    #[case(json!(null), "None")]
    #[case(json!(1.5), "1.5")]
    #[case(json!("us-east-1"), "us-east-1")]
    fn values_print_as_python_str_prints_them(#[case] value: Value, #[case] expected: &str) {
        assert_eq!(py_str(&value), expected);
    }

    #[rstest]
    fn no_key_is_unset_everywhere() {
        assert_eq!(params(json!({})), S3CallbackParams::default());
    }

    #[rstest]
    fn each_key_is_read_into_its_own_field() {
        let read = params(json!({
            "s3_bucket_name": "logs",
            "s3_region_name": "eu-west-1",
            "s3_endpoint_url": "https://s3.example",
            "s3_path": "base",
            "s3_aws_access_key_id": "key-id",
            "s3_aws_secret_access_key": "secret",
            "s3_aws_session_token": "session",
            "s3_aws_session_name": "session-name",
            "s3_aws_profile_name": "profile",
            "s3_aws_role_name": "role",
            "s3_aws_web_identity_token": "token",
            "s3_aws_sts_endpoint": "https://sts.example",
            "s3_use_team_prefix": true,
            "s3_use_key_prefix": 1,
            "s3_strip_base64_files": "yes",
            "s3_use_virtual_hosted_style": [0],
            "s3_log_prompts_only": "true",
            "s3_partition_granularity": "hour",
            "s3_server_side_encryption": "aws:kms",
            "s3_sse_kms_key_id": "kms-1",
            "s3_max_concurrent_uploads": 4,
            "s3_max_queue_size": "100",
            "s3_max_adaptive_concurrency": 50,
            "s3_max_retry_age_seconds": 60,
            "s3_drop_on_terminal_error": false,
            "s3_adaptive_concurrency": "on",
            "s3_batch_file_upload": 1,
            "s3_use_ssl": true,
        }));
        let count = |repr: &str, value: i128| {
            Some(WrittenCount {
                repr: repr.to_owned(),
                boolean: false,
                value: Some(value),
            })
        };
        let flag = |repr: &str, value: bool| {
            Some(Written {
                repr: repr.to_owned(),
                value: Some(value),
            })
        };
        assert_eq!(
            read,
            S3CallbackParams {
                s3_bucket_name: Some("logs".to_owned()),
                s3_region_name: Some(S3Region::Name("eu-west-1".to_owned())),
                s3_endpoint_url: Some("https://s3.example".to_owned()),
                s3_path: Some("base".to_owned()),
                s3_aws_access_key_id: Some(SecretValue::new("key-id")),
                s3_aws_secret_access_key: Some(SecretValue::new("secret")),
                s3_aws_session_token: Some(SecretValue::new("session")),
                s3_aws_session_name: Some("session-name".to_owned()),
                s3_aws_profile_name: Some("profile".to_owned()),
                s3_aws_role_name: Some("role".to_owned()),
                s3_aws_web_identity_token: Some(SecretValue::new("token")),
                s3_aws_sts_endpoint: Some("https://sts.example".to_owned()),
                s3_use_team_prefix: true,
                s3_use_key_prefix: true,
                s3_strip_base64_files: true,
                s3_use_virtual_hosted_style: true,
                s3_log_prompts_only: flag("'true'", true),
                s3_partition_granularity: Some(Written {
                    repr: "'hour'".to_owned(),
                    value: Some(S3PartitionGranularity::Hour),
                }),
                s3_server_side_encryption: Some(WrittenText::Text("aws:kms".to_owned())),
                s3_sse_kms_key_id: Some(WrittenText::Text("kms-1".to_owned())),
                s3_max_concurrent_uploads: count("4", 4),
                s3_max_queue_size: count("'100'", 100),
                s3_max_adaptive_concurrency: count("50", 50),
                s3_max_retry_age_seconds: count("60", 60),
                s3_drop_on_terminal_error: flag("False", false),
                s3_adaptive_concurrency: flag("'on'", true),
                s3_batch_file_upload: flag("1", true),
            }
        );
    }

    #[rstest]
    #[case::falsy_text_is_unset(json!({"s3_bucket_name": ""}), None)]
    #[case::null_is_unset(json!({"s3_bucket_name": null}), None)]
    #[case::a_number_prints_as_python_prints_it(json!({"s3_bucket_name": 7}), Some("7"))]
    fn text_is_the_value_python_finds_truthy(
        #[case] written: Value,
        #[case] expected: Option<&str>,
    ) {
        assert_eq!(params(written).s3_bucket_name.as_deref(), expected);
    }

    #[rstest]
    #[case::not_text(json!(123), Some(S3Region::NotText("123".to_owned())))]
    #[case::empty(json!(""), None)]
    #[case::null(json!(null), None)]
    fn a_region_that_is_not_text_is_kept_as_python_prints_it(
        #[case] written: Value,
        #[case] expected: Option<S3Region>,
    ) {
        assert_eq!(
            params(json!({"s3_region_name": written})).s3_region_name,
            expected
        );
    }

    #[rstest]
    #[case::not_a_boolean(json!("maybe"), Some(Written { repr: "'maybe'".to_owned(), value: None }))]
    #[case::empty_text_is_unset(json!(""), None)]
    #[case::null_is_unset(json!(null), None)]
    fn a_flag_python_cannot_read_keeps_its_repr(
        #[case] written: Value,
        #[case] expected: Option<Written<bool>>,
    ) {
        assert_eq!(
            params(json!({"s3_batch_file_upload": written})).s3_batch_file_upload,
            expected
        );
    }

    #[rstest]
    #[case::a_boolean_is_marked(
        json!(true),
        Some(WrittenCount { repr: "True".to_owned(), boolean: true, value: Some(1) })
    )]
    #[case::not_an_integer(
        json!("many"),
        Some(WrittenCount { repr: "'many'".to_owned(), boolean: false, value: None })
    )]
    #[case::empty_text_is_unset(json!(""), None)]
    fn a_count_python_cannot_read_keeps_its_repr(
        #[case] written: Value,
        #[case] expected: Option<WrittenCount>,
    ) {
        assert_eq!(
            params(json!({"s3_max_queue_size": written})).s3_max_queue_size,
            expected
        );
    }

    #[rstest]
    #[case::empty_text_is_off(json!(""), Some(Written { repr: "''".to_owned(), value: Some(false) }))]
    #[case::null_is_unset(json!(null), None)]
    #[case::not_a_boolean(json!("maybe"), Some(Written { repr: "'maybe'".to_owned(), value: None }))]
    fn prompts_only_tells_an_empty_text_from_an_unset_one(
        #[case] written: Value,
        #[case] expected: Option<Written<bool>>,
    ) {
        assert_eq!(log_prompts_only(&written), expected);
    }

    #[rstest]
    #[case::day(json!("day"), Some(S3PartitionGranularity::Day))]
    #[case::empty_text_is_a_day(json!(""), Some(S3PartitionGranularity::Day))]
    #[case::another_word(json!("minute"), None)]
    #[case::not_text(json!(1), None)]
    fn only_day_and_hour_are_granularities(
        #[case] written: Value,
        #[case] expected: Option<S3PartitionGranularity>,
    ) {
        assert_eq!(partition_granularity(&written).unwrap().value, expected);
    }

    #[rstest]
    #[case::not_text(json!(3), Some(WrittenText::Other { type_name: "int".to_owned() }))]
    #[case::falsy_is_unset(json!(0), None)]
    fn an_encryption_setting_that_is_not_text_keeps_its_python_type(
        #[case] written: Value,
        #[case] expected: Option<WrittenText>,
    ) {
        assert_eq!(
            params(json!({"s3_server_side_encryption": written})).s3_server_side_encryption,
            expected
        );
    }

    #[rstest]
    fn the_params_read_from_yaml_like_from_json() {
        let read: S3CallbackParams =
            serde_yaml_ng::from_str("s3_bucket_name: logs\ns3_max_queue_size: '100'\n").unwrap();
        assert_eq!(
            read,
            params(json!({"s3_bucket_name": "logs", "s3_max_queue_size": "100"}))
        );
    }
}
