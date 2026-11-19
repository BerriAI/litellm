use std::ops::RangeInclusive;

use litellm_python_compat::{lenient::UintOrUnset, unicode::isspace};
use serde::{Deserialize, Deserializer};
use serde_with::serde_as;

/// The environment variables the proxy reads outside its config file, as written. Each field is
/// its variable's name in lower case.
#[serde_as]
#[derive(Clone, Debug, Default, Deserialize, PartialEq)]
#[serde(default)]
pub struct Environment {
    pub litellm_cors_origins: Option<String>,
    pub litellm_cors_allow_credentials: Option<Flag>,
    pub server_root_path: String,
    pub server_root_paths: Option<String>,
    pub max_decompressed_request_body_size_mb: Option<f64>,
    pub max_request_body_decompression_ratio: Option<u64>,
    pub max_decompressed_request_bodies_in_flight_mb: Option<f64>,
    #[serde_as(as = "UintOrUnset")]
    pub non_llm_connection_timeout: Option<u64>,
    #[serde_as(as = "UintOrUnset")]
    pub default_batch_size: Option<u64>,
    #[serde_as(as = "UintOrUnset")]
    pub default_flush_interval_seconds: Option<u64>,
    #[serde_as(as = "UintOrUnset")]
    pub default_s3_batch_size: Option<u64>,
    #[serde_as(as = "UintOrUnset")]
    pub default_s3_flush_interval_seconds: Option<u64>,
    #[serde_as(as = "UintOrUnset")]
    pub default_s3_max_concurrent_uploads: Option<u64>,
    #[serde_as(as = "UintOrUnset")]
    pub default_s3_max_adaptive_concurrency: Option<u64>,
    #[serde_as(as = "UintOrUnset")]
    pub s3_upload_attempt_timeout_seconds: Option<u64>,
}

const DEFAULT_BATCH_SIZE: u64 = 512;
const DEFAULT_FLUSH_INTERVAL_SECONDS: u64 = 5;
const DEFAULT_S3_BATCH_SIZE: u64 = 512;
const DEFAULT_S3_FLUSH_INTERVAL_SECONDS: u64 = 10;
const S3_UPLOAD_ATTEMPT_TIMEOUT_SECONDS: RangeInclusive<u64> = 1..=3600;

/// Python's `value or fallback()` for a count.
fn or_else(value: u64, fallback: impl FnOnce() -> u64) -> u64 {
    if value == 0 { fallback() } else { value }
}

impl Environment {
    /// `DEFAULT_S3_BATCH_SIZE or DEFAULT_BATCH_SIZE`, the batch size `S3Logger()` is built with.
    pub fn s3_batch_size(&self) -> u64 {
        or_else(
            self.default_s3_batch_size.unwrap_or(DEFAULT_S3_BATCH_SIZE),
            || self.default_batch_size.unwrap_or(DEFAULT_BATCH_SIZE),
        )
    }

    /// `DEFAULT_S3_FLUSH_INTERVAL_SECONDS or DEFAULT_FLUSH_INTERVAL_SECONDS`, how often
    /// `S3Logger()` flushes its queue.
    pub fn s3_flush_interval_seconds(&self) -> u64 {
        or_else(
            self.default_s3_flush_interval_seconds
                .unwrap_or(DEFAULT_S3_FLUSH_INTERVAL_SECONDS),
            || {
                self.default_flush_interval_seconds
                    .unwrap_or(DEFAULT_FLUSH_INTERVAL_SECONDS)
            },
        )
    }

    /// `S3_UPLOAD_ATTEMPT_TIMEOUT_SECONDS` when it is within the range `get_env_int_in_range`
    /// accepts, and unset when it is not.
    pub fn s3_upload_attempt_timeout(&self) -> Option<u64> {
        self.s3_upload_attempt_timeout_seconds
            .filter(|seconds| S3_UPLOAD_ATTEMPT_TIMEOUT_SECONDS.contains(seconds))
    }
}

/// The proxy's `os.getenv(name).strip().lower() == "true"`: anything but `true` is off.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Flag(pub bool);

impl Flag {
    pub fn read(raw: &str) -> Self {
        Self(raw.trim_matches(isspace).eq_ignore_ascii_case("true"))
    }
}

impl<'de> Deserialize<'de> for Flag {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        Ok(Self::read(&String::deserialize(deserializer)?))
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use serde_json::json;

    use super::*;

    fn from_variables(variables: &[(&str, &str)]) -> envy::Result<Environment> {
        envy::from_iter(
            variables
                .iter()
                .map(|(key, value)| (key.to_string(), value.to_string())),
        )
    }

    #[rstest]
    fn the_proxy_environment_is_read_by_variable_name() {
        let environment = from_variables(&[
            ("LITELLM_CORS_ORIGINS", "https://a.example"),
            ("LITELLM_CORS_ALLOW_CREDENTIALS", "yes"),
            ("SERVER_ROOT_PATH", "/api"),
            ("SERVER_ROOT_PATHS", "/tenant-a,/tenant-b"),
            ("MAX_DECOMPRESSED_REQUEST_BODY_SIZE_MB", "1.5"),
            ("MAX_REQUEST_BODY_DECOMPRESSION_RATIO", "10"),
            ("MAX_DECOMPRESSED_REQUEST_BODIES_IN_FLIGHT_MB", "20"),
            ("NON_LLM_CONNECTION_TIMEOUT", "40"),
        ])
        .unwrap();

        assert_eq!(
            environment,
            Environment {
                litellm_cors_origins: Some("https://a.example".into()),
                litellm_cors_allow_credentials: Some(Flag(false)),
                server_root_path: "/api".into(),
                server_root_paths: Some("/tenant-a,/tenant-b".into()),
                max_decompressed_request_body_size_mb: Some(1.5),
                max_request_body_decompression_ratio: Some(10),
                max_decompressed_request_bodies_in_flight_mb: Some(20.0),
                non_llm_connection_timeout: Some(40),
                ..Environment::default()
            }
        );
    }

    #[rstest]
    fn the_s3_logger_constants_are_read_by_variable_name() {
        let environment = from_variables(&[
            ("DEFAULT_BATCH_SIZE", "32"),
            ("DEFAULT_FLUSH_INTERVAL_SECONDS", "7"),
            ("DEFAULT_S3_BATCH_SIZE", " 64 "),
            ("DEFAULT_S3_FLUSH_INTERVAL_SECONDS", "3"),
            ("DEFAULT_S3_MAX_CONCURRENT_UPLOADS", "4"),
            ("DEFAULT_S3_MAX_ADAPTIVE_CONCURRENCY", "50"),
            ("S3_UPLOAD_ATTEMPT_TIMEOUT_SECONDS", "45"),
        ])
        .unwrap();

        assert_eq!(
            environment,
            Environment {
                default_batch_size: Some(32),
                default_flush_interval_seconds: Some(7),
                default_s3_batch_size: Some(64),
                default_s3_flush_interval_seconds: Some(3),
                default_s3_max_concurrent_uploads: Some(4),
                default_s3_max_adaptive_concurrency: Some(50),
                s3_upload_attempt_timeout_seconds: Some(45),
                ..Environment::default()
            }
        );
    }

    #[rstest]
    #[case::unset(&[], 512)]
    #[case::configured(&[("DEFAULT_S3_BATCH_SIZE", "64")], 64)]
    #[case::zero_falls_back_to_the_general_size(
        &[("DEFAULT_S3_BATCH_SIZE", "0"), ("DEFAULT_BATCH_SIZE", "32")],
        32
    )]
    #[case::zero_falls_back_to_the_general_default(&[("DEFAULT_S3_BATCH_SIZE", "0")], 512)]
    #[case::not_a_number_is_the_default(&[("DEFAULT_S3_BATCH_SIZE", "many")], 512)]
    fn the_s3_batch_size_is_read_like_the_proxy_reads_it(
        #[case] variables: &[(&str, &str)],
        #[case] expected: u64,
    ) {
        assert_eq!(from_variables(variables).unwrap().s3_batch_size(), expected);
    }

    #[rstest]
    #[case::unset(&[], 10)]
    #[case::configured(&[("DEFAULT_S3_FLUSH_INTERVAL_SECONDS", "3")], 3)]
    #[case::zero_falls_back_to_the_general_default(&[("DEFAULT_S3_FLUSH_INTERVAL_SECONDS", "0")], 5)]
    #[case::zero_falls_back_to_the_general_interval(
        &[("DEFAULT_S3_FLUSH_INTERVAL_SECONDS", "0"), ("DEFAULT_FLUSH_INTERVAL_SECONDS", "7")],
        7
    )]
    fn the_s3_flush_interval_is_read_like_the_proxy_reads_it(
        #[case] variables: &[(&str, &str)],
        #[case] expected: u64,
    ) {
        assert_eq!(
            from_variables(variables)
                .unwrap()
                .s3_flush_interval_seconds(),
            expected
        );
    }

    #[rstest]
    #[case::unset(&[], None)]
    #[case::within_the_range(&[("S3_UPLOAD_ATTEMPT_TIMEOUT_SECONDS", "45")], Some(45))]
    #[case::below_the_range(&[("S3_UPLOAD_ATTEMPT_TIMEOUT_SECONDS", "0")], None)]
    #[case::above_the_range(&[("S3_UPLOAD_ATTEMPT_TIMEOUT_SECONDS", "3601")], None)]
    fn an_s3_upload_timeout_out_of_range_is_unset(
        #[case] variables: &[(&str, &str)],
        #[case] expected: Option<u64>,
    ) {
        assert_eq!(
            from_variables(variables)
                .unwrap()
                .s3_upload_attempt_timeout(),
            expected
        );
    }

    #[rstest]
    fn an_empty_environment_leaves_every_proxy_setting_unset() {
        assert_eq!(from_variables(&[]).unwrap(), Environment::default());
    }

    #[rstest]
    #[case::blank("", None)]
    #[case::whitespace("  ", None)]
    #[case::words("soon", None)]
    #[case::fraction("15.5", None)]
    #[case::negative("-1", None)]
    #[case::padded(" 20 ", Some(20))]
    #[case::grouped_digits("1_000", Some(1000))]
    fn a_timeout_is_read_the_way_python_reads_an_integer(
        #[case] raw: &str,
        #[case] expected: Option<u64>,
    ) {
        let environment = from_variables(&[("NON_LLM_CONNECTION_TIMEOUT", raw)]).unwrap();

        assert_eq!(environment.non_llm_connection_timeout, expected);
    }

    #[rstest]
    fn a_decompression_limit_that_is_not_a_number_fails_startup() {
        assert!(from_variables(&[("MAX_REQUEST_BODY_DECOMPRESSION_RATIO", "many")]).is_err());
    }

    #[rstest]
    #[case::lowercase("true", true)]
    #[case::any_case_with_spaces(" TRUE ", true)]
    #[case::python_only_whitespace("\u{1c}True\u{1f}", true)]
    #[case::false_word("false", false)]
    #[case::one("1", false)]
    #[case::yes("yes", false)]
    #[case::empty("", false)]
    fn a_flag_is_on_only_when_it_reads_true(#[case] value: &str, #[case] expected: bool) {
        assert_eq!(
            serde_json::from_value::<Flag>(json!(value)).unwrap(),
            Flag(expected)
        );
    }

    #[rstest]
    fn an_unset_flag_is_told_apart_from_one_that_is_off() {
        let environment: Environment =
            serde_json::from_value(json!({"litellm_cors_allow_credentials": "no"})).unwrap();
        assert_eq!(
            environment.litellm_cors_allow_credentials,
            Some(Flag(false))
        );
        assert_eq!(Environment::default().litellm_cors_allow_credentials, None);
    }
}
