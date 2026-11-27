use litellm_env::Lookup;
use litellm_python_compat::{
    lenient::{FloatOrUnset, UintOrUnset},
    unicode::isspace,
};
use serde::{Deserialize, Deserializer};
use serde_with::serde_as;
use struct_field_names_as_array::FieldNamesAsSlice;

use crate::Error;

/// The settings only the environment holds, as written: the variables the proxy reads that no
/// config.yaml key sets. Each field is its variable's name in lower case. A setting config.yaml
/// can also set is a field of its config section instead, which `Config::with_environment` fills.
#[serde_as]
#[derive(Clone, Debug, Default, Deserialize, PartialEq, FieldNamesAsSlice)]
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
    pub legacy_multi_instance_rate_limiting: Option<String>,
    pub use_aws_kms: Option<String>,
    pub litellm_ssl_ciphers: Option<String>,
    pub litellm_worker_startup_hooks: Option<String>,
    pub litellm_otel_v2: Option<String>,
    pub use_ddtrace: Option<String>,
    pub litellm_lens_url: Option<String>,
    pub litellm_telemetry_endpoint: Option<String>,
    pub litellm_max_streaming_duration_seconds: Option<String>,
    pub router_max_fallbacks: Option<String>,
    pub default_max_retries: Option<String>,
    pub default_allowed_fails: Option<String>,
    pub default_cooldown_time_seconds: Option<String>,
    #[serde_as(as = "FloatOrUnset")]
    pub default_cooldown_redis_read_interval_seconds: Option<f64>,
    #[serde_as(as = "UintOrUnset")]
    pub default_max_redis_batch_cache_size: Option<u64>,
    pub litellm_rate_limit_window_size: Option<i64>,
    pub litellm_tpm_token_reservation_enabled: Option<Flag>,
    #[serde_as(as = "UintOrUnset")]
    pub proxy_batch_write_at: Option<u64>,
    #[serde_as(as = "UintOrUnset")]
    pub spend_log_queue_max_bytes: Option<u64>,
    #[serde_as(as = "UintOrUnset")]
    pub spend_log_write_batch_max_bytes: Option<u64>,
    #[serde_as(as = "UintOrUnset")]
    pub spend_log_write_batch_max_rows: Option<u64>,
    pub litellm_local_model_cost_map: Flag,
    pub litellm_offline_mode: Flag,
    #[serde_as(as = "UintOrUnset")]
    pub model_cost_map_min_model_count: Option<u64>,
    #[serde_as(as = "FloatOrUnset")]
    pub model_cost_map_max_shrink_ratio: Option<f64>,
    pub aws_region_name: Option<String>,
}

impl Environment {
    /// Each variable asked of `environment` by its field's name, so `environment` can be one
    /// that cannot be listed, as a config's `environment_variables` over a secret manager is.
    pub fn read(environment: &dyn Lookup) -> Result<Self, Error> {
        let set = Self::FIELD_NAMES_AS_SLICE.iter().filter_map(|field| {
            let variable = field.to_uppercase();
            let value = environment.get(&variable)?;
            Some((variable, value))
        });
        Ok(envy::from_iter(set)?)
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
            ("LITELLM_RATE_LIMIT_WINDOW_SIZE", "30"),
            ("LITELLM_TPM_TOKEN_RESERVATION_ENABLED", "false"),
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
                litellm_rate_limit_window_size: Some(30),
                litellm_tpm_token_reservation_enabled: Some(Flag(false)),
                ..Environment::default()
            }
        );
    }

    #[rstest]
    fn each_variable_is_asked_for_by_its_fields_name() {
        let lookup = |name: &str| match name {
            "USE_AWS_KMS" => Some("1".to_owned()),
            "NON_LLM_CONNECTION_TIMEOUT" => Some("40".to_owned()),
            _ => None,
        };

        assert_eq!(
            Environment::read(&lookup).unwrap(),
            Environment {
                use_aws_kms: Some("1".into()),
                non_llm_connection_timeout: Some(40),
                ..Environment::default()
            }
        );
    }

    #[rstest]
    fn a_variable_that_does_not_read_as_its_type_is_an_error() {
        let lookup = |name: &str| {
            (name == "MAX_REQUEST_BODY_DECOMPRESSION_RATIO").then(|| "many".to_owned())
        };

        assert!(Environment::read(&lookup).is_err());
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
    #[case::blank("", "", None, None)]
    #[case::words("soon", "many", None, None)]
    #[case::numbers(" 2.5 ", "42", Some(2.5), Some(42))]
    fn the_router_read_tuning_is_unset_where_python_falls_back_to_its_default(
        #[case] interval: &str,
        #[case] size: &str,
        #[case] expected_interval: Option<f64>,
        #[case] expected_size: Option<u64>,
    ) {
        let environment = from_variables(&[
            ("DEFAULT_COOLDOWN_REDIS_READ_INTERVAL_SECONDS", interval),
            ("DEFAULT_MAX_REDIS_BATCH_CACHE_SIZE", size),
        ])
        .unwrap();

        assert_eq!(
            (
                environment.default_cooldown_redis_read_interval_seconds,
                environment.default_max_redis_batch_cache_size,
            ),
            (expected_interval, expected_size)
        );
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
