//! `litellm_settings.s3_callback_params`, read the way the proxy's `S3Logger` reads it: values
//! may be `os.environ/` references, and a number or flag the proxy would not parse falls back to
//! its default with a warning instead of stopping startup.

use std::time::Duration;

use litellm_auth_aws::AwsAuthConfig;
use litellm_config::{Config, Value};
use litellm_http::SslVerify;
use litellm_secrets::source::SecretSource;

use super::key::Granularity;
use crate::Error;

/// The names a config's callback lists turn this observer on with.
const NAMES: [&str; 2] = ["s3_v2", "s3_log"];

const DEFAULT_FLUSH_INTERVAL_SECONDS: u64 = 10;
const DEFAULT_MAX_CONCURRENT_UPLOADS: usize = 16;
const DEFAULT_MAX_QUEUE_SIZE: usize = 50_000;
const DEFAULT_MAX_RETRY_AGE_SECONDS: u64 = 3600;
const DEFAULT_UPLOAD_ATTEMPT_TIMEOUT_SECONDS: u64 = 30;
/// Where the proxy signs non-LLM AWS calls when no region is configured.
const DEFAULT_REGION: &str = "us-west-2";

/// A configured value once its `os.environ/` reference is resolved.
#[derive(Clone, Debug, PartialEq)]
enum Param {
    Bool(bool),
    Int(i64),
    Float(f64),
    Text(String),
}

impl Param {
    /// Python's `bool(value)`, which the proxy applies to the prefix and style flags: any
    /// non-empty string is true, `"false"` included.
    fn truthy(&self) -> bool {
        match self {
            Self::Bool(flag) => *flag,
            Self::Int(number) => *number != 0,
            Self::Float(number) => *number != 0.0,
            Self::Text(text) => !text.is_empty(),
        }
    }

    /// Pydantic's lax `bool`.
    fn boolean(&self) -> Option<bool> {
        match self {
            Self::Bool(flag) => Some(*flag),
            Self::Int(0) => Some(false),
            Self::Int(1) => Some(true),
            Self::Float(number) if *number == 0.0 => Some(false),
            Self::Float(number) if *number == 1.0 => Some(true),
            Self::Text(text) => match text.trim().to_ascii_lowercase().as_str() {
                "0" | "off" | "f" | "false" | "n" | "no" => Some(false),
                "1" | "on" | "t" | "true" | "y" | "yes" => Some(true),
                _ => None,
            },
            _ => None,
        }
    }

    /// Pydantic's lax `int`.
    fn integer(&self) -> Option<i64> {
        match self {
            Self::Bool(flag) => Some(i64::from(*flag)),
            Self::Int(number) => Some(*number),
            Self::Float(number) if number.fract() == 0.0 && number.is_finite() => {
                Some(*number as i64)
            }
            Self::Text(text) => text.trim().parse().ok(),
            Self::Float(_) => None,
        }
    }

    fn is_blank(&self) -> bool {
        matches!(self, Self::Text(text) if text.is_empty())
    }
}

/// The resolved `s3_callback_params`.
struct Params(Vec<(String, Param)>);

impl Params {
    fn get(&self, name: &str) -> Option<&Param> {
        self.0
            .iter()
            .find(|(key, _)| key == name)
            .map(|(_, value)| value)
    }

    /// `params.get(name) or None`: an empty or false value counts as unset.
    fn text(&self, name: &str) -> Result<Option<String>, Error> {
        match self.get(name) {
            None => Ok(None),
            Some(Param::Text(text)) => Ok(Some(text.clone()).filter(|text| !text.is_empty())),
            Some(Param::Int(number)) => Ok(Some(number.to_string())),
            Some(Param::Float(number)) => Ok(Some(number.to_string())),
            Some(Param::Bool(_)) => Err(Error::S3Log(format!("{name} must be a string"))),
        }
    }

    fn truthy(&self, name: &str) -> bool {
        self.get(name).is_some_and(Param::truthy)
    }

    /// A flag the proxy parses with pydantic; an invalid value warns and takes `invalid`.
    fn flag(&self, name: &str, default: bool, invalid: bool) -> bool {
        match self.get(name) {
            None => default,
            Some(value) if value.is_blank() => default,
            Some(value) => value.boolean().unwrap_or_else(|| {
                tracing::warn!(
                    setting = name,
                    ?value,
                    "s3 logging: not a boolean, using {invalid}"
                );
                invalid
            }),
        }
    }

    /// A positive integer; anything else warns and takes `default`.
    fn positive(&self, name: &str, default: usize, reject_bool: bool) -> usize {
        let Some(value) = self.get(name).filter(|value| !value.is_blank()) else {
            return default;
        };
        let parsed = match value {
            Param::Bool(_) if reject_bool => None,
            value => value.integer(),
        };
        match parsed.and_then(|number| usize::try_from(number).ok()) {
            Some(number) if number >= 1 => number,
            _ => {
                tracing::warn!(
                    setting = name,
                    ?value,
                    "s3 logging: not a positive integer, using {default}"
                );
                default
            }
        }
    }
}

/// How each request is logged and where it goes.
#[derive(Clone, Debug, PartialEq)]
pub(super) struct Settings {
    pub bucket: String,
    /// `s3_path`, the prefix every key starts with.
    pub path: String,
    /// The region requests are signed for and the default endpoint names.
    pub region: String,
    pub endpoint_url: Option<String>,
    pub virtual_hosted_style: bool,
    /// `s3_verify`; unset keeps the gateway's own TLS settings.
    pub verify: Option<SslVerify>,
    pub auth: AwsAuthConfig,
    pub team_prefix: bool,
    pub key_prefix: bool,
    pub strip_base64_files: bool,
    pub prompts_only: bool,
    pub granularity: Granularity,
    pub server_side_encryption: Option<String>,
    pub sse_kms_key_id: Option<String>,
    pub max_concurrent_uploads: usize,
    pub max_queue_size: usize,
    /// How long an upload that keeps failing is retried; `None` retries it until the queue
    /// overflows.
    pub max_retry_age: Option<Duration>,
    pub drop_on_terminal_error: bool,
    pub batch_file_upload: bool,
    pub flush_interval: Duration,
    pub attempt_timeout: Duration,
    pub log_success: bool,
    pub log_failure: bool,
    pub turn_off_message_logging: bool,
    pub redact_user_api_key_info: bool,
    pub user_agent_tags: bool,
}

impl Settings {
    /// `Ok(None)` when no callback list names `s3_v2` or `s3_log`. A config the proxy could
    /// never upload with stops the gateway.
    pub async fn from_config(
        config: &Config,
        secrets: &dyn SecretSource,
        env: &(dyn Fn(&str) -> Option<String> + Sync),
    ) -> Result<Option<Self>, Error> {
        let litellm = &config.litellm_settings;
        let listed = |list: &Option<Vec<Value>>| {
            list.iter().flatten().any(|callback| {
                callback
                    .as_str()
                    .is_some_and(|name| NAMES.contains(&name.trim()))
            })
        };
        let everywhere = listed(&litellm.callbacks);
        let log_success = everywhere || listed(&litellm.success_callback);
        let log_failure = everywhere || listed(&litellm.failure_callback);
        if !log_success && !log_failure {
            return Ok(None);
        }
        let params = resolve(litellm.additional_fields.get("s3_callback_params"), secrets).await?;
        for (name, why) in [
            (
                "s3_flush_interval",
                "the proxy reads DEFAULT_S3_FLUSH_INTERVAL_SECONDS instead",
            ),
            ("s3_batch_size", "the proxy flushes on its interval alone"),
            (
                "s3_adaptive_concurrency",
                "this gateway uploads s3_max_concurrent_uploads at a time",
            ),
        ] {
            if params.get(name).is_some() {
                tracing::warn!(setting = name, "s3 logging: ignored, {why}");
            }
        }
        let bucket = params
            .text("s3_bucket_name")?
            .ok_or_else(|| Error::S3Log("s3_bucket_name is required".into()))?;
        let configured_region = params.text("s3_region_name")?;
        let region = configured_region
            .clone()
            .or_else(|| env("AWS_REGION"))
            .or_else(|| env("AWS_REGION_NAME"))
            .unwrap_or_else(|| DEFAULT_REGION.into());
        let (server_side_encryption, sse_kms_key_id) = sse(
            params.text("s3_server_side_encryption")?,
            params.text("s3_sse_kms_key_id")?,
        );
        let verify = match params.get("s3_verify") {
            None => None,
            Some(Param::Bool(true)) => Some(SslVerify::Enabled),
            Some(Param::Bool(false)) => Some(SslVerify::Disabled),
            Some(Param::Text(text)) => Some(SslVerify::parse(text)),
            Some(other) => {
                return Err(Error::S3Log(format!(
                    "s3_verify must be a boolean or a CA bundle path, not {other:?}"
                )));
            }
        };
        let env_number = |name: &str, default: u64| {
            env(name)
                .and_then(|value| value.trim().parse::<u64>().ok())
                .unwrap_or(default)
        };
        let max_retry_age = match params.get("s3_max_retry_age_seconds") {
            None => Some(DEFAULT_MAX_RETRY_AGE_SECONDS),
            Some(value) if value.is_blank() => Some(DEFAULT_MAX_RETRY_AGE_SECONDS),
            Some(value) => match value {
                Param::Bool(_) => None,
                value => value.integer(),
            }
            .filter(|seconds| *seconds >= 0)
            .map(|seconds| seconds as u64)
            .or_else(|| {
                tracing::warn!(?value, "s3 logging: s3_max_retry_age_seconds is not a non-negative integer, using {DEFAULT_MAX_RETRY_AGE_SECONDS}");
                Some(DEFAULT_MAX_RETRY_AGE_SECONDS)
            }),
        }
        .filter(|seconds| *seconds > 0)
        .map(Duration::from_secs);
        let prompts_only = match params.get("s3_log_prompts_only").cloned() {
            None => env("S3_LOG_PROMPTS_ONLY").map(Param::Text),
            value => value,
        }
        .filter(|value| !value.is_blank())
        .is_some_and(|value| {
            value.boolean().unwrap_or_else(|| {
                tracing::warn!(
                    ?value,
                    "s3 logging: s3_log_prompts_only is not a boolean, logging prompts only"
                );
                true
            })
        });
        let granularity = match params.get("s3_partition_granularity").cloned() {
            None => env("S3_PARTITION_GRANULARITY").map(Param::Text),
            value => value,
        };
        let granularity = match granularity {
            Some(Param::Text(text)) if text == "hour" => Granularity::Hour,
            None => Granularity::Day,
            Some(Param::Text(text)) if text.is_empty() || text == "day" => Granularity::Day,
            Some(value) => {
                tracing::warn!(
                    ?value,
                    "s3 logging: s3_partition_granularity is not one of day, hour, using day"
                );
                Granularity::Day
            }
        };
        let litellm_flag =
            |name: &str| matches!(litellm.additional_fields.get(name), Some(Value::Bool(true)));
        Ok(Some(Self {
            bucket,
            path: params.text("s3_path")?.unwrap_or_default(),
            region: region.clone(),
            endpoint_url: params.text("s3_endpoint_url")?,
            virtual_hosted_style: params.truthy("s3_use_virtual_hosted_style"),
            verify,
            auth: AwsAuthConfig {
                access_key_id: params.text("s3_aws_access_key_id")?,
                secret_access_key: params.text("s3_aws_secret_access_key")?,
                session_token: params.text("s3_aws_session_token")?,
                region_name: Some(region),
                session_name: params.text("s3_aws_session_name")?,
                profile_name: params.text("s3_aws_profile_name")?,
                role_name: params.text("s3_aws_role_name")?,
                web_identity_token: params.text("s3_aws_web_identity_token")?,
                sts_endpoint: params.text("s3_aws_sts_endpoint")?,
                external_id: None,
            },
            team_prefix: params.truthy("s3_use_team_prefix"),
            key_prefix: params.truthy("s3_use_key_prefix"),
            strip_base64_files: params.truthy("s3_strip_base64_files"),
            prompts_only,
            granularity,
            server_side_encryption,
            sse_kms_key_id,
            max_concurrent_uploads: params.positive(
                "s3_max_concurrent_uploads",
                env_number(
                    "DEFAULT_S3_MAX_CONCURRENT_UPLOADS",
                    DEFAULT_MAX_CONCURRENT_UPLOADS as u64,
                ) as usize,
                false,
            ),
            max_queue_size: params.positive("s3_max_queue_size", DEFAULT_MAX_QUEUE_SIZE, true),
            max_retry_age,
            drop_on_terminal_error: params.flag("s3_drop_on_terminal_error", true, true),
            batch_file_upload: params.flag("s3_batch_file_upload", false, false),
            flush_interval: Duration::from_secs(
                env_number(
                    "DEFAULT_S3_FLUSH_INTERVAL_SECONDS",
                    DEFAULT_FLUSH_INTERVAL_SECONDS,
                )
                .max(1),
            ),
            attempt_timeout: Duration::from_secs(
                env_number(
                    "S3_UPLOAD_ATTEMPT_TIMEOUT_SECONDS",
                    DEFAULT_UPLOAD_ATTEMPT_TIMEOUT_SECONDS,
                )
                .clamp(1, 3600),
            ),
            log_success,
            log_failure,
            turn_off_message_logging: litellm_flag("turn_off_message_logging"),
            redact_user_api_key_info: litellm_flag("redact_user_api_key_info"),
            user_agent_tags: !litellm_flag("disable_add_user_agent_to_request_tags"),
        }))
    }
}

/// The encryption headers to send: a KMS key alone means `aws:kms`, and a key next to another
/// algorithm is dropped, as the proxy's `resolve_sse_params` does.
fn sse(algorithm: Option<String>, key_id: Option<String>) -> (Option<String>, Option<String>) {
    let algorithm = algorithm.or_else(|| key_id.as_ref().map(|_| "aws:kms".into()));
    match (algorithm, key_id) {
        (None, _) => (None, None),
        (Some(algorithm), Some(_)) if !algorithm.starts_with("aws:kms") => {
            tracing::warn!(%algorithm, "s3 logging: ignoring s3_sse_kms_key_id, set s3_server_side_encryption to aws:kms to use it");
            (Some(algorithm), None)
        }
        (algorithm, key_id) => (algorithm, key_id),
    }
}

async fn resolve(value: Option<&Value>, secrets: &dyn SecretSource) -> Result<Params, Error> {
    let entries = match value {
        None | Some(Value::Null) => return Ok(Params(Vec::new())),
        Some(Value::Mapping(entries)) => entries,
        Some(_) => return Err(Error::S3Log("s3_callback_params must be a mapping".into())),
    };
    let mut params = Vec::with_capacity(entries.len());
    for (key, value) in entries {
        let Some(name) = key.as_str() else {
            return Err(Error::S3Log(
                "s3_callback_params keys must be strings".into(),
            ));
        };
        let param = match value {
            Value::Null => continue,
            Value::Bool(flag) => Param::Bool(*flag),
            Value::Number(number) => match number.as_i64() {
                Some(number) => Param::Int(number),
                None => Param::Float(number.as_f64().unwrap_or(f64::NAN)),
            },
            Value::String(text) => match text.strip_prefix("os.environ/") {
                Some(variable) => match secrets.get_secret_str(variable).await? {
                    Some(secret) => Param::Text(secret.expose().to_owned()),
                    None => continue,
                },
                None => Param::Text(text.clone()),
            },
            _ => return Err(Error::S3Log(format!("{name} must be a scalar"))),
        };
        params.push((name.to_owned(), param));
    }
    Ok(Params(params))
}

#[cfg(test)]
pub(super) mod tests {
    use std::collections::HashMap;

    use futures_util::future::BoxFuture;
    use litellm_auth_types::SecretValue;
    use rstest::rstest;

    use super::*;

    struct Secrets(HashMap<&'static str, &'static str>);

    impl SecretSource for Secrets {
        fn get_secret_str<'a>(
            &'a self,
            name: &'a str,
        ) -> BoxFuture<'a, Result<Option<SecretValue>, litellm_secrets::Error>> {
            Box::pin(async move { Ok(self.0.get(name).map(|value| SecretValue::new(*value))) })
        }
    }

    pub(crate) async fn settings_with(
        yaml: &str,
        environment: &[(&'static str, &'static str)],
    ) -> Result<Option<Settings>, String> {
        let config = Config::from_yaml(&format!("model_list: []\n{yaml}")).unwrap();
        let environment: HashMap<&str, &str> = environment.iter().copied().collect();
        let env = |name: &str| environment.get(name).map(|value| (*value).to_owned());
        Settings::from_config(&config, &Secrets(environment.clone()), &env)
            .await
            .map_err(|error| error.to_string())
    }

    async fn settings(params: &str) -> Settings {
        settings_with(
            &format!(
                "litellm_settings:\n  callbacks: [s3_v2]\n  s3_callback_params:\n    s3_bucket_name: logs\n{params}"
            ),
            &[],
        )
        .await
        .unwrap()
        .unwrap()
    }

    #[rstest]
    #[case::none("", None)]
    #[case::other_callback("  success_callback: [prometheus]\n", None)]
    #[case::callbacks("  callbacks: [s3_v2]\n", Some((true, true)))]
    #[case::gateway_name("  callbacks: [s3_log]\n", Some((true, true)))]
    #[case::success_only("  success_callback: [s3_v2]\n", Some((true, false)))]
    #[case::failure_only("  failure_callback: s3_v2\n", Some((false, true)))]
    #[tokio::test]
    async fn on_for_the_lists_that_name_it(
        #[case] lists: &str,
        #[case] expected: Option<(bool, bool)>,
    ) {
        let yaml =
            format!("litellm_settings:\n{lists}  s3_callback_params:\n    s3_bucket_name: logs\n");
        let settings = settings_with(&yaml, &[]).await.unwrap();
        assert_eq!(
            settings.map(|settings| (settings.log_success, settings.log_failure)),
            expected
        );
    }

    #[tokio::test]
    async fn defaults_are_the_proxys() {
        let settings = settings("").await;
        assert_eq!(settings.path, "");
        assert_eq!(settings.region, "us-west-2");
        assert_eq!(settings.granularity, Granularity::Day);
        assert_eq!(settings.max_concurrent_uploads, 16);
        assert_eq!(settings.max_queue_size, 50_000);
        assert_eq!(settings.max_retry_age, Some(Duration::from_secs(3600)));
        assert!(settings.drop_on_terminal_error);
        assert!(!settings.batch_file_upload);
        assert!(!settings.prompts_only);
        assert_eq!(settings.flush_interval, Duration::from_secs(10));
        assert_eq!(settings.attempt_timeout, Duration::from_secs(30));
        assert_eq!(settings.verify, None);
        assert_eq!(settings.server_side_encryption, None);
    }

    #[tokio::test]
    async fn references_are_resolved_from_the_secret_source() {
        let settings = settings_with(
            "litellm_settings:
  callbacks: [s3_v2]
  s3_callback_params:
    s3_bucket_name: os.environ/BUCKET
    s3_region_name: eu-west-1
    s3_aws_access_key_id: os.environ/KEY_ID
    s3_aws_secret_access_key: os.environ/SECRET
    s3_aws_session_token: os.environ/UNSET
    s3_use_team_prefix: os.environ/TEAM_PREFIX
",
            &[
                ("BUCKET", "audit"),
                ("KEY_ID", "AKID"),
                ("SECRET", "shh"),
                ("TEAM_PREFIX", "false"),
            ],
        )
        .await
        .unwrap()
        .unwrap();
        assert_eq!(settings.bucket, "audit");
        assert_eq!(settings.region, "eu-west-1");
        assert_eq!(settings.auth.access_key_id.as_deref(), Some("AKID"));
        assert_eq!(settings.auth.secret_access_key.as_deref(), Some("shh"));
        assert_eq!(settings.auth.session_token, None);
        assert_eq!(settings.auth.region_name.as_deref(), Some("eu-west-1"));
        // The proxy takes `bool()` of the string, so any non-empty text turns it on.
        assert!(settings.team_prefix);
    }

    #[rstest]
    #[case::configured("    s3_region_name: ap-south-1\n", &[("AWS_REGION", "us-east-2")], "ap-south-1")]
    #[case::aws_region_wins("", &[("AWS_REGION", "us-east-2"), ("AWS_REGION_NAME", "eu-west-3")], "us-east-2")]
    #[case::aws_region_name("", &[("AWS_REGION_NAME", "eu-west-3")], "eu-west-3")]
    #[tokio::test]
    async fn the_signing_region_falls_back_like_the_proxy(
        #[case] params: &str,
        #[case] environment: &[(&'static str, &'static str)],
        #[case] expected: &str,
    ) {
        let settings = settings_with(
            &format!(
                "litellm_settings:\n  callbacks: [s3_v2]\n  s3_callback_params:\n    s3_bucket_name: logs\n{params}"
            ),
            environment,
        )
        .await
        .unwrap()
        .unwrap();
        assert_eq!(settings.region, expected);
    }

    #[rstest]
    #[case::kms_key_alone("    s3_sse_kms_key_id: key-1\n", Some("aws:kms"), Some("key-1"))]
    #[case::aes("    s3_server_side_encryption: AES256\n", Some("AES256"), None)]
    #[case::key_with_aes(
        "    s3_server_side_encryption: AES256\n    s3_sse_kms_key_id: key-1\n",
        Some("AES256"),
        None
    )]
    #[case::dsse(
        "    s3_server_side_encryption: aws:kms:dsse\n    s3_sse_kms_key_id: key-1\n",
        Some("aws:kms:dsse"),
        Some("key-1")
    )]
    #[tokio::test]
    async fn encryption_settings_combine_like_the_proxy(
        #[case] params: &str,
        #[case] algorithm: Option<&str>,
        #[case] key: Option<&str>,
    ) {
        let settings = settings(params).await;
        assert_eq!(settings.server_side_encryption.as_deref(), algorithm);
        assert_eq!(settings.sse_kms_key_id.as_deref(), key);
    }

    #[rstest]
    #[case::text_number("    s3_max_concurrent_uploads: ' 4 '\n", 4, 50_000)]
    #[case::zero_falls_back("    s3_max_concurrent_uploads: 0\n", 16, 50_000)]
    #[case::garbage_falls_back("    s3_max_queue_size: lots\n", 16, 50_000)]
    #[case::bool_queue_falls_back("    s3_max_queue_size: true\n", 16, 50_000)]
    #[case::bool_width_is_one("    s3_max_concurrent_uploads: true\n", 1, 50_000)]
    #[case::queue("    s3_max_queue_size: 10\n", 16, 10)]
    #[tokio::test]
    async fn numbers_parse_or_fall_back(
        #[case] params: &str,
        #[case] uploads: usize,
        #[case] queue: usize,
    ) {
        let settings = settings(params).await;
        assert_eq!(settings.max_concurrent_uploads, uploads);
        assert_eq!(settings.max_queue_size, queue);
    }

    #[rstest]
    #[case::zero_disables("    s3_max_retry_age_seconds: 0\n", None)]
    #[case::seconds("    s3_max_retry_age_seconds: '60'\n", Some(60))]
    #[case::negative_falls_back("    s3_max_retry_age_seconds: -1\n", Some(3600))]
    #[case::blank_is_default("    s3_max_retry_age_seconds: ''\n", Some(3600))]
    #[tokio::test]
    async fn retry_age(#[case] params: &str, #[case] seconds: Option<u64>) {
        assert_eq!(
            settings(params).await.max_retry_age,
            seconds.map(Duration::from_secs)
        );
    }

    #[rstest]
    #[case::configured("    s3_log_prompts_only: 'yes'\n", &[], true)]
    #[case::invalid_logs_prompts_only("    s3_log_prompts_only: sometimes\n", &[], true)]
    #[case::environment("", &[("S3_LOG_PROMPTS_ONLY", "true")], true)]
    #[case::configured_beats_environment("    s3_log_prompts_only: false\n", &[("S3_LOG_PROMPTS_ONLY", "true")], false)]
    #[case::unset("", &[], false)]
    #[tokio::test]
    async fn prompts_only(
        #[case] params: &str,
        #[case] environment: &[(&'static str, &'static str)],
        #[case] expected: bool,
    ) {
        let settings = settings_with(
            &format!(
                "litellm_settings:\n  callbacks: [s3_v2]\n  s3_callback_params:\n    s3_bucket_name: logs\n{params}"
            ),
            environment,
        )
        .await
        .unwrap()
        .unwrap();
        assert_eq!(settings.prompts_only, expected);
    }

    #[rstest]
    #[case::hour("    s3_partition_granularity: hour\n", &[], Granularity::Hour)]
    #[case::environment("", &[("S3_PARTITION_GRANULARITY", "hour")], Granularity::Hour)]
    #[case::unknown_is_day("    s3_partition_granularity: minute\n", &[], Granularity::Day)]
    #[tokio::test]
    async fn partition_granularity(
        #[case] params: &str,
        #[case] environment: &[(&'static str, &'static str)],
        #[case] expected: Granularity,
    ) {
        let settings = settings_with(
            &format!(
                "litellm_settings:\n  callbacks: [s3_v2]\n  s3_callback_params:\n    s3_bucket_name: logs\n{params}"
            ),
            environment,
        )
        .await
        .unwrap()
        .unwrap();
        assert_eq!(settings.granularity, expected);
    }

    #[rstest]
    #[case::no_bucket(
        "litellm_settings:\n  callbacks: [s3_v2]\n",
        "s3_bucket_name is required"
    )]
    #[case::unset_bucket_reference(
        "litellm_settings:\n  callbacks: [s3_v2]\n  s3_callback_params:\n    s3_bucket_name: os.environ/NOPE\n",
        "s3_bucket_name is required"
    )]
    #[case::not_a_mapping(
        "litellm_settings:\n  callbacks: [s3_v2]\n  s3_callback_params: [a]\n",
        "s3_callback_params must be a mapping"
    )]
    #[case::nested_value(
        "litellm_settings:\n  callbacks: [s3_v2]\n  s3_callback_params:\n    s3_bucket_name: logs\n    s3_path: [a, b]\n",
        "s3_path must be a scalar"
    )]
    #[case::bad_verify(
        "litellm_settings:\n  callbacks: [s3_v2]\n  s3_callback_params:\n    s3_bucket_name: logs\n    s3_verify: 3\n",
        "s3_verify must be a boolean or a CA bundle path"
    )]
    #[tokio::test]
    async fn a_config_that_cannot_upload_stops_the_gateway(
        #[case] yaml: &str,
        #[case] message: &str,
    ) {
        let error = settings_with(yaml, &[]).await.unwrap_err();
        assert!(error.contains(message), "{error}");
    }
}
