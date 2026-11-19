use std::time::Duration;

use litellm_auth_aws::AwsAuthConfig;
pub use litellm_config::S3PartitionGranularity as Granularity;
use litellm_config::{
    S3CallbackParams, S3Region, Written, WrittenCount, WrittenText, log_prompts_only,
    partition_granularity,
};
use serde::Deserialize;
use serde_json::Value;

fn py_optional(value: Option<u64>) -> String {
    value.map_or_else(|| "None".to_owned(), |value| value.to_string())
}

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

#[derive(Clone, Debug, PartialEq)]
pub struct Settings {
    pub bucket_name: Option<String>,
    pub region_name: Option<S3Region>,
    pub endpoint_url: Option<String>,
    pub path: Option<String>,
    pub auth: AwsAuthConfig,
    pub use_team_prefix: bool,
    pub use_key_prefix: bool,
    pub strip_base64_files: bool,
    pub use_virtual_hosted_style: bool,
    pub log_prompts_only: Option<Written<bool>>,
    pub partition_granularity: Option<Written<Granularity>>,
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

fn positive_int(
    setting: &str,
    configured: Option<&WrittenCount>,
    fallback: usize,
    reject_bool: bool,
) -> usize {
    let Some(configured) = configured else {
        return fallback;
    };
    let written = &configured.repr;
    if reject_bool && configured.boolean {
        tracing::warn!(
            "s3 logging: {setting}={written} is a boolean, not an integer, using {fallback}"
        );
        return fallback;
    }
    match configured.value {
        None => {
            tracing::warn!("s3 logging: {setting}={written} is not an integer, using {fallback}");
            fallback
        }
        Some(bound) if bound < 1 => {
            tracing::warn!("s3 logging: {setting}={written} must be at least 1, using {fallback}");
            fallback
        }
        Some(bound) => usize::try_from(bound).unwrap_or(usize::MAX),
    }
}

fn retry_age(configured: &WrittenCount, fallback: Option<u64>) -> Option<u64> {
    let written = &configured.repr;
    if configured.boolean {
        tracing::warn!(
            "s3 logging: s3_max_retry_age_seconds={written} is a boolean, not an integer, falling back to {}",
            py_optional(fallback)
        );
        return fallback;
    }
    match configured.value {
        None => {
            tracing::warn!(
                "s3 logging: s3_max_retry_age_seconds={written} is not an integer, falling back to {}",
                py_optional(fallback)
            );
            fallback
        }
        Some(bound) if bound < 0 => {
            tracing::warn!(
                "s3 logging: s3_max_retry_age_seconds={written} must be at least 0, falling back to {}",
                py_optional(fallback)
            );
            fallback
        }
        Some(0) => None,
        Some(bound) => Some(u64::try_from(bound).unwrap_or(u64::MAX)),
    }
}

fn flag(
    setting: &str,
    configured: Option<&Written<bool>>,
    unset: bool,
    invalid: bool,
    consequence: &str,
) -> bool {
    let Some(configured) = configured else {
        return unset;
    };
    configured.value.unwrap_or_else(|| {
        tracing::warn!(
            "s3 logging: {setting}={} is not a boolean, {consequence}",
            configured.repr
        );
        invalid
    })
}

fn sse_value(name: &str, value: Option<&WrittenText>) -> Option<String> {
    match value {
        None => None,
        Some(WrittenText::Text(text)) => Some(text.clone()),
        Some(WrittenText::Other { type_name }) => {
            tracing::warn!(
                "s3 logging: ignoring {name} because it has invalid type {type_name}; expected a string"
            );
            None
        }
    }
}

pub fn resolve_sse(
    algorithm: Option<&WrittenText>,
    key_id: Option<&WrittenText>,
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

pub fn resolve_partition_granularity(written: Option<&Written<Granularity>>) -> Granularity {
    match written {
        None => Granularity::Day,
        Some(Written {
            value: Some(granularity),
            ..
        }) => *granularity,
        Some(Written { repr, value: None }) => {
            tracing::warn!(
                "s3 logging: s3_partition_granularity={repr} is not one of day, hour, using day"
            );
            Granularity::Day
        }
    }
}

impl Settings {
    pub fn new(params: &S3CallbackParams, constants: &Constants) -> Self {
        let (server_side_encryption, sse_kms_key_id) = resolve_sse(
            params.s3_server_side_encryption.as_ref(),
            params.s3_sse_kms_key_id.as_ref(),
        );
        let default_concurrent_uploads = WrittenCount {
            repr: constants.max_concurrent_uploads.to_string(),
            boolean: false,
            value: i128::try_from(constants.max_concurrent_uploads).ok(),
        };
        let region_text = params
            .s3_region_name
            .as_ref()
            .map(|region| region.text().to_owned());
        Self {
            bucket_name: params.s3_bucket_name.clone(),
            region_name: params.s3_region_name.clone(),
            endpoint_url: params.s3_endpoint_url.clone(),
            path: params.s3_path.clone(),
            auth: AwsAuthConfig {
                access_key_id: params.s3_aws_access_key_id.clone(),
                secret_access_key: params.s3_aws_secret_access_key.clone(),
                session_token: params.s3_aws_session_token.clone(),
                region_name: region_text,
                session_name: params.s3_aws_session_name.clone(),
                profile_name: params.s3_aws_profile_name.clone(),
                role_name: params.s3_aws_role_name.clone(),
                web_identity_token: params.s3_aws_web_identity_token.clone(),
                sts_endpoint: params.s3_aws_sts_endpoint.clone(),
                external_id: None,
                session_tags: None,
                session_policy: None,
            },
            use_team_prefix: params.s3_use_team_prefix,
            use_key_prefix: params.s3_use_key_prefix,
            strip_base64_files: params.s3_strip_base64_files,
            use_virtual_hosted_style: params.s3_use_virtual_hosted_style,
            log_prompts_only: params.s3_log_prompts_only.clone(),
            partition_granularity: params.s3_partition_granularity.clone(),
            server_side_encryption,
            sse_kms_key_id,
            max_concurrent_uploads: positive_int(
                "s3_max_concurrent_uploads",
                Some(
                    params
                        .s3_max_concurrent_uploads
                        .as_ref()
                        .unwrap_or(&default_concurrent_uploads),
                ),
                constants.max_concurrent_uploads,
                false,
            ),
            max_queue_size: positive_int(
                "s3_max_queue_size",
                params.s3_max_queue_size.as_ref(),
                DEFAULT_MAX_QUEUE_SIZE,
                true,
            ),
            max_retry_age: match &params.s3_max_retry_age_seconds {
                Some(configured) => retry_age(configured, Some(DEFAULT_MAX_RETRY_AGE_SECONDS)),
                None => Some(DEFAULT_MAX_RETRY_AGE_SECONDS),
            }
            .map(Duration::from_secs),
            drop_on_terminal_error: flag(
                "s3_drop_on_terminal_error",
                params.s3_drop_on_terminal_error.as_ref(),
                true,
                true,
                "dropping terminal-failed uploads",
            ),
            adaptive_concurrency: flag(
                "s3_adaptive_concurrency",
                params.s3_adaptive_concurrency.as_ref(),
                false,
                false,
                "keeping the fixed upload width",
            ),
            max_adaptive_concurrency: positive_int(
                "s3_max_adaptive_concurrency",
                params.s3_max_adaptive_concurrency.as_ref(),
                constants.max_adaptive_concurrency,
                true,
            ),
            batch_file_upload: flag(
                "s3_batch_file_upload",
                params.s3_batch_file_upload.as_ref(),
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
        resolve_partition_granularity(self.partition_granularity_written(environment).as_ref())
    }

    /// The granularity the params write, else the one `S3_PARTITION_GRANULARITY` does.
    pub fn partition_granularity_written(
        &self,
        environment: &Environment,
    ) -> Option<Written<Granularity>> {
        self.partition_granularity.clone().or_else(|| {
            environment(PARTITION_GRANULARITY_ENV_VAR)
                .and_then(|text| partition_granularity(&Value::String(text)))
        })
    }

    pub fn log_prompts_only(&self, environment: &Environment) -> bool {
        let written = self.log_prompts_only.clone().or_else(|| {
            environment(LOG_PROMPTS_ONLY_ENV_VAR)
                .and_then(|text| log_prompts_only(&Value::String(text)))
        });
        match written {
            None => false,
            Some(Written {
                value: Some(prompts_only),
                ..
            }) => prompts_only,
            Some(Written { repr, value: None }) => {
                tracing::warn!(
                    "s3 logging: s3_log_prompts_only={repr} is not a boolean, logging prompts only"
                );
                true
            }
        }
    }

    pub fn signing_region(
        &self,
        from_environment: impl FnOnce() -> Result<Option<String>, String>,
    ) -> Result<String, String> {
        let region = match &self.region_name {
            Some(S3Region::Name(region)) => region.clone(),
            Some(S3Region::NotText(written)) => return Err(invalid_region(written)),
            None => from_environment()?.unwrap_or_else(|| DEFAULT_REGION.to_owned()),
        };
        if valid_region(&region) {
            Ok(region)
        } else {
            Err(invalid_region(&region))
        }
    }

    /// The region as `str()` prints it, `None` for an unset one.
    pub fn region_text(&self) -> Option<&str> {
        self.region_name.as_ref().map(S3Region::text)
    }

    pub fn url_region(&self) -> String {
        self.region_text().unwrap_or("None").to_owned()
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
        Settings::new(
            &serde_json::from_value(params).unwrap(),
            &Constants::default(),
        )
    }

    fn no_env(_: &str) -> Option<String> {
        None
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
