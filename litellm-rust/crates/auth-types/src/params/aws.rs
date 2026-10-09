use std::sync::LazyLock;

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use veil::Redact;

use super::ParamSpec;

const fn spec(
    setting: &'static str,
    wire: &'static [&'static str],
    env: &'static [&'static str],
) -> ParamSpec {
    ParamSpec {
        setting,
        wire,
        module_global: None,
        env,
    }
}

/// The `aws_*` fields of Python's `GenericLiteLLMParams`.
#[derive(Redact, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct AwsParams {
    #[serde(default)]
    pub aws_access_key_id: Option<String>,
    #[serde(default)]
    #[redact(with = "[REDACTED]")]
    pub aws_secret_access_key: Option<String>,
    #[serde(default)]
    #[redact(with = "[REDACTED]")]
    pub aws_session_token: Option<String>,
    #[serde(default)]
    pub aws_region_name: Option<String>,
    #[serde(default)]
    pub aws_session_name: Option<String>,
    #[serde(default)]
    pub aws_profile_name: Option<String>,
    #[serde(default)]
    pub aws_role_name: Option<String>,
    #[serde(default)]
    #[redact(with = "[REDACTED]")]
    pub aws_web_identity_token: Option<String>,
    #[serde(default)]
    pub aws_sts_endpoint: Option<String>,
    #[serde(default)]
    #[redact(with = "[REDACTED]")]
    pub aws_external_id: Option<String>,
    #[serde(default)]
    pub aws_bedrock_runtime_endpoint: Option<String>,
}

impl AwsParams {
    pub const ACCESS_KEY_ID: ParamSpec = spec(
        "access key id",
        &["aws_access_key_id"],
        &["AWS_ACCESS_KEY_ID"],
    );
    pub const SECRET_ACCESS_KEY: ParamSpec = spec(
        "secret access key",
        &["aws_secret_access_key"],
        &["AWS_SECRET_ACCESS_KEY"],
    );
    pub const SESSION_TOKEN: ParamSpec = spec(
        "session token",
        &["aws_session_token"],
        &["AWS_SESSION_TOKEN"],
    );
    pub const REGION: ParamSpec = spec(
        "region",
        &["aws_region_name"],
        &["AWS_REGION_NAME", "AWS_REGION"],
    );
    pub const SESSION_NAME: ParamSpec =
        spec("session name", &["aws_session_name"], &["AWS_SESSION_NAME"]);
    pub const PROFILE_NAME: ParamSpec =
        spec("profile name", &["aws_profile_name"], &["AWS_PROFILE_NAME"]);
    pub const ROLE_NAME: ParamSpec = spec("role name", &["aws_role_name"], &["AWS_ROLE_NAME"]);
    pub const WEB_IDENTITY_TOKEN: ParamSpec = spec(
        "web identity token",
        &["aws_web_identity_token"],
        &["AWS_WEB_IDENTITY_TOKEN"],
    );
    pub const STS_ENDPOINT: ParamSpec =
        spec("STS endpoint", &["aws_sts_endpoint"], &["AWS_STS_ENDPOINT"]);
    pub const EXTERNAL_ID: ParamSpec =
        spec("external id", &["aws_external_id"], &["AWS_EXTERNAL_ID"]);
    pub const BEDROCK_RUNTIME_ENDPOINT: ParamSpec = spec(
        "Bedrock runtime endpoint",
        &["aws_bedrock_runtime_endpoint"],
        &["AWS_BEDROCK_RUNTIME_ENDPOINT"],
    );

    /// Every `aws_*` param Python reads, in declaration order, with the environment names
    /// each falls back to.
    pub const SPECS: [ParamSpec; 11] = [
        Self::ACCESS_KEY_ID,
        Self::SECRET_ACCESS_KEY,
        Self::SESSION_TOKEN,
        Self::REGION,
        Self::SESSION_NAME,
        Self::PROFILE_NAME,
        Self::ROLE_NAME,
        Self::WEB_IDENTITY_TOKEN,
        Self::STS_ENDPOINT,
        Self::EXTERNAL_ID,
        Self::BEDROCK_RUNTIME_ENDPOINT,
    ];

    /// The wire names a host projects out of a caller's kwargs, derived from [`Self::SPECS`].
    pub fn fields() -> impl Iterator<Item = &'static str> {
        Self::SPECS
            .iter()
            .flat_map(|spec| spec.wire.iter().copied())
    }

    /// The environment names the specs read, for hosts that resolve secrets up front.
    pub fn secret_names() -> &'static [&'static str] {
        static NAMES: LazyLock<Vec<&'static str>> = LazyLock::new(|| {
            AwsParams::SPECS
                .iter()
                .flat_map(|spec| spec.env.iter().copied())
                .fold(Vec::new(), |mut names, name| {
                    if !names.contains(&name) {
                        names.push(name);
                    }
                    names
                })
        });
        &NAMES
    }

    /// The value under a wire name, read through serde so the names can never drift from
    /// the struct.
    pub fn get(&self, wire: &str) -> Option<String> {
        serde_json::to_value(self)
            .ok()?
            .get(wire)?
            .as_str()
            .map(str::to_string)
    }

    /// The spec's value from these params, then the environment.
    pub fn resolve(
        &self,
        spec: &ParamSpec,
        env_lookup: &dyn Fn(&str) -> Option<String>,
    ) -> Option<String> {
        spec.resolve(&|name| self.get(name), env_lookup)
    }

    /// Reads the string-valued `aws_*` keys of an untyped params map, ignoring anything else.
    pub fn from_optional_params(optional_params: &Map<String, Value>) -> Self {
        let value = |key: &str| {
            optional_params
                .get(key)
                .and_then(Value::as_str)
                .map(str::to_string)
        };
        Self {
            aws_access_key_id: value("aws_access_key_id"),
            aws_secret_access_key: value("aws_secret_access_key"),
            aws_session_token: value("aws_session_token"),
            aws_region_name: value("aws_region_name"),
            aws_session_name: value("aws_session_name"),
            aws_profile_name: value("aws_profile_name"),
            aws_role_name: value("aws_role_name"),
            aws_web_identity_token: value("aws_web_identity_token"),
            aws_sts_endpoint: value("aws_sts_endpoint"),
            aws_external_id: value("aws_external_id"),
            aws_bedrock_runtime_endpoint: value("aws_bedrock_runtime_endpoint"),
        }
    }
}
