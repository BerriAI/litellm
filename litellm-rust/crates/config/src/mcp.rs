use std::{collections::BTreeMap, fmt};

use litellm_auth_types::SecretValue;
use serde::Deserialize;

use crate::Object;

#[derive(Clone, Default, Deserialize)]
#[serde(default)]
pub struct McpServer {
    pub server_id: Option<String>,
    pub alias: Option<String>,
    pub description: Option<String>,
    pub mcp_info: Object,
    pub transport: McpTransport,
    pub url: Option<SecretValue>,
    pub command: Option<String>,
    pub args: Box<[String]>,
    pub env: BTreeMap<String, SecretValue>,
    pub auth_type: Option<McpAuth>,
    #[serde(alias = "auth_value")]
    pub authentication_token: Option<SecretValue>,
    pub static_headers: BTreeMap<String, SecretValue>,
    pub upstream_token_header: Option<String>,
    pub allowed_tools: Option<Box<[String]>>,
    pub timeout: Option<f64>,
    pub max_concurrent_requests: Option<usize>,
    #[serde(flatten)]
    pub unsupported: Object,
}

impl fmt::Debug for McpServer {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("McpServer")
            .field("transport", &self.transport)
            .field("auth_type", &self.auth_type)
            .field("timeout", &self.timeout)
            .field("max_concurrent_requests", &self.max_concurrent_requests)
            .finish_non_exhaustive()
    }
}

#[derive(Clone, Copy, Debug, Default, Deserialize, PartialEq, Eq, strum::IntoStaticStr)]
#[serde(rename_all = "snake_case")]
pub enum McpTransport {
    #[default]
    #[strum(serialize = "http")]
    Http,
    #[strum(serialize = "sse")]
    Sse,
    #[strum(serialize = "stdio")]
    Stdio,
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq, strum::IntoStaticStr)]
#[serde(rename_all = "snake_case")]
pub enum McpAuth {
    #[strum(serialize = "none")]
    None,
    #[strum(serialize = "api_key")]
    ApiKey,
    #[strum(serialize = "bearer_token")]
    BearerToken,
    #[strum(serialize = "basic")]
    Basic,
    #[strum(serialize = "authorization")]
    Authorization,
    #[strum(serialize = "token")]
    Token,
    #[strum(serialize = "oauth2")]
    Oauth2,
    #[strum(serialize = "aws_sigv4")]
    AwsSigv4,
    #[strum(serialize = "oauth2_token_exchange")]
    Oauth2TokenExchange,
    #[strum(serialize = "oauth2_id_jag")]
    Oauth2IdJag,
    #[strum(serialize = "true_passthrough")]
    TruePassthrough,
    #[strum(serialize = "oauth_delegate")]
    OauthDelegate,
}
