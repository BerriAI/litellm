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

#[derive(Clone, Copy, Debug, Default, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum McpTransport {
    #[default]
    Http,
    Sse,
    Stdio,
}

impl McpTransport {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Http => "http",
            Self::Sse => "sse",
            Self::Stdio => "stdio",
        }
    }
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum McpAuth {
    None,
    ApiKey,
    BearerToken,
    Basic,
    Authorization,
    Token,
    Oauth2,
    AwsSigv4,
    Oauth2TokenExchange,
    Oauth2IdJag,
    TruePassthrough,
    OauthDelegate,
}

impl McpAuth {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::ApiKey => "api_key",
            Self::BearerToken => "bearer_token",
            Self::Basic => "basic",
            Self::Authorization => "authorization",
            Self::Token => "token",
            Self::Oauth2 => "oauth2",
            Self::AwsSigv4 => "aws_sigv4",
            Self::Oauth2TokenExchange => "oauth2_token_exchange",
            Self::Oauth2IdJag => "oauth2_id_jag",
            Self::TruePassthrough => "true_passthrough",
            Self::OauthDelegate => "oauth_delegate",
        }
    }
}
