use std::collections::BTreeMap;

use litellm_auth_types::SecretValue;
use serde::Deserialize;

use crate::Object;

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default)]
pub struct McpServer {
    pub server_id: Option<String>,
    pub alias: Option<String>,
    pub description: Option<String>,
    pub mcp_info: Object,
    pub transport: McpTransport,
    pub url: Option<SecretValue>,
    pub command: Option<SecretValue>,
    pub args: Box<[SecretValue]>,
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

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use serde_json::json;

    use crate::{Config, McpAuth, McpTransport};

    #[rstest]
    #[case::http(McpTransport::Http, "http")]
    #[case::sse(McpTransport::Sse, "sse")]
    #[case::stdio(McpTransport::Stdio, "stdio")]
    fn mcp_transport_as_str_matches_the_serde_spelling(
        #[case] transport: McpTransport,
        #[case] name: &str,
    ) {
        assert_eq!(<&'static str>::from(transport), name);
        assert_eq!(
            serde_json::from_value::<McpTransport>(json!(name)).unwrap(),
            transport
        );
    }

    #[rstest]
    #[case::none(McpAuth::None, "none")]
    #[case::api_key(McpAuth::ApiKey, "api_key")]
    #[case::bearer_token(McpAuth::BearerToken, "bearer_token")]
    #[case::basic(McpAuth::Basic, "basic")]
    #[case::authorization(McpAuth::Authorization, "authorization")]
    #[case::token(McpAuth::Token, "token")]
    #[case::oauth2(McpAuth::Oauth2, "oauth2")]
    #[case::aws_sigv4(McpAuth::AwsSigv4, "aws_sigv4")]
    #[case::oauth2_token_exchange(McpAuth::Oauth2TokenExchange, "oauth2_token_exchange")]
    #[case::oauth2_id_jag(McpAuth::Oauth2IdJag, "oauth2_id_jag")]
    #[case::true_passthrough(McpAuth::TruePassthrough, "true_passthrough")]
    #[case::oauth_delegate(McpAuth::OauthDelegate, "oauth_delegate")]
    fn mcp_auth_as_str_matches_the_serde_spelling(#[case] auth: McpAuth, #[case] name: &str) {
        assert_eq!(<&'static str>::from(auth), name);
        assert_eq!(
            serde_json::from_value::<McpAuth>(json!(name)).unwrap(),
            auth
        );
    }

    #[rstest]
    fn mcp_config_redacts_nested_credentials_and_preserves_policy_for_validation() {
        let config = Config::from_yaml("mcp_servers:\n  docs:\n    url: https://example.test/private-secret/mcp\n    authentication_token: upstream-secret\n    static_headers: {x-token: header-secret}\n    env: {TOKEN: env-secret}\n    args: [argument-secret]\n    client_secret: oauth-secret\n    allowed_tools: [search]\n").unwrap();
        let server = &config.mcp_servers["docs"];
        assert_eq!(server.allowed_tools.as_deref().unwrap(), ["search"]);
        assert!(server.unsupported.contains_key("client_secret"));
        let debug = format!("{config:?}");
        for secret in [
            "private-secret",
            "upstream-secret",
            "header-secret",
            "env-secret",
            "oauth-secret",
            "argument-secret",
        ] {
            assert!(!debug.contains(secret));
        }
    }

    #[rstest]
    #[case::transport("transport: invalid")]
    #[case::auth("auth_type: invalid")]
    #[case::concurrency("max_concurrent_requests: -1")]
    #[case::headers("static_headers: {x-token: [not, a, string]}")]
    fn rejects_invalid_typed_mcp_settings(#[case] setting: &str) {
        assert!(Config::from_yaml(&format!("mcp_servers:\n  docs:\n    {setting}\n")).is_err());
    }
}
