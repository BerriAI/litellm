use litellm_config::{McpAuth, McpTransport};
use rstest::rstest;
use serde_json::json;

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
