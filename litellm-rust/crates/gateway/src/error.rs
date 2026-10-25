#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error(transparent)]
    Http(#[from] litellm_http::Error),
    #[error("environment_variables values must be strings, numbers or booleans")]
    Environment,
    #[error(transparent)]
    Mcp(#[from] litellm_gateway_mcp::ConnectError),
    #[error("MCP requires a configured master key")]
    Auth(#[from] litellm_gateway_auth::Error),
}
