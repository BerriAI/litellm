#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error(
        "LITELLM_RUST_GATEWAY=true requires LITELLM_PYTHON_UPSTREAM to be a valid http:// origin"
    )]
    PythonUpstreamConfig,
    #[error(transparent)]
    Http(#[from] litellm_http::Error),
    #[error("environment_variables values must be strings, numbers or booleans")]
    Environment,
    #[error("MCP host does not yet support configured setting {0}")]
    McpSetting(String),
    #[error("MCP host does not yet support guardrails or policies")]
    McpGuardrails,
    #[error(transparent)]
    Mcp(#[from] litellm_gateway_mcp::ConnectError),
    #[error("MCP requires working gateway authentication")]
    Auth(#[from] litellm_gateway_auth::Error),
}
