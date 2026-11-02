#[derive(Debug, thiserror::Error)]
pub enum Error {
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
    #[error(transparent)]
    Config(#[from] litellm_config::Error),
    #[error(transparent)]
    Redis(#[from] litellm_config::RedisError),
    #[error(transparent)]
    RedisConnect(#[from] litellm_redis_connect::Error),
    #[error("MCP requires working gateway authentication")]
    Auth(#[from] litellm_gateway_auth::Error),
}
