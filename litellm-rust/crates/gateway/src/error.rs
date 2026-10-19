#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error(transparent)]
    Http(#[from] litellm_http::Error),
    #[error("environment_variables values must be strings, numbers or booleans")]
    Environment,
    #[error("MCP host does not yet support configured setting {0}")]
    McpSetting(String),
    #[error(transparent)]
    Routing(#[from] litellm_gateway_inference::RoutingError),
    #[error(transparent)]
    Mcp(#[from] litellm_gateway_mcp::ConnectError),
    #[error("virtual key database unavailable")]
    Database(#[from] litellm_db::Error),
    #[error(
        "virtual keys cannot be enabled while guardrails, policies or policy_attachments are configured: this gateway does not run them on inference, so virtual-key requests would bypass them"
    )]
    GuardedVirtualKeys,
    #[error("database_url references {0}, which is not set")]
    DatabaseUrl(String),
    #[error("reading a gateway setting from the secret source failed")]
    Secret(#[from] litellm_secrets::Error),
    #[error("litellm_settings.hooks: {0}")]
    Hooks(String),
    #[error("litellm_settings.s3_callback_params: {0}")]
    S3Log(String),
    #[error("coordination Redis: {0}")]
    Redis(String),
    #[error("MCP requires a configured master key")]
    Auth(#[from] litellm_gateway_auth::Error),
}
