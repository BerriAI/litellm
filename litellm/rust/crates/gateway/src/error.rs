use std::path::PathBuf;

#[derive(Debug, thiserror::Error)]
pub enum ConfigSourceError {
    #[error("could not read config {}", path.display())]
    Read {
        path: PathBuf,
        source: std::io::Error,
    },
    #[error("invalid YAML config {}", path.display())]
    Parse {
        path: PathBuf,
        source: serde_yaml_ng::Error,
    },
    #[error("config object {key} cannot be read because bucket configs are not supported")]
    Object { key: String },
}

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
    ConfigSource(#[from] ConfigSourceError),
    #[error(transparent)]
    ConfigLoad(#[from] litellm_config_load::Error<ConfigSourceError>),
    #[error(transparent)]
    Redis(#[from] litellm_redis_resolve::Error),
    #[error("Redis setting {setting} is not plain data")]
    RedisNotPlainData { setting: String },
    #[error(transparent)]
    RedisConnect(#[from] litellm_redis_connect::Error),
    #[error("MCP requires working gateway authentication")]
    Auth(#[from] litellm_gateway_auth::Error),
}
