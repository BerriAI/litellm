#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("could not read config")]
    Read(#[from] std::io::Error),
    #[error("invalid YAML config")]
    Parse(#[from] serde_yaml_ng::Error),
    #[error("Redis host does not yet support configured setting {0}")]
    RedisSetting(String),
    #[error("invalid Redis setting {setting}: {reason}")]
    RedisInvalid {
        setting: &'static str,
        reason: &'static str,
    },
}
