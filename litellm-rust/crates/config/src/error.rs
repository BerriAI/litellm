#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("could not read config")]
    Read(#[from] std::io::Error),
    #[error("invalid YAML config")]
    Parse(#[from] serde_yaml_ng::Error),
}
