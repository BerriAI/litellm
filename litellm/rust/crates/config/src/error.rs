#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("invalid YAML config")]
    Parse(#[from] serde_yaml_ng::Error),
}
