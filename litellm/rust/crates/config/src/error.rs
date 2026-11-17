#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("invalid YAML config")]
    Parse(#[from] serde_yaml_ng::Error),
    #[error("invalid YAML")]
    Load(#[from] litellm_yaml::Error),
}
