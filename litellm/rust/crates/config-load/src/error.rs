use std::{io, path::PathBuf};

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("could not read config {}", path.display())]
    Read { path: PathBuf, source: io::Error },
    #[error("invalid YAML in {}", path.display())]
    LoadFile {
        path: PathBuf,
        source: litellm_yaml::Error,
    },
    #[error("invalid YAML config {}", path.display())]
    ParseFile {
        path: PathBuf,
        source: serde_yaml_ng::Error,
    },
    #[error("could not read {reference}")]
    Secret {
        reference: String,
        source: litellm_secrets::Error,
    },
    #[error("environment_variables.{name} must be a string, number or boolean")]
    Environment { name: String },
    #[error("invalid config value")]
    Value(#[from] serde_yaml_ng::Error),
    #[error(transparent)]
    Config(#[from] litellm_config::Error),
}
