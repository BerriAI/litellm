use std::{io, path::PathBuf};

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("could not read config {}", path.display())]
    Read { path: PathBuf, source: io::Error },
    #[error("invalid YAML config")]
    Parse(#[from] serde_yaml_ng::Error),
    #[error("invalid YAML config {}", path.display())]
    ParseFile {
        path: PathBuf,
        source: serde_yaml_ng::Error,
    },
}
