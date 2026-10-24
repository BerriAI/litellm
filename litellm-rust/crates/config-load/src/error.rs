use std::fmt;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("could not read config")]
    Read(#[from] std::io::Error),
    #[error("invalid YAML config")]
    Parse(#[from] serde_yaml_ng::Error),
    #[error("invalid config")]
    Config(#[from] litellm_config::Error),
    #[error("could not load config {key}")]
    Document {
        key: String,
        #[source]
        source: Box<Error>,
    },
}

impl Error {
    pub(crate) fn document(key: &impl fmt::Debug, source: Error) -> Self {
        Self::Document {
            key: format!("{key:?}"),
            source: Box::new(source),
        }
    }
}
