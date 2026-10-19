#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("could not read config")]
    Read(#[from] std::io::Error),
    #[error("invalid YAML config")]
    Parse(#[from] ParseError),
    #[error("YAML value requires the Python compatibility loader")]
    UnsupportedYaml,
    #[error("'include' must be a list of file paths")]
    Includes,
    #[error("Included config file is not a YAML mapping: {0}")]
    Mapping(std::path::PathBuf),
}

#[derive(Debug, thiserror::Error)]
pub enum ParseError {
    #[error("invalid config value")]
    Value(#[from] serde_yaml_ng::Error),
    #[error("invalid YAML syntax")]
    Syntax(#[from] yaml_rust2::scanner::ScanError),
}

impl From<serde_yaml_ng::Error> for Error {
    fn from(error: serde_yaml_ng::Error) -> Self {
        Self::Parse(ParseError::Value(error))
    }
}

impl From<yaml_rust2::scanner::ScanError> for Error {
    fn from(error: yaml_rust2::scanner::ScanError) -> Self {
        Self::Parse(ParseError::Syntax(error))
    }
}
