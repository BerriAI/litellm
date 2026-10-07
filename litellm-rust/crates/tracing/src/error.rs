#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("invalid diagnostic configuration")]
    ConfigurationJson(#[from] serde_json::Error),
    #[error("diagnostic destination names must be nonempty and unique")]
    DestinationName,
    #[error("diagnostic configuration requires a value for {0}")]
    MissingValue(String),
    #[error("diagnostic export transport is unavailable in this build")]
    UnavailableTransport,
    #[cfg(feature = "posthog")]
    #[error("could not configure diagnostic export")]
    Configuration(#[from] posthog_rs::ClientOptionsBuilderError),
    #[error("sample rate must be finite and between zero and one")]
    InvalidSampleRate,
    #[error("could not configure diagnostic export")]
    ExportBuild(#[from] opentelemetry_otlp::ExporterBuildError),
    #[error("diagnostic exporter operation failed")]
    Export(#[from] opentelemetry_sdk::error::OTelSdkError),
}
