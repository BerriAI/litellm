use litellm_host::failure::{Classify, Kind, UpstreamResponse};

#[derive(Clone, Debug, thiserror::Error)]
pub enum Error {
    #[error(transparent)]
    Upstream(#[from] UpstreamResponse),
    #[error("OCR document download failed with status {0}")]
    DocumentDownloadStatus(u16),
    #[error("OCR document download timed out")]
    DocumentDownloadTimeout,
    #[error("File is empty or could not be read")]
    EmptyFile,
    #[error("Failed to read OCR file {}: {source}", path.display())]
    FileRead {
        path: std::path::PathBuf,
        #[source]
        source: std::sync::Arc<std::io::Error>,
    },
    #[error("OCR document preparation task failed: {0}")]
    DocumentTask(#[source] std::sync::Arc<tokio::task::JoinError>),
    #[error("Invalid MIME type: {0}")]
    InvalidMimeType(String),
    #[error(
        "Cohere Parse only accepts `image_url` documents; document_url and PDF inputs are not supported"
    )]
    CohereImageOnly,
    #[error("Invalid `req_format`: {0:?}. Expected 'native' or 'litellm'.")]
    RequestFormat(String),
    #[error("invalid OCR request field: {path}")]
    RequestField { path: String },
    #[error("missing required field: {0}")]
    MissingField(&'static str),
    #[error("Document URL is required")]
    MissingDocumentUrl,
    #[error("invalid OCR document data URI")]
    InvalidDataUri,
    #[error(
        "Reducto requires a reducto:// id or a data URI; plain HTTP URLs are not supported, upload the file first"
    )]
    ReductoSource,
    #[error("inline OCR document exceeds the size limit")]
    InlineDocumentTooLarge,
    #[error("OCR document URL is blocked by network policy")]
    BlockedDocumentUrl,
    #[error("OCR document downloads are disabled")]
    DownloadDisabled,
    #[error("OCR document download exceeds the size limit")]
    DownloadTooLarge,
    #[error("OCR document download exceeded the redirect limit")]
    TooManyRedirects,
    #[error("invalid OCR pages: {0}")]
    Pages(String),
    #[error("invalid OCR features")]
    Features,
    #[error("OCR model cannot be a dot segment")]
    DotModel,
    #[error("OCR response exceeds the size limit of {limit} bytes")]
    TooLarge { limit: usize },
    #[error("invalid OCR response field: {path}")]
    ResponseField { path: String },
    #[error("OCR response is missing non-empty content")]
    EmptyContent,
    #[error("OCR document redirect is missing a location")]
    MissingRedirectLocation,
    #[error("OCR document redirect location is invalid")]
    InvalidRedirect,
    #[error("OCR operation ended with status {0}")]
    OperationStatus(String),
    #[error("OCR response numeric value is out of range: {0}")]
    NumericRange(&'static str),
    #[error("OCR accepted response is missing a valid operation-location")]
    PollLocation,
    #[error("OCR operation-location must use the submission origin without credentials")]
    PollOrigin,
    #[error("OCR polling timed out")]
    PollTimeout,
    #[error("unsupported by the rust path: {0}")]
    Unsupported(&'static str),
    #[error("invalid provider: {0}")]
    InvalidProvider(String),
    #[error("invalid model: {provider} has no model {model:?} - use one of: {}", supported.join(", "))]
    InvalidModel {
        provider: &'static str,
        model: String,
        supported: &'static [&'static str],
    },
    #[error("invalid request: {0}")]
    InvalidRequest(String),
    #[error("invalid response: {0}")]
    InvalidResponse(String),
    #[error(
        "invalid authentication configuration: Missing Azure AI credentials - set AZURE_AI_API_KEY or configure Entra ID"
    )]
    MissingAzureAiCredentials,
    #[error(
        "invalid authentication configuration: Missing Azure Document Intelligence credentials - set AZURE_DOCUMENT_INTELLIGENCE_API_KEY or configure Entra ID"
    )]
    MissingAzureDocumentIntelligenceCredentials,
    #[error(
        "Missing REDUCTO_API_KEY - set it in the environment or pass api_key to litellm.ocr()/litellm.aocr()"
    )]
    MissingReductoApiKey,
    #[error("secret resolution failed: {0}")]
    Secret(#[source] std::sync::Arc<litellm_secrets::Error>),
    #[error(transparent)]
    Auth(#[from] litellm_auth::Error),
    #[error(transparent)]
    Transport(#[from] litellm_http::transport::Error),
    #[error(transparent)]
    Params(#[from] litellm_core_utils::params::Error),
    #[error(transparent)]
    Headers(#[from] litellm_http::request::HeaderError),
    #[error(transparent)]
    Http(#[from] litellm_http::Error),
}

impl From<litellm_host::machine::MachineFault> for Error {
    fn from(fault: litellm_host::machine::MachineFault) -> Self {
        use litellm_host::machine::MachineFault;
        Self::InvalidRequest(match fault {
            MachineFault::Abandoned => "OCR host driver was abandoned".into(),
            MachineFault::Protocol(message) => format!("OCR {message}"),
        })
    }
}

impl From<litellm_core_utils::call_arguments::ArgumentError> for Error {
    fn from(error: litellm_core_utils::call_arguments::ArgumentError) -> Self {
        Self::RequestField {
            path: format!("optional_params.{}", error.path),
        }
    }
}

impl Classify for Error {
    fn kind(&self) -> Kind {
        match self {
            Self::Upstream(response) => Kind::Upstream(response.clone()),
            Self::Unsupported(_) => Kind::Unsupported,
            Self::Auth(litellm_auth::Error::MissingApiKey { .. })
            | Self::MissingAzureAiCredentials
            | Self::MissingAzureDocumentIntelligenceCredentials
            | Self::MissingReductoApiKey => Kind::Auth,
            Self::Auth(_) => Kind::Request,
            Self::FileRead { path, source } => Kind::File {
                path: path.display().to_string(),
                not_found: source.kind() == std::io::ErrorKind::NotFound,
            },
            Self::Transport(litellm_http::transport::Error::Timeout(_))
            | Self::DocumentDownloadTimeout
            | Self::PollTimeout => Kind::Timeout,
            Self::Transport(_) => Kind::Connection,
            Self::EmptyFile
            | Self::InvalidMimeType(_)
            | Self::CohereImageOnly
            | Self::ReductoSource
            | Self::RequestFormat(_)
            | Self::RequestField { .. }
            | Self::MissingField(_)
            | Self::MissingDocumentUrl
            | Self::InvalidDataUri
            | Self::InlineDocumentTooLarge
            | Self::BlockedDocumentUrl
            | Self::DownloadDisabled
            | Self::DownloadTooLarge
            | Self::TooManyRedirects
            | Self::DocumentDownloadStatus(_)
            | Self::Pages(_)
            | Self::Features
            | Self::DotModel
            | Self::InvalidRequest(_)
            | Self::InvalidProvider(_)
            | Self::InvalidModel { .. }
            | Self::Params(_)
            | Self::Headers(_)
            | Self::Http(_) => Kind::Request,
            Self::TooLarge { .. }
            | Self::ResponseField { .. }
            | Self::EmptyContent
            | Self::MissingRedirectLocation
            | Self::InvalidRedirect
            | Self::OperationStatus(_)
            | Self::NumericRange(_)
            | Self::PollLocation
            | Self::PollOrigin
            | Self::InvalidResponse(_) => Kind::Response,
            Self::DocumentTask(_) | Self::Secret(_) => Kind::Internal,
        }
    }
}
