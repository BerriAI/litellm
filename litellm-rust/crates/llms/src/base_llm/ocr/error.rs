#[derive(Clone, Debug, thiserror::Error)]
pub enum Error {
    #[error("upstream OCR error ({status}): {body}")]
    Provider {
        status: u16,
        body: String,
        headers: Vec<(String, String)>,
    },
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
    #[error("Invalid `req_format`. Expected 'native' or 'litellm'.")]
    RequestFormat,
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
    #[error(transparent)]
    Auth(#[from] litellm_auth::Error),
    #[error(transparent)]
    Transport(#[from] crate::custom_httpx::transport::Error),
    #[error(transparent)]
    Params(#[from] litellm_core_utils::params::Error),
    #[error(transparent)]
    Headers(#[from] crate::custom_httpx::http_handler::HeaderError),
}

impl From<litellm_core_utils::call_arguments::ArgumentError> for Error {
    fn from(error: litellm_core_utils::call_arguments::ArgumentError) -> Self {
        Self::RequestField {
            path: format!("optional_params.{}", error.path),
        }
    }
}

impl Error {
    pub fn http_status_code(&self) -> Option<u16> {
        match self {
            Self::Provider { status, .. }
            | Self::Transport(crate::custom_httpx::transport::Error::Http { status, .. }) => {
                Some(*status)
            }
            error if error.is_request() => Some(400),
            _ => None,
        }
    }

    pub fn is_request(&self) -> bool {
        matches!(
            self,
            Self::EmptyFile
                | Self::InvalidMimeType(_)
                | Self::CohereImageOnly
                | Self::RequestFormat
                | Self::RequestField { .. }
                | Self::MissingField(_)
                | Self::MissingDocumentUrl
                | Self::InvalidDataUri
                | Self::ReductoSource
                | Self::InlineDocumentTooLarge
                | Self::BlockedDocumentUrl
                | Self::DownloadDisabled
                | Self::DownloadTooLarge
                | Self::TooManyRedirects
                | Self::Pages(_)
                | Self::Features
                | Self::DotModel
                | Self::InvalidRequest(_)
                | Self::InvalidProvider(_)
                | Self::Params(_)
                | Self::Headers(_)
        )
    }

    pub fn is_response(&self) -> bool {
        matches!(
            self,
            Self::TooLarge { .. }
                | Self::ResponseField { .. }
                | Self::EmptyContent
                | Self::MissingRedirectLocation
                | Self::InvalidRedirect
                | Self::OperationStatus(_)
                | Self::NumericRange(_)
                | Self::PollLocation
                | Self::PollOrigin
                | Self::PollTimeout
                | Self::InvalidResponse(_)
        )
    }
}
