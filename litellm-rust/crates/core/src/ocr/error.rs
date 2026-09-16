use thiserror::Error;

use crate::transport::Error as TransportError;

#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum Error {
    #[error("expected {expected}, got {actual}")]
    InvalidType {
        expected: &'static str,
        actual: &'static str,
    },
    #[error("missing required field: {0}")]
    MissingField(&'static str),
    #[error("Document URL is required")]
    MissingDocumentUrl,
    #[error("invalid response: {0}")]
    InvalidResponse(String),
    #[error("invalid provider: {0}")]
    InvalidProvider(String),
    #[error("invalid request: {0}")]
    InvalidRequest(String),
    #[error("{0}")]
    Auth(String),
    #[error(
        "Missing {provider} API Key - A call is being made to {provider} but no key is set either in the environment variables or via params"
    )]
    MissingApiKey { provider: &'static str },
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
    #[error("upstream request failed with status {status}: {body}")]
    Http { status: u16, body: String },
    #[error("upstream network error: {0}")]
    Network(String),
    /// The provider was never reached: DNS, TCP, TLS or proxy setup failed
    /// before any byte of the request went out. Nothing was billed, so a host
    /// that keeps a reference implementation can serve the request itself.
    /// A timeout is deliberately not this, since the provider may have received
    /// and answered the request already.
    #[error("could not reach the provider: {0}")]
    Connect(String),
    #[error("routing error: {0}")]
    Routing(String),
    /// The request is outside the surface this route covers in Rust. Hosts that
    /// keep a reference implementation treat this as "fall back", not "fail".
    #[error("unsupported by the rust path: {0}")]
    Unsupported(&'static str),
}

impl Error {
    pub const fn http_status_code(&self) -> Option<u16> {
        match self {
            Self::InvalidRequest(_) => Some(400),
            Self::MissingDocumentUrl => Some(500),
            Self::Http { status, .. } => Some(*status),
            _ => None,
        }
    }
}

impl From<OcrRequestError> for Error {
    fn from(error: OcrRequestError) -> Self {
        match error {
            OcrRequestError::MissingField(field) => Self::MissingField(field),
            OcrRequestError::MissingDocumentUrl => Self::MissingDocumentUrl,
            error => Self::InvalidRequest(error.to_string()),
        }
    }
}

impl From<OcrResponseError> for Error {
    fn from(error: OcrResponseError) -> Self {
        Self::InvalidResponse(error.to_string())
    }
}

impl From<TransportError> for Error {
    fn from(error: TransportError) -> Self {
        match error {
            TransportError::Http { status, body } => Self::Http { status, body },
            TransportError::Network(message) => Self::Network(message),
            TransportError::Connect(message) => Self::Connect(message),
        }
    }
}

impl From<litellm_auth::Error> for Error {
    fn from(error: litellm_auth::Error) -> Self {
        match error {
            litellm_auth::Error::MissingApiKey { provider, .. } => Self::MissingApiKey { provider },
            error => Self::Auth(error.to_string()),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum OcrRequestError {
    #[error("File is empty or could not be read")]
    EmptyFile,
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
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum OcrResponseError {
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
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum OcrPollingError {
    #[error("OCR accepted response is missing a valid operation-location")]
    PollLocation,
    #[error("OCR operation-location must use the submission origin without credentials")]
    PollOrigin,
    #[error("OCR polling timed out")]
    PollTimeout,
}

#[derive(Debug, Error)]
pub enum OcrError {
    #[error("{0}")]
    Request(#[from] OcrRequestError),
    #[error("{0}")]
    Response(#[from] OcrResponseError),
    #[error("{0}")]
    Transport(#[from] TransportError),
    #[error("{0}")]
    Polling(#[from] OcrPollingError),
    #[error("{0}")]
    Public(#[from] Error),
}

impl From<OcrError> for Error {
    fn from(error: OcrError) -> Self {
        match error {
            OcrError::Request(error) => error.into(),
            OcrError::Response(error) => error.into(),
            OcrError::Transport(error) => error.into(),
            OcrError::Polling(error) => Error::InvalidResponse(error.to_string()),
            OcrError::Public(error) => error,
        }
    }
}
