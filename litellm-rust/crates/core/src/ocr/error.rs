use thiserror::Error;

use crate::auth::AuthError;
use crate::error::TransportError;

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum PagesError {
    #[error("`pages` must be integers, not booleans")]
    BooleanIndex,
    #[error("`pages` integers must be >= 0 (Mistral 0-based indices)")]
    NegativeIndex,
    #[error("`pages` integer is outside the supported range")]
    IndexOutOfRange,
    #[error("`pages` must be a list[int] (0-based, Mistral-style) or a string like '1-3,5,7-9'.")]
    MixedElementTypes,
    #[error("Invalid `pages` for Azure Document Intelligence. Expected tokens like '1' or '3-5'.")]
    InvalidNativeRange,
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum OcrRequestError {
    #[error("{0}")]
    Pages(#[from] PagesError),
    #[error(
        "Invalid `features` for Azure Document Intelligence. Expected a list of feature names or a comma-separated string."
    )]
    Features,
    #[error("Invalid `req_format`. Expected 'native' or 'litellm'.")]
    RequestFormat,
    #[error("`req_format=native` is not supported for provider {0}")]
    NativeUnsupported(&'static str),
    #[error("invalid OCR request field: {path}")]
    RequestField { path: String },
    #[error("missing required field: {0}")]
    MissingField(&'static str),
    #[error("model_id cannot be a dot path segment")]
    DotModel,
    #[error(
        "Reducto requires a reducto:// id or a data URI after OCR preprocessing; plain http(s) URLs are not supported"
    )]
    ReductoSource,
    #[error("Invalid OCR document data URI")]
    InvalidDataUri,
    #[error("OCR inline document exceeds maximum allowed size")]
    InlineDocumentTooLarge,
    #[error("OCR document URL rejected by SSRF protection")]
    BlockedDocumentUrl,
    #[error("Too many redirects while fetching OCR document URL")]
    TooManyRedirects,
    #[error("OCR document URL download is disabled (MAX_IMAGE_URL_DOWNLOAD_SIZE_MB=0)")]
    DownloadDisabled,
    #[error("OCR document exceeds maximum allowed download size")]
    DownloadTooLarge,
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum OcrResponseError {
    #[error("invalid OCR response field: {path}")]
    ResponseField { path: String },
    #[error("Azure Document Intelligence analysis failed with status: {0}")]
    OperationStatus(String),
    #[error("OCR numeric value is outside the supported range: {0}")]
    NumericRange(&'static str),
    #[error("No content in DeepSeek OCR response")]
    EmptyContent,
    #[error("OCR document redirect missing Location header")]
    MissingRedirectLocation,
    #[error("invalid OCR document redirect")]
    InvalidRedirect,
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum OcrPollingError {
    #[error("Azure Document Intelligence: rejected cross-origin polling URL")]
    PollOrigin,
    #[error("Azure Document Intelligence returned 202 but no Operation-Location header found")]
    PollLocation,
    #[error("Azure Document Intelligence operation polling timed out")]
    PollTimeout,
}

#[derive(Debug, Error)]
pub enum OcrError {
    #[error("{0}")]
    Request(#[from] OcrRequestError),
    #[error("{0}")]
    Response(#[from] OcrResponseError),
    #[error("{0}")]
    Polling(#[from] OcrPollingError),
    #[error("{0}")]
    Auth(#[from] AuthError),
    #[error("{0}")]
    Transport(#[from] TransportError),
}

impl From<OcrError> for crate::Error {
    fn from(error: OcrError) -> Self {
        match error {
            OcrError::Request(error) => error.into(),
            OcrError::Response(error) => error.into(),
            OcrError::Polling(error) => error.into(),
            OcrError::Auth(error) => error.into(),
            OcrError::Transport(error) => error.into(),
        }
    }
}
