use thiserror::Error;

use crate::error::TransportError;

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
    #[error("invalid OCR document data URI")]
    InvalidDataUri,
    #[error("Reducto requires a reducto:// id or a data URI")]
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
    Public(#[from] crate::Error),
}

impl From<OcrError> for crate::Error {
    fn from(error: OcrError) -> Self {
        match error {
            OcrError::Request(error) => error.into(),
            OcrError::Response(error) => error.into(),
            OcrError::Transport(error) => error.into(),
            OcrError::Polling(error) => crate::Error::InvalidResponse(error.to_string()),
            OcrError::Public(error) => error,
        }
    }
}
