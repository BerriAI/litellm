use std::{io, path::PathBuf, sync::Arc};

use litellm_core_utils::call_arguments::ArgumentError;
use litellm_error::Source;
use reqwest::header::HeaderMap;

static NO_HEADERS: std::sync::LazyLock<HeaderMap> = std::sync::LazyLock::new(HeaderMap::new);

#[derive(Clone, Debug, thiserror::Error)]
pub enum Error {
    #[error("upstream OCR error ({status}): {body}")]
    Provider {
        status: u16,
        body: String,
        headers: Box<HeaderMap>,
    },
    #[error("File is empty or could not be read")]
    EmptyFile,
    #[error("could not read OCR file {}", path.display())]
    FileRead {
        path: PathBuf,
        #[source]
        cause: Arc<io::Error>,
    },
    #[error("OCR document preparation task failed")]
    DocumentTask(#[source] Arc<tokio::task::JoinError>),
    #[error("invalid MIME type {0:?}")]
    InvalidMimeType(String),
    #[error("{provider} cannot read this document - {guidance}")]
    UnsupportedDocument {
        provider: &'static str,
        guidance: &'static str,
    },
    #[error("{provider} cannot read this document: {message} - {guidance}")]
    RejectedDocument {
        provider: &'static str,
        message: String,
        guidance: &'static str,
    },
    #[error("Invalid `req_format`. Expected 'native' or 'litellm'.")]
    RequestFormat,
    #[error("invalid OCR request field: {path}")]
    RequestField {
        path: String,
        #[source]
        cause: Option<Source>,
    },
    #[error("invalid OCR optional params")]
    Arguments(#[from] ArgumentError),
    #[error("missing required field: {0}")]
    MissingField(&'static str),
    #[error("document URL is required")]
    MissingDocumentUrl,
    #[error("invalid OCR document data URI")]
    InvalidDataUri,
    #[error("inline OCR document exceeds the size limit")]
    InlineDocumentTooLarge,
    #[error("invalid OCR pages: {0}")]
    Pages(PagesFault),
    #[error("invalid OCR features")]
    Features,
    #[error(
        "invalid OCR outputContentFormat, expected text or markdown. To drop unsupported params, set `litellm.drop_params = True`."
    )]
    OutputContentFormat,
    #[error("OCR model cannot be a dot segment")]
    DotModel,
    #[error("OCR response exceeds the size limit of {limit} bytes")]
    TooLarge { limit: usize },
    #[error("invalid OCR response field: {path}")]
    ResponseField {
        path: String,
        #[source]
        cause: Option<Source>,
    },
    #[error("OCR response is missing non-empty content")]
    EmptyContent,
    #[error("OCR operation ended with status {0}")]
    OperationStatus(String),
    #[error("OCR response numeric value is out of range: {0}")]
    NumericRange(&'static str),
    #[error("OCR accepted response is missing a valid operation-location")]
    PollLocation,
    #[error("OCR {url} is not a valid URL")]
    InvalidPollUrl {
        url: PollUrl,
        #[source]
        source: url::ParseError,
    },
    #[error("OCR operation-location must use the submission origin without credentials")]
    PollOrigin,
    #[error("OCR polling timed out")]
    PollTimeout,
    #[error("invalid provider: {0}")]
    InvalidProvider(String),
    #[error("invalid model: {provider} has no model {model:?} - use one of: {}", supported.join(", "))]
    InvalidModel {
        provider: &'static str,
        model: String,
        supported: &'static [&'static str],
    },
    #[error("Missing {credential} - {guidance}")]
    MissingCredential {
        credential: &'static str,
        guidance: &'static str,
    },
    #[error("secret resolution failed")]
    Secret(#[source] Arc<litellm_secrets::Error>),
    #[error("the OCR host driver failed")]
    Host(#[source] litellm_host::machine::MachineFault),
    #[error("the OCR host call raised an exception")]
    HostCall,
    #[error(transparent)]
    Auth(#[from] litellm_auth::Error),
    #[error(transparent)]
    Params(#[from] litellm_core_utils::params::Error),
    #[error(transparent)]
    Headers(#[from] litellm_http::request::HeaderError),
    #[error(transparent)]
    Http(#[from] litellm_http::Error),
    #[error(transparent)]
    DocumentDownload(litellm_http::Error),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum PagesFault {
    #[error("page index is out of range")]
    OutOfRange,
    #[error("negative page index")]
    Negative,
    #[error("expected only integers or only strings")]
    Mixed,
    #[error("expected an array of integers or strings, or a native page range")]
    Shape,
    #[error("invalid native page range")]
    Range,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum PollUrl {
    #[error("submission URL")]
    Submission,
    #[error("operation-location")]
    OperationLocation,
}

impl From<litellm_host::machine::MachineFault> for Error {
    fn from(fault: litellm_host::machine::MachineFault) -> Self {
        Self::Host(fault)
    }
}

impl Error {
    pub fn request_field(
        path: impl Into<String>,
        cause: impl std::error::Error + Send + Sync + 'static,
    ) -> Self {
        Self::RequestField {
            path: path.into(),
            cause: Some(Source::new(cause)),
        }
    }

    pub fn response_field(
        path: impl Into<String>,
        cause: impl std::error::Error + Send + Sync + 'static,
    ) -> Self {
        Self::ResponseField {
            path: path.into(),
            cause: Some(Source::new(cause)),
        }
    }

    pub fn upstream_response(&self) -> Option<(u16, &str, &HeaderMap)> {
        match self {
            Self::Provider {
                status,
                body,
                headers,
            } => Some((*status, body, headers)),
            Self::DocumentDownload(error) => error.status().map(|status| {
                let body = if error.is_timeout() {
                    "OCR document download timed out"
                } else {
                    "OCR document download failed"
                };
                (status, body, &*NO_HEADERS)
            }),
            Self::Http(error) => error
                .upstream_response()
                .map(|(status, body)| (status, body, &*NO_HEADERS)),
            Self::EmptyFile
            | Self::FileRead { .. }
            | Self::DocumentTask(_)
            | Self::InvalidMimeType(_)
            | Self::UnsupportedDocument { .. }
            | Self::RejectedDocument { .. }
            | Self::RequestFormat
            | Self::RequestField { .. }
            | Self::Arguments(_)
            | Self::MissingField(_)
            | Self::MissingDocumentUrl
            | Self::InvalidDataUri
            | Self::InlineDocumentTooLarge
            | Self::Pages(_)
            | Self::Features
            | Self::OutputContentFormat
            | Self::DotModel
            | Self::TooLarge { .. }
            | Self::ResponseField { .. }
            | Self::EmptyContent
            | Self::OperationStatus(_)
            | Self::NumericRange(_)
            | Self::PollLocation
            | Self::InvalidPollUrl { .. }
            | Self::PollOrigin
            | Self::PollTimeout
            | Self::InvalidProvider(_)
            | Self::InvalidModel { .. }
            | Self::MissingCredential { .. }
            | Self::Secret(_)
            | Self::Host(_)
            | Self::HostCall
            | Self::Auth(_)
            | Self::Params(_)
            | Self::Headers(_) => None,
        }
    }

    pub fn is_missing_credential(&self) -> bool {
        match self {
            Self::MissingCredential { .. } => true,
            Self::Auth(error) => error.is_missing_credential(),
            Self::Provider { .. }
            | Self::EmptyFile
            | Self::FileRead { .. }
            | Self::DocumentTask(_)
            | Self::InvalidMimeType(_)
            | Self::UnsupportedDocument { .. }
            | Self::RejectedDocument { .. }
            | Self::RequestFormat
            | Self::RequestField { .. }
            | Self::Arguments(_)
            | Self::MissingField(_)
            | Self::MissingDocumentUrl
            | Self::InvalidDataUri
            | Self::InlineDocumentTooLarge
            | Self::Pages(_)
            | Self::Features
            | Self::OutputContentFormat
            | Self::DotModel
            | Self::TooLarge { .. }
            | Self::ResponseField { .. }
            | Self::EmptyContent
            | Self::OperationStatus(_)
            | Self::NumericRange(_)
            | Self::PollLocation
            | Self::InvalidPollUrl { .. }
            | Self::PollOrigin
            | Self::PollTimeout
            | Self::InvalidProvider(_)
            | Self::InvalidModel { .. }
            | Self::Secret(_)
            | Self::Host(_)
            | Self::HostCall
            | Self::Params(_)
            | Self::Headers(_)
            | Self::Http(_)
            | Self::DocumentDownload(_) => false,
        }
    }

    pub fn http_status_code(&self) -> Option<u16> {
        if let Some((status, ..)) = self.upstream_response() {
            return Some(status);
        }
        if let Self::Http(error) = self
            && let Some(status) = error.status()
        {
            return Some(status);
        }
        self.is_request().then_some(400)
    }

    pub fn is_request(&self) -> bool {
        match self {
            Self::EmptyFile
            | Self::InvalidMimeType(_)
            | Self::UnsupportedDocument { .. }
            | Self::RejectedDocument { .. }
            | Self::RequestFormat
            | Self::RequestField { .. }
            | Self::Arguments(_)
            | Self::MissingField(_)
            | Self::MissingDocumentUrl
            | Self::InvalidDataUri
            | Self::InlineDocumentTooLarge
            | Self::Pages(_)
            | Self::Features
            | Self::OutputContentFormat
            | Self::DotModel
            | Self::InvalidProvider(_)
            | Self::InvalidModel { .. }
            | Self::Params(_)
            | Self::Headers(_) => true,
            Self::Auth(error) => error.is_request(),
            Self::Http(error) | Self::DocumentDownload(error) => error.is_request(),
            Self::Provider { .. }
            | Self::FileRead { .. }
            | Self::DocumentTask(_)
            | Self::TooLarge { .. }
            | Self::ResponseField { .. }
            | Self::EmptyContent
            | Self::OperationStatus(_)
            | Self::NumericRange(_)
            | Self::PollLocation
            | Self::InvalidPollUrl { .. }
            | Self::PollOrigin
            | Self::PollTimeout
            | Self::MissingCredential { .. }
            | Self::Secret(_)
            | Self::Host(_)
            | Self::HostCall => false,
        }
    }

    pub fn is_response(&self) -> bool {
        match self {
            Self::TooLarge { .. }
            | Self::ResponseField { .. }
            | Self::EmptyContent
            | Self::OperationStatus(_)
            | Self::NumericRange(_)
            | Self::PollLocation
            | Self::InvalidPollUrl { .. }
            | Self::PollOrigin
            | Self::PollTimeout => true,
            Self::Http(error) | Self::DocumentDownload(error) => error.is_response(),
            Self::Provider { .. }
            | Self::EmptyFile
            | Self::FileRead { .. }
            | Self::DocumentTask(_)
            | Self::InvalidMimeType(_)
            | Self::UnsupportedDocument { .. }
            | Self::RejectedDocument { .. }
            | Self::RequestFormat
            | Self::RequestField { .. }
            | Self::Arguments(_)
            | Self::MissingField(_)
            | Self::MissingDocumentUrl
            | Self::InvalidDataUri
            | Self::InlineDocumentTooLarge
            | Self::Pages(_)
            | Self::Features
            | Self::OutputContentFormat
            | Self::DotModel
            | Self::InvalidProvider(_)
            | Self::InvalidModel { .. }
            | Self::MissingCredential { .. }
            | Self::Secret(_)
            | Self::Host(_)
            | Self::HostCall
            | Self::Auth(_)
            | Self::Params(_)
            | Self::Headers(_) => false,
        }
    }
}
