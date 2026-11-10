use litellm_s3::sdk::config::http::HttpResponse;
use litellm_s3::sdk::error::SdkError;
use litellm_s3::sdk::operation::get_object::GetObjectError;
use litellm_s3::sdk::primitives::ByteStreamError;
use thiserror::Error;
use tokio::time::error::Elapsed;

#[derive(Debug, Error)]
#[error(transparent)]
pub struct Error(Box<Repr>);

#[derive(Debug, Error)]
enum Repr {
    #[error(transparent)]
    Http(#[from] litellm_http::Error),
    #[error("invalid catalog URL")]
    InvalidUrl(#[source] url::ParseError),
    #[error("unsupported catalog URL scheme {scheme}")]
    UnsupportedScheme { scheme: String },
    #[error("invalid S3 URL")]
    S3Uri(#[source] litellm_s3::Error),
    #[error("S3 GetObject failed")]
    S3GetObject(#[source] Box<SdkError<GetObjectError, HttpResponse>>),
    #[error("S3 GetObject body read failed")]
    S3Body(#[source] ByteStreamError),
    #[error("S3 GetObject timed out")]
    S3Timeout(#[source] Elapsed),
    #[error("response body is over the {limit} byte limit")]
    BodyTooLarge { limit: u64 },
}

impl Error {
    pub fn unsupported_scheme(scheme: String) -> Self {
        Self(Box::new(Repr::UnsupportedScheme { scheme }))
    }

    pub(crate) fn body_too_large(limit: u64) -> Self {
        Self(Box::new(Repr::BodyTooLarge { limit }))
    }
}

impl From<SdkError<GetObjectError, HttpResponse>> for Error {
    fn from(cause: SdkError<GetObjectError, HttpResponse>) -> Self {
        Self(Box::new(Repr::S3GetObject(Box::new(cause))))
    }
}

impl From<ByteStreamError> for Error {
    fn from(cause: ByteStreamError) -> Self {
        Self(Box::new(Repr::S3Body(cause)))
    }
}

impl From<Elapsed> for Error {
    fn from(cause: Elapsed) -> Self {
        Self(Box::new(Repr::S3Timeout(cause)))
    }
}

impl From<litellm_http::Error> for Error {
    fn from(cause: litellm_http::Error) -> Self {
        Self(Box::new(Repr::Http(cause)))
    }
}

impl From<litellm_s3::Error> for Error {
    fn from(cause: litellm_s3::Error) -> Self {
        Self(Box::new(Repr::S3Uri(cause)))
    }
}

impl From<url::ParseError> for Error {
    fn from(cause: url::ParseError) -> Self {
        Self(Box::new(Repr::InvalidUrl(cause)))
    }
}

#[cfg(test)]
mod tests {
    use std::io;

    use litellm_error::{Report, find};
    use rstest::rstest;

    use super::*;

    #[rstest]
    fn an_s3_failure_keeps_its_typed_cause_once() {
        let error = Error::from(SdkError::<GetObjectError, HttpResponse>::timeout_error(
            io::Error::from(io::ErrorKind::ConnectionReset),
        ));
        let cause = find::<io::Error>(&error).expect("typed I/O cause");
        assert_eq!(cause.kind(), io::ErrorKind::ConnectionReset);
        assert_eq!(
            Report::new(&error)
                .to_string()
                .matches(&cause.to_string())
                .count(),
            1
        );
        assert!(!error.to_string().contains(&cause.to_string()));
    }

    #[rstest]
    fn invalid_source_keeps_its_typed_cause() {
        let error = Error::from(url::Url::parse("http://[invalid").unwrap_err());
        assert!(find::<url::ParseError>(&error).is_some());
    }

    #[rstest]
    #[tokio::test]
    async fn http_failures_keep_the_cause_without_the_url() {
        let cause = litellm_http::Client::for_test()
            .get("http://localhost:invalid/private?api_key=secret")
            .send()
            .await
            .unwrap_err();
        let error = Error::from(litellm_http::Error::from(cause));
        assert!(find::<reqwest::Error>(&error).is_some());
        let report = Report::new(&error).to_string();
        assert!(!report.contains("secret"), "{report}");
        assert!(!report.contains("private"), "{report}");
        let debug = format!("{error:?}");
        assert!(!debug.contains("secret"), "{debug}");
        assert!(!debug.contains("private"), "{debug}");
    }
}
