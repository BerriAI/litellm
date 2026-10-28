use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use litellm_model_catalog::Error as CatalogError;

use crate::source::Source;

const TRANSIENT_STATUSES: [u16; 5] = [429, 500, 502, 503, 504];

#[derive(Clone, Debug, thiserror::Error)]
#[error(transparent)]
pub struct Error(Arc<Repr>);

#[derive(Debug, thiserror::Error)]
pub(crate) enum Repr {
    #[error("{text:?} is not a URL")]
    InvalidUrl {
        text: String,
        source: url::ParseError,
    },
    #[error("{text:?} is not a local file URL")]
    NotAFilePath { text: String },
    #[error("{text:?} is not an s3://bucket/key URL")]
    InvalidS3Url { text: String },
    #[error("model cost map URLs are http, https, file or s3, not {scheme}")]
    UnsupportedScheme { scheme: String },
    #[error("fetching the model cost map failed")]
    Http { source: litellm_http::Error },
    #[error("reading {} failed", path.display())]
    File {
        path: PathBuf,
        source: std::io::Error,
    },
    #[error("fetching {from} failed")]
    S3 {
        from: Source,
        source: litellm_error::Source,
    },
    #[error("fetching {from} timed out")]
    S3Timeout { from: Source },
    #[error("{from} redirected twice, last to {region}")]
    S3Redirect { from: Source, region: String },
    #[error("reading {from} failed")]
    S3Body {
        from: Source,
        source: litellm_error::Source,
    },
    #[error("the model cost map source returned HTTP {status}")]
    Status {
        status: u16,
        retry_after: Option<Duration>,
    },
    #[error("the model cost map source is not modified")]
    NotModified,
    #[error("the model cost map is unusable")]
    Unusable(#[source] CatalogError),
    #[error("gave up after {attempts} attempts")]
    Exhausted { attempts: u32, source: Error },
    #[error("revalidation is not awaiting {0}")]
    OutOfOrder(Awaiting),
}

#[derive(Clone, Copy, Debug, thiserror::Error)]
pub(crate) enum Awaiting {
    #[error("a reply")]
    Reply,
    #[error("a sleep")]
    Sleep,
}

impl From<Repr> for Error {
    fn from(repr: Repr) -> Self {
        Self(Arc::new(repr))
    }
}

impl Error {
    pub fn transient(&self) -> bool {
        match &*self.0 {
            Repr::Http { .. }
            | Repr::File { .. }
            | Repr::S3 { .. }
            | Repr::S3Timeout { .. }
            | Repr::S3Redirect { .. }
            | Repr::S3Body { .. } => true,
            Repr::Status { status, .. } => TRANSIENT_STATUSES.contains(status),
            Repr::InvalidUrl { .. }
            | Repr::NotAFilePath { .. }
            | Repr::InvalidS3Url { .. }
            | Repr::UnsupportedScheme { .. }
            | Repr::NotModified { .. }
            | Repr::Unusable { .. }
            | Repr::Exhausted { .. }
            | Repr::OutOfOrder(_) => false,
        }
    }

    pub fn retry_after(&self) -> Option<Duration> {
        match &*self.0 {
            Repr::Status { retry_after, .. } => *retry_after,
            Repr::InvalidUrl { .. }
            | Repr::NotAFilePath { .. }
            | Repr::InvalidS3Url { .. }
            | Repr::UnsupportedScheme { .. }
            | Repr::Http { .. }
            | Repr::File { .. }
            | Repr::S3 { .. }
            | Repr::S3Timeout { .. }
            | Repr::S3Redirect { .. }
            | Repr::S3Body { .. }
            | Repr::NotModified { .. }
            | Repr::Unusable { .. }
            | Repr::Exhausted { .. }
            | Repr::OutOfOrder(_) => None,
        }
    }

    pub fn status(&self) -> Option<u16> {
        match &*self.0 {
            Repr::Status { status, .. } => Some(*status),
            Repr::Exhausted { source, .. } => source.status(),
            Repr::InvalidUrl { .. }
            | Repr::NotAFilePath { .. }
            | Repr::InvalidS3Url { .. }
            | Repr::UnsupportedScheme { .. }
            | Repr::Http { .. }
            | Repr::File { .. }
            | Repr::S3 { .. }
            | Repr::S3Timeout { .. }
            | Repr::S3Redirect { .. }
            | Repr::S3Body { .. }
            | Repr::NotModified { .. }
            | Repr::Unusable { .. }
            | Repr::OutOfOrder(_) => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::io;

    use litellm_error::{Report, find};
    use rstest::rstest;

    use super::*;
    use crate::{Installed, Origin, Revalidation, Reply, RetryPolicy, Step};

    fn finish_failed_reply(error: Error) -> Error {
        let source = Source::parse("https://example.com/map.json").expect("valid source");
        let installed = Installed {
            origin: Origin::Bundled,
            etag: None,
            revision: String::new(),
            model_count: 0,
        };
        let (mut revalidation, _) = Revalidation::new(
            &source,
            &installed,
            RetryPolicy {
                max_attempts: 1,
                ..RetryPolicy::default()
            },
            Some(0),
        );
        match revalidation.on_reply(Reply::Failed(error)) {
            Step::Done { error, .. } => error,
            _ => panic!("one failed attempt is done"),
        }
    }

    #[rstest]
    fn failed_file_reply_keeps_its_typed_cause() {
        let error = finish_failed_reply(
            Repr::File {
                path: PathBuf::from("catalog.json"),
                source: io::Error::from(io::ErrorKind::ConnectionReset),
            }
            .into(),
        );
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
        let error = Source::parse("http://[invalid").expect_err("invalid source");
        assert!(find::<url::ParseError>(&error).is_some());
    }

    #[rstest]
    #[tokio::test]
    async fn failed_http_reply_keeps_its_typed_cause_without_the_url() {
        let cause = litellm_http::Client::for_test()
            .get("http://localhost:invalid/private?api_key=secret")
            .send()
            .await
            .unwrap_err();
        let error = finish_failed_reply(
            Repr::Http {
                source: litellm_http::Error::from(cause),
            }
            .into(),
        );
        let cause = find::<reqwest::Error>(&error).expect("typed HTTP cause");
        let report = Report::new(&error).to_string();
        assert_eq!(report.matches(&cause.to_string()).count(), 1);
        assert!(!report.contains("secret"), "{report}");
        assert!(!report.contains("private"), "{report}");
        let debug = format!("{error:?}");
        assert!(!debug.contains("secret"), "{debug}");
        assert!(!debug.contains("private"), "{debug}");
    }
}
