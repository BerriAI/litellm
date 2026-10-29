use std::io;

use thiserror::Error;

#[derive(Debug, Error)]
#[error(transparent)]
pub struct Error(Box<Repr>);

#[derive(Debug, Error)]
enum Repr {
    #[error(transparent)]
    Http(#[from] litellm_http::Error),
    #[error("could not read the catalog")]
    Read(#[source] io::Error),
    #[error("invalid catalog URL")]
    InvalidUrl(#[source] url::ParseError),
    #[error("unsupported catalog URL scheme {scheme}")]
    UnsupportedScheme { scheme: String },
}

impl Error {
    pub fn unsupported_scheme(scheme: String) -> Self {
        Self(Box::new(Repr::UnsupportedScheme { scheme }))
    }
}

impl From<litellm_http::Error> for Error {
    fn from(cause: litellm_http::Error) -> Self {
        Self(Box::new(Repr::Http(cause)))
    }
}

impl From<io::Error> for Error {
    fn from(cause: io::Error) -> Self {
        Self(Box::new(Repr::Read(cause)))
    }
}

impl From<url::ParseError> for Error {
    fn from(cause: url::ParseError) -> Self {
        Self(Box::new(Repr::InvalidUrl(cause)))
    }
}

#[cfg(test)]
mod tests {
    use litellm_error::{Report, find};
    use rstest::rstest;

    use super::*;
    use crate::{DoneReason, Reply, Step};

    #[rstest]
    fn exhausted_transport_keeps_its_typed_cause() {
        let reply = Reply::Transport {
            error: io::Error::from(io::ErrorKind::ConnectionReset).into(),
        };
        let Reply::Transport { error } = reply else {
            panic!("transport failure")
        };
        let step = Step::Done {
            reason: DoneReason::Exhausted(error),
        };
        let Step::Done {
            reason: DoneReason::Exhausted(error),
        } = step
        else {
            panic!("exhausted attempts")
        };
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
