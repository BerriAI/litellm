use std::fmt;
use std::path::PathBuf;

use url::Url;

use crate::error::{Error, Repr};

/// Where a fetched catalog comes from
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Source {
    Http(Url),
    File(PathBuf),
    S3(S3Object),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct S3Object {
    pub bucket: String,
    pub key: String,
}

impl Source {
    pub fn parse(text: &str) -> Result<Self, Error> {
        let url = Url::parse(text).map_err(|source| Repr::InvalidUrl {
            text: text.to_owned(),
            source,
        })?;
        match url.scheme() {
            "http" | "https" => Ok(Self::Http(url)),
            "file" => url.to_file_path().map(Self::File).map_err(|()| {
                Repr::NotAFilePath {
                    text: text.to_owned(),
                }
                .into()
            }),
            "s3" => {
                let bucket = url.host_str().unwrap_or_default();
                let key = url.path().trim_start_matches('/');
                if bucket.is_empty() || key.is_empty() {
                    return Err(Repr::InvalidS3Url {
                        text: text.to_owned(),
                    }
                    .into());
                }
                Ok(Self::S3(S3Object {
                    bucket: bucket.to_owned(),
                    key: key.to_owned(),
                }))
            }
            scheme => Err(Repr::UnsupportedScheme {
                scheme: scheme.to_owned(),
            }
            .into()),
        }
    }
}

impl fmt::Display for Source {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Http(url) => write!(formatter, "{url}"),
            Self::File(path) => write!(formatter, "file://{}", path.display()),
            Self::S3(object) => write!(formatter, "s3://{}/{}", object.bucket, object.key),
        }
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    #[rstest]
    #[case::https("https://example.com/map.json", Source::Http(Url::parse("https://example.com/map.json").unwrap()))]
    #[case::file("file:///tmp/map.json", Source::File(PathBuf::from("/tmp/map.json")))]
    #[case::s3(
        "s3://bucket/prices/map.json",
        Source::S3(S3Object { bucket: "bucket".into(), key: "prices/map.json".into() })
    )]
    fn parses_each_supported_scheme(#[case] text: &str, #[case] expected: Source) {
        assert_eq!(Source::parse(text).expect("a supported URL"), expected);
    }

    #[rstest]
    #[case::no_key("s3://bucket", "s3://bucket/key")]
    #[case::no_bucket("s3:///key", "s3://bucket/key")]
    #[case::not_a_url("http://[bad/model_prices.json", "is not a URL")]
    #[case::unknown_scheme("ftp://example.com/map.json", "not ftp")]
    fn an_unusable_url_names_what_is_wrong_and_is_not_retried(
        #[case] text: &str,
        #[case] says: &str,
    ) {
        let error = Source::parse(text).expect_err("an unusable URL");
        let report = litellm_error::Report::new(&error).to_string();
        assert!(report.contains(says), "{report}");
        assert!(!error.transient());
    }
}
