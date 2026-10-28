use std::fmt;
use std::path::PathBuf;

use url::Url;

use crate::error::SourceError;

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
    pub fn parse(text: &str) -> Result<Self, SourceError> {
        let url = Url::parse(text).map_err(|error| SourceError::Invalid {
            reason: format!("invalid URL {text}: {error}"),
        })?;
        match url.scheme() {
            "http" | "https" => Ok(Self::Http(url)),
            "file" => url
                .to_file_path()
                .map(Self::File)
                .map_err(|()| SourceError::Invalid {
                    reason: format!("invalid file URL {text}"),
                }),
            "s3" => {
                let bucket = url.host_str().unwrap_or_default();
                let key = url.path().trim_start_matches('/');
                if bucket.is_empty() || key.is_empty() {
                    return Err(SourceError::Invalid {
                        reason: format!("invalid S3 URL {text}: expected s3://bucket/key"),
                    });
                }
                Ok(Self::S3(S3Object {
                    bucket: bucket.to_owned(),
                    key: key.to_owned(),
                }))
            }
            scheme => Err(SourceError::UnsupportedScheme {
                scheme: scheme.to_owned(),
            }),
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
        assert_eq!(Source::parse(text), Ok(expected));
    }

    #[rstest]
    #[case::no_key("s3://bucket")]
    #[case::no_bucket("s3:///key")]
    #[case::not_a_url("http://[bad/model_prices.json")]
    fn rejects_an_unusable_url(#[case] text: &str) {
        assert!(matches!(
            Source::parse(text),
            Err(SourceError::Invalid { .. })
        ));
    }

    #[rstest]
    fn rejects_an_unknown_scheme() {
        assert_eq!(
            Source::parse("ftp://example.com/map.json"),
            Err(SourceError::UnsupportedScheme {
                scheme: "ftp".into()
            })
        );
    }
}
