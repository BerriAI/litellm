use litellm_s3::S3Uri;
use url::Url;

use crate::Error;

/// Where a fetched catalog comes from
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Source {
    Http(Url),
    S3(S3Uri),
}

impl Source {
    pub fn parse(url: &str) -> Result<Self, Error> {
        let parsed = Url::parse(url)?;
        match parsed.scheme() {
            "http" | "https" => Ok(Self::Http(parsed)),
            "s3" => Ok(Self::S3(S3Uri::try_from(url)?)),
            scheme => Err(Error::unsupported_scheme(scheme.to_owned())),
        }
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    #[rstest]
    #[case::https("https://example.com/cost.json")]
    #[case::http("http://127.0.0.1:8000/cost.json")]
    fn http_urls_are_fetched_over_http(#[case] url: &str) {
        assert_eq!(
            Source::parse(url).unwrap(),
            Source::Http(Url::parse(url).unwrap())
        );
    }

    #[rstest]
    fn s3_urls_name_a_bucket_and_key() {
        assert_eq!(
            Source::parse("s3://bucket/prices/cost.json").unwrap(),
            Source::S3(S3Uri {
                bucket: "bucket".into(),
                key: "prices/cost.json".into(),
            })
        );
    }

    #[rstest]
    #[case::s3_without_a_key("s3://bucket/")]
    #[case::unsupported("ftp://example.com/cost.json")]
    #[case::file("file:///tmp/cost.json")]
    #[case::not_a_url("not a url")]
    fn other_urls_are_refused(#[case] url: &str) {
        assert!(Source::parse(url).is_err());
    }
}
