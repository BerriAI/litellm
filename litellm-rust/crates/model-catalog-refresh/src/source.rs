use litellm_s3::S3Uri;
use url::Url;

use crate::Error;

/// Where a fetched catalog comes from
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Source {
    Http(Url),
    S3(S3Uri),
    Gcs(GcsUri),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GcsUri {
    pub bucket: String,
    pub object: String,
}

impl TryFrom<&str> for GcsUri {
    type Error = Error;

    fn try_from(uri: &str) -> Result<Self, Error> {
        let rest = uri.split_once("://").map_or(uri, |(_, rest)| rest);
        let (bucket, path) = rest.split_at(rest.find(['/', '?', '#']).unwrap_or(rest.len()));
        let object = path
            .split(['?', '#'])
            .next()
            .unwrap_or_default()
            .trim_start_matches('/');
        if bucket.is_empty() || object.is_empty() {
            return Err(Error::invalid_gcs_url());
        }
        Ok(Self {
            bucket: bucket.to_owned(),
            object: object.to_owned(),
        })
    }
}

impl Source {
    pub fn parse(url: &str) -> Result<Self, Error> {
        let parsed = Url::parse(url)?;
        match parsed.scheme() {
            "http" | "https" => Ok(Self::Http(parsed)),
            "s3" => Ok(Self::S3(S3Uri::try_from(url)?)),
            "gs" => Ok(Self::Gcs(GcsUri::try_from(url)?)),
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
    #[case::object("gs://bucket/cost.json", "bucket", "cost.json")]
    #[case::nested_object("gs://bucket/prices/cost.json", "bucket", "prices/cost.json")]
    #[case::space_kept_raw("gs://bucket/my prices.json", "bucket", "my prices.json")]
    #[case::query_dropped("gs://bucket/cost.json?generation=1", "bucket", "cost.json")]
    fn gs_urls_name_a_bucket_and_object(
        #[case] url: &str,
        #[case] bucket: &str,
        #[case] object: &str,
    ) {
        assert_eq!(
            Source::parse(url).unwrap(),
            Source::Gcs(GcsUri {
                bucket: bucket.into(),
                object: object.into(),
            })
        );
    }

    #[rstest]
    #[case::s3_without_a_key("s3://bucket/")]
    #[case::gs_without_an_object("gs://bucket/")]
    #[case::gs_without_a_path("gs://bucket")]
    #[case::gs_without_a_bucket("gs:///cost.json")]
    #[case::unsupported("ftp://example.com/cost.json")]
    #[case::file("file:///tmp/cost.json")]
    #[case::not_a_url("not a url")]
    fn other_urls_are_refused(#[case] url: &str) {
        assert!(Source::parse(url).is_err());
    }
}
