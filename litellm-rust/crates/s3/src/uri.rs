use url::Url;

use crate::Error;

/// The object an `s3://bucket/key` URL names
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct S3Uri {
    pub bucket: String,
    pub key: String,
}

impl TryFrom<&Url> for S3Uri {
    type Error = Error;

    fn try_from(url: &Url) -> Result<Self, Error> {
        let bucket = url.host_str().unwrap_or_default();
        let key = url.path().trim_start_matches('/');
        if url.scheme() != "s3" || bucket.is_empty() || key.is_empty() {
            return Err(Error::InvalidUri);
        }
        Ok(Self {
            bucket: bucket.to_owned(),
            key: key.to_owned(),
        })
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    #[rstest]
    #[case::key("s3://bucket/cost.json", "bucket", "cost.json")]
    #[case::nested_key("s3://bucket/prices/cost.json", "bucket", "prices/cost.json")]
    #[case::dotted_bucket("s3://my.bucket/cost.json", "my.bucket", "cost.json")]
    fn an_s3_url_names_its_bucket_and_key(
        #[case] url: &str,
        #[case] bucket: &str,
        #[case] key: &str,
    ) {
        assert_eq!(
            S3Uri::try_from(&Url::parse(url).unwrap()).unwrap(),
            S3Uri {
                bucket: bucket.into(),
                key: key.into(),
            }
        );
    }

    #[rstest]
    #[case::no_key("s3://bucket/")]
    #[case::no_path("s3://bucket")]
    #[case::other_scheme("https://bucket/cost.json")]
    fn a_url_without_a_bucket_and_key_is_refused(#[case] url: &str) {
        assert!(matches!(
            S3Uri::try_from(&Url::parse(url).unwrap()),
            Err(Error::InvalidUri)
        ));
    }
}
