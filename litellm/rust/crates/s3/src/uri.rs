use crate::Error;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct S3Uri {
    pub bucket: String,
    pub key: String,
}

/// Splits the way Python's `urllib.parse.urlsplit` does, so the key keeps dot segments and characters a URL parser
/// would normalize or percent-encode. S3 keys are opaque strings
impl TryFrom<&str> for S3Uri {
    type Error = Error;

    fn try_from(uri: &str) -> Result<Self, Error> {
        let rest = match uri.split_once("://") {
            Some((scheme, rest)) if scheme.eq_ignore_ascii_case("s3") => rest,
            _ => return Err(Error::InvalidUri),
        };
        let (bucket, path) = rest.split_at(rest.find(['/', '?', '#']).unwrap_or(rest.len()));
        let key = path
            .split(['?', '#'])
            .next()
            .unwrap_or_default()
            .trim_start_matches('/');
        if bucket.is_empty() || key.is_empty() {
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
    #[case::dot_segments("s3://bucket/archives/../cost.json", "bucket", "archives/../cost.json")]
    #[case::current_dir_segment("s3://bucket/./cost.json", "bucket", "./cost.json")]
    #[case::space_kept_raw("s3://bucket/my prices.json", "bucket", "my prices.json")]
    #[case::query_dropped("s3://bucket/cost.json?versionId=1", "bucket", "cost.json")]
    #[case::fragment_dropped("s3://bucket/cost.json#top", "bucket", "cost.json")]
    #[case::leading_slashes_trimmed("s3://bucket//cost.json", "bucket", "cost.json")]
    #[case::uppercase_scheme("S3://bucket/cost.json", "bucket", "cost.json")]
    fn an_s3_url_names_its_bucket_and_key(
        #[case] url: &str,
        #[case] bucket: &str,
        #[case] key: &str,
    ) {
        assert_eq!(
            S3Uri::try_from(url).unwrap(),
            S3Uri {
                bucket: bucket.into(),
                key: key.into(),
            }
        );
    }

    #[rstest]
    #[case::no_key("s3://bucket/")]
    #[case::no_path("s3://bucket")]
    #[case::query_only("s3://bucket?cost.json")]
    #[case::no_bucket("s3:///cost.json")]
    #[case::other_scheme("https://bucket/cost.json")]
    #[case::no_scheme("bucket/cost.json")]
    fn a_url_without_a_bucket_and_key_is_refused(#[case] url: &str) {
        assert!(matches!(S3Uri::try_from(url), Err(Error::InvalidUri)));
    }
}
