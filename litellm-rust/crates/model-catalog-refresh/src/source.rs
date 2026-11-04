use std::path::PathBuf;

use url::Url;

use crate::Error;

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
    /// Main splits an `s3` URL the way `urllib.parse.urlsplit` does, the bucket is the netloc and the key is the path
    /// without its leading slashes
    pub fn parse(location: &str) -> Result<Self, Error> {
        let scheme = location
            .split_once(':')
            .map(|(scheme, _)| scheme.to_ascii_lowercase());
        match scheme.as_deref() {
            Some("s3") => s3_object(location).map(Source::S3),
            Some("file") => Url::parse(location)?
                .to_file_path()
                .map(Source::File)
                .map_err(|()| Error::unsupported_scheme("file".to_owned())),
            Some("http" | "https") => Ok(Source::Http(Url::parse(location)?)),
            other => Err(Error::unsupported_scheme(
                other.unwrap_or_default().to_owned(),
            )),
        }
    }
}

fn s3_object(location: &str) -> Result<S3Object, Error> {
    let after_scheme = location.split_once(':').map_or("", |(_, rest)| rest);
    let Some(rest) = after_scheme.strip_prefix("//") else {
        return Err(Error::invalid_s3_url());
    };
    let rest = rest.split(['?', '#']).next().unwrap_or_default();
    let (bucket, path) = rest.split_once('/').unwrap_or((rest, ""));
    let key = path.trim_start_matches('/');
    if bucket.is_empty() || key.is_empty() {
        return Err(Error::invalid_s3_url());
    }
    Ok(S3Object {
        bucket: bucket.to_owned(),
        key: key.to_owned(),
    })
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    fn s3(bucket: &str, key: &str) -> Option<Source> {
        Some(Source::S3(S3Object {
            bucket: bucket.to_owned(),
            key: key.to_owned(),
        }))
    }

    #[rstest]
    #[case::s3("s3://bucket/path/map.json", s3("bucket", "path/map.json"))]
    #[case::s3_upper_scheme("S3://bucket/map.json", s3("bucket", "map.json"))]
    #[case::s3_leading_slashes("s3://bucket//map.json", s3("bucket", "map.json"))]
    #[case::s3_query_dropped("s3://bucket/map.json?versionId=1#x", s3("bucket", "map.json"))]
    #[case::s3_no_key("s3://bucket/", None)]
    #[case::s3_no_bucket("s3:///map.json", None)]
    #[case::s3_no_slashes("s3:bucket/map.json", None)]
    #[case::empty("", None)]
    #[case::ftp("ftp://host/map.json", None)]
    #[case::relative("map.json", None)]
    fn sources_split_like_urlsplit(#[case] location: &str, #[case] expected: Option<Source>) {
        assert_eq!(Source::parse(location).ok(), expected);
    }

    #[rstest]
    fn file_urls_are_decoded_paths() {
        assert_eq!(
            Source::parse("file:///tmp/a%20b.json").ok(),
            Some(Source::File(PathBuf::from("/tmp/a b.json")))
        );
    }

    #[rstest]
    fn http_urls_keep_their_query() {
        let Ok(Source::Http(url)) = Source::parse("https://example.com/map.json?ref=main") else {
            panic!("an HTTP source")
        };
        assert_eq!(url.as_str(), "https://example.com/map.json?ref=main");
    }
}
