use litellm_auth_aws::AwsAuthConfig;

use crate::source::Source;
use crate::{Error, Fetch, FetchRequest, FileFetch, HttpFetch, Reply, S3Fetch};

const DEFAULT_S3_REGION: &str = "us-east-1";

/// The `Fetch` a catalog location names
#[derive(Clone, Debug)]
pub enum SourceFetch {
    Http(HttpFetch),
    File(FileFetch),
    S3(Box<S3Fetch>),
}

/// What a host supplies for whichever source a location names
pub struct SourceClients<'a> {
    pub http: litellm_http::Client,
    pub env: &'a dyn Fn(&str) -> Option<String>,
}

impl SourceFetch {
    /// Fails for a location no source can serve, which main reports as a final failure
    pub fn new(location: &str, clients: SourceClients<'_>) -> Result<Self, Error> {
        let owned = location.to_owned();
        Ok(match Source::parse(location)? {
            Source::Http(_) => Self::Http(HttpFetch::new(clients.http, owned)),
            Source::File(path) => Self::File(FileFetch::new(owned, path)),
            Source::S3(object) => {
                let region = s3_region(clients.env);
                let auth = AwsAuthConfig {
                    region_name: Some(region.clone()),
                    ..AwsAuthConfig::default()
                };
                let config = litellm_cache_s3::s3_config(
                    region,
                    s3_endpoint(clients.env),
                    clients.http,
                    auth,
                );
                Self::S3(Box::new(S3Fetch::new(config, owned, object)))
            }
        })
    }
}

/// `AWS_REGION_NAME` as main reads it, then the AWS SDK's own variables
fn s3_region(env: &dyn Fn(&str) -> Option<String>) -> String {
    ["AWS_REGION_NAME", "AWS_REGION", "AWS_DEFAULT_REGION"]
        .into_iter()
        .find_map(|name| env(name).filter(|region| !region.is_empty()))
        .unwrap_or_else(|| DEFAULT_S3_REGION.to_owned())
}

/// `AWS_ENDPOINT_URL_S3` then `AWS_ENDPOINT_URL`, which boto3 also reads, for MinIO and other S3 compatible stores
fn s3_endpoint(env: &dyn Fn(&str) -> Option<String>) -> Option<String> {
    ["AWS_ENDPOINT_URL_S3", "AWS_ENDPOINT_URL"]
        .into_iter()
        .find_map(|name| env(name).filter(|endpoint| !endpoint.is_empty()))
}

impl Fetch for SourceFetch {
    fn location(&self) -> &str {
        match self {
            Self::Http(fetch) => fetch.location(),
            Self::File(fetch) => fetch.location(),
            Self::S3(fetch) => fetch.location(),
        }
    }

    async fn fetch(&self, request: &FetchRequest) -> Reply {
        match self {
            Self::Http(fetch) => fetch.fetch(request).await,
            Self::File(fetch) => fetch.fetch(request).await,
            Self::S3(fetch) => fetch.fetch(request).await,
        }
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    #[rstest]
    #[case::region_name(&[("AWS_REGION_NAME", "eu-west-1"), ("AWS_REGION", "us-west-2")], "eu-west-1")]
    #[case::sdk_region(&[("AWS_REGION_NAME", ""), ("AWS_REGION", "us-west-2")], "us-west-2")]
    #[case::sdk_default_region(&[("AWS_DEFAULT_REGION", "ap-south-1")], "ap-south-1")]
    #[case::none(&[], DEFAULT_S3_REGION)]
    fn s3_region_prefers_main_s_variable(#[case] env: &[(&str, &str)], #[case] expected: &str) {
        let lookup = |name: &str| {
            env.iter()
                .find(|(key, _)| *key == name)
                .map(|(_, value)| (*value).to_owned())
        };
        assert_eq!(s3_region(&lookup), expected);
    }
}
