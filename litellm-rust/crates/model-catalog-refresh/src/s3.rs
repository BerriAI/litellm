use litellm_cache_s3::aws_sdk_s3;

use crate::source::S3Object;
use crate::{Error, Fetch, FetchRequest, Reply};

/// A catalog in an S3 object. Every failure is final, as main's boto3 fetch treats them
#[derive(Clone, Debug)]
pub struct S3Fetch {
    client: aws_sdk_s3::Client,
    location: String,
    object: S3Object,
}

impl S3Fetch {
    /// `config` is built by `litellm_cache_s3::s3_config`, and retries are left to the refresh
    pub fn new(config: aws_sdk_s3::config::Builder, location: String, object: S3Object) -> Self {
        let config = config.retry_config(aws_sdk_s3::config::retry::RetryConfig::disabled());
        Self {
            client: aws_sdk_s3::Client::from_conf(config.build()),
            location,
            object,
        }
    }
}

impl Fetch for S3Fetch {
    fn location(&self) -> &str {
        &self.location
    }

    async fn fetch(&self, request: &FetchRequest) -> Reply {
        let get = async {
            let output = self
                .client
                .get_object()
                .bucket(&self.object.bucket)
                .key(&self.object.key)
                .send()
                .await
                .map_err(Error::s3)?;
            let etag = output.e_tag().map(str::to_owned);
            let body = output.body.collect().await.map_err(Error::s3)?;
            Ok::<_, Error>((etag, body.into_bytes().to_vec()))
        };
        match tokio::time::timeout(request.timeout, get).await {
            Ok(Ok((etag, body))) => Reply::Response {
                status: 200,
                etag,
                retry_after: None,
                body,
            },
            Ok(Err(error)) => Reply::Refused { error },
            Err(_) => Reply::Refused {
                error: Error::timed_out(),
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use litellm_auth_aws::AwsAuthConfig;
    use rstest::rstest;
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    use super::*;

    async fn fetch_from(status: u16) -> (Reply, Vec<wiremock::Request>) {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/bucket/prices/map.json"))
            .respond_with(
                ResponseTemplate::new(status)
                    .insert_header("etag", "\"s3-etag\"")
                    .set_body_string("{}"),
            )
            .mount(&server)
            .await;
        let config = litellm_cache_s3::s3_config(
            "us-east-1".to_owned(),
            Some(server.uri()),
            litellm_http::Client::for_test(),
            AwsAuthConfig {
                access_key_id: Some("access".to_owned()),
                secret_access_key: Some("secret".to_owned()),
                ..AwsAuthConfig::default()
            },
        );
        let fetch = S3Fetch::new(
            config,
            "s3://bucket/prices/map.json".to_owned(),
            S3Object {
                bucket: "bucket".to_owned(),
                key: "prices/map.json".to_owned(),
            },
        );
        let request = FetchRequest {
            if_none_match: None,
            timeout: Duration::from_secs(5),
        };
        let reply = fetch.fetch(&request).await;
        (reply, server.received_requests().await.unwrap_or_default())
    }

    #[rstest]
    #[tokio::test]
    async fn reads_the_object_and_its_etag() {
        let (reply, requests) = fetch_from(200).await;
        let Reply::Response {
            status, etag, body, ..
        } = reply
        else {
            panic!("an object")
        };
        assert_eq!(
            (status, etag.as_deref(), body.as_slice()),
            (200, Some("\"s3-etag\""), b"{}".as_slice())
        );
        assert_eq!(requests.len(), 1);
    }

    #[rstest]
    #[tokio::test]
    async fn a_server_error_is_final_and_not_retried_by_the_sdk() {
        let (reply, requests) = fetch_from(503).await;
        assert!(matches!(reply, Reply::Refused { .. }), "{reply:?}");
        assert_eq!(requests.len(), 1);
    }
}
