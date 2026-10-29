use aws_config::BehaviorVersion;
use aws_sdk_s3::config::SharedCredentialsProvider;
use aws_sdk_s3::config::retry::RetryConfig;
use aws_sdk_s3::error::SdkError;
use aws_smithy_runtime_api::http::Response;
use litellm_s3::{ReqwestHttpClient, S3ClientConfig};

use crate::error::Repr;
use crate::source::{S3Object, Source};
use crate::step::{FetchRequest, Reply};

const BUCKET_REGION_HEADER: &str = "x-amz-bucket-region";
const DEFAULT_REGION: &str = "us-east-1";

/// One S3 object as a catalog source. SDK retries are off, so `RetryPolicy` owns every attempt
#[derive(Clone, Debug)]
pub struct S3Source {
    object: S3Object,
    http: litellm_http::Client,
    credentials: SharedCredentialsProvider,
    endpoint: Option<String>,
    client: aws_sdk_s3::Client,
}

impl S3Source {
    pub fn new(
        object: S3Object,
        http: litellm_http::Client,
        credentials: SharedCredentialsProvider,
        config: S3ClientConfig,
    ) -> Self {
        let endpoint = config.endpoint.clone();
        let client = without_retries(litellm_s3::client(
            config,
            http.clone(),
            credentials.clone(),
        ));
        Self {
            object,
            http,
            credentials,
            endpoint,
            client,
        }
    }

    /// The chain `boto3.client("s3")` used on main. Region from `AWS_REGION_NAME` first, then the SDK chain
    /// (`AWS_REGION`, `AWS_DEFAULT_REGION`, the profile). `AWS_ENDPOINT_URL_S3` or `AWS_ENDPOINT_URL` for
    /// S3-compatible stores
    pub async fn from_environment(object: S3Object, http: litellm_http::Client) -> Self {
        let shared = aws_config::defaults(BehaviorVersion::latest())
            .http_client(ReqwestHttpClient(http.clone()))
            .load()
            .await;
        let region = non_empty_env("AWS_REGION_NAME")
            .or_else(|| shared.region().map(ToString::to_string))
            .unwrap_or_else(|| DEFAULT_REGION.to_owned());
        let endpoint =
            non_empty_env("AWS_ENDPOINT_URL_S3").or_else(|| non_empty_env("AWS_ENDPOINT_URL"));
        let credentials = shared.credentials_provider().unwrap_or_else(|| {
            SharedCredentialsProvider::new(aws_sdk_s3::config::Credentials::new(
                "",
                "",
                None,
                None,
                "no AWS credentials found",
            ))
        });
        Self::new(
            object,
            http,
            credentials,
            S3ClientConfig { region, endpoint },
        )
    }

    pub async fn fetch(&self, request: FetchRequest) -> Reply {
        match self.attempt(&self.client, &request).await {
            Redirect::To(region) => {
                let client = without_retries(litellm_s3::client(
                    S3ClientConfig {
                        region,
                        endpoint: self.endpoint.clone(),
                    },
                    self.http.clone(),
                    self.credentials.clone(),
                ));
                match self.attempt(&client, &request).await {
                    Redirect::To(region) => Reply::Failed(
                        Repr::S3Redirect {
                            from: self.source(),
                            region,
                        }
                        .into(),
                    ),
                    Redirect::No(reply) => reply,
                }
            }
            Redirect::No(reply) => reply,
        }
    }

    async fn attempt(&self, client: &aws_sdk_s3::Client, request: &FetchRequest) -> Redirect {
        let call = client
            .get_object()
            .bucket(&self.object.bucket)
            .key(&self.object.key)
            .set_if_none_match(request.if_none_match.clone())
            .send();
        let output = match tokio::time::timeout(request.timeout, call).await {
            Err(_) => {
                return Redirect::No(Reply::Failed(
                    Repr::S3Timeout {
                        from: self.source(),
                    }
                    .into(),
                ));
            }
            Ok(Err(SdkError::ServiceError(error))) => return rejected(error.raw()),
            Ok(Err(SdkError::ResponseError(error))) => return rejected(error.raw()),
            Ok(Err(error)) => {
                return Redirect::No(Reply::Failed(
                    Repr::S3 {
                        from: self.source(),
                        source: litellm_error::Source::new(error),
                    }
                    .into(),
                ));
            }
            Ok(Ok(output)) => output,
        };
        let etag = output.e_tag().map(str::to_owned);
        match output.body.collect().await {
            Ok(body) => Redirect::No(Reply::Response {
                status: 200,
                etag,
                retry_after: None,
                body: Some(body.into_bytes().to_vec()),
            }),
            Err(error) => Redirect::No(Reply::Failed(
                Repr::S3Body {
                    from: self.source(),
                    source: litellm_error::Source::new(error),
                }
                .into(),
            )),
        }
    }

    fn source(&self) -> Source {
        Source::S3(self.object.clone())
    }
}

enum Redirect {
    /// The bucket lives in another region, repeated once as boto3 does
    To(String),
    No(Reply),
}

fn rejected(raw: &Response) -> Redirect {
    let status = raw.status().as_u16();
    let header = |name: &str| raw.headers().get(name).map(str::to_owned);
    match (status, header(BUCKET_REGION_HEADER)) {
        (301, Some(region)) => Redirect::To(region),
        _ => Redirect::No(Reply::Response {
            status,
            etag: header("etag"),
            retry_after: None,
            body: None,
        }),
    }
}

fn without_retries(client: aws_sdk_s3::Client) -> aws_sdk_s3::Client {
    aws_sdk_s3::Client::from_conf(
        client
            .config()
            .to_builder()
            .retry_config(RetryConfig::disabled())
            .build(),
    )
}

fn non_empty_env(name: &str) -> Option<String> {
    std::env::var(name).ok().filter(|value| !value.is_empty())
}
