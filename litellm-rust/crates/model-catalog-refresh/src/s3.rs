#![allow(
    clippy::disallowed_types,
    reason = "the fork-local client uses HttpClientConfig without the shared pool"
)]

use std::sync::Arc;

use async_trait::async_trait;
use aws_sdk_s3::{
    config::{BehaviorVersion, Region, retry::RetryConfig, timeout::TimeoutConfig},
    error::SdkError,
    operation::get_object::{GetObjectError, GetObjectOutput},
};
use litellm_http::HttpClientConfig;
use litellm_model_catalog::{FetchRequest, Reply, SourceId};
use litellm_s3::{ReqwestHttpClient, S3ClientConfig};
use tokio::sync::OnceCell;

use crate::{Fetch, SourceError, process_local::ProcessLocal};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct S3Object {
    pub bucket: String,
    pub key: String,
    source: String,
}

impl S3Object {
    pub(crate) fn parse(value: &str) -> Result<Self, SourceError> {
        let (bucket, key) = value
            .split_once("://")
            .and_then(|(_, rest)| rest.split(['?', '#']).next())
            .and_then(|rest| rest.split_once('/'))
            .unwrap_or_default();
        let key = key.trim_start_matches('/');
        if bucket.is_empty() || key.is_empty() {
            return Err(SourceError::Invalid {
                reason: "expected s3://bucket/key".to_owned(),
            });
        }
        Ok(Self {
            bucket: bucket.to_owned(),
            key: key.to_owned(),
            source: value.to_owned(),
        })
    }

    pub(crate) fn id(&self) -> SourceId {
        SourceId(self.source.clone())
    }
}

struct Connection {
    http: reqwest::Client,
    client: OnceCell<aws_sdk_s3::Client>,
}

type Environment = dyn Fn(&str) -> Option<String> + Send + Sync;

pub struct S3Source {
    object: S3Object,
    config: HttpClientConfig,
    env: Arc<Environment>,
    connection: ProcessLocal<Result<Connection, litellm_http::Error>>,
}

impl S3Source {
    pub fn new(object: S3Object, config: &HttpClientConfig) -> Result<Self, litellm_http::Error> {
        Self::with_env(object, config, |name| std::env::var(name).ok())
    }

    fn with_env(
        object: S3Object,
        config: &HttpClientConfig,
        env: impl Fn(&str) -> Option<String> + Send + Sync + 'static,
    ) -> Result<Self, litellm_http::Error> {
        let source = Self {
            object,
            config: config.clone(),
            env: Arc::new(env),
            connection: ProcessLocal::new(),
        };
        source.connection()?;
        Ok(source)
    }

    fn connection(&self) -> Result<&Connection, litellm_http::Error> {
        self.connection
            .get_or_init(|| {
                let http = reqwest::ClientBuilder::try_from(&self.config)?
                    .redirect(reqwest::redirect::Policy::none())
                    .build()?;
                Ok(Connection {
                    http,
                    client: OnceCell::new(),
                })
            })
            .as_ref()
            .map_err(Clone::clone)
    }

    async fn client(&self, connection: &Connection) -> aws_sdk_s3::Client {
        connection
            .client
            .get_or_init(|| async {
                let loader = aws_config::defaults(BehaviorVersion::latest())
                    .http_client(ReqwestHttpClient(connection.http.clone()))
                    .retry_config(RetryConfig::disabled());
                let loader = match (self.env)("AWS_REGION_NAME").filter(|s| !s.is_empty()) {
                    Some(region) => loader.region(Region::new(region)),
                    None => loader,
                };
                let defaults = loader.load().await;
                litellm_s3::client(
                    S3ClientConfig {
                        region: defaults
                            .region()
                            .map_or("us-east-1", Region::as_ref)
                            .to_owned(),
                        endpoint: (self.env)("AWS_ENDPOINT_URL_S3")
                            .filter(|s| !s.is_empty())
                            .or_else(|| (self.env)("AWS_ENDPOINT_URL").filter(|s| !s.is_empty())),
                    },
                    connection.http.clone(),
                    defaults
                        .credentials_provider()
                        .expect("the default AWS loader enables credentials"),
                )
            })
            .await
            .clone()
    }

    async fn get(
        &self,
        client: &aws_sdk_s3::Client,
        request: &FetchRequest,
    ) -> Result<GetObjectOutput, Box<SdkError<GetObjectError>>> {
        client
            .get_object()
            .bucket(&self.object.bucket)
            .key(&self.object.key)
            .set_if_none_match(request.if_none_match.clone())
            .customize()
            .config_override(
                aws_sdk_s3::config::Builder::new()
                    .retry_config(RetryConfig::disabled())
                    .timeout_config(
                        TimeoutConfig::builder()
                            .operation_timeout(request.timeout)
                            .build(),
                    ),
            )
            .send()
            .await
            .map_err(Box::new)
    }

    async fn fetch_inner(&self, request: &FetchRequest) -> Reply {
        let connection = match self.connection() {
            Ok(connection) => connection,
            Err(_) => {
                return Reply::Failed {
                    reason: "could not configure S3 transport".to_owned(),
                };
            }
        };
        let client = self.client(connection).await;
        let output = self.get(&client, request).await;
        let region = output
            .as_ref()
            .err()
            .and_then(|error| redirect_region(error, &client));
        let output = if let Some(region) = region {
            let redirected = aws_sdk_s3::Client::from_conf(
                client
                    .config()
                    .to_builder()
                    .region(Region::new(region))
                    .build(),
            );
            self.get(&redirected, request).await
        } else {
            output
        };
        match output {
            Ok(output) => {
                let validator = output.e_tag;
                match output.body.collect().await {
                    Ok(body) => Reply::Fetched {
                        body: body.into_bytes(),
                        validator,
                    },
                    Err(_) => retry("could not read S3 model catalog body"),
                }
            }
            Err(error) => classify(*error),
        }
    }
}

#[async_trait]
impl Fetch for S3Source {
    async fn fetch(&self, request: FetchRequest) -> Reply {
        tokio::time::timeout(request.timeout, self.fetch_inner(&request))
            .await
            .unwrap_or_else(|_| retry("S3 model catalog fetch timed out"))
    }
}

fn redirect_region(
    error: &SdkError<GetObjectError>,
    client: &aws_sdk_s3::Client,
) -> Option<String> {
    let response = error.raw_response()?;
    if !matches!(response.status().as_u16(), 301 | 302 | 307 | 400) {
        return None;
    }
    let region = response.headers().get("x-amz-bucket-region")?;
    (!region.is_empty() && client.config().region().map(Region::as_ref) != Some(region))
        .then(|| region.to_owned())
}

fn classify(error: SdkError<GetObjectError>) -> Reply {
    if let Some(response) = error.raw_response() {
        let status = response.status().as_u16();
        if status == 304 {
            return Reply::NotModified;
        }
        let throttled = error.as_service_error().is_some_and(|error| {
            matches!(
                error.meta().code(),
                Some(
                    "SlowDown"
                        | "Throttling"
                        | "ThrottlingException"
                        | "RequestLimitExceeded"
                        | "RequestThrottled"
                        | "RequestThrottledException"
                        | "TooManyRequestsException"
                        | "RequestTimeout"
                )
            )
        });
        let reason = format!("S3 HTTP {status} fetching model catalog");
        return if status >= 500 || matches!(status, 408 | 429) || throttled {
            Reply::Retry {
                after: None,
                reason,
            }
        } else {
            Reply::Failed { reason }
        };
    }
    match error {
        SdkError::TimeoutError(_) => retry("S3 model catalog fetch timed out"),
        SdkError::DispatchFailure(error) if error.is_io() || error.is_timeout() => {
            retry("S3 model catalog connection failed")
        }
        _ => Reply::Failed {
            reason: "S3 model catalog credentials or request configuration failed".to_owned(),
        },
    }
}

fn retry(reason: &str) -> Reply {
    Reply::Retry {
        after: None,
        reason: reason.to_owned(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{process::Command, time::Duration};

    use litellm_http::{EnvironmentProxies, HttpSettings, Resolution};
    use rstest::{fixture, rstest};
    use tokio::{
        io::{AsyncBufReadExt, AsyncWriteExt, BufReader},
        net::TcpListener,
    };
    use wiremock::{
        Mock, MockServer, ResponseTemplate,
        matchers::{header, header_regex, method, path},
    };

    const KEYS: [(&str, &str); 2] = [
        ("AWS_ACCESS_KEY_ID", "test-access-key"),
        ("AWS_SECRET_ACCESS_KEY", "test-secret-key"),
    ];

    fn isolated(env: &[(&str, &str)]) -> bool {
        if std::env::var_os("S3_TEST_CHILD").is_some() {
            return true;
        }
        let directory = tempfile::tempdir().unwrap();
        let config = directory.path().join("config");
        let credentials = directory.path().join("credentials");
        std::fs::write(
            &config,
            "[profile selected]\nregion = eu-north-1\n\
             [profile process]\nregion = eu-north-1\n\
             credential_process = /bin/echo '{\"Version\":1,\"AccessKeyId\":\"process-key\",\"SecretAccessKey\":\"test-secret-key\"}'\n\
             [profile broken]\ncredential_process = /nonexistent/s3-test-command\n",
        ).unwrap();
        std::fs::write(
            &credentials,
            "[selected]\naws_access_key_id = profile-key\naws_secret_access_key = test-secret-key\n",
        ).unwrap();
        let output = Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                std::thread::current().name().unwrap(),
                "--nocapture",
            ])
            .env_clear()
            .env("S3_TEST_CHILD", "1")
            .env("AWS_EC2_METADATA_DISABLED", "true")
            .env("AWS_CONFIG_FILE", config)
            .env("AWS_SHARED_CREDENTIALS_FILE", credentials)
            .env("HOME", directory.path())
            .envs(env.iter().copied())
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        false
    }

    #[fixture]
    fn config() -> HttpClientConfig {
        Resolution::from(&HttpSettings::default()).config
    }

    #[fixture]
    fn request() -> FetchRequest {
        FetchRequest {
            if_none_match: Some("\"old\"".to_owned()),
            timeout: Duration::from_secs(3),
        }
    }

    fn source(config: &HttpClientConfig, endpoint: String) -> S3Source {
        S3Source::with_env(
            S3Object::parse("s3://bucket/key").unwrap(),
            config,
            move |name| match name {
                "AWS_ENDPOINT_URL_S3" => Some(endpoint.clone()),
                "AWS_REGION_NAME" => Some("auto".to_owned()),
                _ => None,
            },
        )
        .unwrap()
    }

    #[rstest]
    #[case::object(200, "{}", "fetched")]
    #[case::not_modified(304, "", "not_modified")]
    #[case::server_error(500, "InternalError", "retry")]
    #[case::slow_down(503, "SlowDown", "retry")]
    #[case::throttling(400, "Throttling", "retry")]
    #[case::rate_limited(429, "TooManyRequests", "retry")]
    #[case::forbidden(403, "AccessDenied", "failed")]
    #[case::missing_key(404, "NoSuchKey", "failed")]
    #[case::missing_bucket(404, "NoSuchBucket", "failed")]
    #[case::bad_request(400, "InvalidArgument", "failed")]
    #[tokio::test]
    async fn status_mapping(
        config: HttpClientConfig,
        request: FetchRequest,
        #[case] status: u16,
        #[case] body: &str,
        #[case] expected: &str,
    ) {
        if !isolated(&KEYS) {
            return;
        }
        let server = MockServer::start().await;
        let payload = if status >= 400 {
            format!("<Error><Code>{body}</Code><Message>test error</Message></Error>")
        } else {
            body.to_owned()
        };
        Mock::given(method("GET"))
            .and(path("/bucket/key"))
            .and(header("if-none-match", "\"old\""))
            .and(header_regex(
                "authorization",
                r"Credential=test-access-key/\d+/auto/s3/aws4_request",
            ))
            .respond_with(
                ResponseTemplate::new(status)
                    .set_body_string(payload)
                    .insert_header("etag", "\"new\""),
            )
            .expect(1)
            .mount(&server)
            .await;
        let reply = source(&config, server.uri()).fetch(request).await;
        match (expected, &reply) {
            ("fetched", Reply::Fetched { body, validator }) => {
                assert_eq!(body.as_ref(), b"{}");
                assert_eq!(validator.as_deref(), Some("\"new\""));
            }
            ("not_modified", Reply::NotModified)
            | ("retry", Reply::Retry { after: None, .. })
            | ("failed", Reply::Failed { .. }) => {}
            _ => panic!("expected {expected}, got {reply:?}"),
        }
        assert_eq!(server.received_requests().await.unwrap().len(), 1);
        server.verify().await;
    }

    #[rstest]
    #[case::absent(&[])]
    #[case::broken_process(&[("AWS_PROFILE", "broken")])]
    #[tokio::test]
    async fn missing_credentials_send_nothing(
        config: HttpClientConfig,
        request: FetchRequest,
        #[case] env: &[(&str, &str)],
    ) {
        if !isolated(env) {
            return;
        }
        let server = MockServer::start().await;
        let reply = source(&config, server.uri()).fetch(request).await;
        assert!(matches!(reply, Reply::Failed { .. }), "{reply:?}");
        assert!(server.received_requests().await.unwrap().is_empty());
    }

    #[rstest]
    #[case::env_region(&[("AWS_REGION", "ap-south-1"), ("AWS_DEFAULT_REGION", "us-east-2")], "ap-south-1")]
    #[case::default_region(&[("AWS_DEFAULT_REGION", "us-east-2")], "us-east-2")]
    #[case::litellm_override(&[("AWS_REGION_NAME", "auto"), ("AWS_REGION", "ap-south-1")], "auto")]
    #[case::profile(&[("AWS_PROFILE", "selected")], "eu-north-1")]
    #[case::fallback(&[], "us-east-1")]
    #[tokio::test]
    async fn region_precedence(
        config: HttpClientConfig,
        request: FetchRequest,
        #[case] env: &[(&str, &str)],
        #[case] region: &str,
    ) {
        let vars = [KEYS.as_slice(), env].concat();
        if !isolated(&vars) {
            return;
        }
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/bucket/key"))
            .and(header_regex(
                "authorization",
                &format!(r"/{region}/s3/aws4_request"),
            ))
            .respond_with(ResponseTemplate::new(200).set_body_string("{}"))
            .expect(1)
            .mount(&server)
            .await;
        let endpoint = server.uri();
        let source = S3Source::with_env(
            S3Object::parse("s3://bucket/key").unwrap(),
            &config,
            move |name| {
                if name == "AWS_ENDPOINT_URL_S3" {
                    Some(endpoint.clone())
                } else {
                    std::env::var(name).ok()
                }
            },
        )
        .unwrap();
        assert!(matches!(source.fetch(request).await, Reply::Fetched { .. }));
        server.verify().await;
    }

    #[rstest]
    #[case::shared_file("selected", "profile-key")]
    #[case::credential_process("process", "process-key")]
    #[tokio::test]
    async fn profile_credentials(
        config: HttpClientConfig,
        request: FetchRequest,
        #[case] profile: &str,
        #[case] key: &str,
    ) {
        if !isolated(&[
            ("AWS_PROFILE", profile),
            ("AWS_PROFILE_NAME", "ignored"),
            ("AWS_ROLE_NAME", "ignored"),
        ]) {
            return;
        }
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/bucket/key"))
            .and(header_regex(
                "authorization",
                &format!(r"Credential={key}/"),
            ))
            .respond_with(ResponseTemplate::new(200).set_body_string("{}"))
            .expect(1)
            .mount(&server)
            .await;
        let reply = source(&config, server.uri()).fetch(request).await;
        assert!(matches!(reply, Reply::Fetched { .. }), "{reply:?}");
        server.verify().await;
    }

    #[rstest]
    #[case::service_endpoint(true)]
    #[case::global_endpoint(false)]
    #[tokio::test]
    async fn endpoint_precedence(
        config: HttpClientConfig,
        request: FetchRequest,
        #[case] service_endpoint: bool,
    ) {
        if !isolated(&KEYS) {
            return;
        }
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/bucket/key"))
            .respond_with(ResponseTemplate::new(200).set_body_string("{}"))
            .expect(1)
            .mount(&server)
            .await;
        let endpoint = server.uri();
        let source = S3Source::with_env(
            S3Object::parse("s3://bucket/key").unwrap(),
            &config,
            move |name| match name {
                "AWS_ENDPOINT_URL_S3" if service_endpoint => Some(endpoint.clone()),
                "AWS_ENDPOINT_URL" if service_endpoint => Some("http://127.0.0.1:1".to_owned()),
                "AWS_ENDPOINT_URL" => Some(endpoint.clone()),
                _ => None,
            },
        )
        .unwrap();
        assert!(matches!(source.fetch(request).await, Reply::Fetched { .. }));
        server.verify().await;
    }

    #[rstest]
    #[case::redirect(301)]
    #[case::wrong_signing_region(400)]
    #[tokio::test]
    async fn follows_region_once(
        config: HttpClientConfig,
        request: FetchRequest,
        #[case] status: u16,
    ) {
        if !isolated(&KEYS) {
            return;
        }
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/bucket/key"))
            .and(header_regex("authorization", "/auto/s3/aws4_request"))
            .respond_with(
                ResponseTemplate::new(status).insert_header("x-amz-bucket-region", "eu-west-1"),
            )
            .expect(1)
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .and(path("/bucket/key"))
            .and(header("if-none-match", "\"old\""))
            .and(header_regex("authorization", "/eu-west-1/s3/aws4_request"))
            .respond_with(ResponseTemplate::new(200).set_body_string("{}"))
            .expect(1)
            .mount(&server)
            .await;
        let reply = source(&config, server.uri()).fetch(request).await;
        assert!(matches!(reply, Reply::Fetched { .. }), "{reply:?}");
        assert_eq!(server.received_requests().await.unwrap().len(), 2);
        server.verify().await;
    }

    #[rstest]
    #[tokio::test]
    async fn redirect_is_bounded(config: HttpClientConfig, request: FetchRequest) {
        if !isolated(&KEYS) {
            return;
        }
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/bucket/key"))
            .respond_with(
                ResponseTemplate::new(301).insert_header("x-amz-bucket-region", "eu-west-1"),
            )
            .expect(2)
            .mount(&server)
            .await;
        assert!(matches!(
            source(&config, server.uri()).fetch(request).await,
            Reply::Failed { .. }
        ));
        server.verify().await;
    }

    #[rstest]
    #[tokio::test]
    async fn timeout_does_not_retry_inside_sdk(config: HttpClientConfig) {
        if !isolated(&KEYS) {
            return;
        }
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/bucket/key"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_string("{}")
                    .set_delay(Duration::from_secs(2)),
            )
            .expect(1)
            .mount(&server)
            .await;
        let reply = source(&config, server.uri())
            .fetch(FetchRequest {
                if_none_match: None,
                timeout: Duration::from_millis(200),
            })
            .await;
        assert!(
            matches!(reply, Reply::Retry { after: None, .. }),
            "{reply:?}"
        );
        server.verify().await;
    }

    #[rstest]
    #[tokio::test]
    async fn refused_connection_is_retry(config: HttpClientConfig, request: FetchRequest) {
        if !isolated(&KEYS) {
            return;
        }
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let endpoint = format!("http://{}", listener.local_addr().unwrap());
        drop(listener);
        let reply = source(&config, endpoint).fetch(request).await;
        assert!(
            matches!(reply, Reply::Retry { after: None, .. }),
            "{reply:?}"
        );
    }

    #[rstest]
    #[case::cut_off(false)]
    #[case::stalled(true)]
    #[tokio::test]
    async fn incomplete_body_is_retry(config: HttpClientConfig, #[case] stall: bool) {
        if !isolated(&KEYS) {
            return;
        }
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let endpoint = format!("http://{}", listener.local_addr().unwrap());
        let (sent, received) = tokio::sync::oneshot::channel();
        let server = tokio::spawn(async move {
            let (stream, _) = listener.accept().await.unwrap();
            let mut reader = BufReader::new(stream);
            let mut request = String::new();
            reader.read_line(&mut request).await.unwrap();
            sent.send(request).unwrap();
            let mut stream = reader.into_inner();
            stream
                .write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 1000\r\nConnection: close\r\n\r\n{")
                .await
                .unwrap();
            if stall {
                tokio::time::sleep(Duration::from_secs(2)).await;
            }
        });
        let reply = source(&config, endpoint)
            .fetch(FetchRequest {
                if_none_match: None,
                timeout: Duration::from_millis(200),
            })
            .await;
        assert!(
            matches!(reply, Reply::Retry { after: None, .. }),
            "{reply:?}"
        );
        let line = received.await.unwrap();
        assert_eq!(line.split_whitespace().next(), Some("GET"));
        assert_eq!(
            line.split_whitespace().nth(1).unwrap().split('?').next(),
            Some("/bucket/key")
        );
        if stall {
            server.abort();
        } else {
            server.await.unwrap();
        }
    }

    #[rstest]
    #[tokio::test]
    async fn respects_http_proxy(config: HttpClientConfig, request: FetchRequest) {
        if !isolated(&KEYS) {
            return;
        }
        let server = MockServer::start().await;
        let config = HttpClientConfig {
            proxies: EnvironmentProxies::from_environment(&|name: &str| {
                (name == "HTTP_PROXY").then(|| server.uri())
            }),
            ..config
        };
        Mock::given(method("GET"))
            .and(path("/bucket/key"))
            .respond_with(ResponseTemplate::new(200).set_body_string("{}"))
            .expect(1)
            .mount(&server)
            .await;
        assert!(matches!(
            source(&config, "http://unresolvable.invalid".to_owned())
                .fetch(request)
                .await,
            Reply::Fetched { .. }
        ));
        server.verify().await;
    }

    #[rstest]
    #[case::ordinary("s3://bucket/key", "bucket", "key")]
    #[case::nested("s3://bucket/folder/key.json", "bucket", "folder/key.json")]
    #[case::spelling("S3://Bucket/a%2Fb?query#fragment", "Bucket", "a%2Fb")]
    #[case::literal_dots("s3://bucket/folder/../key", "bucket", "folder/../key")]
    fn source_keeps_original_id(#[case] value: &str, #[case] bucket: &str, #[case] key: &str) {
        let source = crate::Source::parse(value).unwrap();
        assert_eq!(source.id().0, value);
        let crate::Source::S3(object) = source else {
            panic!("expected S3");
        };
        assert_eq!(object.bucket, bucket);
        assert_eq!(object.key, key);
    }
}
