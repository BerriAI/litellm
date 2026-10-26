#![allow(
    clippy::disallowed_types,
    reason = "accepts clients built from HttpClientConfig, including fork-local clients"
)]

mod transport;

use aws_credential_types::provider::ProvideCredentials;
use aws_sdk_s3::config::{
    BehaviorVersion, Region, RequestChecksumCalculation, ResponseChecksumValidation,
};

pub use transport::ReqwestHttpClient;

pub struct S3ClientConfig {
    pub region: String,
    pub endpoint: Option<String>,
}

pub fn client(
    config: S3ClientConfig,
    http: reqwest::Client,
    credentials: impl ProvideCredentials + 'static,
) -> aws_sdk_s3::Client {
    let base = aws_sdk_s3::Config::builder()
        .behavior_version(BehaviorVersion::latest())
        .region(Region::new(config.region))
        .http_client(ReqwestHttpClient(http))
        .credentials_provider(credentials)
        .request_checksum_calculation(RequestChecksumCalculation::WhenRequired)
        .response_checksum_validation(ResponseChecksumValidation::WhenRequired);
    let builder = match config.endpoint {
        Some(url) => base.endpoint_url(url).force_path_style(true),
        None => base,
    };
    aws_sdk_s3::Client::from_conf(builder.build())
}
