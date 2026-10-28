mod transport;

use aws_credential_types::provider::ProvideCredentials;
use aws_sdk_s3::config::{
    BehaviorVersion, Region, RequestChecksumCalculation, ResponseChecksumValidation,
};

pub use transport::ReqwestHttpClient;

/// Where an S3 client sends its requests
pub struct S3ClientConfig {
    pub region: String,
    /// An S3-compatible endpoint (MinIO, R2, Ceph), addressed path-style
    pub endpoint: Option<String>,
}

/// One way to build an S3 client over a reqwest client, shared by every S3 caller
pub fn client(
    config: S3ClientConfig,
    http: litellm_http::Client,
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
