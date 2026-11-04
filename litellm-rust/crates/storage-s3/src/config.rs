use aws_sdk_s3::config::{
    BehaviorVersion, Region, RequestChecksumCalculation, ResponseChecksumValidation,
};
use litellm_auth_aws::AwsAuthConfig;

use crate::{auth::S3Credentials, transport::ReqwestHttpClient};

/// An S3 client config over litellm's HTTP client and AWS credential chain. A custom `endpoint` is addressed path
/// style, as MinIO and other S3 compatible stores expect
pub fn s3_config(
    region: String,
    endpoint: Option<String>,
    http: litellm_http::Client,
    auth: AwsAuthConfig,
) -> aws_sdk_s3::config::Builder {
    let base = aws_sdk_s3::Config::builder()
        .behavior_version(BehaviorVersion::latest())
        .region(Region::new(region))
        .http_client(ReqwestHttpClient(http))
        .credentials_provider(S3Credentials::new(auth))
        .request_checksum_calculation(RequestChecksumCalculation::WhenRequired)
        .response_checksum_validation(ResponseChecksumValidation::WhenRequired);
    match endpoint {
        Some(url) => base.endpoint_url(url).force_path_style(true),
        None => base,
    }
}
