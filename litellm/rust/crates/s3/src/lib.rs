mod error;
pub mod signed;
mod transport;
mod uri;

pub use aws_credential_types::provider::ProvideCredentials;
pub use aws_sdk_s3 as sdk;
pub use error::Error;
pub use uri::S3Uri;

use aws_config::meta::region::RegionProviderChain;
use aws_sdk_s3::config::{BehaviorVersion, Builder, Region};
use aws_types::SdkConfig;
use aws_types::service_config::ServiceConfigKey;

use crate::transport::ReqwestHttpClient;

/// A custom endpoint is addressed path style, as boto3 addresses one
pub fn config(
    region: String,
    endpoint_url: Option<&str>,
    http: litellm_http::Client,
    credentials: impl ProvideCredentials + 'static,
) -> Builder {
    let builder = aws_sdk_s3::Config::builder()
        .behavior_version(BehaviorVersion::latest())
        .region(Region::new(region))
        .http_client(ReqwestHttpClient(http))
        .credentials_provider(credentials);
    match endpoint_url {
        Some(url) => builder.endpoint_url(url).force_path_style(true),
        None => builder,
    }
}

/// The region, endpoint and credentials an AWS SDK finds in the environment and shared config files, as boto3's
/// default `s3` client found them. `region` wins over that chain, and `us-east-1` is used when nothing names one
pub async fn environment_config(http: litellm_http::Client, region: Option<String>) -> Builder {
    let region = RegionProviderChain::first_try(region.map(Region::new))
        .or_default_provider()
        .or_else(Region::from_static("us-east-1"));
    let shared = aws_config::defaults(BehaviorVersion::latest())
        .region(region)
        .http_client(ReqwestHttpClient(http))
        .load()
        .await;
    Builder::from(&shared).force_path_style(has_custom_endpoint(&shared))
}

fn has_custom_endpoint(shared: &SdkConfig) -> bool {
    let s3_endpoint = ServiceConfigKey::builder()
        .service_id("S3")
        .env("AWS_ENDPOINT_URL")
        .profile("endpoint_url")
        .build()
        .ok();
    shared.endpoint_url().is_some()
        || s3_endpoint
            .and_then(|key| shared.service_config()?.load_config(key))
            .is_some()
}
