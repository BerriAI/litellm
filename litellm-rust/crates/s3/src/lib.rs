mod error;
mod transport;
mod uri;

pub use aws_credential_types::provider::ProvideCredentials;
pub use aws_sdk_s3 as sdk;
pub use error::Error;
pub use uri::S3Uri;

use aws_sdk_s3::config::{BehaviorVersion, Builder, Region};

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
