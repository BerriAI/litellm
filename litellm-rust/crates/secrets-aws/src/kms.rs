use std::sync::Arc;

use aws_sdk_kms::{
    Client,
    config::{BehaviorVersion, Region},
    primitives::Blob,
};
use litellm_core_utils::settings::Lookup;
use litellm_secrets_types::KeyManagementSettings;

use crate::{Error, auth};

#[derive(Clone)]
pub struct AwsKms {
    client: Client,
}

impl AwsKms {
    pub fn new(client: Client) -> Self {
        Self { client }
    }

    pub async fn decrypt(&self, ciphertext: Vec<u8>) -> Result<Vec<u8>, Error> {
        let response = self
            .client
            .decrypt()
            .ciphertext_blob(Blob::new(ciphertext))
            .send()
            .await
            .map_err(|error| Error::Decrypt(Box::new(error)))?;
        Ok(response
            .plaintext
            .ok_or(Error::MissingPlaintext)?
            .into_inner())
    }
}

pub fn validate_environment(environment: &dyn Lookup) -> Result<(), Error> {
    auth::region(&KeyManagementSettings::default(), environment).map(|_| ())
}

pub fn load_aws_kms(
    use_aws_kms: Option<bool>,
    settings: &KeyManagementSettings,
    environment: Arc<dyn Lookup + Send + Sync>,
) -> Result<Option<AwsKms>, Error> {
    if use_aws_kms != Some(true) {
        return Ok(None);
    }
    let config = aws_sdk_kms::Config::builder()
        .behavior_version(BehaviorVersion::latest())
        .region(Region::new(auth::region(settings, environment.as_ref())?))
        .credentials_provider(auth::Credentials::new(settings, environment))
        .build();
    Ok(Some(AwsKms::new(Client::from_conf(config))))
}
