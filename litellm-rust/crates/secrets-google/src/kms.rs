use std::sync::Arc;

use google_cloud_kms_v1::client::KeyManagementService;
use litellm_core_utils::settings::Lookup;
use litellm_secrets_types::SecretValue;

use crate::{Error, auth};

const GOOGLE_APPLICATION_CREDENTIALS: &str = "GOOGLE_APPLICATION_CREDENTIALS";
const GOOGLE_KMS_RESOURCE_NAME: &str = "GOOGLE_KMS_RESOURCE_NAME";

#[derive(Clone)]
pub struct GoogleKms {
    client: KeyManagementService,
    resource_name: String,
}

impl GoogleKms {
    pub fn new(client: KeyManagementService, resource_name: String) -> Self {
        Self {
            client,
            resource_name,
        }
    }

    pub async fn decrypt(&self, ciphertext: Vec<u8>) -> Result<Vec<u8>, Error> {
        let response = self
            .client
            .decrypt()
            .set_name(&self.resource_name)
            .set_ciphertext(ciphertext)
            .send()
            .await?;
        Ok(response.plaintext.to_vec())
    }
}

pub fn validate_environment(environment: &dyn Lookup) -> Result<(), Error> {
    environment
        .get(GOOGLE_KMS_RESOURCE_NAME)
        .map(|_| ())
        .ok_or(Error::MissingEnvironment(GOOGLE_KMS_RESOURCE_NAME))
}

pub async fn load_google_kms(
    use_google_kms: Option<bool>,
    environment: Arc<dyn Lookup + Send + Sync>,
) -> Result<Option<GoogleKms>, Error> {
    if use_google_kms != Some(true) {
        return Ok(None);
    }
    validate_environment(environment.as_ref())?;
    let resource_name = environment
        .get(GOOGLE_KMS_RESOURCE_NAME)
        .ok_or(Error::MissingEnvironment(GOOGLE_KMS_RESOURCE_NAME))?;
    let credentials = auth::credentials(
        None,
        environment
            .get(GOOGLE_APPLICATION_CREDENTIALS)
            .map(SecretValue::new),
        environment,
    );
    let client = KeyManagementService::builder()
        .with_credentials(credentials)
        .build()
        .await?;
    Ok(Some(GoogleKms::new(client, resource_name)))
}
