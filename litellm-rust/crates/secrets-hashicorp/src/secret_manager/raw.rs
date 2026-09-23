use super::*;
use rustify::{client::Client as _, endpoint::Endpoint};
use vaultrs::api::kv2::requests::{
    DeleteLatestSecretVersionRequest, ReadSecretRequest, SetSecretRequest,
};

#[derive(veil::Redact)]
pub enum RawOperationError {
    Local(Error),
    Authentication {
        source: Error,
        url: String,
        certificate: bool,
    },
    Http {
        method: String,
        url: String,
        status: u16,
        #[redact]
        body: Vec<u8>,
    },
    Transport(#[redact] RustifyClientError),
    Timeout {
        method: String,
        elapsed: Duration,
    },
}

impl HashicorpVault {
    pub async fn write_raw(
        &self,
        name: &str,
        value: &SecretValue,
        context: &SecretWriteContext<HashicorpOperationContext>,
    ) -> Result<Vec<u8>, RawOperationError> {
        let location = self
            .secret_location_with_context(name, &context.operation)
            .map_err(RawOperationError::Local)?;
        let data = super::write::write_data(value, context).map_err(RawOperationError::Local)?;
        let response = self
            .raw_request(
                &location,
                SetSecretRequest {
                    mount: location.mount.clone(),
                    path: location.path.clone(),
                    data,
                    options: None,
                },
                &context.operation,
            )
            .await?;
        self.cache
            .invalidate_where(move |key| key.location == location);
        Ok(response)
    }

    pub async fn delete_raw(
        &self,
        name: &str,
        context: &HashicorpOperationContext,
    ) -> Result<(), RawOperationError> {
        let location = self
            .secret_location_with_context(name, context)
            .map_err(RawOperationError::Local)?;
        self.raw_request(
            &location,
            DeleteLatestSecretVersionRequest {
                mount: location.mount.clone(),
                path: location.path.clone(),
            },
            context,
        )
        .await?;
        self.cache
            .invalidate_where(move |key| key.location == location);
        Ok(())
    }

    pub async fn read_raw(
        &self,
        name: &str,
        context: &HashicorpOperationContext,
    ) -> Result<Vec<u8>, RawOperationError> {
        let location = self
            .secret_location_with_context(name, context)
            .map_err(RawOperationError::Local)?;
        self.raw_request(
            &location,
            ReadSecretRequest {
                mount: location.mount.clone(),
                path: location.path.clone(),
                version: None,
            },
            context,
        )
        .await
    }

    async fn raw_request(
        &self,
        location: &SecretLocation,
        endpoint: impl Endpoint,
        context: &HashicorpOperationContext,
    ) -> Result<Vec<u8>, RawOperationError> {
        let client = self.client_for_location(location).await.map_err(|source| {
            let certificate = self.config.approle.is_none();
            let mount = self
                .config
                .approle
                .as_ref()
                .map_or("cert", |auth| auth.mount_path.as_str());
            RawOperationError::Authentication {
                source,
                certificate,
                url: format!("{}/v1/auth/{mount}/login", self.config.address),
            }
        })?;
        let request = endpoint
            .with_middleware(&client.middle)
            .request(client.http.base())
            .map_err(RawOperationError::Transport)?;
        let method = request.method().to_string();
        let url = format!(
            "{}/v1/{}{}/data/{}",
            self.config.address,
            location
                .namespace
                .as_ref()
                .map(|ns| format!("{ns}/"))
                .unwrap_or_default(),
            location.mount,
            location.path
        );
        let started = Instant::now();
        let response = match context.timeout {
            Some(timeout) => tokio::time::timeout(timeout, client.http.send(request))
                .await
                .map_err(|_| RawOperationError::Timeout {
                    method: method.clone(),
                    elapsed: started.elapsed(),
                })?,
            None => client.http.send(request).await,
        }
        .map_err(RawOperationError::Transport)?;
        if !response.status().is_success() {
            return Err(RawOperationError::Http {
                method,
                url,
                status: response.status().as_u16(),
                body: response.into_body(),
            });
        }
        Ok(response.into_body())
    }
}
