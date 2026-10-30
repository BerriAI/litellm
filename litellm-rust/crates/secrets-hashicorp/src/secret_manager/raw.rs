use super::*;
use rustify::{client::Client as _, endpoint::Endpoint};
use vaultrs::api::kv2::requests::{
    DeleteLatestSecretVersionRequest, ReadSecretRequest, SetSecretRequest,
};

#[derive(thiserror::Error, veil::Redact)]
pub enum RawOperationError {
    #[error(transparent)]
    Local(Error),
    #[error("HashiCorp Vault authentication failed")]
    Authentication {
        source: Error,
        url: String,
        certificate: bool,
    },
    #[error("HashiCorp Vault {} request returned HTTP {}", .0.method, .0.status)]
    Http(HttpFailure),
    #[error("HashiCorp Vault request failed")]
    Transport(
        #[source]
        #[redact]
        litellm_http::Error,
    ),
    #[error("HashiCorp Vault client failed")]
    Client(
        #[source]
        #[redact]
        RustifyClientError,
    ),
    #[error("HashiCorp Vault {method} request timed out")]
    Timeout { method: String, elapsed: Duration },
}

#[derive(veil::Redact)]
pub struct HttpFailure {
    pub method: String,
    pub url: String,
    pub status: u16,
    #[redact]
    pub body: Vec<u8>,
}

pub struct LoginFailure<'a> {
    pub source: &'a Error,
    pub url: &'a str,
    pub certificate: bool,
}

impl RawOperationError {
    pub fn is_unsafe_secret_name(&self) -> bool {
        match self {
            Self::Local(error) => error.is_unsafe_secret_name(),
            Self::Authentication { .. }
            | Self::Http(_)
            | Self::Transport(_)
            | Self::Client(_)
            | Self::Timeout { .. } => false,
        }
    }

    pub fn local(&self) -> Option<&Error> {
        match self {
            Self::Local(error) => Some(error),
            Self::Authentication { .. }
            | Self::Http(_)
            | Self::Transport(_)
            | Self::Client(_)
            | Self::Timeout { .. } => None,
        }
    }

    pub fn authentication(&self) -> Option<LoginFailure<'_>> {
        match self {
            Self::Authentication {
                source,
                url,
                certificate,
            } => Some(LoginFailure {
                source,
                url,
                certificate: *certificate,
            }),
            Self::Local(_)
            | Self::Http(_)
            | Self::Transport(_)
            | Self::Client(_)
            | Self::Timeout { .. } => None,
        }
    }

    pub fn http_failure(&self) -> Option<&HttpFailure> {
        match self {
            Self::Http(failure) => Some(failure),
            Self::Local(_)
            | Self::Authentication { .. }
            | Self::Transport(_)
            | Self::Client(_)
            | Self::Timeout { .. } => None,
        }
    }

    pub fn timeout(&self) -> Option<(&str, Duration)> {
        match self {
            Self::Timeout { method, elapsed } => Some((method, *elapsed)),
            Self::Local(_)
            | Self::Authentication { .. }
            | Self::Http(_)
            | Self::Transport(_)
            | Self::Client(_) => None,
        }
    }

    pub fn transport(&self) -> Option<&litellm_http::Error> {
        match self {
            Self::Transport(error) => Some(error),
            Self::Local(_)
            | Self::Authentication { .. }
            | Self::Http(_)
            | Self::Client(_)
            | Self::Timeout { .. } => None,
        }
    }
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
            .map_err(RawOperationError::Client)?;
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
        .map_err(|error| match crate::error::transport(error) {
            Ok(error) => RawOperationError::Transport(error),
            Err(error) => RawOperationError::Client(error),
        })?;
        if !response.status().is_success() {
            return Err(RawOperationError::Http(HttpFailure {
                method,
                url,
                status: response.status().as_u16(),
                body: response.into_body(),
            }));
        }
        Ok(response.into_body())
    }
}
