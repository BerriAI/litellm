use crate::{Error, SecretValue, SecretWriteContext};

pub fn validate_secret_name(name: &str) -> Result<(), Error> {
    if name.split('/').any(|segment| segment == "..")
        || name
            .chars()
            .any(|c| c.is_control() || matches!(c, '\u{2028}' | '\u{2029}'))
    {
        return Err(Error::UnsafeSecretName);
    }
    Ok(())
}

pub trait BaseSecretManager {
    type Error: From<Error>;
    type Context: Clone + Default + Send + Sync;

    fn async_read_secret(
        &self,
        name: &str,
        context: &Self::Context,
    ) -> impl std::future::Future<Output = Result<Option<SecretValue>, Self::Error>> + Send;
}

pub trait SecretWriter: BaseSecretManager {
    type WriteResponse;

    fn async_write_secret(
        &self,
        name: &str,
        value: &SecretValue,
        context: &SecretWriteContext<Self::Context>,
    ) -> impl std::future::Future<Output = Result<Self::WriteResponse, Self::Error>> + Send;
}

pub trait SecretDeleter: BaseSecretManager {
    type DeleteResponse;

    fn async_delete_secret(
        &self,
        name: &str,
        context: &Self::Context,
    ) -> impl std::future::Future<Output = Result<Self::DeleteResponse, Self::Error>> + Send;
}

pub trait SecretRotator: SecretDeleter {
    type RotationResponse;

    fn async_write_replacement(
        &self,
        current_name: &str,
        new_name: &str,
        value: &SecretValue,
        context: &Self::Context,
    ) -> impl std::future::Future<Output = Result<Self::RotationResponse, Self::Error>> + Send;

    fn async_read_secret_fresh(
        &self,
        name: &str,
        context: &Self::Context,
    ) -> impl std::future::Future<Output = Result<Option<SecretValue>, Self::Error>> + Send;
}

#[derive(Debug, PartialEq, Eq, thiserror::Error)]
pub enum RotationError<R, E> {
    #[error("could not read the current secret")]
    Read(#[source] E),
    #[error("replacement write failed; provider state may be unknown")]
    Write(#[source] E),
    #[error("replacement was written but could not be verified")]
    Verification {
        response: Box<R>,
        #[source]
        source: E,
    },
    #[error("replacement was verified but retiring the old secret failed")]
    Retirement {
        response: Box<R>,
        #[source]
        source: E,
    },
}

pub async fn async_rotate_secret<M: SecretRotator>(
    manager: &M,
    current_name: &str,
    new_name: &str,
    value: &SecretValue,
    context: &M::Context,
) -> Result<M::RotationResponse, RotationError<M::RotationResponse, M::Error>> {
    if manager
        .async_read_secret_fresh(current_name, context)
        .await
        .map_err(RotationError::Read)?
        .is_none()
    {
        return Err(RotationError::Read(Error::CurrentSecretMissing.into()));
    }
    let response = manager
        .async_write_replacement(current_name, new_name, value, context)
        .await
        .map_err(RotationError::Write)?;
    let verification = match manager.async_read_secret_fresh(new_name, context).await {
        Ok(None) => Err(Error::NewSecretMissing.into()),
        Ok(Some(actual)) if actual != *value => Err(Error::NewSecretMismatch.into()),
        Ok(Some(_)) => Ok(()),
        Err(error) => Err(error),
    };
    if let Err(source) = verification {
        return Err(RotationError::Verification {
            response: Box::new(response),
            source,
        });
    }
    if current_name != new_name
        && let Err(source) = manager.async_delete_secret(current_name, context).await
    {
        return Err(RotationError::Retirement {
            response: Box::new(response),
            source,
        });
    }
    Ok(response)
}
