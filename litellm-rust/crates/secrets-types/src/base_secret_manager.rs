use crate::{Error, SecretOperationContext, SecretValue, SecretWriteContext};

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

    fn async_read_secret(
        &self,
        name: &str,
        context: &SecretOperationContext,
    ) -> impl std::future::Future<Output = Result<Option<SecretValue>, Self::Error>> + Send;
}

pub trait SecretWriter: BaseSecretManager {
    type WriteResponse;

    fn async_write_secret(
        &self,
        name: &str,
        value: &SecretValue,
        context: &SecretWriteContext,
    ) -> impl std::future::Future<Output = Result<Self::WriteResponse, Self::Error>> + Send;
}

pub trait SecretDeleter: BaseSecretManager {
    type DeleteResponse;

    fn async_delete_secret(
        &self,
        name: &str,
        recovery_window_in_days: Option<u32>,
        context: &SecretOperationContext,
    ) -> impl std::future::Future<Output = Result<Self::DeleteResponse, Self::Error>> + Send;
}

pub async fn async_rotate_secret<M: SecretWriter + SecretDeleter>(
    manager: &M,
    current_name: &str,
    new_name: &str,
    value: &SecretValue,
    context: &SecretOperationContext,
) -> Result<M::WriteResponse, M::Error> {
    if manager
        .async_read_secret(current_name, context)
        .await?
        .is_none()
    {
        return Err(Error::CurrentSecretMissing.into());
    }
    let response = manager
        .async_write_secret(
            new_name,
            value,
            &SecretWriteContext::rotated_from(current_name, context.clone()),
        )
        .await?;
    match manager.async_read_secret(new_name, context).await? {
        None => return Err(Error::NewSecretMissing.into()),
        Some(actual) if actual != *value => return Err(Error::NewSecretMismatch.into()),
        Some(_) => {}
    }
    if current_name != new_name {
        manager
            .async_delete_secret(current_name, Some(7), context)
            .await?;
    }
    Ok(response)
}
