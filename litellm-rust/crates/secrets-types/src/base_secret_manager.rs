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

#[expect(
    async_fn_in_trait,
    reason = "closed backend dispatch does not require Send bounds on generic rotation"
)]
pub trait BaseSecretManager {
    type Error: From<Error>;
    type WriteResponse;
    type DeleteResponse;

    async fn async_read_secret(
        &self,
        name: &str,
        context: &SecretOperationContext,
    ) -> Result<Option<SecretValue>, Self::Error>;
    async fn async_write_secret(
        &self,
        name: &str,
        value: &SecretValue,
        context: &SecretWriteContext,
    ) -> Result<Self::WriteResponse, Self::Error>;
    async fn async_delete_secret(
        &self,
        name: &str,
        recovery_window_in_days: Option<u32>,
        context: &SecretOperationContext,
    ) -> Result<Self::DeleteResponse, Self::Error>;
}

pub async fn async_rotate_secret<M: BaseSecretManager>(
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
    if manager
        .async_read_secret(new_name, context)
        .await?
        .is_none()
    {
        return Err(Error::NewSecretMissing.into());
    }
    manager
        .async_delete_secret(current_name, Some(7), context)
        .await?;
    Ok(response)
}
