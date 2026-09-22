use crate::{Error, SecretValue};

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

    async fn async_read_secret(&self, name: &str) -> Result<Option<SecretValue>, Self::Error>;
    async fn async_write_secret(
        &self,
        name: &str,
        value: &SecretValue,
        description: Option<&str>,
    ) -> Result<Self::WriteResponse, Self::Error>;
    async fn async_delete_secret(
        &self,
        name: &str,
        recovery_window_in_days: i64,
    ) -> Result<Self::DeleteResponse, Self::Error>;
}

pub async fn async_rotate_secret<M: BaseSecretManager>(
    manager: &M,
    current_name: &str,
    new_name: &str,
    value: &SecretValue,
) -> Result<M::WriteResponse, M::Error> {
    if manager.async_read_secret(current_name).await?.is_none() {
        return Err(Error::CurrentSecretMissing.into());
    }
    let response = manager
        .async_write_secret(
            new_name,
            value,
            Some(&format!("Rotated from {current_name}")),
        )
        .await?;
    if manager.async_read_secret(new_name).await?.is_none() {
        return Err(Error::NewSecretMissing.into());
    }
    manager.async_delete_secret(current_name, 7).await?;
    Ok(response)
}
