use aws_sdk_secretsmanager::error::SdkError;
use litellm_secret_value::Redacted;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("AWS authentication failed")]
    Auth(#[source] Redacted<litellm_auth_aws::Error>),
    #[error("AWS region is not configured")]
    MissingRegion,
    #[error("AWS Secrets Manager was constructed without context-aware configuration")]
    OperationContextUnavailable,
    #[error("KMS response has no plaintext")]
    MissingPlaintext,
    #[error("AWS request timed out")]
    Timeout(#[source] Redacted<Box<SdkError<aws_sdk_secretsmanager::operation::get_secret_value::GetSecretValueError>>>),
    #[error("AWS KMS decrypt failed")]
    Decrypt(#[source] Redacted<Box<SdkError<aws_sdk_kms::operation::decrypt::DecryptError>>>),
    #[error("AWS Secrets Manager read failed")]
    Read(#[source] Redacted<Box<SdkError<aws_sdk_secretsmanager::operation::get_secret_value::GetSecretValueError>>>),
    #[error("AWS Secrets Manager create failed")]
    Create(#[source] Redacted<Box<SdkError<aws_sdk_secretsmanager::operation::create_secret::CreateSecretError>>>),
    #[error("AWS Secrets Manager restore failed")]
    Restore(#[source] Redacted<Box<SdkError<aws_sdk_secretsmanager::operation::restore_secret::RestoreSecretError>>>),
    #[error("AWS Secrets Manager restored update failed")]
    Update(#[source] Redacted<Box<SdkError<aws_sdk_secretsmanager::operation::update_secret::UpdateSecretError>>>),
    #[error("AWS Secrets Manager tagging failed")]
    Tag(#[source] Redacted<Box<SdkError<aws_sdk_secretsmanager::operation::tag_resource::TagResourceError>>>),
    #[error("AWS Secrets Manager update failed")]
    Put(#[source] Redacted<Box<SdkError<aws_sdk_secretsmanager::operation::put_secret_value::PutSecretValueError>>>),
    #[error("AWS Secrets Manager delete failed")]
    Delete(#[source] Redacted<Box<SdkError<aws_sdk_secretsmanager::operation::delete_secret::DeleteSecretError>>>),
    #[error("AWS Secrets Manager replication failed")]
    Replicate(#[source] Redacted<Box<SdkError<aws_sdk_secretsmanager::operation::replicate_secret_to_regions::ReplicateSecretToRegionsError>>>),
    #[error("AWS Secrets Manager response has no string payload")]
    MissingString,
    #[error("primary secret is not a JSON object")]
    PrimarySecret(#[source] Option<Redacted<serde_json::Error>>),
    #[error(transparent)]
    Operation(#[from] litellm_secrets_types::Error),
}
