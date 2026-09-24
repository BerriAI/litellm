use aws_sdk_secretsmanager::error::SdkError;

#[derive(thiserror::Error, veil::Redact)]
pub enum Error {
    #[error("AWS authentication failed")]
    Auth(#[from] #[redact] litellm_auth_aws::Error),
    #[error("AWS region is not configured")]
    MissingRegion,
    #[error("AWS Secrets Manager was constructed without context-aware configuration")]
    OperationContextUnavailable,
    #[error("KMS response has no plaintext")]
    MissingPlaintext,
    #[error("AWS request timed out")]
    Timeout,
    #[error("AWS KMS decrypt failed")]
    Decrypt(#[from] #[redact] Box<SdkError<aws_sdk_kms::operation::decrypt::DecryptError>>),
    #[error("AWS Secrets Manager read failed")]
    Read(#[from] #[redact] Box<SdkError<aws_sdk_secretsmanager::operation::get_secret_value::GetSecretValueError>>),
    #[error("AWS Secrets Manager create failed")]
    Create(#[from] #[redact] Box<SdkError<aws_sdk_secretsmanager::operation::create_secret::CreateSecretError>>),
    #[error("AWS Secrets Manager restore failed")]
    Restore(#[from] #[redact] Box<SdkError<aws_sdk_secretsmanager::operation::restore_secret::RestoreSecretError>>),
    #[error("AWS Secrets Manager restored update failed")]
    Update(#[from] #[redact] Box<SdkError<aws_sdk_secretsmanager::operation::update_secret::UpdateSecretError>>),
    #[error("AWS Secrets Manager tagging failed")]
    Tag(#[from] #[redact] Box<SdkError<aws_sdk_secretsmanager::operation::tag_resource::TagResourceError>>),
    #[error("AWS Secrets Manager update failed")]
    Put(#[from] #[redact] Box<SdkError<aws_sdk_secretsmanager::operation::put_secret_value::PutSecretValueError>>),
    #[error("AWS Secrets Manager delete failed")]
    Delete(#[from] #[redact] Box<SdkError<aws_sdk_secretsmanager::operation::delete_secret::DeleteSecretError>>),
    #[error("AWS Secrets Manager replication failed")]
    Replicate(#[from] #[redact] Box<SdkError<aws_sdk_secretsmanager::operation::replicate_secret_to_regions::ReplicateSecretToRegionsError>>),
    #[error("AWS Secrets Manager response has no string payload")]
    MissingString,
    #[error("primary secret is not a JSON object")]
    PrimarySecret,
    #[error(transparent)]
    Operation(#[from] litellm_secrets_types::Error),
}
