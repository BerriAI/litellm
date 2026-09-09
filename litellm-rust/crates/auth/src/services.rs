use crate::{AuthServiceError, SecretString};

pub trait AuthValueLookup: Send + Sync {
    fn lookup(&self, key: &str) -> Result<Option<SecretString>, AuthServiceError>;
}

pub trait CallerTokenProvider: Send + Sync {
    fn invoke(&self) -> Result<Option<SecretString>, AuthServiceError>;
}

pub trait ExecutionHeaders: Send + Sync {
    fn read(&self) -> Result<Vec<(String, String)>, AuthServiceError>;
}
