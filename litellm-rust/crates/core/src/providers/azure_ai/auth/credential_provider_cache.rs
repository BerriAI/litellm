use std::future::Future;
use std::sync::Arc;

use azure_core::credentials::TokenCredential;
use moka::future::Cache;

use crate::AuthError;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) struct AzureCredentialProviderCacheKey {
    pub(crate) mechanism: &'static str,
    pub(crate) authority: String,
    pub(crate) tenant_id: String,
    pub(crate) client_id: String,
    pub(crate) scope: String,
    pub(crate) secret_identity: String,
}

pub(crate) struct AzureCredentialProviderCache {
    entries: Cache<AzureCredentialProviderCacheKey, Arc<dyn TokenCredential>>,
}

impl AzureCredentialProviderCache {
    pub(crate) fn new(capacity: u64) -> Self {
        Self {
            entries: Cache::builder().max_capacity(capacity).build(),
        }
    }

    pub(crate) async fn get_or_create<F>(
        &self,
        key: AzureCredentialProviderCacheKey,
        create: F,
    ) -> Result<Arc<dyn TokenCredential>, AuthError>
    where
        F: Future<Output = Result<Arc<dyn TokenCredential>, AuthError>>,
    {
        self.entries
            .try_get_with(key, create)
            .await
            .map_err(|error| (*error).clone())
    }
}
