use std::{sync::Arc, time::Duration};

use azure_core::{
    credentials::TokenCredential,
    error::ErrorKind,
    http::{ClientOptions, RequestContent, Transport},
};
use azure_storage_blob::{
    BlobContainerClient, BlobContainerClientOptions,
    models::{BlobClientUploadOptions, StorageErrorCode},
};
use futures_util::{TryStreamExt, future::try_join_all};
use litellm_cache::{
    BaseCache, BatchCache, CacheCodec, DisconnectCache, Error, ExactCacheContext, FlushCache,
};
use tokio::runtime::Handle;
use url::Url;

use crate::{credential::AzureBlobCredential, transport::ReqwestTransport};

pub struct AzureBlobCache<C> {
    container: BlobContainerClient,
    codec: C,
    runtime: Handle,
    account_url: String,
    container_name: String,
}

impl<C: CacheCodec> AzureBlobCache<C> {
    /// `http` is the host's pooled client; the SDK sends every request through it.
    pub async fn connect(
        account_url: &str,
        container: &str,
        http: litellm_http::Client,
        codec: C,
        runtime: Handle,
    ) -> Result<Self, Error> {
        Self::connect_with_options(
            account_url,
            container,
            Some(Arc::new(AzureBlobCredential::default())),
            ClientOptions {
                transport: Some(Transport::new(Arc::new(ReqwestTransport(http)))),
                ..ClientOptions::default()
            },
            codec,
            runtime,
        )
        .await
    }

    pub async fn connect_with_options(
        account_url: &str,
        container: &str,
        credential: Option<Arc<dyn TokenCredential>>,
        client_options: ClientOptions,
        codec: C,
        runtime: Handle,
    ) -> Result<Self, Error> {
        let parsed = Url::parse(account_url).map_err(|_| Error::Unavailable)?;
        let account_url = parsed.as_str().trim_end_matches('/').to_string();
        let container_url = {
            let mut url = parsed;
            url.path_segments_mut()
                .map_err(|()| Error::Unavailable)?
                .pop_if_empty()
                .push(container);
            url
        };
        let client = BlobContainerClient::new(
            container_url,
            credential,
            Some(BlobContainerClientOptions {
                client_options,
                ..BlobContainerClientOptions::default()
            }),
        )
        .map_err(|_| Error::Unavailable)?;
        let cache = Self {
            container: client,
            codec,
            runtime,
            account_url,
            container_name: container.to_string(),
        };
        cache.create_container().await?;
        Ok(cache)
    }

    pub fn account_url(&self) -> &str {
        &self.account_url
    }

    pub fn container_name(&self) -> &str {
        &self.container_name
    }

    async fn create_container(&self) -> Result<(), Error> {
        match self.container.create(None).await {
            Ok(_) => Ok(()),
            Err(error) if is_storage_error(&error, StorageErrorCode::ContainerAlreadyExists) => {
                Ok(())
            }
            Err(_) => Err(Error::Unavailable),
        }
    }

    async fn upload(&self, key: &str, value: &C::Value, overwrite: bool) -> Result<(), Error> {
        let payload = self.codec.encode(value)?;
        let options = (!overwrite).then(|| BlobClientUploadOptions::default().if_not_exists());
        match self
            .container
            .blob_client(key)
            .upload(RequestContent::from(payload), options)
            .await
        {
            Ok(_) => Ok(()),
            Err(error) if !overwrite && is_already_present(&error) => Ok(()),
            Err(_) => Err(Error::Unavailable),
        }
    }

    async fn download(&self, key: &str) -> Result<Option<C::Value>, Error> {
        let response = match self.container.blob_client(key).download(None).await {
            Ok(response) => response,
            Err(error) if is_storage_error(&error, StorageErrorCode::BlobNotFound) => {
                return Ok(None);
            }
            Err(_) => return Err(Error::Unavailable),
        };
        let bytes = response
            .body
            .collect()
            .await
            .map_err(|_| Error::Unavailable)?;
        self.codec.decode(&bytes).map(Some)
    }

    async fn delete_all_blobs(&self) -> Result<(), Error> {
        let mut pages = self
            .container
            .list_blobs(None)
            .map_err(|_| Error::Unavailable)?
            .into_pages();
        while let Some(page) = pages.try_next().await.map_err(|_| Error::Unavailable)? {
            let page = page.into_model().map_err(|_| Error::Unavailable)?;
            for name in page.blob_items.into_iter().filter_map(|item| item.name) {
                self.container
                    .blob_client(&name)
                    .delete(None)
                    .await
                    .map_err(|_| Error::Unavailable)?;
            }
        }
        Ok(())
    }

    fn block_on<T>(&self, future: impl Future<Output = T>) -> T {
        if Handle::try_current().is_ok() {
            tokio::task::block_in_place(|| self.runtime.block_on(future))
        } else {
            self.runtime.block_on(future)
        }
    }
}

fn is_already_present(error: &azure_core::Error) -> bool {
    is_storage_error(error, StorageErrorCode::BlobAlreadyExists)
        || is_storage_error(error, StorageErrorCode::ConditionNotMet)
}

fn is_storage_error(error: &azure_core::Error, code: StorageErrorCode) -> bool {
    matches!(
        error.kind(),
        ErrorKind::HttpResponse {
            error_code: Some(error_code),
            ..
        } if error_code == code.as_ref()
    )
}

impl<C: CacheCodec> BaseCache for AzureBlobCache<C> {
    type Value = C::Value;
    type Context = ExactCacheContext;

    fn get_ttl(&self, _: &ExactCacheContext) -> Option<Duration> {
        None
    }

    fn set_cache(&self, key: &str, value: C::Value, _: &ExactCacheContext) -> Result<(), Error> {
        self.block_on(self.upload(key, &value, false))
    }

    fn get_cache(&self, key: &str, _: &ExactCacheContext) -> Result<Option<C::Value>, Error> {
        self.block_on(self.download(key))
    }

    async fn async_set_cache(
        &self,
        key: &str,
        value: C::Value,
        _: ExactCacheContext,
    ) -> Result<(), Error> {
        self.upload(key, &value, true).await
    }

    async fn async_get_cache(
        &self,
        key: &str,
        _: &ExactCacheContext,
    ) -> Result<Option<C::Value>, Error> {
        self.download(key).await
    }

    async fn async_set_cache_pipeline(
        &self,
        entries: Vec<(String, C::Value)>,
        _: ExactCacheContext,
    ) -> Result<(), Error> {
        try_join_all(
            entries
                .iter()
                .map(|(key, value)| self.upload(key, value, true)),
        )
        .await
        .map(drop)
    }
}

impl<C: CacheCodec> BatchCache for AzureBlobCache<C> {}

impl<C: CacheCodec> FlushCache for AzureBlobCache<C> {
    fn flush_cache(&self) -> Result<(), Error> {
        self.block_on(self.delete_all_blobs())
    }

    async fn async_flush_cache(&self) -> Result<(), Error> {
        self.delete_all_blobs().await
    }
}

impl<C: CacheCodec> DisconnectCache for AzureBlobCache<C> {
    /// Python closes its two SDK clients; the Rust clients hold no connection of their own
    /// (the pooled transport belongs to the host), so there is nothing to release.
    async fn disconnect(&self) -> Result<(), Error> {
        Ok(())
    }
}
