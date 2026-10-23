use std::{
    collections::HashMap,
    sync::{Arc, Mutex as StdMutex},
};

use litellm_inference::Error;
use litellm_llms::ErrorDetail;
use litellm_token_counter::raw_text::{LocalTokenizerConfig, RawTextCacheKey, RawTextTokenizer};
use tokio::{
    sync::{Mutex, Semaphore, watch},
    task::JoinHandle,
};

use crate::error::LocalTokenizerRegistryError as RegistryError;

const DEFAULT_CAPACITY: usize = 32;

type SharedResult = Result<Arc<RawTextTokenizer>, Arc<RegistryError>>;

struct Entry {
    sender: watch::Sender<Option<SharedResult>>,
    receiver: watch::Receiver<Option<SharedResult>>,
    task: StdMutex<Option<JoinHandle<()>>>,
}

struct RegistryInner {
    entries: Mutex<HashMap<RawTextCacheKey, Arc<Entry>>>,
    semaphore: Arc<Semaphore>,
    capacity: usize,
}

#[derive(Clone)]
pub struct LocalTokenizerRegistry {
    inner: Arc<RegistryInner>,
}

impl Default for LocalTokenizerRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl LocalTokenizerRegistry {
    pub fn new() -> Self {
        let concurrency = std::thread::available_parallelism()
            .map(usize::from)
            .unwrap_or(1);
        Self::with_limits(DEFAULT_CAPACITY, concurrency)
    }

    pub fn with_capacity(capacity: usize) -> Self {
        let concurrency = std::thread::available_parallelism()
            .map(usize::from)
            .unwrap_or(1);
        Self::with_limits(capacity, concurrency)
    }

    pub fn with_limits(capacity: usize, blocking_workers: usize) -> Self {
        Self {
            inner: Arc::new(RegistryInner {
                entries: Mutex::new(HashMap::new()),
                semaphore: Arc::new(Semaphore::new(blocking_workers.max(1))),
                capacity,
            }),
        }
    }

    pub async fn preload(
        &self,
        config: Arc<LocalTokenizerConfig>,
    ) -> Result<Arc<RawTextTokenizer>, Error> {
        let config_for_key = Arc::clone(&config);
        let key = run_blocking(Arc::clone(&self.inner.semaphore), move || {
            config_for_key
                .cache_key()
                .map_err(|error| RegistryError::Tokenizer(Arc::new(error)))
        })
        .await
        .map_err(route_error)?;
        let (entry, initialize) = {
            let mut entries = self.inner.entries.lock().await;
            if let Some(entry) = entries.get(&key) {
                (Arc::clone(entry), false)
            } else if entries.len() >= self.inner.capacity {
                return Err(route_error(RegistryError::Capacity));
            } else {
                let (sender, receiver) = watch::channel(None);
                let entry = Arc::new(Entry {
                    sender,
                    receiver,
                    task: StdMutex::new(None),
                });
                entries.insert(key.clone(), Arc::clone(&entry));
                (entry, true)
            }
        };
        if initialize {
            let inner = Arc::clone(&self.inner);
            let sender = entry.sender.clone();
            let key_for_prepare = key.clone();
            let key_for_removal = key.clone();
            let config_for_task = Arc::clone(&config);
            let task = tokio::spawn(async move {
                let tokenizer = run_blocking(Arc::clone(&inner.semaphore), move || {
                    config_for_task
                        .prepare_for_key(key_for_prepare)
                        .and_then(RawTextTokenizer::load_prepared)
                        .map(Arc::new)
                        .map_err(|error| RegistryError::Tokenizer(Arc::new(error)))
                })
                .await
                .map_err(Arc::new);
                if tokenizer.is_err() {
                    let mut entries = inner.entries.lock().await;
                    entries.remove(&key_for_removal);
                }
                sender.send_replace(Some(tokenizer));
            });
            if let Ok(mut handle) = entry.task.lock() {
                *handle = Some(task);
            }
        }
        let mut result = entry.receiver.clone();
        loop {
            if let Some(value) = result.borrow_and_update().clone() {
                return value.map_err(|error| route_error(error.as_ref().clone()));
            }
            result
                .changed()
                .await
                .map_err(|_| route_error(RegistryError::Worker))?;
        }
    }

    pub async fn count(
        &self,
        tokenizer: Arc<RawTextTokenizer>,
        text: String,
    ) -> Result<usize, Error> {
        run_blocking(Arc::clone(&self.inner.semaphore), move || {
            tokenizer
                .count(&text)
                .map_err(|error| RegistryError::Tokenizer(Arc::new(error)))
        })
        .await
        .map_err(route_error)
    }

    pub async fn encode(
        &self,
        tokenizer: Arc<RawTextTokenizer>,
        text: String,
        include_ids: bool,
        include_tokens: bool,
    ) -> Result<litellm_token_counter::raw_text::RawTextEncoding, Error> {
        run_blocking(Arc::clone(&self.inner.semaphore), move || {
            tokenizer
                .encode(&text, include_ids, include_tokens)
                .map_err(|error| RegistryError::Tokenizer(Arc::new(error)))
        })
        .await
        .map_err(route_error)
    }

    #[cfg(test)]
    async fn len(&self) -> usize {
        self.inner.entries.lock().await.len()
    }
}

async fn run_blocking<T, F>(semaphore: Arc<Semaphore>, work: F) -> Result<T, RegistryError>
where
    T: Send + 'static,
    F: FnOnce() -> Result<T, RegistryError> + Send + 'static,
{
    let permit = semaphore
        .acquire_owned()
        .await
        .map_err(|_| RegistryError::Worker)?;
    tokio::task::spawn_blocking(move || {
        let _permit = permit;
        work()
    })
    .await
    .map_err(|_| RegistryError::Worker)?
}

fn route_error(error: RegistryError) -> Error {
    Error::InvalidRequest(ErrorDetail::invalid("local tokenizer", error))
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use sha2::{Digest, Sha256};

    use super::*;
    use litellm_token_counter::raw_text::{RawTextArtifact, RawTextEngine, SpecialTextPolicy};

    const HF_JSON: &str = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../../litellm/litellm_core_utils/tokenizers/anthropic_tokenizer.json"
    ));

    fn config(id: &str, special_text: Option<SpecialTextPolicy>) -> Arc<LocalTokenizerConfig> {
        Arc::new(LocalTokenizerConfig {
            id: id.into(),
            revision: "revision".into(),
            engine: RawTextEngine::HuggingFace,
            artifact: RawTextArtifact::Inline {
                content: HF_JSON.into(),
            },
            special_text,
            add_special_tokens: None,
        })
    }

    #[rstest]
    #[tokio::test]
    async fn concurrent_misses_reuse_the_loaded_instance() {
        let registry = LocalTokenizerRegistry::with_limits(4, 2);
        let config = config("shared", None);
        let (first, second) = tokio::join!(
            registry.preload(Arc::clone(&config)),
            registry.preload(Arc::clone(&config))
        );

        assert!(Arc::ptr_eq(&first.unwrap(), &second.unwrap()));
        assert_eq!(registry.len().await, 1);
    }

    #[rstest]
    #[tokio::test]
    async fn content_and_effective_policy_isolate_registry_entries() {
        let registry = LocalTokenizerRegistry::with_capacity(4);
        let ordinary = registry.preload(config("same-id", None)).await.unwrap();
        let changed_content = Arc::new(LocalTokenizerConfig {
            artifact: RawTextArtifact::Inline {
                content: format!("{HF_JSON}\n "),
            },
            ..(*config("same-id", None)).clone()
        });
        let different_artifact = registry.preload(changed_content).await.unwrap();
        let literal = registry
            .preload(config("same-id", Some(SpecialTextPolicy::Literal)))
            .await
            .unwrap();

        assert!(!Arc::ptr_eq(&ordinary, &literal));
        assert!(!Arc::ptr_eq(&ordinary, &different_artifact));
        assert_eq!(registry.len().await, 3);
    }

    #[rstest]
    #[tokio::test]
    async fn failed_loads_are_not_cached() {
        let registry = LocalTokenizerRegistry::with_capacity(2);
        let config = Arc::new(LocalTokenizerConfig {
            id: "missing".into(),
            revision: "revision".into(),
            engine: RawTextEngine::HuggingFace,
            artifact: RawTextArtifact::File {
                path: "/path/that/does/not/exist".into(),
                sha256: "0".repeat(64),
            },
            special_text: None,
            add_special_tokens: None,
        });

        assert!(registry.preload(Arc::clone(&config)).await.is_err());
        assert!(registry.preload(config).await.is_err());
        assert_eq!(registry.len().await, 0);
    }

    #[rstest]
    #[tokio::test]
    async fn warm_file_entry_survives_source_removal() {
        let registry = LocalTokenizerRegistry::with_capacity(2);
        let source = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(source.path(), HF_JSON).unwrap();
        let config = Arc::new(LocalTokenizerConfig {
            id: "file-tokenizer".into(),
            revision: "revision".into(),
            engine: RawTextEngine::HuggingFace,
            artifact: RawTextArtifact::File {
                path: source.path().to_path_buf(),
                sha256: format!("{:x}", Sha256::digest(HF_JSON.as_bytes())).to_ascii_uppercase(),
            },
            special_text: None,
            add_special_tokens: None,
        });
        let loaded = registry.preload(Arc::clone(&config)).await.unwrap();
        std::fs::remove_file(source.path()).unwrap();

        let warm = registry.preload(config).await.unwrap();

        assert!(Arc::ptr_eq(&loaded, &warm));
        assert!(registry.count(warm, "hello".into()).await.unwrap() > 0);
    }

    #[rstest]
    #[tokio::test]
    async fn capacity_errors_instead_of_growing_the_registry() {
        let registry = LocalTokenizerRegistry::with_capacity(1);
        registry.preload(config("first", None)).await.unwrap();

        assert!(registry.preload(config("second", None)).await.is_err());
        assert_eq!(registry.len().await, 1);
    }
}
