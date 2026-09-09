//! Input token counting kept off the async worker threads.
//!
//! Large inputs are encoded on the blocking pool behind a semaphore, so a burst of 100K-token
//! requests can never stall the threads that accept and answer small requests.

use std::path::Path;
use std::sync::Arc;

use tokio::sync::Semaphore;

use crate::constants::{APPROX_BYTES_PER_TOKEN, TOKENIZE_INLINE_MAX_BYTES};

#[derive(Debug, thiserror::Error)]
pub enum TokenizerError {
    #[error("failed to load tokenizer: {0}")]
    Load(String),
    #[error("tokenization failed: {0}")]
    Encode(String),
}

#[derive(Clone)]
enum Backend {
    HuggingFace(Arc<tokenizers::Tokenizer>),
    Approximate,
}

/// Counts input tokens with a bounded number of concurrent encodes.
#[derive(Clone)]
pub struct TokenCounter {
    backend: Backend,
    permits: Arc<Semaphore>,
}

impl TokenCounter {
    /// Load a HuggingFace `tokenizer.json` (the proxy ships the Anthropic one under
    /// `litellm/litellm_core_utils/tokenizers/anthropic_tokenizer.json`).
    pub fn from_file(path: &Path, concurrency: usize) -> Result<Self, TokenizerError> {
        let tokenizer = tokenizers::Tokenizer::from_file(path)
            .map_err(|error| TokenizerError::Load(error.to_string()))?;
        Ok(Self {
            backend: Backend::HuggingFace(Arc::new(tokenizer)),
            permits: Arc::new(Semaphore::new(concurrency.max(1))),
        })
    }

    /// `len / APPROX_BYTES_PER_TOKEN`, for hosts without a tokenizer file.
    pub fn approximate() -> Self {
        Self {
            backend: Backend::Approximate,
            permits: Arc::new(Semaphore::new(1)),
        }
    }

    pub fn is_exact(&self) -> bool {
        matches!(self.backend, Backend::HuggingFace(_))
    }

    pub async fn count(&self, text: String) -> Result<usize, TokenizerError> {
        let tokenizer = match &self.backend {
            Backend::Approximate => return Ok(text.len().div_ceil(APPROX_BYTES_PER_TOKEN)),
            Backend::HuggingFace(tokenizer) => Arc::clone(tokenizer),
        };
        if text.len() <= TOKENIZE_INLINE_MAX_BYTES {
            return encode_len(&tokenizer, &text);
        }
        let _permit = self
            .permits
            .acquire()
            .await
            .map_err(|_| TokenizerError::Encode("tokenizer pool closed".to_string()))?;
        tokio::task::spawn_blocking(move || encode_len(&tokenizer, &text))
            .await
            .map_err(|error| TokenizerError::Encode(error.to_string()))?
    }
}

fn encode_len(tokenizer: &tokenizers::Tokenizer, text: &str) -> Result<usize, TokenizerError> {
    tokenizer
        .encode_fast(text, false)
        .map(|encoding| encoding.len())
        .map_err(|error| TokenizerError::Encode(error.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn approximate_counter_rounds_up() {
        let counter = TokenCounter::approximate();
        assert_eq!(counter.count("abcde".to_string()).await.unwrap(), 2);
        assert_eq!(counter.count(String::new()).await.unwrap(), 0);
    }

    #[tokio::test]
    async fn loads_anthropic_tokenizer_and_counts_off_thread() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../../litellm/litellm_core_utils/tokenizers/anthropic_tokenizer.json");
        let counter = TokenCounter::from_file(&path, 1).expect("tokenizer loads");
        assert!(counter.is_exact());
        let small = counter.count("hello world".to_string()).await.unwrap();
        assert!((1..=4).contains(&small), "got {small}");
        let large = "the quick brown fox ".repeat(2000);
        let large_len = large.len();
        let count = counter.count(large).await.unwrap();
        assert!(
            count > large_len / 8 && count < large_len / 2,
            "got {count}"
        );
    }
}
