//! The embeddings call, the Rust equivalent of Python's `litellm.embeddings()`.
//!
//! [`embeddings`] is the top-level entrypoint: give it a model, input text, and
//! credentials, and it resolves the provider, transforms the request, calls the
//! provider, and returns a typed response with embedding vectors.

mod client;
mod handler;
mod prepare;
pub mod transformation;
pub mod types;

use crate::error::CoreResult;

use handler::execute_embeddings_provider_call;
use prepare::prepare_embeddings_call;
use types::{EmbeddingsRequest, EmbeddingsResponse};

pub async fn embeddings(request: EmbeddingsRequest<'_>) -> CoreResult<EmbeddingsResponse> {
    execute_embeddings_provider_call(prepare_embeddings_call(request)?).await
}

#[cfg(test)]
mod tests;
