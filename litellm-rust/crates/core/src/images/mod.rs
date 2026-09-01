//! The images call, the Rust equivalent of Python's `litellm.images_generation()` and `litellm.images_edit()`.
//!
//! [`images_generation`] is the entrypoint for image generation: give it a model, prompt, and
//! credentials, and it resolves the provider, transforms the request, calls the provider, and
//! returns a typed response with image URLs or base64 data. [`images_edit`] is for image editing
//! with a mask.

mod client;
mod handler;
mod prepare;
pub mod transformation;
pub mod types;

use crate::error::CoreResult;

use handler::{execute_images_edit_provider_call, execute_images_generation_provider_call};
use prepare::{prepare_images_edit_call, prepare_images_generation_call};
use types::{
    ImagesEditRequest, ImagesEditResponse, ImagesGenerationRequest, ImagesGenerationResponse,
};

pub async fn images_generation(
    request: ImagesGenerationRequest<'_>,
) -> CoreResult<ImagesGenerationResponse> {
    execute_images_generation_provider_call(prepare_images_generation_call(request)?).await
}

pub async fn images_edit(request: ImagesEditRequest<'_>) -> CoreResult<ImagesEditResponse> {
    execute_images_edit_provider_call(prepare_images_edit_call(request)?).await
}

#[cfg(test)]
mod tests;
