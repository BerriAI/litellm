use std::{path::Path, sync::Arc};

use axum::{
    Json,
    extract::{Request, State},
    response::{IntoResponse, Response},
};
use base64::{Engine, engine::general_purpose::STANDARD};
use litellm_core::audio_transcription::{audio_transcription, types::AudioTranscriptionRequest};
use serde_json::{Value, json};

use crate::{Error, Gateway, request};

pub(crate) async fn create(State(gateway): State<Arc<Gateway>>, request: Request) -> Response {
    match handle(&gateway, request).await {
        Ok(response) => Json(response).into_response(),
        Err(error) => error.openai_response(),
    }
}

async fn handle(gateway: &Gateway, request: Request) -> Result<Value, Error> {
    let (body, upload) = request::parse(request).await?;
    let deployment = request::deployment(gateway, &body)?;
    let audio = match upload {
        Some(upload) => {
            let format = upload
                .file_name
                .as_deref()
                .and_then(|name| Path::new(name).extension())
                .and_then(|extension| extension.to_str())
                .ok_or_else(|| {
                    Error::InvalidBody("audio file requires a filename extension".into())
                })?;
            json!({"data": STANDARD.encode(upload.bytes), "format": format.to_ascii_lowercase()})
        }
        None => body
            .get("audio")
            .cloned()
            .ok_or_else(|| Error::InvalidBody("audio is required".into()))?,
    };
    Ok(audio_transcription(
        &gateway.resources,
        &gateway.http,
        AudioTranscriptionRequest {
            model: &deployment.model,
            audio,
            api_key: deployment.api_key.as_deref(),
            api_base: deployment.api_base.as_deref(),
            custom_llm_provider: deployment.custom_llm_provider.as_deref(),
            extra_headers: None,
            optional_params: body
                .into_iter()
                .filter(|(name, _)| !matches!(name.as_str(), "model" | "audio"))
                .collect(),
            timeout: deployment.timeout,
        },
    )
    .await?)
}
