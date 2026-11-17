use litellm_gateway_auth::AuthenticatedRequest;
use std::{path::Path, sync::Arc};

use axum::{
    Json,
    extract::State,
    response::{IntoResponse, Response},
};
use base64::{Engine, engine::general_purpose::STANDARD};
use litellm_inference_transcription::{AudioTranscriptionRoute, types::AudioTranscriptionRequest};
use serde_json::{Value, json};

use crate::{
    Error, Gateway,
    request::InferenceBody,
    routing::{self, Target},
};

pub(crate) async fn create(
    State(gateway): State<Arc<Gateway>>,
    identity: AuthenticatedRequest,
    InferenceBody {
        fields: body,
        upload,
    }: InferenceBody,
) -> Result<Response, Error> {
    let audio = match upload {
        Some(upload) => {
            let format = upload
                .file_name
                .as_deref()
                .and_then(|name| Path::new(name).extension())
                .and_then(|extension| extension.to_str())
                .map(str::to_ascii_lowercase);
            json!({"data": STANDARD.encode(upload.bytes), "format": format})
        }
        None => body.get("audio").cloned().unwrap_or_default(),
    };
    let route = gateway.audio_transcription.clone();
    routing::call(
        gateway,
        &identity,
        "atranscription",
        "/audio/transcriptions",
        body,
        move |target| attempt(route.clone(), audio.clone(), target),
    )
    .await
}

async fn attempt(
    route: AudioTranscriptionRoute,
    audio: Value,
    target: Target,
) -> Result<Response, Error> {
    let api_key = target.api_key();
    let params = &target.deployment.params;
    let transcript = route
        .execute(AudioTranscriptionRequest {
            model: &params.model,
            audio,
            api_key: api_key.as_deref(),
            api_base: params.api_base.as_deref(),
            custom_llm_provider: params.custom_llm_provider.as_deref(),
            extra_headers: None,
            optional_params: target
                .body
                .iter()
                .filter(|(name, _)| !matches!(name.as_str(), "model" | "audio"))
                .map(|(name, value)| (name.clone(), value.clone()))
                .collect(),
            timeout: target.timeout,
        })
        .await?;
    Ok(Json(transcript).into_response())
}
