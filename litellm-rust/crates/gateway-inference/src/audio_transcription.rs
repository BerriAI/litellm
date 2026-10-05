use litellm_gateway_auth::AuthenticatedRequest;
use std::{path::Path, sync::Arc};

use axum::{Json, extract::State, response::IntoResponse};
use base64::{Engine, engine::general_purpose::STANDARD};
use litellm_core::audio_transcription::types::AudioTranscriptionRequest;
use serde_json::{Value, json};

use crate::{
    Error, Gateway,
    request::{self, InferenceBody},
};

pub(crate) async fn create(
    State(gateway): State<Arc<Gateway>>,
    identity: AuthenticatedRequest,
    body: InferenceBody,
) -> Result<impl IntoResponse, Error> {
    handle(&gateway, &identity, body).await.map(Json)
}

async fn handle(
    gateway: &Gateway,
    identity: &AuthenticatedRequest,
    InferenceBody {
        fields: body,
        upload,
    }: InferenceBody,
) -> Result<Value, Error> {
    let deployment = request::resolve_deployment(gateway, &body)?;
    request::authorize_model(identity, deployment, &body).await?;
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
    Ok(gateway
        .audio_transcription
        .execute(AudioTranscriptionRequest {
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
        })
        .await?)
}
