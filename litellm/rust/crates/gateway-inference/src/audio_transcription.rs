use litellm_gateway_auth::AuthenticatedRequest;
use std::{path::Path, sync::Arc, time::Duration};

use axum::{Json, body::Bytes, extract::State, http::request::Parts, response::IntoResponse};
use base64::{Engine, engine::general_purpose::STANDARD};
use litellm_inference_transcription::types::AudioTranscriptionRequest;
use serde_json::{Map, Value, json};

use crate::{
    Error, Gateway,
    admit::{Planned, Refusal},
    request::{self, InferenceBody},
};

pub(crate) struct Call {
    model: String,
    audio: Value,
    api_key: Option<String>,
    api_base: Option<String>,
    custom_llm_provider: Option<String>,
    optional_params: Map<String, Value>,
    timeout: Option<Duration>,
}

pub(crate) async fn plan(
    gateway: &Gateway,
    identity: &AuthenticatedRequest,
    parts: &Parts,
    body: Bytes,
) -> Result<Call, Refusal> {
    let InferenceBody {
        fields: body,
        upload,
    } = request::inference_body(parts, body).await?;
    let deployment = request::resolve_deployment(gateway, "/audio/transcriptions", &body)?;
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
    Ok(Call {
        model: deployment.model.clone(),
        audio,
        api_key: deployment.api_key.clone(),
        api_base: deployment.api_base.clone(),
        custom_llm_provider: deployment.custom_llm_provider.clone(),
        optional_params: body
            .into_iter()
            .filter(|(name, _)| !matches!(name.as_str(), "model" | "audio"))
            .collect(),
        timeout: deployment.timeout,
    })
}

pub(crate) async fn create(
    State(gateway): State<Arc<Gateway>>,
    Planned(call): Planned<Call>,
) -> Result<impl IntoResponse, Error> {
    Ok(Json(
        gateway
            .audio_transcription
            .execute(AudioTranscriptionRequest {
                model: &call.model,
                audio: call.audio,
                api_key: call.api_key.as_deref(),
                api_base: call.api_base.as_deref(),
                custom_llm_provider: call.custom_llm_provider.as_deref(),
                extra_headers: None,
                optional_params: call.optional_params,
                timeout: call.timeout,
            })
            .await?,
    ))
}
