use std::sync::Arc;

use axum::{
    extract::{Path, State},
    http::StatusCode,
    response::{IntoResponse, Response},
};

use crate::{Error, Gateway, JsonObject, chat_completions};

pub(crate) async fn dispatch(
    State(gateway): State<Arc<Gateway>>,
    Path(path): Path<String>,
    body: Result<JsonObject, Error>,
) -> Result<Response, Error> {
    if let Some(model) = path
        .strip_suffix("/chat/completions")
        .filter(|model| !model.is_empty())
    {
        return chat_completions::create_from_model_path(&gateway, model, body?).await;
    }
    if path.ends_with("/embeddings") || path.ends_with("/completions") {
        return Err(Error::Unsupported(path));
    }
    Ok(StatusCode::NOT_FOUND.into_response())
}
