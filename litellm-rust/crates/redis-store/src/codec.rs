use litellm_python_compat::{Value, json, literal};

#[derive(Debug, thiserror::Error)]
pub enum DecodeError {
    #[error("Redis value is not UTF-8")]
    NotUtf8,
    #[error("Redis value is neither JSON nor a Python literal: {0}")]
    Unparsed(#[source] litellm_python_compat::Error),
    #[error("Redis value has the wrong shape: {0}")]
    Shape(&'static str),
}

/// `RedisCache._get_cache_logic`: `json.loads`, falling back to `ast.literal_eval`.
pub(crate) fn python_value(raw: &[u8]) -> Result<Value, DecodeError> {
    let text = std::str::from_utf8(raw).map_err(|_| DecodeError::NotUtf8)?;
    match serde_json::from_str::<serde_json::Value>(text) {
        Ok(document) => Ok(json::from_json(document)),
        Err(_) => literal::literal_eval(text).map_err(DecodeError::Unparsed),
    }
}
