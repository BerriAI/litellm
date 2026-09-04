use litellm_core::Error;

pub(crate) fn invalid_json_response(operation: &'static str, error: serde_json::Error) -> Error {
    Error::InvalidResponse(format!("invalid {operation} response JSON: {error}"))
}
