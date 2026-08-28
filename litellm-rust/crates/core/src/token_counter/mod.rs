mod counting;
pub(crate) mod encoding;
mod formatting;
mod hf_tokenizer;
#[cfg(test)]
mod tests;
pub mod types;

use crate::{CoreError, CoreResult};
use types::TokenCounterRequest;

pub fn token_counter(request: &TokenCounterRequest<'_>) -> CoreResult<usize> {
    if request.text.is_some() && request.messages.is_some() {
        return Err(CoreError::InvalidRequest(
            "text and messages cannot both be set".to_string(),
        ));
    }

    let tokenizer = encoding::resolve(request.model);

    if let Some(text) = request.text {
        if request.tools.is_some() || request.tool_choice.is_some() {
            return Err(CoreError::InvalidRequest(
                "tools or tool_choice cannot be set if using text".to_string(),
            ));
        }
        return Ok(tokenizer.count(text));
    }

    if let Some(messages) = request.messages {
        return counting::count_messages(
            &tokenizer,
            messages,
            request.tools,
            request.tool_choice,
            request.count_response_tokens,
            request.model,
            request.default_token_count,
        );
    }

    Err(CoreError::InvalidRequest(
        "Either text or messages must be provided".to_string(),
    ))
}
