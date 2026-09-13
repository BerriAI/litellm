mod deepseek;
mod mistral;

use crate::Error;
use crate::auth::InputSource;
use crate::auth::error::AuthConfigurationError;
use crate::ocr::error::OcrError;
use crate::ocr::types::OcrConnection;

pub(crate) use deepseek::VertexDeepSeekAdapter;
pub(crate) use mistral::VertexMistralAdapter;

fn validate_destination(connection: &OcrConnection) -> Result<(), OcrError> {
    if connection.api_base.is_some() && connection.api_base_source == InputSource::Request {
        return Err(Error::from(crate::AuthError::Configuration(
            AuthConfigurationError::RequestVertexCredentialDestination,
        ))
        .into());
    }
    Ok(())
}
