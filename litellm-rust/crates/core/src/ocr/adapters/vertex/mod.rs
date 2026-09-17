mod deepseek;
mod mistral;

use crate::ocr::Error;
use litellm_auth::InputSource;

use crate::ocr::error::OcrError;
use crate::ocr::types::OcrConnection;

pub(crate) use deepseek::VertexDeepSeekAdapter;
pub(crate) use mistral::VertexMistralAdapter;

fn validate_destination(connection: &OcrConnection) -> Result<(), OcrError> {
    if connection.api_base.is_some() && connection.api_base_source == InputSource::Request {
        return Err(Error::from(litellm_auth::Error::RequestVertexCredentialDestination).into());
    }
    Ok(())
}
