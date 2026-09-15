use crate::Error;
use crate::ocr::error::OcrError;
use crate::ocr::types::OcrConnection;
use litellm_auth::InputSource;

pub(super) fn validate_destination(connection: &OcrConnection) -> Result<(), OcrError> {
    if connection.api_base.is_some() && connection.api_base_source == InputSource::Request {
        return Err(Error::from(crate::AuthError::RequestVertexCredentialDestination).into());
    }
    Ok(())
}
