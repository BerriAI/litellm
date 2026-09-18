use litellm_auth::InputSource;

use crate::ocr::types::OcrConnection;

pub(super) fn validate_destination(connection: &OcrConnection) -> Result<(), crate::ocr::Error> {
    if connection.api_base.is_some() && connection.api_base_source == InputSource::Request {
        return Err(litellm_auth::Error::RequestVertexCredentialDestination.into());
    }
    Ok(())
}
