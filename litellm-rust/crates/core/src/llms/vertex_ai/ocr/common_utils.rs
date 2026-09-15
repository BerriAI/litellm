use crate::ocr::Error;
use crate::ocr::types::OcrConnection;
use litellm_auth::InputSource;

pub(super) fn validate_destination(connection: &OcrConnection) -> Result<(), Error> {
    if connection.api_base.is_some() && connection.api_base_source == InputSource::Request {
        return Err(litellm_auth::Error::RequestVertexCredentialDestination.into());
    }
    Ok(())
}
