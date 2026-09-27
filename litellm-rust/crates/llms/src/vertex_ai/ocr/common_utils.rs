use litellm_auth::InputSource;
use litellm_auth_gcp::VertexConfig;

use crate::base_llm::ocr::{
    error::Error,
    transformation::{OcrConnection, PreparedOcrRequest},
};

pub(super) fn vertex_config(request: &PreparedOcrRequest) -> Result<VertexConfig, Error> {
    let settings = &request.connection.settings;
    Ok(VertexConfig::from_sourced_optional_params(
        &request.optional_params,
        &request.input_sources,
    )?
    .or_configured(
        settings.vertex_project.as_deref(),
        settings.vertex_location.as_deref(),
    ))
}

pub(super) fn validate_destination(connection: &OcrConnection) -> Result<(), Error> {
    if connection.api_base.is_some() && connection.api_base_source == InputSource::Request {
        return Err(litellm_auth::Error::RequestVertexCredentialDestination.into());
    }
    Ok(())
}
