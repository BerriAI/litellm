use litellm_auth::InputSource;
use litellm_auth_gcp::VertexConfig;

use crate::base_llm::ocr::{
    error::Error,
    transformation::{OcrConnection, PreparedOcrRequest},
};

pub(super) fn vertex_config(request: &PreparedOcrRequest) -> Result<VertexConfig, Error> {
    Ok(VertexConfig::from_sourced_optional_params(
        &request.optional_params,
        &request.input_sources,
    )?)
}

pub(super) fn validate_destination(connection: &OcrConnection) -> Result<(), Error> {
    if connection.api_base.is_some() && connection.api_base_source == InputSource::Request {
        return Err(litellm_auth::Error::InvalidConfiguration(
            "credentials cannot be sent to a request-controlled Vertex AI endpoint".into(),
        )
        .into());
    }
    Ok(())
}
