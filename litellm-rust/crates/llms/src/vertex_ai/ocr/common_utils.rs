use litellm_auth::InputSource;
use litellm_auth_gcp::VertexConfig;

use crate::base_llm::ocr::{
    error::Error,
    transformation::{OcrConnection, PreparedOcrRequest},
};
use crate::vertex_ai::connection::VertexConnection;

pub(super) fn vertex_connection(request: &PreparedOcrRequest) -> Result<VertexConnection, Error> {
    let config = VertexConfig::from_sourced_optional_params(
        &request.optional_params,
        &request.input_sources,
    )?;
    Ok(VertexConnection::resolve(
        config,
        request.connection.api_base.as_deref(),
        &request.runtime_defaults,
        &|name: &str| request.connection.secret(name),
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
