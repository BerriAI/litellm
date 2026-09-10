use std::future::Future;

use serde_json::{Map, Value};

use super::OcrClient;
use super::backends::{OcrBackend, PreparedOcrBackend};
use super::error::{OcrError, OcrRequestError};
use super::formats::OcrFormat;
use super::types::{OcrConnection, OcrDocument, OcrRequestFormat};
use super::wire::DecodedOcrResponse;

mod azure_document_intelligence;
mod azure_mistral;
mod mistral_direct;
mod reducto_legacy;
mod reducto_v3;
mod vertex_deepseek;
mod vertex_mistral;

pub use super::backends::reducto::ReductoUpload;
pub use super::document::{DocumentPreparation, PassThrough, RequireInline};
pub use azure_document_intelligence::AzureDocumentIntelligence;
pub use azure_mistral::AzureMistral;
pub use mistral_direct::MistralDirect;
pub use reducto_legacy::ReductoLegacy;
pub use reducto_v3::ReductoV3;
pub use vertex_deepseek::VertexDeepSeek;
pub use vertex_mistral::VertexMistral;

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum GuardrailStage {
    Document,
    RequestBody,
}

pub trait OcrIntegration: Send + Sync + Sized + 'static {
    type Backend: OcrBackend;
    type Format: OcrFormat;
    type DocumentPreparation: DocumentPreparation<
        Output: Into<<Self::Format as OcrFormat>::PreparedDocument>,
    >;
    const GUARDRAIL_STAGE: GuardrailStage = GuardrailStage::RequestBody;

    fn supports_native_request_format(&self) -> bool {
        false
    }

    fn decode_input_params(
        &self,
        params: Map<String, Value>,
        prefix: &str,
    ) -> Result<InputParams<Self>, OcrRequestError> {
        validate_request_format(
            &params,
            self.supports_native_request_format(),
            Self::Backend::PROVIDER.as_str(),
        )?;
        Self::Format::validate_input_params(&params)?;
        super::wire::decode_request_value(Value::Object(params), prefix)
    }

    fn prepare(
        &self,
        connection: &OcrConnection,
        config: &BackendConfig<Self>,
        model: &str,
        params: &MappedParams<Self>,
        env_lookup: &(dyn Fn(&str) -> Option<String> + Sync),
    ) -> impl Future<Output = Result<PreparedOcrBackend, OcrError>> + Send;

    fn prepare_document(
        &self,
        client: &OcrClient,
        document: OcrDocument,
        connection: &OcrConnection,
        headers: &[(String, String)],
    ) -> impl Future<
        Output = Result<<Self::DocumentPreparation as DocumentPreparation>::Output, OcrError>,
    > + Send {
        Self::DocumentPreparation::prepare(client, document, connection, headers)
    }

    fn validate_request_body(
        &self,
        _body: &<Self::Format as OcrFormat>::RequestBody,
    ) -> Result<(), OcrRequestError> {
        Ok(())
    }

    fn preserve_native_response(&self, _params: &MappedParams<Self>) -> bool {
        false
    }

    fn read_response(
        &self,
        _client: &OcrClient,
        response: reqwest::Response,
        _url: &str,
        _headers: &[(String, String)],
        _connection: &OcrConnection,
        params: &MappedParams<Self>,
    ) -> impl Future<
        Output = Result<DecodedOcrResponse<<Self::Format as OcrFormat>::ResponseBody>, OcrError>,
    > + Send {
        super::client::read_json_response(response, self.preserve_native_response(params))
    }
}

fn validate_request_format(
    params: &Map<String, Value>,
    supports_native: bool,
    provider: &'static str,
) -> Result<(), OcrRequestError> {
    let Some(format) = params.get("req_format") else {
        return Ok(());
    };
    let format: OcrRequestFormat =
        serde_json::from_value(format.clone()).map_err(|_| OcrRequestError::RequestFormat)?;
    if format == OcrRequestFormat::Native && !supports_native {
        return Err(OcrRequestError::NativeUnsupported(provider));
    }
    Ok(())
}

pub type InputParams<I> = <<I as OcrIntegration>::Format as OcrFormat>::InputParams;
pub type MappedParams<I> = <<I as OcrIntegration>::Format as OcrFormat>::MappedParams;
pub type BackendConfig<I> = <<I as OcrIntegration>::Backend as OcrBackend>::Config;

macro_rules! for_each_ocr_integration {
    ($callback:ident) => {
        $callback! {
            Mistral, $crate::ocr::integrations::MistralDirect, $crate::ocr::integrations::MistralDirect, Mistral;
            AzureMistral, $crate::ocr::integrations::AzureMistral, $crate::ocr::integrations::AzureMistral, AzureAi;
            AzureDocumentIntelligence, $crate::ocr::integrations::AzureDocumentIntelligence, $crate::ocr::integrations::AzureDocumentIntelligence, AzureAi;
            VertexMistral, $crate::ocr::integrations::VertexMistral, $crate::ocr::integrations::VertexMistral, VertexAi;
            VertexDeepSeek, $crate::ocr::integrations::VertexDeepSeek, $crate::ocr::integrations::VertexDeepSeek, VertexAi;
            ReductoV3, $crate::ocr::integrations::ReductoV3, $crate::ocr::integrations::ReductoV3, Reducto;
            ReductoLegacy, $crate::ocr::integrations::ReductoLegacy, $crate::ocr::integrations::ReductoLegacy, Reducto;
        }
    };
}

pub(crate) use for_each_ocr_integration;
