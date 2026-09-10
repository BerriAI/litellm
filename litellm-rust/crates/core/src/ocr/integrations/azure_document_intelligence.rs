use crate::constants::AZURE_DI_API_VERSION;
use crate::ocr::backends::PreparedOcrBackend;
use crate::ocr::backends::azure_ai::{self, AzureBackend};
use crate::ocr::error::{OcrError, OcrRequestError};
use crate::ocr::formats::document_intelligence::{
    AzureDocumentIntelligenceOcrFormat,
    types::{AzureDocumentIntelligenceOperation, DocumentIntelligenceParams},
};
use crate::ocr::types::{OcrConnection, OcrRequestFormat};
use crate::ocr::wire::DecodedOcrResponse;
use crate::url_utils::ApiUrl;

use super::{BackendConfig, OcrIntegration};

#[derive(Clone, Debug)]
pub struct AzureDocumentIntelligence;

impl OcrIntegration for AzureDocumentIntelligence {
    type Backend = AzureBackend;
    type Format = AzureDocumentIntelligenceOcrFormat;
    type DocumentPreparation = super::PassThrough;
    const FORMAT: Self::Format = AzureDocumentIntelligenceOcrFormat;

    async fn prepare(
        &self,
        connection: &OcrConnection,
        config: &BackendConfig<Self>,
        model: &str,
        params: &DocumentIntelligenceParams,
        env_lookup: &(dyn Fn(&str) -> Option<String> + Sync),
    ) -> Result<PreparedOcrBackend, OcrError> {
        let (endpoint, headers) =
            azure_ai::prepare_document_intelligence_auth(connection, config, env_lookup).await?;
        let model = format!("{}:analyze", model_id(model)?);
        let url = ApiUrl::parse(&endpoint)
            .and_then(|url| url.complete_path(&["documentintelligence", "documentModels", &model]))
            .map(|url| {
                url.append_query_pairs(
                    [("api-version", AZURE_DI_API_VERSION)]
                        .into_iter()
                        .chain(params.pages.iter().map(|pages| ("pages", pages.0.as_str())))
                        .chain(
                            params
                                .features
                                .iter()
                                .map(|features| ("features", features.0.as_str())),
                        ),
                )
                .into_string()
            })
            .map_err(|_| OcrRequestError::RequestField {
                path: "api_base".into(),
            })?;
        Ok(PreparedOcrBackend { url, headers })
    }

    fn preserve_native_response(&self, params: &DocumentIntelligenceParams) -> bool {
        params.request_format == OcrRequestFormat::Native
    }

    fn supports_native_request_format(&self) -> bool {
        true
    }

    async fn read_response(
        &self,
        client: &crate::ocr::OcrClient,
        response: reqwest::Response,
        url: &str,
        headers: &[(String, String)],
        connection: &OcrConnection,
        params: &DocumentIntelligenceParams,
    ) -> Result<DecodedOcrResponse<AzureDocumentIntelligenceOperation>, OcrError> {
        crate::ocr::backends::azure_document_intelligence::polling::read_operation_response(
            client.provider_http(),
            response,
            url,
            headers,
            connection,
            params.request_format == OcrRequestFormat::Native,
        )
        .await
    }
}

fn model_id(model: &str) -> Result<&str, OcrRequestError> {
    let model = model.rsplit('/').next().unwrap_or(model);
    if matches!(model, "." | "..") {
        return Err(OcrRequestError::DotModel);
    }
    Ok(model)
}
