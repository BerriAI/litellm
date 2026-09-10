use super::super::OcrAdapter;
use crate::Error;
use crate::auth::azure::AzureAuthInputs;
use crate::constants::{AZURE_DI_API_VERSION, AZURE_DI_SUBSCRIPTION_HEADER};
use crate::ocr::OcrClient;
use crate::ocr::codecs::document_intelligence::{
    self, AzureDocumentIntelligenceOperation, DocumentIntelligenceParams,
};
use crate::ocr::error::{OcrError, OcrRequestError, OcrResponseError};
use crate::ocr::prepare::{credential_env, transform_request_body};
use crate::ocr::registry::OcrProvider;
use crate::ocr::types::{
    LiteLLMOcrRequest, LiteLLMOcrResponse, OcrConnection, OcrResponseFormat,
};
use crate::ocr::wire::DecodedOcrResponse;
use crate::url_utils::ApiUrl;

mod polling;

const AZURE_DI_API_KEY_ENV: &str = "AZURE_DOCUMENT_INTELLIGENCE_API_KEY";
const AZURE_DI_ENDPOINT_ENV: &str = "AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT";

#[derive(Clone, Debug)]
pub(crate) struct AzureDocumentIntelligenceAdapter;

impl OcrAdapter for AzureDocumentIntelligenceAdapter {
    type ProviderResponse = AzureDocumentIntelligenceOperation;
    const PROVIDER: OcrProvider = OcrProvider::AzureAi;
    const SUPPORTS_NATIVE_RESPONSE: bool = true;

    #[tracing::instrument(target = "litellm::function_trace", level = "trace", skip_all)]
    async fn transform_ocr_request(
        &self,
        request: &LiteLLMOcrRequest,
        client: &OcrClient,
    ) -> Result<reqwest::Request, OcrError> {
        let params = map_ocr_params(request)?;
        let config =
            AzureAuthInputs::from_optional_params(&request.optional_params).map_err(Error::from)?;
        let headers = validate_environment(&request.connection, &config, &credential_env).await?;
        let endpoint = nonblank(request.connection.api_base.clone())
            .or_else(|| nonblank(credential_env(AZURE_DI_ENDPOINT_ENV)))
            .ok_or_else(|| Error::Auth("Missing Azure Document Intelligence API Base - Set AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT or pass api_base".into()))?;
        let url = get_complete_url(&endpoint, &request.model, &params)?;
        let body = document_intelligence::transform_ocr_request(request.document.clone())?;
        transform_request_body(client, request, &url, &headers, body, |_| Ok(())).await
    }

    fn transform_ocr_response(
        &self,
        request: &LiteLLMOcrRequest,
        response: Self::ProviderResponse,
    ) -> Result<LiteLLMOcrResponse, OcrResponseError> {
        document_intelligence::transform_ocr_response(&request.model, response)
    }

    async fn read_response(
        &self,
        client: &OcrClient,
        response: reqwest::Response,
        url: &str,
        headers: &[(String, String)],
        request: &LiteLLMOcrRequest,
    ) -> Result<DecodedOcrResponse<Self::ProviderResponse>, OcrError> {
        polling::read_operation_response(
            client.provider_http(),
            response,
            url,
            headers,
            &request.connection,
            request.response_format()? == OcrResponseFormat::Native,
        )
        .await
    }
}

#[tracing::instrument(target = "litellm::function_trace", level = "trace", skip_all)]
fn map_ocr_params(
    request: &LiteLLMOcrRequest,
) -> Result<DocumentIntelligenceParams, OcrRequestError> {
    let params = document_intelligence::decode_input_params(
        request.optional_params.clone(),
        "optional_params",
    )?;
    document_intelligence::map_ocr_params(params)
}

fn get_complete_url(
    endpoint: &str,
    model: &str,
    params: &DocumentIntelligenceParams,
) -> Result<String, OcrError> {
    let model = format!("{}:analyze", model_id(model)?);
    ApiUrl::parse(endpoint)
        .and_then(|url| url.complete_path(&["documentintelligence", "documentModels", &model]))
        .map(|url| {
            url.append_query_pairs(
                [("api-version", AZURE_DI_API_VERSION)]
                    .into_iter()
                    .chain(params.pages.iter().map(|pages| ("pages", pages.as_str())))
                    .chain(
                        params
                            .features
                            .iter()
                            .map(|features| ("features", features.as_str())),
                    ),
            )
            .into_string()
        })
        .map_err(|_| OcrRequestError::RequestField {
            path: "api_base".into(),
        })
        .map_err(OcrError::from)
}

async fn validate_environment(
    connection: &OcrConnection,
    config: &AzureAuthInputs,
    env_lookup: &(dyn Fn(&str) -> Option<String> + Sync),
) -> Result<Vec<(String, String)>, OcrError> {
    if crate::http_utils::has_header(&connection.extra_headers, "authorization")
        || crate::http_utils::has_header(&connection.extra_headers, AZURE_DI_SUBSCRIPTION_HEADER)
    {
        return Ok(connection.extra_headers.clone());
    }
    if let Some(key) =
        nonblank(connection.api_key.clone()).or_else(|| nonblank(env_lookup(AZURE_DI_API_KEY_ENV)))
    {
        return Ok(std::iter::once((AZURE_DI_SUBSCRIPTION_HEADER.into(), key))
            .chain(connection.extra_headers.clone())
            .collect());
    }
    let token = super::resolve_entra(config, env_lookup)
        .await?
        .ok_or(Error::MissingAzureDocumentIntelligenceCredentials)?;
    Ok(
        std::iter::once(("Authorization".into(), format!("Bearer {token}")))
            .chain(connection.extra_headers.clone())
            .collect(),
    )
}

fn model_id(model: &str) -> Result<&str, OcrRequestError> {
    let model = model.rsplit('/').next().unwrap_or(model);
    if matches!(model, "." | "..") {
        return Err(OcrRequestError::DotModel);
    }
    Ok(model)
}

fn nonblank(value: Option<String>) -> Option<String> {
    value
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
}
