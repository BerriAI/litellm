use std::sync::Arc;

use super::OcrClient;
use super::backends::OcrBackend;
use super::formats::{OcrFormat, request_error};
use super::hooks::{OcrDuringCallRequest, OcrHooks, OcrLifecycleHooks};
use super::prepare::{PreparedOcrRequest, prepare_ocr_call};
use super::registry::{
    AZURE_DOCUMENT_INTELLIGENCE, AZURE_MISTRAL, MISTRAL, OcrIntegrationInput,
    OcrIntegrationRequest, REDUCTO_LEGACY, REDUCTO_V3, VERTEX_DEEPSEEK, VERTEX_MISTRAL,
};
use super::types::{OcrConnection, OcrDocument, OcrRequest, OcrResponseData};
use crate::Error;
use crate::call_lifecycle::{CallLifecycle, CallLifecycleContext};

struct OcrExecution<'a> {
    client: &'a OcrClient,
    model: String,
    document: OcrDocument,
    connection: OcrConnection,
    lifecycle_context: CallLifecycleContext,
    hooks: Arc<dyn OcrHooks>,
}

pub(crate) async fn perform_ocr_request(
    client: &OcrClient,
    request: OcrRequest,
) -> Result<OcrResponseData, Error> {
    let context = CallLifecycleContext::new(
        "ocr",
        request.model.clone(),
        request.integration.kind().provider().as_str(),
        request
            .litellm_call_id
            .unwrap_or_else(|| format!("ocr-{:032x}", rand::random::<u128>())),
    );
    let execution = OcrExecution {
        client,
        model: request.model,
        document: request.document,
        connection: request.connection,
        lifecycle_context: context,
        hooks: request.hooks,
    };
    match request.integration {
        OcrIntegrationRequest::Mistral(input) => execution.run(MISTRAL, input).await,
        OcrIntegrationRequest::AzureMistral(input) => execution.run(AZURE_MISTRAL, input).await,
        OcrIntegrationRequest::AzureDocumentIntelligence(input) => {
            execution.run(AZURE_DOCUMENT_INTELLIGENCE, input).await
        }
        OcrIntegrationRequest::VertexMistral(input) => execution.run(VERTEX_MISTRAL, input).await,
        OcrIntegrationRequest::VertexDeepSeek(input) => execution.run(VERTEX_DEEPSEEK, input).await,
        OcrIntegrationRequest::ReductoV3(input) => execution.run(REDUCTO_V3, input).await,
        OcrIntegrationRequest::ReductoLegacy(input) => execution.run(REDUCTO_LEGACY, input).await,
    }
}

impl OcrExecution<'_> {
    async fn run<F, B>(
        self,
        integration: super::registry::OcrIntegration<F, B>,
        input: OcrIntegrationInput<F::InputParams, B::Config>,
    ) -> Result<OcrResponseData, Error>
    where
        F: OcrFormat,
        B: OcrBackend<F>,
    {
        let Self {
            client,
            model,
            document,
            connection,
            lifecycle_context,
            hooks,
        } = self;
        let request = prepare_ocr_call(
            integration,
            model,
            document,
            input.params,
            input.backend_config,
            connection,
        );
        let lifecycle_hooks = OcrLifecycleHooks {
            hooks: hooks.clone(),
            provider_name: lifecycle_context.custom_llm_provider.clone(),
            marker: std::marker::PhantomData,
        };
        CallLifecycle::default()
            .run(lifecycle_context, request, &lifecycle_hooks, |request| {
                execute_ocr_provider_call(
                    client,
                    request,
                    hooks.as_ref(),
                    &lifecycle_hooks.provider_name,
                )
            })
            .await
    }
}

#[tracing::instrument(target = "litellm::function_trace", level = "trace", skip_all)]
async fn execute_ocr_provider_call<F, B>(
    client: &OcrClient,
    request: PreparedOcrRequest<F, B>,
    hooks: &dyn OcrHooks,
    provider_name: &str,
) -> Result<OcrResponseData, Error>
where
    F: OcrFormat,
    B: OcrBackend<F>,
{
    let backend = &request.integration.backend;
    let format = &request.integration.format;
    let prepared_backend = backend
        .prepare(
            &request.connection,
            &request.backend_config,
            &request.model,
            &request.params,
            &super::prepare::credential_env,
        )
        .await?;
    let url = prepared_backend.url;
    let headers = prepared_backend.headers;
    let document = if backend.guard_document_before_preparation() && hooks.has_guardrails() {
        let guarded = hooks
            .during_call(OcrDuringCallRequest {
                model: request.model.clone(),
                custom_llm_provider: provider_name.into(),
                url: url.clone(),
                body: serde_json::to_value(request.document)
                    .map_err(|_| request_error("document"))?,
            })
            .await?;
        super::wire::decode_request_value(guarded.body, "guardrail.body")?
    } else {
        request.document
    };
    let document = backend
        .prepare_document(client, document, &request.connection, &headers)
        .await?;
    let body = serde_json::to_value(format.transform_ocr_request(
        &request.model,
        document,
        &request.params,
    )?)
    .map_err(|_| request_error("body"))?;
    let body = if !backend.guard_document_before_preparation() && hooks.has_guardrails() {
        let guarded = hooks
            .during_call(OcrDuringCallRequest {
                model: request.model.clone(),
                custom_llm_provider: provider_name.into(),
                url: url.clone(),
                body,
            })
            .await?;
        guarded.body
    } else {
        body
    };
    let mut builder = client
        .provider_http()
        .post(&url)
        .json(&body)
        .timeout(request.connection.timeout);
    for (name, value) in &headers {
        builder = builder.header(name, value);
    }
    let response = crate::http_utils::http_request(builder)
        .await
        .map_err(crate::error::TransportError::from)?;
    let decoded = backend
        .read_response(
            client,
            response,
            &url,
            &headers,
            &request.connection,
            &request.params,
        )
        .await?;
    let response = format.transform_ocr_response(&request.model, decoded.data, &request.params)?;
    Ok(OcrResponseData {
        provider_native_response: decoded.native,
        ..response
    })
}
