use crate::error::Error;
use crate::lifecycle::{
    ActionResult, CallLifecycleContext, RequestPolicy, TerminalDispatcher, TerminalRecord,
};
use crate::providers::reducto::ocr::transformation::{
    build_upload_request, extract_document_source, extract_upload_file_id,
};
use serde_json::{Map, Value, json};
use std::future::Future;
use std::pin::Pin;

use super::client::http_client;
use super::common_utils::{convert_document_url_to_data_uri, string_headers, truncate_error_body};
use super::runtime_types::{PreparedOcrRequest, ProviderOcrRequest};
use crate::integrations::custom_guardrail::{
    CustomGuardrailRunner, GuardrailContext, GuardrailError, GuardrailRequest,
};
use crate::integrations::custom_logger::{CallType, CustomLoggerRunner, LogFuture};
use crate::integrations::types::RequestMetadata;

pub(crate) struct OcrLifecycleHooks {
    logger_runner: CustomLoggerRunner,
    guardrail_runner: CustomGuardrailRunner,
    request_metadata: RequestMetadata,
}

type OcrFuture<'a, T> = Pin<Box<dyn Future<Output = ActionResult<T, Error>> + Send + 'a>>;

impl OcrLifecycleHooks {
    pub(crate) fn new(
        logger_runner: CustomLoggerRunner,
        guardrail_runner: CustomGuardrailRunner,
        request_metadata: RequestMetadata,
    ) -> Self {
        Self {
            logger_runner,
            guardrail_runner,
            request_metadata,
        }
    }

    async fn run_pre_call_guardrails(
        &self,
        request: PreparedOcrRequest,
    ) -> Result<PreparedOcrRequest, Error> {
        if self.guardrail_runner.is_empty() {
            return Ok(request);
        }

        let context = guardrail_context(&self.request_metadata);
        let guardrail_request = GuardrailRequest::new(json!({
            "model": request.model,
            "custom_llm_provider": request.custom_llm_provider,
            "document": request.document,
            "optional_params": request.optional_params,
        }));
        let (guardrail_request, _) = self
            .guardrail_runner
            .run_pre_call(&context, guardrail_request)
            .await
            .map_err(guardrail_error_to_core_error)?;
        let (document, optional_params) = parse_ocr_pre_call_guardrail_request(guardrail_request)?;
        let optional_params = match &request.config {
            Ok(config) => config.map_ocr_params(&optional_params),
            Err(_) => optional_params,
        };
        Ok(PreparedOcrRequest {
            document,
            optional_params,
            ..request
        })
    }

    pub(crate) async fn prepare_provider_request(
        &self,
        request: PreparedOcrRequest,
    ) -> Result<ProviderOcrRequest, Error> {
        let config = request.config?;
        let env_lookup = |key: &str| std::env::var(key).ok();
        let upstream_headers = config.validate_environment(
            string_headers(request.extra_headers)?,
            request.api_key.as_deref(),
            &env_lookup,
        )?;
        let url = config.complete_url(
            request.api_base.as_deref(),
            &request.model,
            &request.optional_params,
            &env_lookup,
        )?;
        let model = request.model.clone();
        let custom_llm_provider = request.custom_llm_provider.clone();
        let is_reducto = custom_llm_provider == "reducto";
        let document = if is_reducto {
            let guarded_document = self
                .run_during_call_guardrails(&model, &custom_llm_provider, &url, request.document)
                .await?;
            upload_reducto_document(
                &guarded_document,
                request.api_base.as_deref(),
                request.timeout,
                &upstream_headers,
            )
            .await?
        } else if config.requires_data_uri_document() {
            convert_document_url_to_data_uri(request.document).await?
        } else {
            request.document
        };
        let optional_params = request.optional_params;
        let body = config
            .transform_ocr_request(&request.model, document, optional_params.clone())?
            .data;
        let body = if is_reducto {
            body
        } else {
            self.run_during_call_guardrails(&model, &custom_llm_provider, &url, body)
                .await?
        };
        Ok(ProviderOcrRequest {
            model,
            config,
            url,
            body,
            optional_params,
            upstream_headers,
            timeout: request.timeout,
        })
    }

    async fn run_during_call_guardrails(
        &self,
        model: &str,
        custom_llm_provider: &str,
        url: &str,
        body: Value,
    ) -> Result<Value, Error> {
        if self.guardrail_runner.is_empty() {
            return Ok(body);
        }

        let context = guardrail_context(&self.request_metadata);
        let guardrail_request = GuardrailRequest::new(json!({
            "model": model,
            "custom_llm_provider": custom_llm_provider,
            "url": url,
            "body": body,
        }));
        let (guardrail_request, _) = self
            .guardrail_runner
            .run_during_call(&context, guardrail_request)
            .await
            .map_err(guardrail_error_to_core_error)?;
        parse_ocr_during_call_guardrail_request(guardrail_request)
    }

}

async fn upload_reducto_document(
    document: &Value,
    api_base: Option<&str>,
    timeout: Option<std::time::Duration>,
    upstream_headers: &[(String, String)],
) -> Result<Value, Error> {
    let source = extract_document_source(document)?;
    let Some(authorization) = upstream_headers
        .iter()
        .find(|(name, _)| name.eq_ignore_ascii_case("authorization"))
        .map(|(_, value)| value.as_str())
    else {
        return Err(Error::Auth(
            "Reducto upload requires an Authorization header".to_string(),
        ));
    };
    let Some(upload) = build_upload_request(source, authorization, api_base) else {
        return Ok(document.clone());
    };
    let part = reqwest::multipart::Part::bytes(upload.bytes)
        .file_name(upload.file_name)
        .mime_str(&upload.mime_type)
        .map_err(|error| Error::InvalidRequest(error.to_string()))?;
    let form = reqwest::multipart::Form::new().part("file", part);
    let mut request_builder = http_client().post(upload.url).multipart(form);
    for (name, value) in upstream_headers {
        if !name.eq_ignore_ascii_case("content-type")
            && !name.eq_ignore_ascii_case("content-length")
        {
            request_builder = request_builder.header(name, value);
        }
    }
    if let Some(timeout) = timeout {
        request_builder = request_builder.timeout(timeout);
    }
    let response = request_builder
        .send()
        .await
        .map_err(|error| Error::Network(error.to_string()))?;
    let status = response.status();
    let body = response
        .text()
        .await
        .map_err(|error| Error::Network(error.to_string()))?;
    if !status.is_success() {
        return Err(Error::Http {
            status: status.as_u16(),
            body: truncate_error_body(&body),
        });
    }
    let response_json: Value = serde_json::from_str(&body).map_err(|error| {
        Error::InvalidResponse(format!("invalid Reducto upload response JSON: {error}"))
    })?;
    let file_id = extract_upload_file_id(&response_json)?;
    Ok(json!({"type": "document_url", "document_url": file_id}))
}

impl RequestPolicy<PreparedOcrRequest, PreparedOcrRequest> for OcrLifecycleHooks {
    type PreCallFuture<'a> = OcrFuture<'a, PreparedOcrRequest>;
    type DuringCallFuture<'a> = OcrFuture<'a, PreparedOcrRequest>;

    fn async_pre_call_hook<'a>(
        &'a self,
        _context: &'a CallLifecycleContext,
        request: PreparedOcrRequest,
    ) -> Self::PreCallFuture<'a> {
        Box::pin(async move {
            match self.run_pre_call_guardrails(request).await {
                Ok(request) => ActionResult::Replace(request),
                Err(error) => ActionResult::Reject(error),
            }
        })
    }

    fn async_during_call_hook<'a>(
        &'a self,
        _context: &'a CallLifecycleContext,
        request: PreparedOcrRequest,
    ) -> Self::DuringCallFuture<'a> {
        Box::pin(async move { ActionResult::Continue(request) })
    }
}

impl TerminalDispatcher for OcrLifecycleHooks {
    fn dispatch<'a>(&'a self, terminal: &'a TerminalRecord) -> LogFuture<'a> {
        let mut terminal = terminal.clone();
        terminal.cost_inputs.metadata = request_metadata(&self.request_metadata);
        Box::pin(async move { self.logger_runner.dispatch(&terminal).await })
    }
}

fn request_metadata(metadata: &RequestMetadata) -> crate::integrations::types::StandardLoggingMetadata {
    crate::integrations::types::StandardLoggingMetadata {
        user_api_key_hash: metadata.user_api_key_hash.clone(),
        user_api_key_user_id: metadata.user_api_key_user_id.clone(),
        user_api_key_team_id: metadata.user_api_key_team_id.clone(),
        ..Default::default()
    }
}

fn guardrail_context(metadata: &RequestMetadata) -> GuardrailContext {
    GuardrailContext {
        call_type: CallType::Ocr,
        selected_guardrails: Vec::new(),
        metadata: std::collections::HashMap::new(),
        user_api_key_hash: metadata.user_api_key_hash.clone(),
        user_api_key_user_id: metadata.user_api_key_user_id.clone(),
        user_api_key_team_id: metadata.user_api_key_team_id.clone(),
        trace_parent: None,
    }
}

fn parse_ocr_pre_call_guardrail_request(
    request: GuardrailRequest,
) -> Result<(Value, Map<String, Value>), Error> {
    let Value::Object(mut data) = request.data else {
        return Err(Error::InvalidRequest(
            "OCR pre_call guardrail must return an object".to_string(),
        ));
    };
    let document = data.remove("document").ok_or_else(|| {
        Error::InvalidRequest("OCR pre_call guardrail removed document".to_string())
    })?;
    let optional_params = match data.remove("optional_params") {
        Some(Value::Object(params)) => params,
        Some(_) => {
            return Err(Error::InvalidRequest(
                "OCR pre_call guardrail optional_params must be an object".to_string(),
            ));
        }
        None => Map::new(),
    };
    Ok((document, optional_params))
}

fn parse_ocr_during_call_guardrail_request(request: GuardrailRequest) -> Result<Value, Error> {
    let Value::Object(mut data) = request.data else {
        return Err(Error::InvalidRequest(
            "OCR during_call guardrail must return an object".to_string(),
        ));
    };
    data.remove("body")
        .ok_or_else(|| Error::InvalidRequest("OCR during_call guardrail removed body".to_string()))
}

fn guardrail_error_to_core_error(error: GuardrailError) -> Error {
    Error::InvalidRequest(format!("{}: {}", error.kind, error.message))
}

fn core_error_kind(error: &Error) -> &'static str {
    match error {
        Error::Auth(_) => "AuthError",
        Error::InvalidProvider(_) => "InvalidProvider",
        Error::InvalidRequest(_) => "InvalidRequest",
        Error::InvalidType { .. } => "InvalidType",
        Error::MissingField(_) => "MissingField",
        Error::Http { .. } => "HttpError",
        Error::InvalidResponse(_) => "InvalidResponse",
        Error::Network(_) => "NetworkError",
        Error::Connect(_) => "ConnectError",
        Error::Routing(_) => "RoutingError",
        Error::Unsupported(_) => "UnsupportedRequest",
    }
}
