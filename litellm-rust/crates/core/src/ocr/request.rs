pub(super) use crate::providers::dispatch::ocr_provider_config as provider_config;

use std::time::Duration;

use serde_json::{Map, Value};

use crate::Error;
use crate::routing_utils::provider::{CustomLlmProvider, get_custom_llm_provider};

use super::transformation::{OcrProviderConfig, OcrResponseHandling};
use super::types::{OcrAdmissionRequest, OcrEndpoint, OcrPreCallRequest};

fn request_config(
    request: &OcrAdmissionRequest,
) -> Result<(CustomLlmProvider<'_>, &'static dyn OcrProviderConfig), Error> {
    match request.request_format.as_deref() {
        None | Some("litellm") => {}
        Some("native") => return Err(Error::Unsupported("native OCR request format")),
        Some(_) => {
            return Err(Error::Unsupported(
                "OCR request format must be litellm or native",
            ));
        }
    }
    let provider = get_custom_llm_provider(&request.model, request.custom_llm_provider.as_deref())
        .ok_or_else(|| Error::InvalidProvider("unable to resolve OCR provider".into()))?;
    let config = provider_config(provider.custom_llm_provider, provider.model)?;
    if provider.model.trim().is_empty() {
        return Err(Error::InvalidRequest("OCR model must not be empty".into()));
    }
    Duration::try_from_secs_f64(request.timeout_seconds)
        .ok()
        .filter(|timeout| !timeout.is_zero())
        .ok_or_else(|| Error::InvalidRequest("timeout must be positive and finite".into()))?;

    Ok((provider, config))
}

pub(crate) fn admission_capabilities(request: &OcrAdmissionRequest) -> Result<(), Error> {
    check_admission_capabilities(request, &|key| std::env::var(key).ok())
}

fn check_admission_capabilities(
    request: &OcrAdmissionRequest,
    env_lookup: &dyn Fn(&str) -> Option<String>,
) -> Result<(), Error> {
    let (_, config) = request_config(request)?;
    validate_capabilities(config)?;
    request
        .document
        .validate(config.requires_data_uri_document())?;
    if request.stream {
        return Err(Error::Unsupported("OCR streaming response handling"));
    }
    config.check_auth_capabilities(request, env_lookup)?;
    if let Some(operation) = config.credential_acquisition_operation() {
        let supplied = request
            .api_key
            .as_deref()
            .is_some_and(|key| !key.trim().is_empty())
            || crate::http_utils::has_header(&request.extra_headers, "authorization");
        let configured = config.has_configured_credentials(request, env_lookup);
        if !supplied && !configured {
            return Err(Error::Unsupported(operation));
        }
    }
    Ok(())
}

#[cfg(test)]
pub(crate) fn build_pre_call_request(
    request: OcrAdmissionRequest,
) -> Result<OcrPreCallRequest, Error> {
    build_pre_call_request_with_auth(request, &())
}

pub(crate) fn build_pre_call_request_with_auth(
    request: OcrAdmissionRequest,
    auth: &dyn litellm_auth::AuthServices,
) -> Result<OcrPreCallRequest, Error> {
    let (provider, config) = request_config(&request)?;
    let env_lookup = |key: &str| std::env::var(key).ok();
    check_admission_capabilities(&request, &env_lookup)?;
    let headers = config
        .prepare_credentials(&request, auth, &env_lookup)
        .map_err(
            |error| match (error, config.credential_acquisition_operation()) {
                (Error::Auth(_), Some(operation)) => Error::Unsupported(operation),
                (error, _) => error,
            },
        )?;
    validate_capabilities(config)?;
    request
        .document
        .validate(config.requires_data_uri_document())?;
    if request.stream {
        return Err(Error::Unsupported("OCR streaming response handling"));
    }
    let url_params = [
        ("vertex_project", request.vertex_project.as_ref()),
        ("vertex_location", request.vertex_location.as_ref()),
    ]
    .into_iter()
    .filter_map(|(name, value)| value.map(|value| (name.into(), Value::String(value.clone()))))
    .collect();
    let url = config.complete_url(
        request.api_base.as_deref(),
        provider.model,
        &url_params,
        &env_lookup,
    )?;
    let parsed_url = reqwest::Url::parse(&url)
        .map_err(|_| Error::InvalidRequest("invalid OCR API URL".into()))?;
    if !matches!(parsed_url.scheme(), "http" | "https") || parsed_url.host_str().is_none() {
        return Err(Error::InvalidRequest("invalid OCR API URL".into()));
    }
    let template =
        config.transform_ocr_request(provider.model, request.document.clone(), Map::new())?;
    let body_policy = config.request_body_policy();
    if body_policy != crate::lifecycle::RequestBodyPolicy::StructuredAtSend {
        return Err(Error::Unsupported("OCR request body policy"));
    }
    Ok(OcrPreCallRequest {
        endpoint: OcrEndpoint {
            model: provider.model.to_string(),
            custom_llm_provider: provider.custom_llm_provider.to_string(),
            url,
            timeout_seconds: request.timeout_seconds,
        },
        headers,
        body: crate::lifecycle::PreCallBody::StructuredAtSend {
            callback: template.data,
        },
        document_projection: config.document_projection(),
        parameter_fields: config.supported_ocr_params(),
    })
}

pub(super) fn validate_capabilities(config: &dyn OcrProviderConfig) -> Result<(), Error> {
    match config.response_handling() {
        OcrResponseHandling::Json => Ok(()),
        OcrResponseHandling::AzureDocumentIntelligencePoll => Err(Error::Unsupported(
            "Azure Document Intelligence OCR polling",
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ocr::types::OcrDocument;
    use crate::providers::azure_ai::ocr::transformation::AZURE_AI_OCR_CONFIG;
    use crate::providers::mistral::ocr::transformation::MISTRAL_OCR_CONFIG;
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct CountingTokenProvider(AtomicUsize);

    impl litellm_auth::CallerTokenProvider for CountingTokenProvider {
        fn invoke(
            &self,
        ) -> Result<Option<litellm_auth::SecretString>, litellm_auth::AuthServiceError> {
            let count = self.0.fetch_add(1, Ordering::SeqCst) + 1;
            Ok(Some(litellm_auth::SecretString::new(format!(
                "token-{count}"
            ))))
        }
    }

    impl litellm_auth::AuthServices for CountingTokenProvider {
        fn token_provider(
            &self,
            _: litellm_auth::CallerCredential,
        ) -> Option<&dyn litellm_auth::CallerTokenProvider> {
            Some(self)
        }
    }

    fn azure_request() -> OcrAdmissionRequest {
        OcrAdmissionRequest {
            model: "azure_ai/mistral-ocr-latest".into(),
            api_key: None,
            api_base: Some("https://example.test".into()),
            credentials: litellm_auth::CredentialInputs {
                azure: litellm_auth::AzureCredentialInputs {
                    has_token_provider: true,
                    ..Default::default()
                },
            },
            document: OcrDocument::ImageUrl {
                image_url: "data:image/png;base64,AA==".into(),
            },
            ..request()
        }
    }

    #[test]
    fn azure_caller_token_is_acquired_only_during_auth_and_never_cached() {
        let provider = CountingTokenProvider(AtomicUsize::new(0));
        let request = azure_request();
        check_admission_capabilities(&request, &|_| None).unwrap();
        assert_eq!(provider.0.load(Ordering::SeqCst), 0);
        for count in 1..=2 {
            let headers = AZURE_AI_OCR_CONFIG
                .prepare_credentials(&request, &provider, &|_| None)
                .unwrap();
            assert_eq!(
                headers,
                vec![("Authorization".into(), format!("Bearer token-{count}"))]
            );
        }
        assert_eq!(provider.0.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn azure_key_and_missing_endpoint_do_not_invoke_caller() {
        let provider = CountingTokenProvider(AtomicUsize::new(0));
        let request = OcrAdmissionRequest {
            api_key: Some(" key ".into()),
            ..azure_request()
        };
        assert_eq!(
            AZURE_AI_OCR_CONFIG
                .prepare_credentials(&request, &provider, &|_| None)
                .unwrap(),
            vec![("Authorization".into(), "Bearer  key ".into())]
        );
        let request = OcrAdmissionRequest {
            api_base: None,
            ..azure_request()
        };
        assert!(matches!(
            AZURE_AI_OCR_CONFIG.prepare_credentials(&request, &provider, &|_| None),
            Err(Error::InvalidRequest(_))
        ));
        assert_eq!(provider.0.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn azure_oidc_with_caller_requires_native_exchange_before_admission() {
        let request = OcrAdmissionRequest {
            credentials: litellm_auth::CredentialInputs {
                azure: litellm_auth::AzureCredentialInputs {
                    token: Some("oidc/assertion".into()),
                    has_token_provider: true,
                    client_id: Some("client".into()),
                    tenant_id: Some("tenant".into()),
                    ..Default::default()
                },
            },
            ..azure_request()
        };
        assert!(matches!(
            check_admission_capabilities(&request, &|_| None),
            Err(Error::Unsupported(_))
        ));
        let request = OcrAdmissionRequest {
            api_key: Some("key".into()),
            ..request
        };
        assert!(check_admission_capabilities(&request, &|_| None).is_ok());
    }

    fn request() -> OcrAdmissionRequest {
        OcrAdmissionRequest {
            model: "mistral/mistral-ocr-latest".into(),
            custom_llm_provider: None,
            api_key: Some("test-key".into()),
            api_base: Some("not a URL".into()),
            extra_headers: vec![],
            timeout_seconds: 2.0,
            request_format: None,
            document: OcrDocument::DocumentUrl {
                document_url: "https://example.test/document.pdf".into(),
            },
            credentials: Default::default(),
            vertex_project: None,
            vertex_location: None,
            stream: false,
        }
    }

    #[rstest::rstest]
    #[case("openai/model", Some("native"), "native OCR request format")]
    #[case("openai/model", None, "OCR provider")]
    #[case(
        "azure_ai/doc-intelligence/prebuilt-read",
        None,
        "Azure Document Intelligence OCR polling"
    )]
    #[case(
        "vertex_ai/mistral-ocr-latest",
        None,
        "OCR HTTP document URL to data URI conversion"
    )]
    #[case("mistral/mistral-ocr-latest", None, "OCR streaming response handling")]
    fn admission_preserves_unsupported_error_order(
        #[case] model: &str,
        #[case] format: Option<&str>,
        #[case] expected: &str,
    ) {
        let request = OcrAdmissionRequest {
            model: model.into(),
            request_format: format.map(str::to_owned),
            stream: true,
            ..request()
        };
        assert!(
            matches!(check_admission_capabilities(&request, &|_| None), Err(Error::Unsupported(message)) if message == expected)
        );
        assert!(matches!(
            build_pre_call_request(request),
            Err(Error::Unsupported(message)) if message == expected
        ));
    }

    #[test]
    fn admission_leaves_url_building_and_revalidation_until_request_build() {
        let request = request();
        assert!(check_admission_capabilities(&request, &|_| None).is_ok());
        assert!(matches!(
            build_pre_call_request(request),
            Err(Error::InvalidRequest(_))
        ));

        let mut request = self::request();
        assert!(check_admission_capabilities(&request, &|_| None).is_ok());
        request.timeout_seconds = 0.0;
        request.stream = true;
        assert!(matches!(
            check_admission_capabilities(&request, &|_| None),
            Err(Error::InvalidRequest(_))
        ));
        assert!(matches!(
            build_pre_call_request(request),
            Err(Error::InvalidRequest(_))
        ));
    }

    #[test]
    fn admission_does_not_validate_credentials_or_resolve_headers() {
        let request = OcrAdmissionRequest {
            api_key: None,
            ..request()
        };
        assert!(
            check_admission_capabilities(&request, &|_| panic!(
                "Mistral admission needs no key lookup"
            ))
            .is_ok()
        );
        assert!(
            MISTRAL_OCR_CONFIG
                .validate_credentials(vec![], None, None, &|_| None)
                .is_err()
        );
    }

    #[rstest::rstest]
    #[case("azure_ai/model", "AZURE_AI_API_KEY")]
    #[case("vertex_ai/model", "VERTEX_AI_API_KEY")]
    #[case("vertex_ai/model", "VERTEXAI_API_KEY")]
    fn admission_checks_credential_method_using_only_inert_configuration(
        #[case] model: &str,
        #[case] env_name: &str,
    ) {
        let request = OcrAdmissionRequest {
            model: model.into(),
            api_key: None,
            document: OcrDocument::ImageUrl {
                image_url: "data:image/png;base64,AA==".into(),
            },
            ..request()
        };
        assert!(matches!(
            check_admission_capabilities(&request, &|_| None),
            Err(Error::Unsupported(_))
        ));
        assert_eq!(
            check_admission_capabilities(&request, &|_| Some(" ".into())).is_ok(),
            model.starts_with("azure_ai/")
        );
        assert!(
            check_admission_capabilities(&request, &|name| (name == env_name)
                .then(|| "configured".into()))
            .is_ok()
        );
        let request = OcrAdmissionRequest {
            extra_headers: vec![("AUTHORIZATION".into(), "inert".into())],
            ..request
        };
        assert!(check_admission_capabilities(&request, &|_| None).is_ok());
    }
}
