use std::time::Duration;

use serde_json::{Map, Value};

use crate::Error;
use crate::providers::azure_ai::ocr::transformation as azure_ai;
use crate::providers::mistral::ocr::transformation::MISTRAL_OCR_CONFIG;
use crate::providers::vertex_ai::ocr::transformation as vertex_ai;
use crate::routing_utils::provider::{CustomLlmProvider, get_custom_llm_provider};

use super::transformation::{OcrProviderConfig, OcrResponseHandling};
use super::types::OcrRequest;
pub use super::types::PreparedOcr;

fn request_config(
    request: &OcrRequest,
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

pub(crate) fn admission_capabilities(request: &OcrRequest) -> Result<(), Error> {
    check_admission_capabilities(request, &|key| std::env::var(key).ok())
}

fn check_admission_capabilities(
    request: &OcrRequest,
    env_lookup: &dyn Fn(&str) -> Option<String>,
) -> Result<(), Error> {
    let (provider, config) = request_config(request)?;
    validate_capabilities(config)?;
    request
        .document
        .validate(config.requires_data_uri_document())?;
    if request.stream {
        return Err(Error::Unsupported("OCR streaming response handling"));
    }
    if let Some(operation) = config.credential_acquisition_operation() {
        let supplied = request
            .api_key
            .as_deref()
            .is_some_and(|key| !key.trim().is_empty())
            || crate::http_utils::has_header(&request.extra_headers, "authorization");
        let configured = match provider.custom_llm_provider {
            "azure_ai" => {
                crate::http_utils::has_header(&request.extra_headers, "api-key")
                    || request
                        .azure_ad_token
                        .as_deref()
                        .is_some_and(|key| !key.trim().is_empty())
                    || env_lookup("AZURE_AI_API_KEY").is_some_and(|key| !key.trim().is_empty())
            }
            "vertex_ai" => ["VERTEX_AI_API_KEY", "VERTEXAI_API_KEY"]
                .into_iter()
                .any(|name| env_lookup(name).is_some_and(|key| !key.trim().is_empty())),
            _ => false,
        };
        if !supplied && !configured {
            return Err(Error::Unsupported(operation));
        }
    }
    Ok(())
}

pub fn prepare(request: OcrRequest) -> Result<PreparedOcr, Error> {
    let (provider, config) = request_config(&request)?;
    let env_lookup = |key: &str| std::env::var(key).ok();
    let headers = config
        .validate_credentials(
            request.extra_headers.clone(),
            request.api_key.as_deref(),
            request.azure_ad_token.as_deref(),
            &env_lookup,
        )
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
    let document = serde_json::to_value(&request.document)
        .map_err(|_| Error::InvalidRequest("could not project OCR document".into()))?;
    let template = config.transform_ocr_request(provider.model, document, Map::new())?;
    if template.files.is_some() {
        return Err(Error::Unsupported("OCR multipart document preparation"));
    }
    let Value::Object(body) = template.data else {
        return Err(Error::Unsupported("non-object OCR request template"));
    };
    Ok(PreparedOcr {
        model: provider.model.to_string(),
        custom_llm_provider: provider.custom_llm_provider.to_string(),
        url,
        headers,
        body,
        document_projection: config.document_projection(),
        parameter_fields: config.supported_ocr_params(),
        timeout_seconds: request.timeout_seconds,
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

pub(super) fn provider_config(
    provider: &str,
    model: &str,
) -> Result<&'static dyn OcrProviderConfig, Error> {
    match provider {
        "mistral" => Ok(&MISTRAL_OCR_CONFIG),
        "azure_ai" => azure_ai::config_for_model(model),
        "vertex_ai" => vertex_ai::config_for_model(model),
        _ => Err(Error::Unsupported("OCR provider")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ocr::types::OcrDocument;

    fn request() -> OcrRequest {
        OcrRequest {
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
            azure_ad_token: None,
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
        let request = OcrRequest {
            model: model.into(),
            request_format: format.map(str::to_owned),
            stream: true,
            ..request()
        };
        assert!(
            matches!(check_admission_capabilities(&request, &|_| None), Err(Error::Unsupported(message)) if message == expected)
        );
        assert!(
            matches!(prepare(request), Err(Error::Unsupported(message)) if message == expected)
        );
    }

    #[test]
    fn admission_leaves_url_preparation_and_revalidation_until_prepare() {
        let request = request();
        assert!(check_admission_capabilities(&request, &|_| None).is_ok());
        assert!(matches!(prepare(request), Err(Error::InvalidRequest(_))));

        let mut request = self::request();
        assert!(check_admission_capabilities(&request, &|_| None).is_ok());
        request.timeout_seconds = 0.0;
        request.stream = true;
        assert!(matches!(
            check_admission_capabilities(&request, &|_| None),
            Err(Error::InvalidRequest(_))
        ));
        assert!(matches!(prepare(request), Err(Error::InvalidRequest(_))));
    }

    #[test]
    fn admission_does_not_validate_credentials_or_resolve_headers() {
        let request = OcrRequest {
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
        let request = OcrRequest {
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
        assert!(matches!(
            check_admission_capabilities(&request, &|_| Some(" ".into())),
            Err(Error::Unsupported(_))
        ));
        assert!(
            check_admission_capabilities(&request, &|name| (name == env_name)
                .then(|| "configured".into()))
            .is_ok()
        );
        let request = OcrRequest {
            extra_headers: vec![("AUTHORIZATION".into(), "inert".into())],
            ..request
        };
        assert!(check_admission_capabilities(&request, &|_| None).is_ok());
    }
}
