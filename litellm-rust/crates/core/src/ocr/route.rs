use std::sync::{Arc, Mutex};

use litellm_auth::ResolvedCredential;
use litellm_host::{
    event::{CallEvent, RequestContext, WireRequest},
    machine::{HostChannel, HostTokenProvider, MachineFault, RouteMachine, TokenRoute},
    route::Route,
};
use litellm_llms::base_llm::ocr::{
    error::Error, handler::OcrClient, transformation::LiteLLMOcrResponse,
};

use super::handler::perform_ocr_request;
use crate::ocr::types::{LiteLLMOcrRequest, OcrDocumentInput, OcrFileContent, ResolvedOcrRequest};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OcrOp {
    ProjectRequest,
    ReadDocument,
    AcquireAzureAdToken,
}

pub enum OcrOpResult {
    Request {
        request: Box<LiteLLMOcrRequest<OcrDocumentInput>>,
        caller_token: bool,
    },
    Document(OcrFileContent),
    AzureAdToken(ResolvedCredential),
}

pub struct Ocr;

impl Route for Ocr {
    type Response = LiteLLMOcrResponse;
    type Error = Error;
    type Op = OcrOp;
    type OpResult = OcrOpResult;
    type Chunk = std::convert::Infallible;
    type StreamHead = std::convert::Infallible;
}

impl TokenRoute for Ocr {
    fn acquire_token_op() -> OcrOp {
        OcrOp::AcquireAzureAdToken
    }

    fn token_credential(result: OcrOpResult) -> Option<ResolvedCredential> {
        match result {
            OcrOpResult::AzureAdToken(credential) => Some(credential),
            _ => None,
        }
    }
}

pub type OcrHost = HostChannel<Ocr>;
pub type OcrMachine = RouteMachine<Ocr>;

/// The OCR call as a machine: projection, document reading and token acquisition are
/// host operations; everything else runs in Rust.
pub fn ocr_machine(client: OcrClient) -> OcrMachine {
    RouteMachine::new(move |host| Box::pin(execute(client, host)))
}

async fn execute(client: OcrClient, host: OcrHost) -> Result<LiteLLMOcrResponse, Error> {
    let OcrOpResult::Request {
        request,
        caller_token,
    } = host.route(OcrOp::ProjectRequest).await?
    else {
        return Err(MachineFault::Mismatch.into());
    };
    let request = LiteLLMOcrRequest {
        azure_ad_token_provider: caller_token
            .then(|| HostTokenProvider::handle(host.clone()))
            .or(request.azure_ad_token_provider),
        ..*request
    };
    let caller_document = matches!(request.document, OcrDocumentInput::Document(_));
    let request = prepare_request_document(request, &host).await?;
    perform_ocr_request(&client, request, &host, caller_document).await
}

async fn prepare_request_document(
    request: LiteLLMOcrRequest<OcrDocumentInput>,
    host: &OcrHost,
) -> Result<ResolvedOcrRequest, Error> {
    let request = match &request.document {
        OcrDocumentInput::HostReader { mime_type } => {
            let mime_type = mime_type.clone();
            let OcrOpResult::Document(content) = host.route(OcrOp::ReadDocument).await? else {
                return Err(MachineFault::Mismatch.into());
            };
            request.with_document(OcrDocumentInput::Bytes {
                bytes: content.bytes,
                file_name: content.file_name,
                mime_type,
            })
        }
        _ => request,
    };
    if let OcrDocumentInput::Document(_) = &request.document {
        return request.map_document(super::document::prepare_document);
    }
    tokio::task::spawn_blocking(move || request.map_document(super::document::prepare_document))
        .await
        .map_err(|error| Error::DocumentTask(Arc::new(error)))?
}

type Reader = Box<dyn Fn() -> Result<OcrFileContent, Error> + Send + Sync>;
type BeforeSend =
    Box<dyn Fn(WireRequest, &RequestContext) -> Result<WireRequest, Error> + Send + Sync>;
type Observer = Box<dyn Fn(&CallEvent) + Send + Sync>;

/// The in-process host for a request that is already in hand: the request answers
/// projection, and the optional observer sees and may rewrite the wire request.
pub struct LocalOcrHost {
    request: Mutex<Option<LiteLLMOcrRequest<OcrDocumentInput>>>,
    reader: Option<Reader>,
    before_send: Option<BeforeSend>,
    observer: Option<Observer>,
}

impl LocalOcrHost {
    pub fn new(request: LiteLLMOcrRequest<OcrDocumentInput>) -> Self {
        Self {
            request: Mutex::new(Some(request)),
            reader: None,
            before_send: None,
            observer: None,
        }
    }

    pub fn with_reader(
        self,
        reader: impl Fn() -> Result<OcrFileContent, Error> + Send + Sync + 'static,
    ) -> Self {
        Self {
            reader: Some(Box::new(reader)),
            ..self
        }
    }

    pub fn with_before_send(
        self,
        before_send: impl Fn(WireRequest, &RequestContext) -> Result<WireRequest, Error>
        + Send
        + Sync
        + 'static,
    ) -> Self {
        Self {
            before_send: Some(Box::new(before_send)),
            ..self
        }
    }

    pub fn with_observer(self, observer: impl Fn(&CallEvent) + Send + Sync + 'static) -> Self {
        Self {
            observer: Some(Box::new(observer)),
            ..self
        }
    }
}

impl litellm_host::host::Host<Ocr> for LocalOcrHost {
    async fn route(&self, op: OcrOp) -> Result<OcrOpResult, Error> {
        match op {
            OcrOp::ProjectRequest => self
                .request
                .lock()
                .unwrap_or_else(|error| error.into_inner())
                .take()
                .map(|request| OcrOpResult::Request {
                    request: Box::new(request),
                    caller_token: false,
                })
                .ok_or_else(|| Error::InvalidRequest("OCR request was already projected".into())),
            OcrOp::ReadDocument => self
                .reader
                .as_ref()
                .ok_or_else(|| Error::InvalidRequest("OCR host has no document reader".into()))
                .and_then(|reader| reader())
                .map(OcrOpResult::Document),
            OcrOp::AcquireAzureAdToken => {
                Err(Error::Auth(litellm_auth::Error::AzureTokenAcquisition(
                    "OCR host has no Azure AD token provider".into(),
                )))
            }
        }
    }

    async fn before_send(
        &self,
        wire: WireRequest,
        context: &RequestContext,
    ) -> Result<WireRequest, Error> {
        match &self.before_send {
            Some(before_send) => before_send(wire, context),
            None => Ok(wire),
        }
    }

    async fn emit(&self, event: &CallEvent) -> Result<(), Error> {
        if let Some(observer) = &self.observer {
            observer(event);
        }
        Ok(())
    }
}

#[cfg(test)]
mod aws_textract_tests {
    use std::{collections::BTreeMap, time::SystemTime};

    use litellm_auth_aws::{Credentials, aws_signature_headers, sign_post};
    use litellm_llms::base_llm::ocr::error::Error;
    use serde_json::{Value, json};
    use time::{PrimitiveDateTime, format_description};

    use crate::ocr::{
        route::LocalOcrHost,
        test_support::{
            MockResponse, header, mock_server, perform_ocr_with, request_body,
            wire_request_with_document,
        },
        types::LiteLLMOcrRequest,
    };

    const ACCESS_KEY_ID: &str = "AKIDEXAMPLE";
    const SECRET_ACCESS_KEY: &str = "wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY";

    fn textract_request(base: &str) -> LiteLLMOcrRequest {
        textract_request_for("aws_textract/detect-document-text", base)
    }

    fn textract_request_for(model: &str, base: &str) -> LiteLLMOcrRequest {
        wire_request_with_document(
            model,
            &format!("{base}/"),
            json!({"type": "image_url", "image_url": "data:image/png;base64,b3JpZ2luYWw="}),
            json!({
                "aws_access_key_id": ACCESS_KEY_ID,
                "aws_secret_access_key": SECRET_ACCESS_KEY,
                "aws_region_name": "eu-west-1"
            }),
        )
    }

    fn textract_response() -> MockResponse {
        MockResponse::json(json!({
            "DocumentMetadata": {"Pages": 1},
            "Blocks": [{"BlockType": "PAGE"}, {"BlockType": "LINE", "Text": "Invoice 12345"}]
        }))
    }

    /// Recomputes SigV4 over the bytes the server received, at the time the client claimed.
    fn expected_authorization(url: &str, raw_request: &str) -> String {
        let format =
            format_description::parse_borrowed::<2>("[year][month][day]T[hour][minute][second]Z")
                .unwrap();
        let signed_at: SystemTime =
            PrimitiveDateTime::parse(header(raw_request, "x-amz-date").unwrap(), &format)
                .unwrap()
                .assume_utc()
                .into();
        let headers: BTreeMap<String, String> = ["content-type", "x-amz-target"]
            .into_iter()
            .map(|name| {
                (
                    name.to_string(),
                    header(raw_request, name).unwrap().to_string(),
                )
            })
            .collect();
        let body = raw_request.split_once("\r\n\r\n").unwrap().1;
        sign_post(
            url,
            body.as_bytes(),
            &aws_signature_headers(&headers),
            "eu-west-1",
            "textract",
            &Credentials::new(ACCESS_KEY_ID, SECRET_ACCESS_KEY, None, None, "test"),
            signed_at,
        )
        .unwrap()["Authorization"]
            .clone()
    }

    #[tokio::test]
    async fn the_request_is_signed_for_textract_and_lines_become_the_page() {
        let (base, seen, server) = mock_server(vec![textract_response()]).await;

        let response = perform_ocr_with(LocalOcrHost::new(textract_request(&base)))
            .await
            .unwrap();
        server.await.unwrap();

        let raw = seen.lock().unwrap()[0].clone();
        assert_eq!(
            header(&raw, "x-amz-target"),
            Some("Textract.DetectDocumentText")
        );
        assert_eq!(
            header(&raw, "content-type"),
            Some("application/x-amz-json-1.1")
        );
        assert_eq!(
            request_body(&raw),
            json!({"Document": {"Bytes": "b3JpZ2luYWw="}})
        );
        assert_eq!(
            header(&raw, "authorization"),
            Some(expected_authorization(&format!("{base}/"), &raw).as_str())
        );
        assert_eq!(response.pages[0].markdown, "Invoice 12345");
        assert_eq!(response.usage_info.unwrap().pages_processed, Some(1));
    }

    #[tokio::test]
    async fn a_body_rewritten_by_before_send_is_what_gets_signed_and_sent() {
        let (base, seen, server) = mock_server(vec![textract_response()]).await;
        let host = LocalOcrHost::new(textract_request(&base)).with_before_send(|mut wire, _| {
            assert!(
                !wire
                    .headers
                    .iter()
                    .any(|(name, _)| name.eq_ignore_ascii_case("authorization")),
                "the hook ran after signing"
            );
            wire.body["Document"]["Bytes"] = Value::from("cmVkYWN0ZWQ=");
            Ok(wire)
        });

        perform_ocr_with(host).await.unwrap();
        server.await.unwrap();

        let raw = seen.lock().unwrap()[0].clone();
        assert_eq!(
            request_body(&raw),
            json!({"Document": {"Bytes": "cmVkYWN0ZWQ="}})
        );
        assert_eq!(
            header(&raw, "authorization"),
            Some(expected_authorization(&format!("{base}/"), &raw).as_str())
        );
    }

    #[tokio::test]
    async fn a_multi_page_rejection_reaches_the_caller_with_the_single_page_limit() {
        let (base, _, server) = mock_server(vec![MockResponse {
            status: 400,
            headers: vec![],
            body: json!({
                "__type": "UnsupportedDocumentException",
                "Message": "Request has unsupported document format"
            }),
        }])
        .await;

        let error = perform_ocr_with(LocalOcrHost::new(textract_request(&base)))
            .await
            .unwrap_err();
        server.await.unwrap();

        let Error::Provider { status, body, .. } = error else {
            panic!("expected a provider error, got {error:?}");
        };
        assert_eq!(status, 400);
        assert!(
            body.contains("multi-page documents are not supported"),
            "{body}"
        );
    }

    #[tokio::test]
    async fn analyze_document_asks_for_layout_and_tables_and_returns_markdown() {
        let (base, seen, server) = mock_server(vec![MockResponse::json(json!({
            "DocumentMetadata": {"Pages": 1},
            "Blocks": [
                {"Id": "l1", "BlockType": "LINE", "Text": "Quarterly Report"},
                {"Id": "t", "BlockType": "LAYOUT_TITLE",
                    "Relationships": [{"Type": "CHILD", "Ids": ["l1"]}]}
            ]
        }))])
        .await;
        let request = textract_request_for("aws_textract/analyze-document", &base);

        let response = perform_ocr_with(LocalOcrHost::new(request)).await.unwrap();
        server.await.unwrap();

        let raw = seen.lock().unwrap()[0].clone();
        assert_eq!(
            header(&raw, "x-amz-target"),
            Some("Textract.AnalyzeDocument")
        );
        assert_eq!(
            request_body(&raw)["FeatureTypes"],
            json!(["LAYOUT", "TABLES"])
        );
        assert_eq!(
            header(&raw, "authorization"),
            Some(expected_authorization(&format!("{base}/"), &raw).as_str())
        );
        assert_eq!(response.pages[0].markdown, "# Quarterly Report");
    }
}

#[cfg(test)]
mod azure_ai_tests {
    use litellm_llms::base_llm::ocr::error::Error;
    use serde_json::{Value, json};

    use crate::ocr::route::LocalOcrHost;
    use crate::ocr::test_support::{
        MockResponse, mock_server, perform_ocr, perform_ocr_with, wire_request,
    };

    #[tokio::test]
    async fn facade_executes_azure_mistral_with_prepared_auth() {
        let (base, seen, server) = mock_server(vec![MockResponse::json(json!({
            "pages":[{"index":0,"markdown":"hello"}],
            "usage_info":{"pages_processed":1}
        }))])
        .await;
        let mut request = wire_request(
            "azure_ai/model",
            &base,
            json!({"include_image_base64":true}),
        );
        request.credentials.api_key = None;
        request.transport.extra_headers = vec![(
            "Authorization".into(),
            "Bearer python-prepared-token".into(),
        )];

        let result = perform_ocr(request).await.unwrap();
        server.await.unwrap();
        assert_eq!(result.pages[0].markdown, "hello");
        let requests = seen.lock().unwrap();
        assert_eq!(requests.len(), 1);
        assert!(requests[0].starts_with("POST /providers/mistral/azure/ocr "));
        assert!(
            requests[0]
                .to_ascii_lowercase()
                .contains("authorization: bearer python-prepared-token\r\n")
        );
        let body: Value =
            serde_json::from_str(requests[0].split_once("\r\n\r\n").unwrap().1).unwrap();
        assert_eq!(
            body,
            json!({
                "model":"model",
                "document":{"type":"document_url","document_url":"data:application/pdf;base64,YWJj"},
                "include_image_base64":true
            })
        );
    }

    #[tokio::test]
    async fn facade_acquires_supplied_entra_token_for_final_request() {
        let (base, seen, server) = mock_server(vec![MockResponse::json(json!({"pages":[]}))]).await;
        let mut request = wire_request(
            "azure_ai/model",
            &base,
            json!({"azure_ad_token":"rust-owned-token"}),
        );
        request.credentials.api_key = None;

        perform_ocr(request).await.unwrap();
        server.await.unwrap();

        let requests = seen.lock().unwrap();
        assert_eq!(requests.len(), 1);
        assert!(
            requests[0]
                .to_ascii_lowercase()
                .contains("authorization: bearer rust-owned-token\r\n")
        );
    }

    #[tokio::test]
    async fn rejects_non_inline_body_after_guardrails() {
        let request = wire_request("azure_ai/model", "http://127.0.0.1:1", json!({}));
        let host = LocalOcrHost::new(request).with_before_send(|mut wire, _| {
            wire.body["document"] = json!({
                "type":"document_url",
                "document_url":"https://example.com/not-inline.pdf"
            });
            Ok(wire)
        });
        let error = perform_ocr_with(host).await.unwrap_err();
        assert!(error.to_string().contains("data URI"));
    }

    mod transformation {
        use std::sync::{
            Arc,
            atomic::{AtomicUsize, Ordering},
        };

        use litellm_auth::{
            ResolvedCredential, SecretValue, TokenFuture, TokenProvider, TokenProviderHandle,
        };
        use rstest::rstest;
        use serde_json::json;

        use super::*;
        use crate::ocr::{
            test_support::{MockResponse, header, mock_server, perform_ocr},
            types::LiteLLMOcrRequest,
            wire::decode_request,
        };

        #[derive(Debug)]
        struct CountingToken {
            token: fn(usize) -> String,
            calls: AtomicUsize,
        }

        impl CountingToken {
            fn new(token: fn(usize) -> String) -> Arc<Self> {
                Arc::new(Self {
                    token,
                    calls: AtomicUsize::new(0),
                })
            }

            fn calls(&self) -> usize {
                self.calls.load(Ordering::SeqCst)
            }
        }

        impl TokenProvider for CountingToken {
            fn acquire(&self) -> TokenFuture<'_> {
                let call = self.calls.fetch_add(1, Ordering::SeqCst) + 1;
                let token = SecretValue::new((self.token)(call));
                Box::pin(async move {
                    Ok(ResolvedCredential::AccessToken {
                        token,
                        expires_on: None,
                    })
                })
            }
        }

        fn numbered_token(call: usize) -> String {
            format!("callback-{call}")
        }

        fn azure_request(
            provider: &Arc<CountingToken>,
            api_base: Option<&str>,
            api_key: Option<&str>,
            extra_headers: Value,
            optional_params: Value,
        ) -> LiteLLMOcrRequest {
            let wire = serde_json::from_value(json!({
                "model": "azure_ai/mistral-ocr-latest",
                "document": {"type":"document_url","document_url":"data:application/pdf;base64,YWJj"},
                "api_key": api_key,
                "api_base": api_base,
                "custom_llm_provider": null,
                "extra_headers": extra_headers,
                "optional_params": optional_params,
                "timeout_seconds": 2.0
            }))
            .unwrap();
            LiteLLMOcrRequest {
                azure_ad_token_provider: Some(TokenProviderHandle::new(provider.clone())),
                ..decode_request(wire).unwrap()
            }
        }

        fn ocr_page() -> MockResponse {
            MockResponse::json(json!({"pages":[{"index":0,"markdown":"hello"}]}))
        }

        #[tokio::test]
        async fn token_provider_result_is_the_bearer_and_is_acquired_for_each_request() {
            let provider = CountingToken::new(numbered_token);
            let (base, seen, server) = mock_server(vec![ocr_page(), ocr_page()]).await;

            for _ in 0..2 {
                perform_ocr(azure_request(
                    &provider,
                    Some(&base),
                    None,
                    Value::Null,
                    json!({}),
                ))
                .await
                .unwrap();
            }
            server.await.unwrap();

            assert_eq!(provider.calls(), 2);
            let requests = seen.lock().unwrap();
            assert_eq!(
                requests
                    .iter()
                    .map(|request| header(request, "authorization"))
                    .collect::<Vec<_>>(),
                [Some("Bearer callback-1"), Some("Bearer callback-2")]
            );
        }

        #[rstest]
        #[case::api_key_skips_provider(Some("resource-key"), Value::Null, json!({}), "Bearer resource-key", 0)]
        #[case::provider_beats_static_token(
            None,
            Value::Null,
            json!({"azure_ad_token":"static-token"}),
            "Bearer callback-1",
            1
        )]
        #[case::header_wins_on_the_wire_but_provider_still_runs(
            None,
            json!({"Authorization":"Bearer override"}),
            json!({}),
            "Bearer override",
            1
        )]
        #[tokio::test]
        async fn credential_precedence(
            #[case] api_key: Option<&str>,
            #[case] extra_headers: Value,
            #[case] optional_params: Value,
            #[case] expected_authorization: &str,
            #[case] expected_calls: usize,
        ) {
            let provider = CountingToken::new(numbered_token);
            let (base, seen, server) = mock_server(vec![ocr_page()]).await;

            perform_ocr(azure_request(
                &provider,
                Some(&base),
                api_key,
                extra_headers,
                optional_params,
            ))
            .await
            .unwrap();
            server.await.unwrap();

            assert_eq!(provider.calls(), expected_calls);
            let requests = seen.lock().unwrap();
            assert_eq!(requests.len(), 1);
            assert_eq!(
                header(&requests[0], "authorization"),
                Some(expected_authorization)
            );
        }

        #[rstest]
        #[case::missing_api_base(
            false,
            json!({}),
            numbered_token,
            |error: &Error| matches!(error, Error::Auth(litellm_auth::Error::MissingApiBase {
                provider: "Azure AI",
                environment_variable: "AZURE_AI_API_BASE",
            })),
            0
        )]
        #[case::unsupported_oidc_reference(
            true,
            json!({"azure_ad_token":"oidc/assertion","client_id":"client","tenant_id":"tenant"}),
            numbered_token,
            |error: &Error| matches!(error, Error::Auth(litellm_auth::Error::UnsupportedOidcReference)),
            0
        )]
        #[case::empty_provider_token_ignores_static_token(
            true,
            json!({"azure_ad_token":"static-token"}),
            |_| String::new(),
            |error: &Error| matches!(error, Error::MissingAzureAiCredentials),
            1
        )]
        #[tokio::test]
        async fn credential_failures_send_no_provider_request(
            #[case] with_api_base: bool,
            #[case] optional_params: Value,
            #[case] token: fn(usize) -> String,
            #[case] expected: fn(&Error) -> bool,
            #[case] expected_calls: usize,
        ) {
            let provider = CountingToken::new(token);
            let (base, seen, server) = mock_server(vec![ocr_page()]).await;

            let error = perform_ocr(azure_request(
                &provider,
                with_api_base.then_some(base.as_str()),
                None,
                Value::Null,
                optional_params,
            ))
            .await
            .unwrap_err();
            server.abort();

            assert!(expected(&error), "unexpected error: {error:?}");
            assert_eq!(provider.calls(), expected_calls);
            assert!(seen.lock().unwrap().is_empty());
        }
    }
}

#[cfg(test)]
mod azure_document_intelligence_tests {
    use litellm_host::event::{CallEvent, MachineEvent};
    use litellm_llms::base_llm::ocr::{error::Error, settings::OcrSettings};
    use rstest::rstest;
    use serde_json::{Value, json};

    use crate::ocr::route::LocalOcrHost;
    use crate::ocr::{
        test_support::{
            MockResponse, mock_server, ocr_client, perform_ocr, perform_ocr_with, wire_request,
        },
        wire::{OcrWireRequest, decode_request},
    };

    fn query_value(url: &str, key: &str) -> Option<String> {
        url::Url::parse(url)
            .unwrap()
            .query_pairs()
            .find_map(|(name, value)| (name == key).then(|| value.into_owned()))
    }

    #[tokio::test]
    async fn facade_maps_pages_features_and_url_document() {
        let (base, seen, server) = mock_server(vec![MockResponse::json(json!({
            "status":"succeeded",
            "analyzeResult":{"pages":[]}
        }))])
        .await;
        let mut request = wire_request(
            "azure_ai/doc-intelligence/prebuilt-read",
            &base,
            json!({"pages":[2,0,0,1],"features":["keyValuePairs","languages"]}),
        );
        request.document = serde_json::from_value::<
            litellm_llms::base_llm::ocr::transformation::OcrDocument,
        >(json!({
            "type":"document_url",
            "document_url":"https://example.com/document.pdf"
        }))
        .unwrap()
        .into();

        perform_ocr(request).await.unwrap();
        server.await.unwrap();
        let request = &seen.lock().unwrap()[0];
        let target = request.split_whitespace().nth(1).unwrap();
        let url = format!("{base}{target}");
        assert_eq!(query_value(&url, "pages").as_deref(), Some("1,2,3"));
        assert_eq!(
            query_value(&url, "features").as_deref(),
            Some("keyValuePairs,languages")
        );
        let body: Value = serde_json::from_str(request.split_once("\r\n\r\n").unwrap().1).unwrap();
        assert_eq!(
            body,
            json!({"urlSource":"https://example.com/document.pdf"})
        );
    }

    #[rstest]
    #[case(json!({"pages":[true]}), Error::Pages("expected only integers or only strings".into()))]
    #[case(json!({"pages":[1,"2"]}), Error::Pages("expected only integers or only strings".into()))]
    #[case(json!({"pages":[-1]}), Error::Pages("negative page index".into()))]
    #[case(json!({"pages":"1&&features=bad"}), Error::Pages("invalid native page range".into()))]
    #[case(json!({"features":"languages&pages=1"}), Error::Features)]
    #[case(json!({"req_format":"azure"}), Error::RequestFormat)]
    #[tokio::test]
    async fn rejects_invalid_pages_features_and_format(
        #[case] options: Value,
        #[case] expected: Error,
    ) {
        let (base, seen, server) = mock_server(vec![MockResponse::json(json!({}))]).await;
        let result = decode_request(OcrWireRequest {
            model: "azure_ai/doc-intelligence/prebuilt-read".into(),
            document: json!({"type":"document_url","document_url":"https://example.com/a.pdf"}),
            api_key: Some(litellm_auth::SecretValue::new("key")),
            api_base: Some(base),
            custom_llm_provider: None,
            extra_headers: None,
            optional_params: options.as_object().unwrap().clone(),
            input_sources: Default::default(),
            timeout_seconds: Some(2.0),
        });
        let result = match result {
            Ok(request) => perform_ocr(request).await,
            Err(error) => Err(error),
        };
        server.abort();
        let _ = server.await;
        assert!(
            seen.lock().unwrap().is_empty(),
            "sent invalid options: {options}"
        );
        let error = result.unwrap_err();
        assert_eq!(
            std::mem::discriminant(&error),
            std::mem::discriminant(&expected)
        );
        assert_eq!(error.http_status_code(), Some(400));
        assert_eq!(error.to_string(), expected.to_string());
    }

    #[rstest]
    #[case(json!({}))]
    #[case(json!({"req_format":"litellm"}))]
    #[tokio::test]
    async fn missing_native_fields_keep_page_text_without_retaining_raw_response(
        #[case] options: Value,
    ) {
        let operation = json!({
            "status":"succeeded",
            "analyzeResult":{"pages":[{"pageNumber":1,"lines":[{"content":"hello"}]}]}
        });
        let (base, seen, server) = mock_server(vec![MockResponse::json(operation)]).await;
        let response = perform_ocr(wire_request(
            "azure_ai/doc-intelligence/prebuilt-read",
            &base,
            options,
        ))
        .await
        .unwrap();
        server.await.unwrap();

        assert_eq!(response.pages.len(), 1);
        assert_eq!(response.pages[0].index, 0);
        assert_eq!(response.pages[0].markdown, "hello");
        assert_eq!(response.provider_native_response, None);
        let serialized = response.into_json();
        assert_eq!(serialized.get("content"), Some(&Value::Null));
        assert_eq!(serialized.get("tables"), Some(&Value::Null));
        assert_eq!(serialized.get("keyValuePairs"), Some(&Value::Null));
        let requests = seen.lock().unwrap();
        assert_eq!(requests.len(), 1);
        let target = requests[0].split_whitespace().nth(1).unwrap();
        let url = format!("{base}{target}");
        for field in ["pages", "features", "req_format"] {
            assert_eq!(query_value(&url, field), None);
        }
        let body: Value =
            serde_json::from_str(requests[0].split_once("\r\n\r\n").unwrap().1).unwrap();
        assert_eq!(body, json!({"base64Source":"YWJj"}));
    }

    #[tokio::test]
    async fn inline_document_decodes_to_base64_source() {
        let (base, seen, server) = mock_server(vec![MockResponse::json(json!({
            "status":"succeeded"
        }))])
        .await;
        let request = wire_request("azure_ai/doc-intelligence/prebuilt-read", &base, json!({}));

        perform_ocr(request).await.unwrap();
        server.await.unwrap();
        let request = &seen.lock().unwrap()[0];
        let body: Value = serde_json::from_str(request.split_once("\r\n\r\n").unwrap().1).unwrap();
        assert_eq!(body, json!({"base64Source":"YWJj"}));
    }

    #[tokio::test]
    async fn immediate_response_normalizes_pages_and_preserves_native() {
        let operation = json!({
            "status":"succeeded",
            "operationExtension":42,
            "analyzeResult":{
                "content":"A\n\nB",
                "tables":[{"cells":[]}],
                "keyValuePairs":[{"key":{"content":"A"}}],
                "pages":[{
                    "pageNumber":"2",
                    "width":"8.5",
                    "height":11,
                    "unit":"inch",
                    "lines":[{"content":"A"},{"content":null},{"content":"B"}]
                }]
            }
        });
        let (base, _, server) = mock_server(vec![MockResponse::json(operation.clone())]).await;
        let result = perform_ocr(wire_request(
            "azure_ai/doc-intelligence/prebuilt-read",
            &base,
            json!({"req_format":"native"}),
        ))
        .await
        .unwrap();
        server.await.unwrap();

        assert_eq!(result.pages[0].index, 1);
        assert_eq!(result.pages[0].markdown, "A\n\nB");
        assert_eq!(
            serde_json::to_value(&result.pages[0].dimensions).unwrap(),
            json!({"width":816,"height":1056,"dpi":96})
        );
        assert_eq!(result.usage_info.as_ref().unwrap().pages_processed, Some(1));
        let serialized = result.clone().into_json();
        assert_eq!(serialized["content"], "A\n\nB");
        assert_eq!(serialized["tables"], json!([{"cells":[]}]));
        assert_eq!(
            serialized["keyValuePairs"],
            json!([{"key":{"content":"A"}}])
        );
        assert!(serialized.get("key_value_pairs").is_none());
        assert_eq!(
            result.provider_native_response.map(Value::Object),
            Some(operation)
        );
    }

    #[tokio::test]
    async fn client_settings_choose_the_api_version_and_the_inch_to_pixel_dpi() {
        let (base, seen, server) = mock_server(vec![MockResponse::json(json!({
            "status":"succeeded",
            "analyzeResult":{"pages":[{"pageNumber":1,"width":8.5,"height":11,"unit":"inch"}]}
        }))])
        .await;
        let client = ocr_client().with_settings(OcrSettings {
            document_intelligence_api_version: "2099-01-01".into(),
            document_intelligence_dpi: 72,
            ..OcrSettings::default()
        });

        let result = crate::ocr::client::perform(
            &client,
            wire_request("azure_ai/doc-intelligence/prebuilt-read", &base, json!({})),
        )
        .await
        .unwrap();
        server.await.unwrap();

        let target = seen.lock().unwrap()[0]
            .split_whitespace()
            .nth(1)
            .unwrap()
            .to_string();
        assert_eq!(
            query_value(&format!("{base}{target}"), "api-version").as_deref(),
            Some("2099-01-01")
        );
        assert_eq!(
            serde_json::to_value(&result.pages[0].dimensions).unwrap(),
            json!({"width":612,"height":792,"dpi":72})
        );
    }

    #[tokio::test]
    async fn accepted_response_polls_to_success_with_only_credentials() {
        let operation = json!({"status":"succeeded","analyzeResult":{"pages":[]}});
        let (base, seen, server) = mock_server(vec![
            MockResponse {
                status: 202,
                headers: vec![("Operation-Location", "{base}/operation".into())],
                body: json!({}),
            },
            MockResponse {
                status: 200,
                headers: vec![("Retry-After", "0".into())],
                body: json!({"status":"running"}),
            },
            MockResponse::json(operation.clone()),
        ])
        .await;
        let mut request = wire_request(
            "azure_ai/doc-intelligence/prebuilt-read",
            &base,
            json!({"req_format":"native"}),
        );
        request
            .transport
            .extra_headers
            .push(("X-Trace".into(), "initial-only".into()));

        let result = perform_ocr(request).await.unwrap();
        server.await.unwrap();
        assert_eq!(
            result.provider_native_response.map(Value::Object),
            Some(operation)
        );
        let requests = seen.lock().unwrap();
        assert_eq!(requests.len(), 3);
        assert!(requests[0].to_ascii_lowercase().contains("x-trace:"));
        for poll in &requests[1..] {
            assert!(!poll.to_ascii_lowercase().contains("x-trace:"));
            assert!(
                poll.to_ascii_lowercase()
                    .contains("ocp-apim-subscription-key: test-key")
            );
        }
    }

    #[tokio::test]
    async fn accepted_response_emits_response_received_before_polling() {
        let (base, seen, server) = mock_server(vec![
            MockResponse {
                status: 202,
                headers: vec![("Operation-Location", "{base}/operation".into())],
                body: json!({"submitted": true}),
            },
            MockResponse::json(json!({"status":"succeeded"})),
        ])
        .await;
        let request_count = seen.clone();
        let host = LocalOcrHost::new(wire_request(
            "azure_ai/doc-intelligence/prebuilt-read",
            &base,
            json!({}),
        ))
        .with_observer(move |event| {
            let CallEvent::Machine(MachineEvent::ResponseReceived { raw }) = event else {
                return;
            };
            match request_count.lock().unwrap().len() {
                1 => assert_eq!(raw.body, r#"{"submitted":true}"#),
                2 => assert!(raw.body.contains("succeeded")),
                count => panic!("unexpected callback after {count} requests"),
            }
        });

        perform_ocr_with(host).await.unwrap();
        server.await.unwrap();
        assert_eq!(seen.lock().unwrap().len(), 2);
    }

    #[tokio::test]
    async fn polling_forwards_bearer_credentials() {
        let (base, seen, server) = mock_server(vec![
            MockResponse {
                status: 202,
                headers: vec![("Operation-Location", "{base}/operation".into())],
                body: json!({}),
            },
            MockResponse::json(json!({"status":"succeeded"})),
        ])
        .await;
        let mut request = wire_request("azure_ai/doc-intelligence/prebuilt-read", &base, json!({}));
        request.credentials.api_key = None;
        request.transport.extra_headers = vec![("Authorization".into(), "Bearer token".into())];

        perform_ocr(request).await.unwrap();
        server.await.unwrap();
        let requests = seen.lock().unwrap();
        assert!(
            requests[1]
                .to_ascii_lowercase()
                .contains("authorization: bearer token")
        );
    }

    #[tokio::test]
    async fn polling_does_not_follow_redirects() {
        let (base, seen, server) = mock_server(vec![
            MockResponse {
                status: 202,
                headers: vec![("Operation-Location", "{base}/operation".into())],
                body: json!({}),
            },
            MockResponse {
                status: 302,
                headers: vec![("Location", "{base}/redirected".into())],
                body: json!({}),
            },
            MockResponse::json(json!({"status":"succeeded"})),
        ])
        .await;

        let error = perform_ocr(wire_request(
            "azure_ai/doc-intelligence/prebuilt-read",
            &base,
            json!({}),
        ))
        .await
        .unwrap_err();

        assert!(error.to_string().contains("status 302"), "{error}");
        assert_eq!(seen.lock().unwrap().len(), 2);
        server.abort();
    }

    #[tokio::test]
    async fn polling_rejects_terminal_failure() {
        let (base, _, server) = mock_server(vec![
            MockResponse {
                status: 202,
                headers: vec![("Operation-Location", "{base}/operation".into())],
                body: json!({}),
            },
            MockResponse::json(json!({"status":"failed"})),
        ])
        .await;

        let error = perform_ocr(wire_request(
            "azure_ai/doc-intelligence/prebuilt-read",
            &base,
            json!({}),
        ))
        .await
        .unwrap_err();
        server.await.unwrap();
        assert!(error.to_string().contains("status failed"));
    }

    #[tokio::test]
    async fn malformed_provider_pages_report_response_paths() {
        for (analysis, path) in [
            (json!({"pages":null}), "pages"),
            (json!({"pages":[null]}), "pages[0]"),
            (json!({"pages":[{"lines":null}]}), "lines"),
            (json!({"pages":[{"width":"bad"}]}), "width"),
        ] {
            let (base, _, server) = mock_server(vec![MockResponse::json(json!({
                "status":"succeeded",
                "analyzeResult":analysis
            }))])
            .await;
            let error = perform_ocr(wire_request(
                "azure_ai/doc-intelligence/prebuilt-read",
                &base,
                json!({}),
            ))
            .await
            .unwrap_err();
            server.await.unwrap();
            assert!(error.to_string().contains(path), "{error}");
        }
    }

    #[tokio::test]
    async fn rejects_missing_invalid_and_cross_origin_operation_locations() {
        for headers in [
            Vec::new(),
            vec![("Operation-Location", "/relative".into())],
            vec![("Operation-Location", "http://example.com/operation".into())],
            vec![(
                "Operation-Location",
                "http://user:password@127.0.0.1/operation".into(),
            )],
        ] {
            let (base, _, server) = mock_server(vec![MockResponse {
                status: 202,
                headers,
                body: json!({}),
            }])
            .await;
            let error = perform_ocr(wire_request(
                "azure_ai/doc-intelligence/prebuilt-read",
                &base,
                json!({}),
            ))
            .await
            .unwrap_err();
            server.await.unwrap();
            assert!(error.to_string().contains("operation-location"));
        }
    }

    #[tokio::test]
    async fn polling_deadline_bounds_retry_delay() {
        let (base, _, server) = mock_server(vec![
            MockResponse {
                status: 202,
                headers: vec![("Operation-Location", "{base}/operation".into())],
                body: json!({}),
            },
            MockResponse {
                status: 200,
                headers: vec![("Retry-After", "9999".into())],
                body: json!({"status":"notStarted"}),
            },
        ])
        .await;
        let request = wire_request("azure_ai/doc-intelligence/prebuilt-read", &base, json!({}));
        let client = ocr_client().with_settings(OcrSettings {
            poll_timeout: std::time::Duration::from_millis(100),
            ..OcrSettings::default()
        });

        let error = tokio::time::timeout(
            std::time::Duration::from_secs(1),
            crate::ocr::client::perform(&client, request),
        )
        .await
        .unwrap()
        .unwrap_err();
        server.await.unwrap();
        assert!(error.to_string().contains("timed out"));
    }

    #[tokio::test]
    async fn model_id_is_encoded_and_dot_segments_are_rejected() {
        let (base, seen, server) = mock_server(vec![MockResponse::json(json!({
            "status":"succeeded"
        }))])
        .await;
        perform_ocr(wire_request(
            "azure_ai/doc-intelligence/a ?#é",
            &base,
            json!({}),
        ))
        .await
        .unwrap();
        server.await.unwrap();
        assert!(seen.lock().unwrap()[0].contains("a%20%3F%23%C3%A9:analyze"));

        for model in [
            "azure_ai/doc-intelligence/.",
            "azure_ai/doc-intelligence/..",
        ] {
            let error = perform_ocr(wire_request(model, "http://127.0.0.1:1", json!({})))
                .await
                .unwrap_err();
            assert!(error.to_string().contains("dot segment"));
        }
    }

    mod transformation {
        use std::sync::{Arc, Mutex};

        use litellm_host::event::{CallEvent, MachineEvent};
        use litellm_llms::base_llm::ocr::transformation::OcrDocument;
        use serde_json::{Value, json};

        use super::*;
        use crate::ocr::{
            route::LocalOcrHost,
            test_support::{
                MockResponse, mock_server, perform_ocr, perform_ocr_with, wire_request,
            },
        };

        #[tokio::test]
        async fn facade_maps_pages_features_and_url_document() {
            let (base, seen, server) = mock_server(vec![MockResponse::json(json!({
                "status":"succeeded",
                "analyzeResult":{"pages":[]}
            }))])
            .await;
            let mut request = wire_request(
                "azure_ai/doc-intelligence/prebuilt-read",
                &base,
                json!({"pages":[2,0,0,1],"features":["keyValuePairs","languages"], "future_option": {"nested":null}, "extra_body":{"provider_option":false}}),
            );
            request.document = serde_json::from_value::<OcrDocument>(json!({
                "type":"document_url",
                "document_url":"https://example.com/document.pdf"
            }))
            .unwrap()
            .into();

            perform_ocr(request).await.unwrap();
            server.await.unwrap();
            let request = &seen.lock().unwrap()[0];
            let target = request.split_whitespace().nth(1).unwrap();
            let url = format!("{base}{target}");
            assert_eq!(query_value(&url, "pages").as_deref(), Some("1,2,3"));
            assert_eq!(
                query_value(&url, "features").as_deref(),
                Some("keyValuePairs,languages")
            );
            let body: Value =
                serde_json::from_str(request.split_once("\r\n\r\n").unwrap().1).unwrap();
            assert_eq!(
                body,
                json!({"urlSource":"https://example.com/document.pdf", "future_option":{"nested":null}, "provider_option":false})
            );
        }

        #[tokio::test]
        async fn rejects_invalid_pages_features_and_format() {
            for options in [
                json!({"pages":[true]}),
                json!({"pages":[1,"2"]}),
                json!({"pages":[-1]}),
                json!({"pages":"1&&features=bad"}),
                json!({"features":"languages&pages=1"}),
                json!({"req_format":"azure"}),
            ] {
                let request = wire_request(
                    "azure_ai/doc-intelligence/prebuilt-read",
                    "http://127.0.0.1:1",
                    options.clone(),
                );
                let rejected = perform_ocr(request).await.is_err();
                assert!(rejected, "accepted {options}");
            }
        }

        #[tokio::test]
        async fn immediate_response_normalizes_pages_and_preserves_native() {
            let operation = json!({
                "status":"succeeded",
                "operationExtension":42,
                "analyzeResult":{
                    "content":"A\n\nB",
                    "tables":[{"cells":[]}],
                    "keyValuePairs":[{"key":{"content":"A"}}],
                    "pages":[{
                        "pageNumber":"2",
                        "width":"8.5",
                        "height":11,
                        "unit":"inch",
                        "lines":[{"content":"A"},{"content":null},{"content":"B"}]
                    }]
                }
            });
            let (base, _, server) = mock_server(vec![MockResponse::json(operation.clone())]).await;
            let result = perform_ocr(wire_request(
                "azure_ai/doc-intelligence/prebuilt-read",
                &base,
                json!({"req_format":"native"}),
            ))
            .await
            .unwrap();
            server.await.unwrap();

            assert_eq!(result.pages[0].index, 1);
            assert_eq!(result.pages[0].markdown, "A\n\nB");
            assert_eq!(
                serde_json::to_value(&result.pages[0].dimensions).unwrap(),
                json!({"width":816,"height":1056,"dpi":96})
            );
            assert_eq!(result.usage_info.as_ref().unwrap().pages_processed, Some(1));
            let serialized = result.clone().into_json();
            assert_eq!(serialized["content"], "A\n\nB");
            assert_eq!(serialized["tables"], json!([{"cells":[]}]));
            assert_eq!(
                serialized["keyValuePairs"],
                json!([{"key":{"content":"A"}}])
            );
            assert!(serialized.get("key_value_pairs").is_none());
            assert_eq!(
                result.provider_native_response.as_ref(),
                operation.as_object()
            );
        }

        #[tokio::test]
        async fn accepted_response_polls_to_success_with_only_credentials() {
            let operation = json!({"status":"succeeded","analyzeResult":{"pages":[]}});
            let (base, seen, server) = mock_server(vec![
                MockResponse {
                    status: 202,
                    headers: vec![("Operation-Location", "{base}/operation".into())],
                    body: json!({}),
                },
                MockResponse {
                    status: 200,
                    headers: vec![("Retry-After", "0".into())],
                    body: json!({"status":"running"}),
                },
                MockResponse::json(operation.clone()),
            ])
            .await;
            let mut request = wire_request(
                "azure_ai/doc-intelligence/prebuilt-read",
                &base,
                json!({"req_format":"native"}),
            );
            request
                .transport
                .extra_headers
                .push(("X-Trace".into(), "initial-only".into()));

            let result = perform_ocr(request).await.unwrap();
            server.await.unwrap();
            assert_eq!(
                result.provider_native_response.as_ref(),
                operation.as_object()
            );
            let requests = seen.lock().unwrap();
            assert_eq!(requests.len(), 3);
            assert!(requests[0].to_ascii_lowercase().contains("x-trace:"));
            for poll in &requests[1..] {
                assert!(!poll.to_ascii_lowercase().contains("x-trace:"));
                assert!(
                    poll.to_ascii_lowercase()
                        .contains("ocp-apim-subscription-key: test-key")
                );
            }
        }

        #[tokio::test]
        async fn accepted_response_emits_response_received_for_submission_and_completed_poll() {
            let (base, seen, server) = mock_server(vec![
                MockResponse {
                    status: 202,
                    headers: vec![("Operation-Location", "{base}/operation".into())],
                    body: json!({"submitted": true}),
                },
                MockResponse::json(json!({"status":"succeeded"})),
            ])
            .await;
            let responses_received = Arc::new(Mutex::new(Vec::new()));
            let request_count = seen.clone();
            let observed = responses_received.clone();
            let host = LocalOcrHost::new(wire_request(
                "azure_ai/doc-intelligence/prebuilt-read",
                &base,
                json!({}),
            ))
            .with_observer(move |event| {
                if let CallEvent::Machine(MachineEvent::ResponseReceived { raw }) = event {
                    observed
                        .lock()
                        .unwrap()
                        .push((request_count.lock().unwrap().len(), raw.body.clone()));
                }
            });

            perform_ocr_with(host).await.unwrap();
            server.await.unwrap();
            assert_eq!(seen.lock().unwrap().len(), 2);
            assert_eq!(
                *responses_received.lock().unwrap(),
                [
                    (1, r#"{"submitted":true}"#.to_string()),
                    (2, r#"{"status":"succeeded"}"#.to_string()),
                ]
            );
        }
    }
}

#[cfg(test)]
mod cohere_tests {
    mod transformation {
        use litellm_llms::{
            base_llm::ocr::{
                error::Error,
                transformation::{BaseOcrConfig, OcrDocument, OcrResponseFormat},
            },
            cohere::ocr::transformation::*,
        };
        use rstest::rstest;
        use serde_json::{Value, json};

        #[tokio::test]
        async fn composed_body_preserves_native_document_fields_and_untyped_overrides() {
            let request = crate::ocr::test_support::wire_request(
                "cohere/parse",
                "https://example.com",
                json!({
                "output_format":"markdown", "timeout":30,
                    "extra_body":{
                        "output_format": {"future":true},
                        "document":{"type":"image_url","image_url":"https://example.com/a.png",
                            "provider_options":{"nested":[false,0,null]}}
                    }
                }),
            );
            let request = request.with_document(
                serde_json::from_value(json!({
                    "type":"image_url","image_url":"https://example.com/original.png"
                }))
                .unwrap(),
            );
            let request = crate::ocr::prepare::prepare_request_for_test(request);
            let http = CohereParseConfig
                .prepare_request(
                    &request,
                    &crate::ocr::test_support::ocr_client(),
                    &crate::ocr::test_support::NoHooks,
                )
                .await
                .unwrap();
            let body: Value = serde_json::from_slice(http.body()).unwrap();
            assert_eq!(
                body,
                json!({
                    "model":"parse", "output_format":{"future":true},
                    "document":{"type":"image_url","image_url":"https://example.com/a.png",
                        "provider_options":{"nested":[false,0,null]}}
                })
            );
        }

        #[tokio::test]
        async fn explicit_null_options_use_defaults_before_http() {
            let request = crate::ocr::test_support::wire_request(
                "cohere/parse",
                "https://example.com",
                json!({"output_format":null,"req_format":null}),
            );
            let request = request.with_document(
                serde_json::from_value(
                    json!({"type":"image_url","image_url":"https://example.com/a.png"}),
                )
                .unwrap(),
            );
            assert_eq!(
                request.response_format().unwrap(),
                OcrResponseFormat::Litellm
            );
            let request = crate::ocr::prepare::prepare_request_for_test(request);
            let http = CohereParseConfig
                .prepare_request(
                    &request,
                    &crate::ocr::test_support::ocr_client(),
                    &crate::ocr::test_support::NoHooks,
                )
                .await
                .unwrap();
            let body: Value = serde_json::from_slice(http.body()).unwrap();
            assert_eq!(body["output_format"], "markdown");
            assert!(body.get("req_format").is_none());
        }

        #[rstest]
        #[case::cohere("cohere/parse-v5.0", "POST /v2/parse ")]
        #[case::azure_ai("azure_ai/Cohere-parse-v5.0", "POST /providers/cohere/v2/parse ")]
        #[tokio::test]
        async fn route_sends_image_to_its_parse_endpoint_with_the_bearer_key(
            #[case] model: &str,
            #[case] request_line: &str,
        ) {
            use crate::ocr::test_support::{MockResponse, header, mock_server, perform_ocr};

            let (base, seen, server) =
                mock_server(vec![MockResponse::json(json!({"pages":[]}))]).await;
            let request = crate::ocr::test_support::wire_request(model, &base, json!({}))
                .with_document(
                    serde_json::from_value::<OcrDocument>(
                        json!({"type":"image_url","image_url":"data:image/png;base64,YWJj"}),
                    )
                    .unwrap()
                    .into(),
                );

            perform_ocr(request).await.unwrap();
            server.await.unwrap();

            let requests = seen.lock().unwrap();
            assert_eq!(requests.len(), 1);
            assert!(requests[0].starts_with(request_line), "{}", requests[0]);
            assert_eq!(
                header(&requests[0], "authorization"),
                Some("Bearer test-key")
            );
        }

        #[rstest]
        #[tokio::test]
        async fn route_rejects_non_image_document_without_a_request(
            #[values("cohere/parse-v5.0", "azure_ai/Cohere-parse-v5.0")] model: &str,
        ) {
            use crate::ocr::test_support::{MockResponse, mock_server, perform_ocr};

            let (base, seen, server) =
                mock_server(vec![MockResponse::json(json!({"pages":[]}))]).await;

            let error = perform_ocr(crate::ocr::test_support::wire_request(
                model,
                &base,
                json!({}),
            ))
            .await
            .unwrap_err();
            server.abort();

            assert!(matches!(error, Error::CohereImageOnly), "{error:?}");
            assert!(seen.lock().unwrap().is_empty());
        }
    }
}

#[cfg(test)]
mod deepseek_tests {
    use litellm_llms::{
        base_llm::ocr::transformation::{BaseOcrConfig, OcrDocument},
        vertex_ai::ocr::deepseek_transformation::{
            DeepSeekOcrParams, DeepSeekOcrResponse, VertexAIDeepSeekOCRConfig,
            normalize_response as transform_ocr_response,
        },
    };
    use rstest::rstest;
    use serde_json::{Value, json};

    fn document() -> OcrDocument {
        serde_json::from_value(json!({"type":"image_url","image_url":"gs://bucket/a.png"})).unwrap()
    }

    #[rstest]
    #[case("stream", json!(true))]
    #[case("temperature", json!(0.1))]
    #[case("max_tokens", json!(1024))]
    #[case("top_p", json!(0.9))]
    #[case("n", json!(2))]
    #[case("stop", json!("done"))]
    #[case("stop", json!(["done", "stop"]))]
    fn request_mapping_matches_python(#[case] name: &str, #[case] value: Value) {
        let params: DeepSeekOcrParams =
            serde_json::from_value(json!({name: value.clone(), "ignored": true})).unwrap();
        let result = serde_json::to_value(
            VertexAIDeepSeekOCRConfig
                .transform_ocr_request("deepseek-ai/deepseek-ocr-maas", document(), &params, &[])
                .unwrap(),
        )
        .unwrap();
        assert_eq!(result["model"], "deepseek-ai/deepseek-ocr-maas");
        assert_eq!(
            result["messages"][0]["content"][0],
            json!({"type":"image_url","image_url":"gs://bucket/a.png"})
        );
        assert_eq!(result[name], value);
        assert!(result.get("ignored").is_none());
    }

    #[rstest]
    #[case(json!({"type":"image_url","image_url":"data:image/png;base64,AA=="}))]
    #[case(json!({"type":"document_url","document_url":"data:application/pdf;base64,AA=="}))]
    fn request_maps_both_document_types_to_image_content(#[case] document: Value) {
        let source = document
            .get("image_url")
            .or_else(|| document.get("document_url"))
            .unwrap()
            .clone();
        let request = VertexAIDeepSeekOCRConfig
            .transform_ocr_request(
                "deepseek-ai/deepseek-ocr-maas",
                serde_json::from_value(document).unwrap(),
                &DeepSeekOcrParams::default(),
                &[],
            )
            .unwrap();
        let result = serde_json::to_value(request).unwrap();
        assert_eq!(
            result["messages"][0]["content"][0],
            json!({"type":"image_url","image_url":source})
        );
    }

    #[rstest]
    #[case(json!("# hello"), "# hello")]
    #[case(json!("{broken"), "{broken")]
    #[case(json!(" {\"pages\":[]} "), " {\"pages\":[]} ")]
    #[case(json!({"pages":[]}), "")]
    #[case(json!("[]"), "[]")]
    #[case(json!("{\"pages\":[{\"markdown\":\"json text\"}]}"), "json text")]
    #[case(json!({"pages":[{"markdown":"object"}]}), "object")]
    fn response_codec_handles_text_json_and_objects(
        #[case] content: Value,
        #[case] expected: &str,
    ) {
        let structured = content
            .as_object()
            .is_some_and(|object| object.contains_key("pages"))
            || content
                .as_str()
                .is_some_and(|text| text.contains("\"pages\""));
        let response: DeepSeekOcrResponse = serde_json::from_value(
            json!({"choices":[{"message":{"content":content}}],"usage":{"prompt_tokens":1}}),
        )
        .unwrap();
        let result = transform_ocr_response("model", response)
            .unwrap()
            .into_json();
        assert_eq!(result["pages"][0]["markdown"], expected);
        assert_eq!(result["pages"][0]["index"], 0);
        if structured {
            assert!(result["usage_info"].is_null());
        } else {
            assert_eq!(result["usage_info"]["prompt_tokens"], 1);
        }
    }

    #[test]
    fn structured_result_maps_pages_usage_model_and_annotation() {
        let response: DeepSeekOcrResponse = serde_json::from_value(json!({
            "choices":[{"message":{"content":{
                "pages":[{"index":2,"markdown":"page","images":[{"id":"one"}],"dimensions":{"width":10}}],
                "model":"provider-model",
                "usage_info":{"pages_processed":1},
                "document_annotation":{"language":"en"},
                "future":"kept"
            }}}]
        }))
        .unwrap();
        let result = transform_ocr_response("requested", response)
            .unwrap()
            .into_json();
        assert_eq!(result["pages"][0]["index"], 2);
        assert_eq!(result["pages"][0]["images"][0]["id"], "one");
        assert_eq!(result["model"], "provider-model");
        assert_eq!(result["usage_info"]["pages_processed"], 1);
        assert_eq!(result["document_annotation"]["language"], "en");
        assert_eq!(result["future"], "kept");
    }

    #[test]
    fn response_codec_rejects_missing_empty_and_malformed_content() {
        for value in [
            json!({"choices":[{"message":{"content":{}}}]}),
            json!({"choices":[]}),
            json!({"choices":[{"message":{"content":""}}]}),
            json!({"choices":[{"message":{"content":"{\"pages\":[{\"markdown\":42}]}"}}]}),
            json!({"choices":[{"message":{"content":{"pages":[{"markdown":42}]}}}]}),
        ] {
            let result = serde_json::from_value::<DeepSeekOcrResponse>(value)
                .map_err(|_| ())
                .and_then(|response| transform_ocr_response("model", response).map_err(|_| ()));
            assert!(result.is_err());
        }
    }
}

#[cfg(test)]
mod reducto_tests {
    use litellm_host::event::{CallEvent, MachineEvent, WireRequest};
    use litellm_llms::base_llm::ocr::{error::Error, transformation::OcrDocument};
    use rstest::rstest;
    use serde_json::{Value, json};

    use crate::ocr::route::LocalOcrHost;
    use crate::ocr::test_support::{
        MockResponse, mock_server, perform_ocr, perform_ocr_with, wire_request,
    };

    fn request_body(request: &str) -> Value {
        serde_json::from_str(request.split_once("\r\n\r\n").unwrap().1).unwrap()
    }

    #[rstest]
    #[case(
        "reducto/parse-v3",
        json!({
            "formatting":{"table_output_format":"html"},
            "retrieval":{"chunk_mode":"section"},
            "settings":{"ocr_system":"standard"},
            "future_ocr_option":true,
            "extra_body":{"provider_option":"value"}
        }),
        "reducto://already.pdf",
    json!({
        "input":"reducto://already.pdf",
            "formatting":{"table_output_format":"html"},
            "retrieval":{"chunk_mode":"section"},
            "settings":{"ocr_system":"standard"},
            "future_ocr_option":true,
            "provider_option":"value"
        })
    )]
    #[case(
        "reducto/parse-legacy",
        json!({
            "enhance":{"agentic":[{"type":"table"}]},
            "future_ocr_option":true,
            "extra_body":{"provider_option":"value"}
        }),
        "reducto://legacy.pdf",
    json!({
        "document_url":"reducto://legacy.pdf",
            "options":{"enhance":{"agentic":[{"type":"table"}]}},
            "future_ocr_option":true,
            "provider_option":"value"
        })
    )]
    #[tokio::test]
    async fn request_mapping_matches_python(
        #[case] model: &str,
        #[case] options: Value,
        #[case] source: &str,
        #[case] expected: Value,
    ) {
        let (base, seen, server) = mock_server(vec![MockResponse::json(json!({
            "result":{"chunks":[]}
        }))])
        .await;
        let request =
            crate::ocr::test_support::with_source(wire_request(model, &base, options), source);

        perform_ocr(request).await.unwrap();
        server.await.unwrap();
        let requests = seen.lock().unwrap();
        assert_eq!(requests.len(), 1);
        assert!(requests[0].starts_with("POST /parse "));
        assert_eq!(request_body(&requests[0]), expected);
    }

    #[rstest]
    #[case("parse-v3")]
    #[case("parse-legacy")]
    #[tokio::test]
    async fn data_uri_upload_preserves_multipart_headers(
        #[case] model: &str,
        #[values("application/pdf", "image/png")] mime_type: &str,
    ) {
        let (base, seen, server) = mock_server(vec![
            MockResponse::json(json!({"file_id":"reducto://uploaded.pdf"})),
            MockResponse::json(json!({"result":{"chunks":[{"content":"hello"}]}})),
        ])
        .await;
        let document = if mime_type.starts_with("image/") {
            json!({"type":"image_url","image_url":format!("data:{mime_type};base64,YWJj")})
        } else {
            json!({"type":"document_url","document_url":format!("data:{mime_type};base64,YWJj")})
        };
        let mut request = crate::ocr::types::LiteLLMOcrRequest {
            document: serde_json::from_value::<OcrDocument>(document)
                .unwrap()
                .into(),
            ..wire_request(&format!("reducto/{model}"), &base, json!({}))
        };
        request.transport.extra_headers = vec![
            ("Content-Type".into(), "application/json".into()),
            ("X-Trace".into(), "upload-test".into()),
        ];

        let response = perform_ocr(request).await.unwrap();
        server.await.unwrap();
        assert_eq!(response.pages[0].markdown, "hello");
        let requests = seen.lock().unwrap();
        assert_eq!(requests.len(), 2);
        assert!(requests[0].starts_with("POST /upload "));
        assert!(
            requests[0]
                .to_ascii_lowercase()
                .contains("content-type: multipart/form-data; boundary=")
        );
        assert!(requests[0].contains("x-trace: upload-test"));
        let multipart = requests[0].split_once("\r\n\r\n").unwrap().1;
        assert!(multipart.contains(&format!("Content-Type: {mime_type}\r\n")));
        assert!(multipart.contains("\r\n\r\nabc\r\n--"));
        assert!(requests[1].starts_with("POST /parse "));
        let source_field = if model == "parse-legacy" {
            "document_url"
        } else {
            "input"
        };
        assert_eq!(
            request_body(&requests[1]),
            json!({source_field:"reducto://uploaded.pdf"})
        );
        for request in requests.iter() {
            assert!(
                request
                    .to_ascii_lowercase()
                    .contains("authorization: bearer test-key\r\n")
            );
        }
    }

    #[tokio::test]
    async fn response_received_stays_after_reducto_upload_and_parse() {
        let (base, seen, server) = mock_server(vec![
            MockResponse::json(json!({"file_id":"reducto://uploaded.pdf"})),
            MockResponse::json(json!({"result":{"chunks":[]}})),
        ])
        .await;
        let request_count = seen.clone();
        let host = LocalOcrHost::new(wire_request("reducto/parse-v3", &base, json!({})))
            .with_observer(move |event| {
                if let CallEvent::Machine(MachineEvent::ResponseReceived { raw }) = event {
                    assert_eq!(request_count.lock().unwrap().len(), 2);
                    assert_eq!(raw.body, r#"{"result":{"chunks":[]}}"#);
                }
            });

        perform_ocr_with(host).await.unwrap();
        server.await.unwrap();
        assert_eq!(seen.lock().unwrap().len(), 2);
    }

    #[rstest]
    #[case(json!({"file_id":""}))]
    #[case(json!({}))]
    #[case(json!({"file_id":null}))]
    #[tokio::test]
    async fn invalid_upload_ids_stop_before_parse(#[case] response: Value) {
        let (base, seen, server) = mock_server(vec![MockResponse::json(response)]).await;
        let error = perform_ocr(wire_request("reducto/parse-v3", &base, json!({})))
            .await
            .unwrap_err();
        server.await.unwrap();
        assert!(error.to_string().contains("file_id"));
        assert_eq!(seen.lock().unwrap().len(), 1);
    }

    #[tokio::test]
    async fn upload_failure_stops_before_parse() {
        let (base, seen, server) = mock_server(vec![MockResponse {
            status: 503,
            headers: vec![],
            body: json!({"error":"unavailable"}),
        }])
        .await;
        assert!(
            perform_ocr(wire_request("reducto/parse-v3", &base, json!({})))
                .await
                .is_err()
        );
        server.await.unwrap();
        assert_eq!(seen.lock().unwrap().len(), 1);
    }

    #[rstest]
    #[case("https://example.com/a.pdf", Error::ReductoSource)]
    #[case("reducto://", Error::RequestField { path: "document file id".into() })]
    #[case("data:application/pdf;base64", Error::InvalidDataUri)]
    #[case("data:application/pdf;base64,INVALID!", Error::InvalidDataUri)]
    #[tokio::test]
    async fn rejects_invalid_document_sources_before_network(
        #[case] source: &str,
        #[case] expected: Error,
    ) {
        let (base, seen, server) = mock_server(vec![MockResponse::json(json!({}))]).await;
        let request = crate::ocr::test_support::with_source(
            wire_request("reducto/parse-v3", &base, json!({})),
            source,
        );
        let result = perform_ocr(request).await;
        server.abort();
        let _ = server.await;
        assert!(
            seen.lock().unwrap().is_empty(),
            "sent invalid source: {source}"
        );
        let error = result.unwrap_err();
        assert_eq!(
            std::mem::discriminant(&error),
            std::mem::discriminant(&expected)
        );
        assert_eq!(error.http_status_code(), Some(400));
        assert_eq!(error.to_string(), expected.to_string());
    }

    #[test]
    fn response_normalization_groups_blocks_and_distinguishes_null_result() {
        use litellm_llms::reducto::ocr::transformation::{
            ReductoResponse, normalize_response as transform_ocr_response,
        };

        let raw = json!({"usage":{"num_pages":"2","credits":"3"},"result":{"type":"full","chunks":[
            {"blocks":[{
                "type":"Table",
                "content":"B",
                "bbox":{"left":0.1,"top":0.2,"width":0.8,"height":0.3,"page":2,"original_page":4},
                "confidence":"high",
                "granular_confidence":{"parse_confidence":0.95,"extract_confidence":null},
                "image_url":null
            }]},
            {"blocks":[{"content":"A","bbox":{"page":1},"type":"Text"},{"content":"C","bbox":{"page":1}}]}
        ]}});
        let response: ReductoResponse = serde_json::from_value(raw).unwrap();
        let normalized = transform_ocr_response("parse-v3", response)
            .unwrap()
            .into_json();
        assert_eq!(normalized["pages"][0]["markdown"], "A\n\nC");
        assert_eq!(normalized["pages"][1]["markdown"], "B");
        assert_eq!(normalized["pages"][1]["blocks"][0]["type"], "Table");
        assert_eq!(
            normalized["pages"][1]["blocks"][0]["bbox"],
            json!({"left":0.1,"top":0.2,"width":0.8,"height":0.3,"page":2,"original_page":4})
        );
        assert_eq!(normalized["pages"][1]["blocks"][0]["confidence"], "high");
        assert_eq!(
            normalized["pages"][1]["blocks"][0]["granular_confidence"]["parse_confidence"],
            0.95
        );
        assert!(normalized["pages"][1]["blocks"][0]["image_url"].is_null());
        assert_eq!(normalized["usage_info"]["pages_processed"], 2);
        assert_eq!(normalized["usage_info"]["credits"], 3.0);

        let missing: ReductoResponse =
            serde_json::from_value(json!({"chunks":[{"content":"text"}]})).unwrap();
        let missing = transform_ocr_response("parse-v3", missing).unwrap();
        assert_eq!(missing.pages[0].markdown, "text");
        let null: ReductoResponse = serde_json::from_value(
            json!({"result":null,"chunks":[{"content":"ignored"}],"usage":null}),
        )
        .unwrap();
        let null = transform_ocr_response("parse-v3", null).unwrap();
        assert!(null.pages.is_empty());
    }

    #[tokio::test]
    async fn facade_omits_native_response_by_default_and_preserves_auth_priority() {
        let raw = json!({"job_id":"job-1","result":{"chunks":[]}});
        let (base, seen, server) = mock_server(vec![MockResponse::json(raw)]).await;
        let mut request = crate::ocr::test_support::with_source(
            wire_request("reducto/parse-v3", &base, json!({})),
            "reducto://ready.pdf",
        );
        request.transport.extra_headers = vec![("authorization".into(), "Bearer existing".into())];

        let response = perform_ocr(request).await.unwrap();
        server.await.unwrap();
        assert_eq!(response.provider_native_response, None);
        assert!(
            seen.lock().unwrap()[0]
                .to_ascii_lowercase()
                .contains("authorization: bearer existing")
        );
    }

    #[tokio::test]
    async fn native_format_retains_the_provider_response() {
        let raw = json!({
            "result":{"chunks":[{"content":"native OCR response"}]},
            "usage":{"num_pages":1}
        });
        let (base, _, server) = mock_server(vec![MockResponse::json(raw.clone())]).await;
        let request = crate::ocr::test_support::with_source(
            wire_request("reducto/parse-v3", &base, json!({"req_format":"native"})),
            "reducto://ready.pdf",
        );

        let response = perform_ocr(request).await.unwrap();
        server.await.unwrap();

        assert_eq!(response.pages[0].markdown, "native OCR response");
        assert_eq!(response.provider_native_response.as_ref(), raw.as_object());
    }

    #[tokio::test]
    async fn unknown_model_reaches_parse_and_keeps_its_name() {
        let (base, seen, server) = mock_server(vec![MockResponse::json(json!({
            "result":{"chunks":[{"content":"future model response"}]}
        }))])
        .await;
        let request = crate::ocr::test_support::with_source(
            wire_request("reducto/future-parse-model", &base, json!({})),
            "reducto://ready.pdf",
        );

        let response = perform_ocr(request).await.unwrap();
        server.await.unwrap();

        assert_eq!(response.model, "future-parse-model");
        assert_eq!(response.pages[0].markdown, "future model response");
        let requests = seen.lock().unwrap();
        assert!(requests[0].starts_with("POST /parse "));
        assert_eq!(
            request_body(&requests[0]),
            json!({"input":"reducto://ready.pdf"})
        );
    }

    #[tokio::test]
    async fn guardrail_rewrites_document_before_upload() {
        let (base, seen, server) =
            mock_server(vec![MockResponse::json(json!({"result":{"chunks":[]}}))]).await;
        let host = LocalOcrHost::new(wire_request("reducto/parse-v3", &base, json!({})))
            .with_before_send(|wire, _| {
                assert_eq!(
                    wire.body["document_url"],
                    "data:application/pdf;base64,YWJj"
                );
                Ok(WireRequest {
                    body: json!({"type":"document_url","document_url":"reducto://guarded.pdf"}),
                    ..wire
                })
            });

        perform_ocr_with(host).await.unwrap();
        server.await.unwrap();
        let requests = seen.lock().unwrap();
        assert_eq!(requests.len(), 1);
        assert!(requests[0].starts_with("POST /parse "));
        assert!(requests[0].contains("reducto://guarded.pdf"));
    }

    mod transformation {
        use litellm_host::event::{CallEvent, MachineEvent, WireRequest};
        use litellm_llms::{
            base_llm::ocr::transformation::{BaseOcrConfig, OcrConnection, OcrRequestContext},
            reducto::ocr::transformation::*,
        };
        use rstest::rstest;

        use super::*;
        use crate::ocr::{
            route::LocalOcrHost,
            test_support::{
                MockResponse, mock_server, perform_ocr, perform_ocr_with, wire_request,
            },
        };

        #[tokio::test]
        async fn v3_options_preserve_explicit_null() {
            let overrides =
                serde_json::from_value(json!({"formatting":null,"settings":{},"unknown":true}))
                    .unwrap();
            let params = ReductoParseV3Config
                .map_ocr_params(&overrides, "parse-v3")
                .unwrap();
            let client = crate::ocr::test_support::ocr_client();
            let connection = OcrConnection::default();
            let document = serde_json::from_value(
                json!({"type":"document_url","document_url":"reducto://ready.pdf"}),
            )
            .unwrap();
            let body = ReductoParseV3Config
                .async_transform_ocr_request(
                    "parse-v3",
                    document,
                    &params,
                    &[],
                    OcrRequestContext {
                        client: &client,
                        connection: &connection,
                    },
                )
                .await
                .unwrap();
            assert_eq!(
                serde_json::to_value(body).unwrap(),
                json!({
                    "input":"reducto://ready.pdf", "formatting":null, "settings":{}
                })
            );
            let absent = ReductoParseV3Config
                .map_ocr_params(
                    &litellm_core_utils::call_arguments::CallArguments::default(),
                    "parse-v3",
                )
                .unwrap();
            assert_eq!(serde_json::to_value(absent).unwrap(), json!({}));
        }

        #[rstest]
        #[case(
            "reducto/parse-v3",
            json!({
                "formatting":{"table_output_format":"html"},
                "retrieval":{"chunk_mode":"section"},
                "settings":{"ocr_system":"standard"},
                "future_ocr_option":true,
                "extra_body":{"provider_option":"value"}
            }),
            "reducto://already.pdf",
        json!({
            "input":"reducto://already.pdf",
                "formatting":{"table_output_format":"html"},
                "retrieval":{"chunk_mode":"section"},
                "settings":{"ocr_system":"standard"},
                "future_ocr_option":true,
                "provider_option":"value"
            })
        )]
        #[case(
            "reducto/parse-legacy",
            json!({
                "enhance":{"agentic":[{"type":"table"}]},
                "future_ocr_option":true,
                "extra_body":{"provider_option":"value"}
            }),
            "reducto://legacy.pdf",
        json!({
            "document_url":"reducto://legacy.pdf",
                "options":{"enhance":{"agentic":[{"type":"table"}]}},
                "future_ocr_option":true,
                "provider_option":"value"
            })
        )]
        #[tokio::test]
        async fn request_mapping_matches_python(
            #[case] model: &str,
            #[case] options: Value,
            #[case] source: &str,
            #[case] expected: Value,
        ) {
            let (base, seen, server) = mock_server(vec![MockResponse::json(json!({
                "result":{"chunks":[]}
            }))])
            .await;
            let request =
                crate::ocr::test_support::with_source(wire_request(model, &base, options), source);

            perform_ocr(request).await.unwrap();
            server.await.unwrap();
            let requests = seen.lock().unwrap();
            assert_eq!(requests.len(), 1);
            assert!(requests[0].starts_with("POST /parse "));
            assert_eq!(request_body(&requests[0]), expected);
        }

        #[rstest]
        #[case("parse-v3")]
        #[case("parse-legacy")]
        #[tokio::test]
        async fn data_uri_upload_preserves_multipart_headers(#[case] model: &str) {
            let (base, seen, server) = mock_server(vec![
                MockResponse::json(json!({"file_id":"reducto://uploaded.pdf"})),
                MockResponse::json(json!({"result":{"chunks":[{"content":"hello"}]}})),
            ])
            .await;
            let mut request = wire_request(&format!("reducto/{model}"), &base, json!({}));
            request.transport.extra_headers = vec![
                ("Content-Type".into(), "application/json".into()),
                ("X-Trace".into(), "upload-test".into()),
            ];

            let response = perform_ocr(request).await.unwrap();
            server.await.unwrap();
            assert_eq!(response.pages[0].markdown, "hello");
            let requests = seen.lock().unwrap();
            assert_eq!(requests.len(), 2);
            assert!(requests[0].starts_with("POST /upload "));
            assert!(
                requests[0]
                    .to_ascii_lowercase()
                    .contains("content-type: multipart/form-data; boundary=")
            );
            assert!(requests[0].contains("x-trace: upload-test"));
            assert!(requests[0].contains("application/pdf"));
            assert!(requests[0].contains("abc"));
            assert!(requests[1].starts_with("POST /parse "));
        }

        #[tokio::test]
        async fn response_received_stays_after_reducto_upload_and_parse() {
            let (base, seen, server) = mock_server(vec![
                MockResponse::json(json!({"file_id":"reducto://uploaded.pdf"})),
                MockResponse::json(json!({"result":{"chunks":[]}})),
            ])
            .await;
            let request_count = seen.clone();
            let host = LocalOcrHost::new(wire_request("reducto/parse-v3", &base, json!({})))
                .with_observer(move |event| {
                    if let CallEvent::Machine(MachineEvent::ResponseReceived { raw }) = event {
                        assert_eq!(request_count.lock().unwrap().len(), 2);
                        assert_eq!(raw.body, r#"{"result":{"chunks":[]}}"#);
                    }
                });

            perform_ocr_with(host).await.unwrap();
            server.await.unwrap();
            assert_eq!(seen.lock().unwrap().len(), 2);
        }

        #[rstest]
        #[case("https://example.com/a.pdf")]
        #[case("reducto://")]
        #[case("data:application/pdf;base64")]
        #[case("data:application/pdf;base64,INVALID!")]
        #[tokio::test]
        async fn rejects_invalid_document_sources_before_network(#[case] source: &str) {
            let request = crate::ocr::test_support::with_source(
                wire_request("reducto/parse-v3", "http://127.0.0.1:1", json!({})),
                source,
            );
            assert!(perform_ocr(request).await.is_err());
        }

        #[tokio::test]
        async fn facade_omits_native_response_by_default_and_preserves_auth_priority() {
            let raw = json!({"job_id":"job-1","result":{"chunks":[]}});
            let (base, seen, server) = mock_server(vec![MockResponse::json(raw)]).await;
            let mut request = crate::ocr::test_support::with_source(
                wire_request("reducto/parse-v3", &base, json!({})),
                "reducto://ready.pdf",
            );
            request.transport.extra_headers =
                vec![("authorization".into(), "Bearer existing".into())];

            let response = perform_ocr(request).await.unwrap();
            server.await.unwrap();
            assert_eq!(response.provider_native_response, None);
            assert!(
                seen.lock().unwrap()[0]
                    .to_ascii_lowercase()
                    .contains("authorization: bearer existing")
            );
        }

        #[rstest]
        #[case("reducto/parse-v3")]
        #[case("reducto/parse-legacy")]
        #[tokio::test]
        async fn guardrail_headers_reach_upload_and_parse(#[case] model: &str) {
            let (base, seen, server) = mock_server(vec![
                MockResponse::json(json!({"file_id":"reducto://uploaded.pdf"})),
                MockResponse::json(json!({"result":{"chunks":[]}})),
            ])
            .await;
            let mut request = wire_request(model, &base, json!({}));
            request.transport.extra_headers =
                vec![("authorization".into(), "Bearer original".into())];
            let host = LocalOcrHost::new(request).with_before_send(|wire, _| {
                Ok(WireRequest {
                    headers: vec![("authorization".into(), "Bearer guarded".into())],
                    ..wire
                })
            });

            perform_ocr_with(host).await.unwrap();
            server.await.unwrap();
            let requests = seen.lock().unwrap();
            assert_eq!(requests.len(), 2);
            assert!(requests[0].starts_with("POST /upload "));
            assert!(requests[1].starts_with("POST /parse "));
            for request in requests.iter() {
                assert!(request.contains("authorization: Bearer guarded"));
                assert!(!request.contains("Bearer original"));
            }
        }
    }
}

#[cfg(test)]
mod vertex_ai_tests {
    use litellm_auth::InputSource;
    use litellm_llms::base_llm::ocr::{settings::OcrSettings, transformation::OcrResponseFormat};
    use serde_json::{Value, json};

    use crate::ocr::test_support::{
        MockResponse, mock_server, ocr_client, perform_ocr, wire_request,
    };

    fn request_body(request: &str) -> Value {
        serde_json::from_str(request.split_once("\r\n\r\n").unwrap().1).unwrap()
    }

    #[tokio::test]
    async fn facade_executes_vertex_mistral_with_resolved_project_and_location() {
        let (base, seen, server) = mock_server(vec![MockResponse::json(json!({
            "pages":[{"index":0,"markdown":"hello"}],
            "usage_info":{"pages_processed":1}
        }))])
        .await;
        let request = wire_request(
            "vertex_ai/mistral-ocr-maas",
            &base,
            json!({
                "vertex_project":"project-1",
                "vertex_location":"europe-west4",
                "extract_footer":true
            }),
        );

        let response = perform_ocr(request).await.unwrap();
        server.await.unwrap();
        assert_eq!(response.pages[0].markdown, "hello");
        let requests = seen.lock().unwrap();
        assert_eq!(requests.len(), 1);
        assert!(requests[0].starts_with(
            "POST /v1/projects/project-1/locations/europe-west4/publishers/mistralai/models/mistral-ocr-maas:rawPredict "
        ));
        assert!(
            requests[0]
                .to_ascii_lowercase()
                .contains("authorization: bearer test-key")
        );
        assert_eq!(
            request_body(&requests[0]),
            json!({
                "model":"mistral-ocr-maas",
                "document":{"type":"document_url","document_url":"data:application/pdf;base64,YWJj"},
                "extract_footer":true
            })
        );
    }

    #[tokio::test]
    async fn configured_project_and_location_apply_when_the_call_sets_neither() {
        let (base, seen, server) = mock_server(vec![MockResponse::json(json!({"pages":[]}))]).await;
        let client = ocr_client().with_settings(OcrSettings {
            vertex_project: Some("configured-project".into()),
            vertex_location: Some("europe-west4".into()),
            ..OcrSettings::default()
        });

        crate::ocr::client::perform(
            &client,
            wire_request("vertex_ai/mistral-ocr-maas", &base, json!({})),
        )
        .await
        .unwrap();
        server.await.unwrap();
        assert!(seen.lock().unwrap()[0].starts_with(
            "POST /v1/projects/configured-project/locations/europe-west4/publishers/mistralai/models/mistral-ocr-maas:rawPredict "
        ));
    }

    #[tokio::test]
    async fn supplied_authorization_is_forwarded_without_a_static_token() {
        let (base, seen, server) = mock_server(vec![MockResponse::json(json!({"pages":[]}))]).await;
        let mut request = wire_request(
            "vertex_ai/model",
            &base,
            json!({"vertex_project":"project-1"}),
        );
        request.credentials.api_key = None;
        request.transport.extra_headers = vec![("authorization".into(), "Bearer supplied".into())];

        perform_ocr(request).await.unwrap();
        server.await.unwrap();
        assert!(
            seen.lock().unwrap()[0]
                .to_ascii_lowercase()
                .contains("authorization: bearer supplied")
        );
    }

    #[tokio::test]
    async fn invalid_credentials_fail_before_provider_http() {
        let request = wire_request(
            "vertex_ai/model",
            "http://127.0.0.1:1",
            json!({"vertex_credentials": true}),
        );
        let error = perform_ocr(request).await.unwrap_err();
        assert!(error.to_string().contains("vertex_credentials"));
    }

    #[tokio::test]
    async fn request_controlled_api_base_is_rejected_before_vertex_auth() {
        let mut request = wire_request(
            "vertex_ai/mistral-ocr-maas",
            "https://caller.example",
            json!({"vertex_project":"project-1"}),
        );
        request.credentials.api_base = Some(litellm_auth::Sourced::new(
            "https://caller.example".into(),
            InputSource::Request,
        ));

        let error = perform_ocr(request).await.unwrap_err();
        assert!(
            error
                .to_string()
                .contains("request-controlled Vertex AI endpoint")
        );
    }

    #[tokio::test]
    async fn adapters_build_complete_requests_and_share_mistral_normalization() {
        use std::time::Duration;

        use litellm_llms::{
            base_llm::ocr::transformation::BaseOcrConfig,
            mistral::ocr::transformation::MistralOcrConfig,
            vertex_ai::ocr::transformation::VertexAiOcrConfig,
        };

        use crate::ocr::test_support::ocr_client;

        let client = ocr_client();
        let options = json!({
            "pages": [0, 2],
            "include_image_base64": true,
            "vertex_project": "project-1",
            "vertex_location": "us-central1",
            "unknown": "ignored"
        });
        let direct = wire_request(
            "mistral/mistral-ocr-maas",
            "https://mistral.test",
            options.clone(),
        );
        let vertex = wire_request("vertex_ai/mistral-ocr-maas", "https://vertex.test", options);
        let direct = crate::ocr::prepare::prepare_request_for_test(
            crate::ocr::test_support::resolved_request(direct),
        );
        let vertex = crate::ocr::prepare::prepare_request_for_test(
            crate::ocr::test_support::resolved_request(vertex),
        );
        let direct_http = MistralOcrConfig
            .prepare_request(&direct, &client, &crate::ocr::test_support::NoHooks)
            .await
            .unwrap();
        let vertex_http = VertexAiOcrConfig
            .prepare_request(&vertex, &client, &crate::ocr::test_support::NoHooks)
            .await
            .unwrap();
        assert_eq!(direct_http.url(), "https://mistral.test/v1/ocr");
        assert_eq!(
            vertex_http.url(),
            "https://vertex.test/v1/projects/project-1/locations/us-central1/publishers/mistralai/models/mistral-ocr-maas:rawPredict"
        );
        for http in [&direct_http, &vertex_http] {
            assert_eq!(http.header("authorization").unwrap(), "Bearer test-key");
            assert_eq!(http.header("content-type").unwrap(), "application/json");
            assert_eq!(http.timeout(), Some(Duration::from_secs(2)));
            let body: Value = serde_json::from_slice(http.body()).unwrap();
            assert_eq!(
                body,
                json!({
                    "model": "mistral-ocr-maas",
                    "document": {"type": "document_url", "document_url": "data:application/pdf;base64,YWJj"},
                    "pages": [0, 2],
                    "include_image_base64": true,
                    "unknown": "ignored"
                })
            );
        }
        let payload = json!({"pages": [{"index": 0, "markdown": "hello"}], "extra": "preserved"});
        let raw = serde_json::to_vec(&payload).unwrap();
        let direct_response = MistralOcrConfig
            .transform_ocr_response(&direct.model, &raw, OcrResponseFormat::Litellm)
            .unwrap()
            .into_json();
        let vertex_response = VertexAiOcrConfig
            .transform_ocr_response(&vertex.model, &raw, OcrResponseFormat::Litellm)
            .unwrap()
            .into_json();
        assert_eq!(direct_response, vertex_response);
        assert_eq!(direct_response["model"], "mistral-ocr-maas");
        assert_eq!(direct_response["object"], "ocr");
        assert_eq!(direct_response["extra"], "preserved");
    }

    mod transformation {

        use rstest::rstest;
        use serde_json::{Value, json};

        use crate::ocr::test_support::wire_request;

        #[rstest]
        #[case::mistral(false)]
        #[case::vertex(true)]
        #[tokio::test]
        async fn configs_build_complete_requests_and_share_mistral_normalization(
            #[case] use_vertex: bool,
        ) {
            use std::time::Duration;

            use litellm_llms::{
                base_llm::ocr::transformation::BaseOcrConfig,
                mistral::ocr::transformation::MistralOcrConfig,
                vertex_ai::ocr::transformation::VertexAiOcrConfig,
            };

            use crate::ocr::test_support::ocr_client;

            let client = ocr_client();
            let options = json!({
                "pages": [0, 2],
                "include_image_base64": true,
                "vertex_project": "project-1",
                "vertex_location": "us-central1",
                "unknown": "preserved"
            });
            let direct = wire_request(
                "mistral/mistral-ocr-maas",
                "https://mistral.test",
                options.clone(),
            );
            let vertex = wire_request("vertex_ai/mistral-ocr-maas", "https://vertex.test", options);
            let direct = crate::ocr::prepare::prepare_request_for_test(
                crate::ocr::test_support::resolved_request(direct),
            );
            let vertex = crate::ocr::prepare::prepare_request_for_test(
                crate::ocr::test_support::resolved_request(vertex),
            );
            let direct_http = MistralOcrConfig
                .prepare_request(&direct, &client, &crate::ocr::test_support::NoHooks)
                .await
                .unwrap();
            let vertex_http = VertexAiOcrConfig
                .prepare_request(&vertex, &client, &crate::ocr::test_support::NoHooks)
                .await
                .unwrap();
            assert_eq!(direct_http.url(), "https://mistral.test/v1/ocr");
            assert_eq!(
                vertex_http.url(),
                "https://vertex.test/v1/projects/project-1/locations/us-central1/publishers/mistralai/models/mistral-ocr-maas:rawPredict"
            );
            let http = if use_vertex {
                &vertex_http
            } else {
                &direct_http
            };
            assert_eq!(http.header("authorization").unwrap(), "Bearer test-key");
            assert_eq!(http.header("content-type").unwrap(), "application/json");
            assert_eq!(http.timeout(), Some(Duration::from_secs(2)));
            let body: Value = serde_json::from_slice(http.body()).unwrap();
            assert_eq!(
                body,
                json!({
                    "model": "mistral-ocr-maas",
                    "document": {"type": "document_url", "document_url": "data:application/pdf;base64,YWJj"},
                    "pages": [0, 2],
                    "include_image_base64": true,
                    "unknown": "preserved"
                })
            );
            let payload = serde_json::to_vec(
                &json!({"pages": [{"index": 0, "markdown": "hello"}], "extra": "preserved"}),
            )
            .unwrap();
            let direct_response = MistralOcrConfig
                .transform_ocr_response(&direct.model, &payload, Default::default())
                .unwrap()
                .into_json();
            let vertex_response = VertexAiOcrConfig
                .transform_ocr_response(&vertex.model, &payload, Default::default())
                .unwrap()
                .into_json();
            assert_eq!(direct_response, vertex_response);
            assert_eq!(direct_response["model"], "mistral-ocr-maas");
            assert_eq!(direct_response["object"], "ocr");
            assert_eq!(direct_response["extra"], "preserved");
        }
    }
}

#[cfg(test)]
mod vertex_ai_deepseek_tests {
    use litellm_auth::InputSource;
    use serde_json::{Value, json};

    use crate::ocr::test_support::{MockResponse, mock_server, perform_ocr, wire_request};

    fn request_body(request: &str) -> Value {
        serde_json::from_str(request.split_once("\r\n\r\n").unwrap().1).unwrap()
    }

    #[tokio::test]
    async fn facade_executes_vertex_deepseek_at_the_openai_endpoint() {
        let (base, seen, server) = mock_server(vec![MockResponse::json(json!({
            "choices":[{"message":{"content":"recognized"}}],
            "usage":{"prompt_tokens":1}
        }))])
        .await;
        let request = wire_request(
            "vertex_ai/deepseek-ocr-maas",
            &base,
            json!({
                "vertex_project":"project-1",
                "vertex_location":"europe-west4",
                "temperature":0.1,
                "future_ocr_option":true,
                "extra_body":{"provider_option":"value"}
            }),
        );
        let request = crate::ocr::test_support::with_source(request, "gs://bucket/document.pdf");

        let response = perform_ocr(request).await.unwrap();
        server.await.unwrap();
        assert_eq!(response.pages[0].markdown, "recognized");
        assert_eq!(
            response.usage_info.unwrap().extra_fields["prompt_tokens"],
            1
        );
        let requests = seen.lock().unwrap();
        assert!(requests[0].starts_with(
            "POST /v1/projects/project-1/locations/europe-west4/endpoints/openapi/chat/completions "
        ));
        assert!(
            requests[0]
                .to_ascii_lowercase()
                .contains("authorization: bearer test-key")
        );
        let body = request_body(&requests[0]);
        assert_eq!(body["model"], "deepseek-ai/deepseek-ocr-maas");
        assert_eq!(body["temperature"], 0.1);
        assert_eq!(body["future_ocr_option"], true);
        assert!(body.get("extra_body").is_none());
        assert_eq!(
            body["messages"][0]["content"][0],
            json!({"type":"image_url","image_url":"gs://bucket/document.pdf"})
        );
    }

    #[test]
    fn host_registration_selects_deepseek_without_affecting_mistral() {
        assert!(crate::ocr::arguments::is_supported_request(
            "deepseek-ocr-maas",
            Some("vertex_ai")
        ));
        assert!(crate::ocr::arguments::is_supported_request(
            "mistral-ocr-maas",
            Some("vertex_ai")
        ));
    }

    #[tokio::test]
    async fn request_controlled_api_base_is_rejected_before_vertex_auth() {
        let mut request = wire_request(
            "vertex_ai/deepseek-ocr-maas",
            "https://caller.example",
            json!({"vertex_project":"project-1"}),
        );
        request.credentials.api_base = Some(litellm_auth::Sourced::new(
            "https://caller.example".into(),
            InputSource::Request,
        ));

        let error = perform_ocr(request).await.unwrap_err();
        assert!(
            error
                .to_string()
                .contains("request-controlled Vertex AI endpoint")
        );
    }

    mod deepseek_transformation {
        use serde_json::json;

        use super::*;
        use crate::ocr::test_support::{MockResponse, mock_server, perform_ocr, wire_request};

        #[tokio::test]
        async fn facade_executes_vertex_deepseek_at_the_openai_endpoint() {
            let (base, seen, server) = mock_server(vec![MockResponse::json(json!({
                "choices":[{"message":{"content":"recognized"}}],
                "usage":{"prompt_tokens":1}
            }))])
            .await;
            let request = wire_request(
                "vertex_ai/deepseek-ocr-maas",
                &base,
                json!({
                    "vertex_project":"project-1",
                    "vertex_location":"europe-west4",
                    "temperature":0.1,
                    "future_ocr_option":true,
                    "extra_body":{"provider_option":"value"}
                }),
            );
            let request =
                crate::ocr::test_support::with_source(request, "gs://bucket/document.pdf");

            let response = perform_ocr(request).await.unwrap();
            server.await.unwrap();
            assert_eq!(response.pages[0].markdown, "recognized");
            assert_eq!(
                response.usage_info.unwrap().extra_fields["prompt_tokens"],
                1
            );
            let requests = seen.lock().unwrap();
            assert!(requests[0].starts_with(
                "POST /v1/projects/project-1/locations/europe-west4/endpoints/openapi/chat/completions "
            ));
            assert!(
                requests[0]
                    .to_ascii_lowercase()
                    .contains("authorization: bearer test-key")
            );
            let body = request_body(&requests[0]);
            assert_eq!(body["model"], "deepseek-ai/deepseek-ocr-maas");
            assert_eq!(body["temperature"], 0.1);
            assert_eq!(body["future_ocr_option"], true);
            assert_eq!(body["provider_option"], "value");
            assert!(body.get("vertex_project").is_none());
            assert!(body.get("extra_body").is_none());
            assert_eq!(
                body["messages"][0]["content"][0],
                json!({"type":"image_url","image_url":"gs://bucket/document.pdf"})
            );
        }
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use std::sync::{Arc, Mutex};

    use futures_util::future::BoxFuture;
    use litellm_auth_gcp::VertexAuth;
    use litellm_host::{
        event::{CallEvent, MachineEvent, WireRequest},
        host::{Host, HostOp, HostResult},
        machine::{HostFailure, Machine, MachineStep},
    };
    use litellm_http::{
        HttpClientPool, HttpSettings, Resolution,
        media::{PublicDnsResolver, UrlPolicy},
    };
    use litellm_llms::base_llm::ocr::{
        error::Error as OcrError,
        handler::OcrClient,
        settings::OcrSettings,
        transformation::{
            BaseOcrConfig, LiteLLMOcrResponse, OCR_RESPONSE_MAX_BYTES, OcrTransportConfig,
        },
    };
    use litellm_secrets::source::SecretSource;
    use rstest::rstest;
    use serde_json::{Value, json};

    use crate::ocr::route::{LocalOcrHost, OcrOp, OcrOpResult, ocr_machine};
    use crate::ocr::{
        test_support::{
            MockResponse, mock_server, ocr_client, perform_ocr, perform_ocr_with, wire_request,
        },
        wire::{OcrWireRequest, decode_request},
    };

    struct RecordingSecretSource {
        names: Arc<Mutex<Vec<String>>>,
        values: &'static [(&'static str, &'static str)],
        api_base: String,
    }

    impl SecretSource for RecordingSecretSource {
        fn get_secret_str<'a>(
            &'a self,
            name: &'a str,
        ) -> BoxFuture<'a, Result<Option<litellm_secrets::SecretValue>, litellm_secrets::Error>>
        {
            self.names.lock().unwrap().push(name.to_owned());
            Box::pin(async move {
                Ok(match name {
                    "MISTRAL_AZURE_API_BASE" => Some(self.api_base.clone()),
                    _ => self
                        .values
                        .iter()
                        .find(|(key, _)| *key == name)
                        .map(|(_, value)| value.to_string()),
                }
                .map(litellm_secrets::SecretValue::new))
            })
        }
    }

    #[rstest]
    #[case::mistral("mistral/model", json!({}))]
    #[case::vertex("vertex_ai/mistral-ocr-latest", json!({"vertex_project":"test-project", "vertex_location":"us-central1"}))]
    #[tokio::test]
    async fn ocr_contract_upstream_error_preserves_status_body_and_headers(
        #[case] model: &str,
        #[case] options: Value,
    ) {
        let payload = json!({"message": format!("{} END-OF-PROVIDER-BODY", "x".repeat(4096))});
        let expected_body = serde_json::to_string(&payload).unwrap();
        let (base, seen, server) = mock_server(vec![MockResponse {
            status: 422,
            headers: vec![
                ("Retry-After", "17".into()),
                ("X-Request-ID", "request-123".into()),
                ("X-Future-Header", "retained".into()),
            ],
            body: payload,
        }])
        .await;
        let error = perform_ocr(wire_request(model, &base, options))
            .await
            .unwrap_err();
        server.await.unwrap();
        assert_eq!(seen.lock().unwrap().len(), 1);
        let OcrError::Provider {
            status,
            body,
            headers,
        } = error
        else {
            panic!("expected provider error, got {error:?}");
        };
        assert_eq!(status, 422);
        for (name, value) in [
            ("retry-after", "17"),
            ("x-request-id", "request-123"),
            ("x-future-header", "retained"),
        ] {
            assert!(
                headers
                    .iter()
                    .any(|(key, actual)| key.eq_ignore_ascii_case(name) && actual == value)
            );
        }
        assert_eq!(
            body.len(),
            expected_body.len(),
            "provider error body was truncated"
        );
        assert_eq!(body, expected_body);
    }

    #[test]
    fn request_boundary_selects_mistral_and_rejects_unknown_providers() {
        let request = OcrWireRequest {
            model: "mistral/model".into(),
            document: json!({"type":"document_url","document_url":"https://example.com/doc.pdf"}),
            api_key: Some(litellm_auth::SecretValue::new("key")),
            api_base: None,
            custom_llm_provider: None,
            extra_headers: None,
            optional_params: json!({"extract_header":true,"unknown":42})
                .as_object()
                .unwrap()
                .clone(),
            input_sources: Default::default(),
            timeout_seconds: None,
        };
        assert!(decode_request(request).is_ok());
        assert!(
            decode_request(OcrWireRequest {
                model: "model".into(),
                document: json!({"type":"document_url","document_url":"https://example.com/doc.pdf"}),
                api_key: Some(litellm_auth::SecretValue::new("key")),
                api_base: None,
                custom_llm_provider: Some("unknown".into()),
                extra_headers: None,
                optional_params: serde_json::Map::new(),
                input_sources: Default::default(),
                timeout_seconds: None,
            })
            .is_err()
        );
    }

    #[tokio::test]
    async fn facade_executes_direct_mistral_once() {
        let (base, seen, server) = mock_server(vec![MockResponse::json(json!({
            "pages":[{"index":0,"markdown":"hello","custom":"preserved"}],
            "usage_info":{"pages_processed":1}
        }))])
        .await;
        let result = perform_ocr(wire_request(
            "mistral/model",
            &base,
            json!({"pages":"0,2-4","extract_header":true,"unknown":"ignored"}),
        ))
        .await
        .unwrap();
        server.await.unwrap();
        assert_eq!(result.pages[0].markdown, "hello");
        assert_eq!(result.pages[0].extra_fields["custom"], "preserved");
        let requests = seen.lock().unwrap();
        assert_eq!(requests.len(), 1);
        assert!(requests[0].starts_with("POST /v1/ocr "));
        assert!(
            requests[0]
                .to_ascii_lowercase()
                .contains("authorization: bearer test-key\r\n")
        );
        let body: Value =
            serde_json::from_str(requests[0].split_once("\r\n\r\n").unwrap().1).unwrap();
        assert_eq!(
            body,
            json!({
                "model":"model",
                "document":{"type":"document_url","document_url":"data:application/pdf;base64,YWJj"},
                "pages":"0,2-4",
                "extract_header":true,
                "unknown":"ignored"
            })
        );
    }

    #[tokio::test]
    async fn facade_retains_native_response_when_requested() {
        let provider_response = json!({
            "pages":[{"index":0,"markdown":"hello"}],
            "usage_info":{"pages_processed":1},
            "provider_only":"preserved"
        });
        let (base, _, server) =
            mock_server(vec![MockResponse::json(provider_response.clone())]).await;
        let response = perform_ocr(wire_request(
            "mistral/model",
            &base,
            json!({"req_format":"native"}),
        ))
        .await
        .unwrap();

        server.await.unwrap();
        assert_eq!(
            response.provider_native_response.map(Value::Object),
            Some(provider_response)
        );
    }

    #[rstest]
    #[case::plain_key(&[("MISTRAL_API_KEY", "plain")], "plain")]
    #[case::azure_key_wins(&[("MISTRAL_AZURE_API_KEY", "azure"), ("MISTRAL_API_KEY", "plain")], "azure")]
    #[case::empty_azure_key_falls_through(&[("MISTRAL_AZURE_API_KEY", ""), ("MISTRAL_API_KEY", "plain")], "plain")]
    #[tokio::test]
    async fn mistral_env_fallbacks_follow_python_through_the_injected_secret_source(
        #[case] secrets: &'static [(&'static str, &'static str)],
        #[case] expected_key: &str,
    ) {
        let (base, seen, server) = mock_server(vec![MockResponse::json(json!({"pages":[]}))]).await;
        let names = Arc::new(Mutex::new(Vec::new()));
        let client = ocr_client().with_secrets(Arc::new(RecordingSecretSource {
            names: names.clone(),
            values: secrets,
            api_base: base.clone(),
        }));
        let request = decode_request(OcrWireRequest {
            model: "mistral/model".into(),
            document: json!({"type":"document_url","document_url":"data:application/pdf;base64,YWJj"}),
            api_key: None,
            api_base: None,
            custom_llm_provider: None,
            extra_headers: None,
            optional_params: Default::default(),
            input_sources: Default::default(),
            timeout_seconds: Some(2.0),
        })
        .unwrap();

        crate::ocr::client::perform(&client, request).await.unwrap();
        server.await.unwrap();
        assert_eq!(
            *names.lock().unwrap(),
            litellm_llms::mistral::ocr::transformation::MistralOcrConfig.secret_names()
        );
        assert!(seen.lock().unwrap()[0].contains(&format!("authorization: Bearer {expected_key}")));
    }

    #[tokio::test]
    async fn mistral_ocr_resolves_provider_secrets_before_transformation() {
        let (base, seen, server) = mock_server(vec![MockResponse::json(json!({"pages":[]}))]).await;
        let names = Arc::new(Mutex::new(Vec::new()));
        let client = ocr_client().with_secrets(Arc::new(RecordingSecretSource {
            names: names.clone(),
            values: &[("MISTRAL_API_KEY", "source-key")],
            api_base: base.clone(),
        }));
        let request = decode_request(OcrWireRequest {
            model: "mistral/mistral-ocr-latest".into(),
            document: json!({
                "type":"document_url",
                "document_url":"data:application/pdf;base64,YWJj"
            }),
            api_key: None,
            api_base: None,
            custom_llm_provider: None,
            extra_headers: None,
            optional_params: Default::default(),
            input_sources: Default::default(),
            timeout_seconds: Some(2.0),
        })
        .unwrap();

        crate::ocr::client::perform(&client, request).await.unwrap();
        server.await.unwrap();
        assert_eq!(
            *names.lock().unwrap(),
            litellm_llms::mistral::ocr::transformation::MistralOcrConfig.secret_names()
        );
        assert!(seen.lock().unwrap()[0].contains("authorization: Bearer source-key"));
    }

    #[tokio::test]
    async fn ocr_client_uses_the_injected_http_pool_configuration() {
        let (base, seen, server) = mock_server(vec![MockResponse::json(json!({"pages":[]}))]).await;
        let settings = HttpSettings {
            user_agent: Some("host-owned/1".into()),
            ..HttpSettings::default()
        };
        let client = OcrClient::new(
            &HttpClientPool::new(Arc::new(PublicDnsResolver)),
            &Resolution::from(&settings).config,
            UrlPolicy::default(),
            VertexAuth::default(),
            OcrSettings::default(),
            Arc::new(litellm_secrets::source::EnvironmentSecrets::default()),
        )
        .unwrap();
        crate::ocr::client::perform(&client, wire_request("mistral/model", &base, json!({})))
            .await
            .unwrap();
        server.await.unwrap();
        assert!(seen.lock().unwrap()[0].contains("user-agent: host-owned/1"));
    }

    fn event_name(event: &CallEvent) -> &'static str {
        match event {
            CallEvent::Started { .. } => "started",
            CallEvent::Machine(MachineEvent::ResponseReceived { .. }) => "response",
            CallEvent::Succeeded { .. } => "success",
            CallEvent::Failed { .. } => "failure",
        }
    }

    fn recording_host(
        request: crate::ocr::types::LiteLLMOcrRequest,
        events: Arc<Mutex<Vec<&'static str>>>,
        block: bool,
    ) -> LocalOcrHost {
        let before_send_events = events.clone();
        LocalOcrHost::new(request)
            .with_before_send(move |wire, _| {
                before_send_events.lock().unwrap().push("before_send");
                if block {
                    return Err(OcrError::InvalidRequest("blocked".into()));
                }
                Ok(wire)
            })
            .with_observer(move |event| events.lock().unwrap().push(event_name(event)))
    }

    #[tokio::test]
    async fn lifecycle_sends_headers_returned_by_the_before_send_operation() {
        let (base, seen, server) = mock_server(vec![MockResponse::json(json!({"pages":[]}))]).await;
        let host = LocalOcrHost::new(wire_request("mistral/model", &base, json!({})))
            .with_before_send(|mut wire, _| {
                wire.headers
                    .push(("x-core-callback".into(), "edited".into()));
                Ok(wire)
            });

        perform_ocr_with(host).await.unwrap();
        server.await.unwrap();

        assert!(seen.lock().unwrap()[0].contains("x-core-callback: edited"));
    }

    #[tokio::test]
    async fn before_send_context_names_the_route_and_its_secrets() {
        let (base, _, server) = mock_server(vec![MockResponse::json(json!({"pages":[]}))]).await;
        let observed = Arc::new(Mutex::new(None));
        let captured = observed.clone();
        let host = LocalOcrHost::new(wire_request(
            "mistral/model",
            &base,
            json!({"pages": [0], "req_format": "native"}),
        ))
        .with_before_send(move |wire, context| {
            *captured.lock().unwrap() = Some((wire.clone(), context.clone()));
            Ok(wire)
        });
        perform_ocr_with(host).await.unwrap();
        server.await.unwrap();
        let (wire, context) = observed.lock().unwrap().take().unwrap();
        assert_eq!(context.custom_llm_provider, "mistral");
        assert_eq!(context.model, "model");
        assert_eq!(wire.body["pages"], json!([0]));
        assert!(context.secret_fields.is_empty());
        assert_eq!(context.optional_params["req_format"], "native");

        let (base, _, server) = mock_server(vec![MockResponse::json(json!({"pages":[]}))]).await;
        let observed = Arc::new(Mutex::new(None));
        let captured = observed.clone();
        let request = wire_request(
            "azure_ai/model",
            &base,
            json!({"client_secret": "shh", "tenant_id": "t"}),
        );
        let request = request.with_document(crate::ocr::types::OcrDocumentInput::Bytes {
            bytes: b"abc".as_slice().into(),
            file_name: None,
            mime_type: Some("application/pdf".into()),
        });
        let host = LocalOcrHost::new(request).with_before_send(move |wire, context| {
            *captured.lock().unwrap() = Some(context.clone());
            Ok(wire)
        });
        perform_ocr_with(host).await.unwrap();
        server.await.unwrap();
        let context = observed.lock().unwrap().take().unwrap();
        assert_eq!(context.secret_fields, ["client_secret"]);
    }

    #[tokio::test]
    async fn lifecycle_orders_hooks_and_emits_one_success() {
        let (base, seen, server) = mock_server(vec![MockResponse::json(json!({"pages":[]}))]).await;
        let events = Arc::new(Mutex::new(Vec::new()));
        let host = recording_host(
            wire_request("mistral/model", &base, json!({})),
            events.clone(),
            false,
        );
        perform_ocr_with(host).await.unwrap();
        server.await.unwrap();
        assert_eq!(
            *events.lock().unwrap(),
            ["started", "before_send", "response", "success"]
        );
        assert_eq!(seen.lock().unwrap().len(), 1);
    }

    #[tokio::test]
    async fn lifecycle_blocking_prevents_execution_and_emits_one_failure() {
        let events = Arc::new(Mutex::new(Vec::new()));
        let host = recording_host(
            wire_request("mistral/model", "http://127.0.0.1:1", json!({})),
            events.clone(),
            true,
        );
        let error = perform_ocr_with(host).await.unwrap_err();
        assert!(matches!(error, OcrError::InvalidRequest(message) if message == "blocked"));
        assert_eq!(
            *events.lock().unwrap(),
            ["started", "before_send", "failure"]
        );
    }

    #[tokio::test]
    async fn upstream_failure_emits_one_terminal_failure() {
        let (base, seen, server) = mock_server(vec![MockResponse {
            status: 500,
            headers: vec![],
            body: json!({"error":"failed"}),
        }])
        .await;
        let events = Arc::new(Mutex::new(Vec::new()));
        let host = recording_host(
            wire_request("mistral/model", &base, json!({})),
            events.clone(),
            false,
        );
        assert!(perform_ocr_with(host).await.is_err());
        server.await.unwrap();
        assert_eq!(
            *events.lock().unwrap(),
            ["started", "before_send", "failure"]
        );
        assert_eq!(seen.lock().unwrap().len(), 1);
    }

    /// Drives the machine by hand, answering every op through `host` except `before_send`,
    /// which `intercept` answers so a test can fail or cancel exactly there.
    async fn drive_until(
        client: OcrClient,
        host: &LocalOcrHost,
        mut intercept: impl FnMut(WireRequest) -> Result<WireRequest, HostFailure<OcrError>>,
    ) -> (
        Result<LiteLLMOcrResponse, OcrError>,
        Vec<&'static str>,
        crate::ocr::route::OcrMachine,
    ) {
        let mut machine = ocr_machine(client);
        let mut result = None;
        let mut ops = Vec::new();
        let outcome = loop {
            let op = match machine.resume(result.take()).await {
                Ok(MachineStep::Host(op)) => op,
                Ok(MachineStep::Complete(response)) => break Ok(response),
                Err(error) => break Err(error),
            };
            let answer = match op {
                HostOp::Route(op) => {
                    ops.push(match op {
                        OcrOp::ProjectRequest => "ProjectRequest",
                        OcrOp::ReadDocument => "ReadDocument",
                        OcrOp::AcquireAzureAdToken => "AcquireAzureAdToken",
                    });
                    host.route(op)
                        .await
                        .map(HostResult::Route)
                        .map_err(HostFailure::Error)
                }
                HostOp::BeforeSend { wire, .. } => {
                    ops.push("BeforeSend");
                    intercept(*wire).map(|wire| HostResult::BeforeSend(Box::new(wire)))
                }
                HostOp::Emit(event) => {
                    let event = CallEvent::Machine(event);
                    ops.push(event_name(&event));
                    host.emit(&event)
                        .await
                        .map(|()| HostResult::Emitted)
                        .map_err(HostFailure::Error)
                }
            };
            match answer {
                Ok(answer) => result = Some(answer),
                Err(failure) => break machine.interrupt(failure).await,
            }
        };
        (outcome, ops, machine)
    }

    #[tokio::test]
    async fn failed_before_send_does_not_replay_or_reach_transport() {
        let host = LocalOcrHost::new(wire_request(
            "mistral/model",
            "http://127.0.0.1:1",
            json!({}),
        ));
        let (outcome, ops, mut machine) = drive_until(ocr_client(), &host, |_| {
            Err(HostFailure::Error(OcrError::InvalidRequest(
                "before_send failed".into(),
            )))
        })
        .await;
        assert!(
            matches!(outcome, Err(OcrError::InvalidRequest(message)) if message == "before_send failed")
        );
        assert_eq!(ops, ["ProjectRequest", "BeforeSend"]);
        assert!(machine.resume(None).await.is_err());
    }

    #[tokio::test]
    async fn invalid_provider_response_emits_response_received_before_normalization_failure() {
        let (base, seen, server) =
            mock_server(vec![MockResponse::json(json!({"pages":"invalid"}))]).await;
        let responses_received = Arc::new(Mutex::new(Vec::new()));
        let observed = responses_received.clone();
        let host = LocalOcrHost::new(wire_request("mistral/model", &base, json!({})))
            .with_observer(move |event| {
                if let CallEvent::Machine(MachineEvent::ResponseReceived { raw }) = event {
                    observed.lock().unwrap().push(raw.body.clone());
                }
            });
        let error = perform_ocr_with(host).await.unwrap_err();
        server.await.unwrap();
        assert!(matches!(error, OcrError::ResponseField { .. }));
        assert_eq!(seen.lock().unwrap().len(), 1);
        assert_eq!(
            *responses_received.lock().unwrap(),
            [r#"{"pages":"invalid"}"#]
        );
    }

    #[tokio::test]
    async fn direct_native_host_drives_the_same_state_machine() {
        let (base, seen, server) = mock_server(vec![MockResponse::json(json!({
            "pages":[{"index":0,"markdown":"native"}]
        }))])
        .await;
        let host = LocalOcrHost::new(wire_request("mistral/model", &base, json!({})));
        let (outcome, ops, mut machine) = drive_until(ocr_client(), &host, Ok).await;
        server.await.unwrap();
        assert_eq!(outcome.unwrap().pages[0].markdown, "native");
        assert_eq!(seen.lock().unwrap().len(), 1);
        assert_eq!(ops, ["ProjectRequest", "BeforeSend", "response"]);
        assert!(matches!(
            machine.resume(None).await,
            Err(OcrError::InvalidRequest(_))
        ));
    }

    async fn drive_native_file_call(
        request: crate::ocr::types::LiteLLMOcrRequest<crate::ocr::types::OcrDocumentInput>,
        content: Result<crate::ocr::types::OcrFileContent, OcrError>,
    ) -> (Result<LiteLLMOcrResponse, OcrError>, usize) {
        let reads = Arc::new(Mutex::new(0));
        let counted = reads.clone();
        let content = Mutex::new(Some(content));
        let host = LocalOcrHost::new(request).with_reader(move || {
            *counted.lock().unwrap() += 1;
            content.lock().unwrap().take().unwrap()
        });
        let outcome = perform_ocr_with(host).await;
        let reads = *reads.lock().unwrap();
        (outcome, reads)
    }

    #[tokio::test]
    async fn host_reader_documents_are_read_once_at_the_core_selected_point_and_encoded() {
        let (base, seen, server) = mock_server(vec![MockResponse::json(json!({
            "pages":[{"index":0,"markdown":"file"}]
        }))])
        .await;
        let request = wire_request("mistral/model", &base, json!({})).with_document(
            crate::ocr::types::OcrDocumentInput::HostReader {
                mime_type: Some("application/pdf".into()),
            },
        );
        let (response, reads) = drive_native_file_call(
            request,
            Ok(crate::ocr::types::OcrFileContent {
                bytes: b"abc".as_slice().into(),
                file_name: Some("scan.png".into()),
            }),
        )
        .await;
        server.await.unwrap();
        assert_eq!(response.unwrap().pages[0].markdown, "file");
        assert_eq!(reads, 1);
        assert!(seen.lock().unwrap()[0].contains("data:application/pdf;base64,YWJj"));
    }

    #[tokio::test]
    async fn host_reader_failures_and_empty_files_fail_before_the_provider_is_called() {
        let (base, seen, _server) = mock_server(vec![]).await;
        let request = wire_request("mistral/model", &base, json!({}));
        let failure = OcrError::InvalidRequest("reader exploded".into());
        let (response, reads) = drive_native_file_call(
            request
                .with_document(crate::ocr::types::OcrDocumentInput::HostReader { mime_type: None }),
            Err(failure.clone()),
        )
        .await;
        assert!(
            matches!(response.unwrap_err(), OcrError::InvalidRequest(message) if message == "reader exploded")
        );
        assert_eq!(reads, 1);

        let request = wire_request("mistral/model", &base, json!({}));
        let (response, _) = drive_native_file_call(
            request
                .with_document(crate::ocr::types::OcrDocumentInput::HostReader { mime_type: None }),
            Ok(crate::ocr::types::OcrFileContent {
                bytes: Default::default(),
                file_name: None,
            }),
        )
        .await;
        assert!(matches!(response.unwrap_err(), OcrError::EmptyFile));
        assert!(seen.lock().unwrap().is_empty());
    }

    #[tokio::test]
    async fn path_documents_are_read_by_core_without_a_host_operation() {
        let (base, seen, server) = mock_server(vec![MockResponse::json(json!({
            "pages":[{"index":0,"markdown":"path"}]
        }))])
        .await;
        let dir = std::env::temp_dir().join(format!("litellm-ocr-{}", rand::random::<u64>()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("scan.png");
        std::fs::write(&path, b"abc").unwrap();
        let request = wire_request("mistral/model", &base, json!({})).with_document(
            crate::ocr::types::OcrDocumentInput::Path {
                path: path.clone(),
                mime_type: None,
            },
        );
        let (response, reads) =
            drive_native_file_call(request, Err(OcrError::InvalidRequest("unused".into()))).await;
        server.await.unwrap();
        std::fs::remove_dir_all(&dir).unwrap();
        assert_eq!(response.unwrap().pages[0].markdown, "path");
        assert_eq!(reads, 0);
        assert!(seen.lock().unwrap()[0].contains("data:image/png;base64,YWJj"));

        let (base, seen, _server) = mock_server(vec![]).await;
        let request = wire_request("mistral/model", &base, json!({}));
        let (response, _) = drive_native_file_call(
            request.with_document(crate::ocr::types::OcrDocumentInput::Path {
                path: path.clone(),
                mime_type: None,
            }),
            Err(OcrError::InvalidRequest("unused".into())),
        )
        .await;
        assert!(matches!(
            response.unwrap_err(),
            OcrError::FileRead { path: failed, source } if failed == path && source.kind() == std::io::ErrorKind::NotFound
        ));
        assert!(seen.lock().unwrap().is_empty());
    }

    #[tokio::test]
    async fn cancellation_at_before_send_prevents_execution_and_further_resumption() {
        let host = LocalOcrHost::new(wire_request(
            "mistral/model",
            "http://127.0.0.1:1",
            json!({}),
        ));
        let (outcome, ops, mut machine) = drive_until(ocr_client(), &host, |_| {
            Err(HostFailure::Cancelled(OcrError::InvalidRequest(
                "cancelled".into(),
            )))
        })
        .await;
        assert!(
            matches!(outcome, Err(OcrError::InvalidRequest(message)) if message == "cancelled")
        );
        assert_eq!(ops, ["ProjectRequest", "BeforeSend"]);
        assert!(machine.resume(Some(HostResult::Emitted)).await.is_err());
    }

    #[tokio::test]
    async fn missing_host_result_preserves_pending_operation() {
        let request = wire_request("mistral/model", "http://127.0.0.1:1", json!({}));
        let mut machine = ocr_machine(ocr_client());
        assert!(matches!(
            machine.resume(None).await.unwrap(),
            MachineStep::Host(HostOp::Route(OcrOp::ProjectRequest))
        ));
        assert!(machine.resume(None).await.is_err());
        assert!(matches!(
            machine
                .resume(Some(HostResult::Route(OcrOpResult::Request {
                    request: Box::new(request),
                    caller_token: false,
                })))
                .await
                .unwrap(),
            MachineStep::Host(HostOp::BeforeSend { .. })
        ));
    }

    async fn read_bounded_response(
        response: Vec<u8>,
        limit: usize,
    ) -> Result<bytes::Bytes, OcrError> {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut request = [0; 4096];
            assert!(socket.read(&mut request).await.unwrap() > 0);
            socket.write_all(&response).await.unwrap();
            std::future::pending::<()>().await;
        });
        let response = reqwest::Client::new()
            .get(format!("http://{address}"))
            .send()
            .await
            .unwrap();
        let result = tokio::time::timeout(
            std::time::Duration::from_secs(2),
            litellm_llms::base_llm::ocr::handler::read_response_bytes(response, limit),
        )
        .await;
        server.abort();
        let _ = server.await;
        result.expect("bounded reads must finish without waiting for the rest of an oversized body")
    }

    #[tokio::test]
    async fn response_limit_accepts_exact_size_and_rejects_declared_and_chunked_overflow() {
        use litellm_llms::base_llm::ocr::error::Error;

        for response in [
            "HTTP/1.1 200 OK\r\nContent-Length: 8\r\n\r\nabcdefgh",
            "HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n4\r\nabcd\r\n4\r\nefgh\r\n0\r\n\r\n",
        ] {
            assert_eq!(
                read_bounded_response(response.as_bytes().to_vec(), 8)
                    .await
                    .unwrap(),
                "abcdefgh"
            );
        }
        for response in [
            "HTTP/1.1 200 OK\r\nContent-Length: 9\r\n\r\n",
            "HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n4\r\nabcd\r\n5\r\nefghi\r\n",
        ] {
            assert!(matches!(
                read_bounded_response(response.as_bytes().to_vec(), 8).await,
                Err(Error::TooLarge { limit: 8 })
            ));
        }
    }

    #[rstest]
    #[case::declared("Content-Length: 1000000")]
    #[case::chunked("Transfer-Encoding: chunked")]
    #[tokio::test]
    async fn oversized_error_retains_http_status_and_bounded_diagnostics_without_draining(
        #[case] headers: &str,
    ) {
        let prefix = "x".repeat(4096);
        let body = if headers.starts_with("Transfer") {
            format!("{:x}\r\n{prefix}\r\n", prefix.len())
        } else {
            prefix.clone()
        };
        let response = format!("HTTP/1.1 429 Too Many Requests\r\n{headers}\r\n\r\n{body}");
        let error = read_bounded_response(response.into_bytes(), prefix.len())
            .await
            .unwrap_err();
        match error {
            OcrError::Transport(litellm_http::transport::Error::Http { status, body }) => {
                assert_eq!(status, 429);
                assert_eq!(body, prefix);
            }
            error => panic!("unexpected error: {error}"),
        }
    }

    #[test]
    fn response_limit_is_validated_and_not_forwarded_to_the_provider() {
        let request = wire_request(
            "mistral/model",
            "http://localhost",
            json!({"max_response_bytes": 123}),
        );
        assert_eq!(request.transport.max_response_bytes, 123);
        assert!(!request.optional_params.contains_key("max_response_bytes"));
        for value in [
            json!(0),
            json!(-1),
            json!(true),
            json!("123"),
            json!(1.5),
            json!(OCR_RESPONSE_MAX_BYTES + 1),
            Value::Null,
        ] {
            let wire = serde_json::from_value(json!({
                "model": "mistral/model", "document": {"type": "document_url", "document_url": "data:application/pdf;base64,YWJj"},
                "optional_params": {"max_response_bytes": value}
            })).unwrap();
            let Err(error) = decode_request(wire) else {
                panic!("invalid response limit accepted")
            };
            assert!(error.to_string().contains("max_response_bytes"));
        }
    }

    #[derive(Debug)]
    struct PendingToken {
        entered: Arc<tokio::sync::Notify>,
        dropped: Arc<std::sync::atomic::AtomicBool>,
    }

    struct TokenFutureDrop(Arc<std::sync::atomic::AtomicBool>);

    impl Drop for TokenFutureDrop {
        fn drop(&mut self) {
            self.0.store(true, std::sync::atomic::Ordering::SeqCst);
        }
    }

    impl litellm_auth::TokenProvider for PendingToken {
        fn acquire(&self) -> litellm_auth::TokenFuture<'_> {
            Box::pin(async move {
                let _guard = TokenFutureDrop(self.dropped.clone());
                self.entered.notify_one();
                std::future::pending().await
            })
        }
    }

    #[tokio::test]
    async fn interrupt_drops_provider_captures_before_returning() {
        use std::sync::atomic::{AtomicBool, Ordering};

        let entered = Arc::new(tokio::sync::Notify::new());
        let dropped = Arc::new(AtomicBool::new(false));
        let request = wire_request("azure_ai/mistral-ocr", "https://example.invalid", json!({}));
        let request = crate::ocr::types::LiteLLMOcrRequest {
            transport: OcrTransportConfig {
                extra_headers: vec![("authorization".into(), "Bearer test-key".into())],
                ..request.transport
            },
            azure_ad_token_provider: Some(litellm_auth::TokenProviderHandle::new(Arc::new(
                PendingToken {
                    entered: entered.clone(),
                    dropped: dropped.clone(),
                },
            ))),
            ..request
        };
        let host = LocalOcrHost::new(request);
        let mut machine = ocr_machine(ocr_client());
        let mut result = None;
        tokio::time::timeout(std::time::Duration::from_secs(2), async {
            loop {
                tokio::select! {
                    _ = entered.notified() => break,
                    step = machine.resume(result.take()) => {
                        result = Some(match step.unwrap() {
                            MachineStep::Host(HostOp::Route(op)) => HostResult::Route(host.route(op).await.unwrap()),
                            MachineStep::Host(HostOp::BeforeSend { wire, .. }) => {
                                HostResult::BeforeSend(wire)
                            }
                            MachineStep::Host(HostOp::Emit(_)) => HostResult::Emitted,
                            MachineStep::Complete(_) => panic!("pending provider completed"),
                        });
                    }
                }
            }
        })
        .await
        .unwrap();
        assert!(!dropped.load(Ordering::SeqCst));
        let selected = OcrError::InvalidRequest("cancelled".into());
        let acknowledgement = machine.interrupt(HostFailure::Cancelled(selected.clone()));
        assert!(
            dropped.load(Ordering::SeqCst),
            "interrupt returned while provider captures were still alive"
        );
        assert!(
            matches!(acknowledgement.await, Err(OcrError::InvalidRequest(message)) if message == "cancelled")
        );
    }

    struct CallerTokenHost {
        request: Mutex<Option<crate::ocr::types::LiteLLMOcrRequest>>,
        trace: Mutex<Vec<String>>,
    }

    impl Host<crate::ocr::route::Ocr> for CallerTokenHost {
        async fn route(&self, op: OcrOp) -> Result<OcrOpResult, OcrError> {
            match op {
                OcrOp::ProjectRequest => {
                    self.trace.lock().unwrap().push("project".into());
                    Ok(OcrOpResult::Request {
                        request: Box::new(self.request.lock().unwrap().take().unwrap()),
                        caller_token: true,
                    })
                }
                OcrOp::AcquireAzureAdToken => {
                    self.trace.lock().unwrap().push("token".into());
                    Ok(OcrOpResult::AzureAdToken(
                        litellm_auth::ResolvedCredential::Static(litellm_auth::SecretValue::new(
                            "caller-token",
                        )),
                    ))
                }
                OcrOp::ReadDocument => Err(OcrError::InvalidRequest("no reader".into())),
            }
        }

        async fn before_send(
            &self,
            wire: WireRequest,
            _: &litellm_host::event::RequestContext,
        ) -> Result<WireRequest, OcrError> {
            let is_authorization = |name: &str| name.eq_ignore_ascii_case("authorization");
            let authorization = wire
                .headers
                .iter()
                .find(|(name, _)| is_authorization(name))
                .map(|(_, value)| value.clone())
                .unwrap_or_default();
            self.trace
                .lock()
                .unwrap()
                .push(format!("before_send:{authorization}"));
            let headers = wire
                .headers
                .into_iter()
                .map(|(name, value)| match is_authorization(&name) {
                    true => (name, "Bearer edited".to_string()),
                    false => (name, value),
                })
                .collect();
            Ok(WireRequest { headers, ..wire })
        }
    }

    #[tokio::test]
    async fn the_callers_azure_token_is_acquired_before_before_send_which_can_still_replace_it() {
        let (base, seen, server) = mock_server(vec![MockResponse::json(json!({"pages":[]}))]).await;
        let mut request = wire_request("azure_ai/model", &base, json!({}));
        request.credentials.api_key = None;
        let host = CallerTokenHost {
            request: Mutex::new(Some(request)),
            trace: Mutex::new(Vec::new()),
        };

        litellm_host::run::run(ocr_machine(ocr_client()), &host)
            .await
            .unwrap();
        server.await.unwrap();

        assert_eq!(
            *host.trace.lock().unwrap(),
            ["project", "token", "before_send:Bearer caller-token"]
        );
        assert!(
            seen.lock().unwrap()[0]
                .to_ascii_lowercase()
                .contains("authorization: bearer edited\r\n")
        );
    }

    #[tokio::test]
    async fn interrupting_an_in_flight_provider_request_closes_its_connection() {
        use tokio::io::AsyncReadExt;

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let base = format!("http://{}", listener.local_addr().unwrap());
        let received = Arc::new(tokio::sync::Notify::new());
        let server_received = received.clone();
        let server = tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut request = Vec::new();
            let mut buffer = [0u8; 4096];
            while !request.windows(4).any(|window| window == b"\r\n\r\n") {
                let read = socket.read(&mut buffer).await.unwrap();
                request.extend_from_slice(&buffer[..read]);
            }
            server_received.notify_one();
            loop {
                if socket.read(&mut buffer).await.unwrap() == 0 {
                    break;
                }
            }
        });
        let host = LocalOcrHost::new(wire_request("mistral/model", &base, json!({})));
        let mut machine = ocr_machine(ocr_client());
        let mut result = None;
        tokio::time::timeout(std::time::Duration::from_secs(2), async {
            loop {
                tokio::select! {
                    _ = received.notified() => break,
                    step = machine.resume(result.take()) => {
                        result = Some(match step.unwrap() {
                            MachineStep::Host(HostOp::Route(op)) => HostResult::Route(host.route(op).await.unwrap()),
                            MachineStep::Host(HostOp::BeforeSend { wire, .. }) => HostResult::BeforeSend(wire),
                            MachineStep::Host(HostOp::Emit(_)) => HostResult::Emitted,
                            MachineStep::Complete(_) => panic!("the stalled provider completed"),
                        });
                    }
                }
            }
        })
        .await
        .unwrap();

        let cancelled = OcrError::InvalidRequest("cancelled".into());
        assert!(
            machine
                .interrupt(HostFailure::Cancelled(cancelled))
                .await
                .is_err()
        );
        tokio::time::timeout(std::time::Duration::from_secs(1), server)
            .await
            .expect("the provider connection stayed open after the interrupt")
            .unwrap();
    }
}
