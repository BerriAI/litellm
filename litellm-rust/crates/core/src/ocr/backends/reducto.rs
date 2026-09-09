use crate::constants::{REDUCTO_ID_PREFIX, REDUCTO_OCR_API_BASE};
use crate::ocr::backends::{HostConfig, OcrHost, OcrIntegration, PreparedOcrBackend};
use crate::ocr::document::InlineDocument;
use crate::ocr::error::{OcrError, OcrRequestError, OcrResponseError};
use crate::ocr::formats::reducto::{
    ReductoParseLegacyFormat, ReductoParseV3Format,
    types::{ReductoFileId, ReductoLegacyParams, ReductoUploadResponse, ReductoV3Params},
};
use crate::ocr::types::{OcrConnection, OcrDocument};
use crate::providers::reducto::auth;

#[derive(Clone, Debug)]
pub struct ReductoHost;

impl OcrHost for ReductoHost {
    type Config = ();
    const PROVIDER: crate::ocr::registry::OcrProvider = crate::ocr::registry::OcrProvider::Reducto;
}

pub fn normalize_api_base(api_base: Option<&str>) -> &str {
    api_base
        .map(str::trim)
        .filter(|base| !base.is_empty())
        .unwrap_or(REDUCTO_OCR_API_BASE)
        .trim_end_matches('/')
}

async fn prepare_reducto_document(
    client: &crate::ocr::OcrClient,
    document: OcrDocument,
    connection: &OcrConnection,
    headers: &[(String, String)],
) -> Result<ReductoFileId, OcrError> {
    let source = document.source();
    if source.starts_with(REDUCTO_ID_PREFIX) {
        return Ok(ReductoFileId(source.to_string()));
    }
    let document = InlineDocument::parse(source)?.ok_or(OcrRequestError::ReductoSource)?;
    let mime = document.mime_type().to_string();
    let bytes = document.decode(crate::constants::OCR_INLINE_MAX_BYTES)?;
    let part = reqwest::multipart::Part::bytes(bytes)
        .file_name("document")
        .mime_str(&mime)
        .map_err(|_| OcrRequestError::InvalidDataUri)?;
    let mut builder = client
        .provider_http()
        .post(format!(
            "{}/upload",
            normalize_api_base(connection.api_base.as_deref())
        ))
        .multipart(reqwest::multipart::Form::new().part("file", part))
        .timeout(connection.timeout);
    for (name, value) in headers {
        if !name.eq_ignore_ascii_case("content-type")
            && !name.eq_ignore_ascii_case("content-length")
        {
            builder = builder.header(name, value);
        }
    }
    let response = crate::http_utils::http_request(builder)
        .await
        .map_err(crate::error::TransportError::from)?;
    let response = crate::ocr::client::read_json_response::<ReductoUploadResponse>(response, false)
        .await?
        .data;
    if response.file_id.is_empty() {
        return Err(OcrResponseError::ResponseField {
            path: "file_id".into(),
        }
        .into());
    }
    Ok(ReductoFileId(response.file_id))
}

macro_rules! impl_reducto_backend {
    ($integration:ident, $format:ident, $params:ty) => {
        #[derive(Clone, Debug)]
        pub struct $integration;

        impl OcrIntegration for $integration {
            type Host = ReductoHost;
            type Format = $format;
            type PreparedDocument = ReductoFileId;
            const FORMAT: Self::Format = $format;

            async fn prepare(
                &self,
                connection: &OcrConnection,
                _config: &HostConfig<Self>,
                _model: &str,
                _params: &$params,
                env_lookup: &(dyn Fn(&str) -> Option<String> + Sync),
            ) -> Result<PreparedOcrBackend, OcrError> {
                let headers = auth::validate_environment(
                    connection.extra_headers.clone(),
                    connection.api_key.as_deref(),
                    env_lookup,
                )?;
                Ok(PreparedOcrBackend {
                    url: format!(
                        "{}/parse",
                        normalize_api_base(connection.api_base.as_deref())
                    ),
                    headers,
                })
            }

            async fn prepare_document(
                &self,
                client: &crate::ocr::OcrClient,
                document: OcrDocument,
                connection: &OcrConnection,
                headers: &[(String, String)],
            ) -> Result<ReductoFileId, OcrError> {
                prepare_reducto_document(client, document, connection, headers).await
            }

            fn guard_document_before_preparation(&self) -> bool {
                true
            }

            fn preserve_native_response(&self, _params: &$params) -> bool {
                true
            }
        }
    };
}

impl_reducto_backend!(ReductoV3, ReductoParseV3Format, ReductoV3Params);
impl_reducto_backend!(ReductoLegacy, ReductoParseLegacyFormat, ReductoLegacyParams);
