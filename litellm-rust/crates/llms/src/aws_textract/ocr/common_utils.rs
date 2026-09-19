use base64::{Engine, engine::general_purpose::STANDARD};
use litellm_auth_aws::{SigV4Signer, resolve_aws_region};
use litellm_http::outbound::RequestSigner;
use serde::{Deserialize, Serialize};

use crate::base_llm::ocr::{
    document::{InlineDocument, inline_remote_document},
    error::Error,
    transformation::{
        OCR_INLINE_MAX_BYTES, OcrDocument, OcrEnvironment, OcrRequestContext, PreparedOcrRequest,
    },
};

const TEXTRACT_SERVICE: &str = "textract";
const AWS_JSON_CONTENT_TYPE: &str = "application/x-amz-json-1.1";
const UNSUPPORTED_DOCUMENT: &str = "UnsupportedDocumentException";

pub(super) const HEALTH_CHECK_IMAGE_DATA_URI: &str = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4//8/AAX+Av4N70a4AAAAAElFTkSuQmCC";

#[derive(Debug, Deserialize, Serialize)]
pub struct TextractDocument {
    #[serde(rename = "Bytes")]
    pub bytes: String,
}

#[derive(Deserialize)]
#[serde(rename_all = "PascalCase")]
pub(super) struct Block {
    #[serde(default)]
    pub id: String,
    pub block_type: String,
    pub text: Option<String>,
    pub page: Option<i64>,
    pub row_index: Option<usize>,
    pub column_index: Option<usize>,
    #[serde(default)]
    pub relationships: Vec<Relationship>,
}

#[derive(Deserialize)]
#[serde(rename_all = "PascalCase")]
pub(super) struct Relationship {
    pub r#type: String,
    #[serde(default)]
    pub ids: Vec<String>,
}

impl Block {
    /// The synchronous API omits `Page` because it only ever reads one.
    pub fn page(&self) -> i64 {
        self.page.unwrap_or(1)
    }

    pub fn children(&self) -> impl Iterator<Item = &str> {
        self.relationships
            .iter()
            .filter(|relationship| relationship.r#type == "CHILD")
            .flat_map(|relationship| relationship.ids.iter().map(String::as_str))
    }
}

#[derive(Deserialize)]
#[serde(rename_all = "PascalCase")]
pub(super) struct DocumentMetadata {
    pub pages: Option<i64>,
}

pub struct TextractEnvironment {
    headers: Vec<(String, String)>,
    region: String,
    signer: SigV4Signer,
}

impl OcrEnvironment for TextractEnvironment {
    fn headers(&self) -> &[(String, String)] {
        &self.headers
    }

    fn signer(&self) -> Option<&dyn RequestSigner> {
        Some(&self.signer)
    }
}

pub(super) async fn environment(
    request: &PreparedOcrRequest,
    target: &'static str,
) -> Result<TextractEnvironment, Error> {
    let env_lookup = |name: &str| request.connection.secret(name);
    let region =
        resolve_aws_region(None, &request.optional_params, &env_lookup).ok_or_else(|| {
            Error::InvalidRequest(
                "Missing AWS region - pass aws_region_name or set AWS_REGION_NAME or AWS_REGION"
                    .into(),
            )
        })?;
    let signer = SigV4Signer::resolve(
        region.clone(),
        TEXTRACT_SERVICE,
        &request.optional_params,
        &env_lookup,
    )
    .await
    .map_err(litellm_auth::Error::from)?;
    Ok(TextractEnvironment {
        headers: request
            .connection
            .extra_headers
            .iter()
            .cloned()
            .chain([
                ("X-Amz-Target".into(), target.into()),
                ("Content-Type".into(), AWS_JSON_CONTENT_TYPE.into()),
            ])
            .collect(),
        region,
        signer,
    })
}

pub(super) fn endpoint(request: &PreparedOcrRequest, environment: &TextractEnvironment) -> String {
    request
        .connection
        .api_base
        .clone()
        .unwrap_or_else(|| format!("https://textract.{}.amazonaws.com/", environment.region))
}

pub(super) fn document_bytes(document: &OcrDocument) -> Result<TextractDocument, Error> {
    let inline = InlineDocument::parse(document.source())?.ok_or(Error::InvalidDataUri)?;
    Ok(TextractDocument {
        bytes: STANDARD.encode(inline.decode(OCR_INLINE_MAX_BYTES)?),
    })
}

pub(super) async fn inline_document(
    document: OcrDocument,
    context: OcrRequestContext<'_>,
) -> Result<OcrDocument, Error> {
    inline_remote_document(
        context.client.document_fetcher(),
        document,
        context.connection,
    )
    .await
}

#[derive(Deserialize)]
struct AwsError {
    #[serde(rename = "__type", default)]
    kind: String,
    #[serde(rename = "Message", alias = "message", default)]
    message: String,
}

/// Textract answers a multi-page PDF or TIFF with a bare "unsupported document
/// format", which reads like a corrupt file. Say what the limit is.
pub(super) fn error_class(body: String, status: u16, headers: Vec<(String, String)>) -> Error {
    let unsupported = serde_json::from_str::<AwsError>(&body)
        .ok()
        .filter(|error| error.kind.ends_with(UNSUPPORTED_DOCUMENT));
    Error::Provider {
        status,
        body: match unsupported {
            Some(error) => format!(
                "{UNSUPPORTED_DOCUMENT}: {}. aws_textract uses Textract's synchronous API, which reads a JPEG, PNG, or a single-page PDF or TIFF; multi-page documents are not supported",
                error.message
            ),
            None => body,
        },
        headers,
    }
}

pub(super) fn lines_by_page(blocks: &[Block]) -> Vec<(i64, String)> {
    let pages: std::collections::BTreeSet<i64> = blocks.iter().map(Block::page).collect();
    pages
        .into_iter()
        .map(|page| {
            let lines: Vec<&str> = blocks
                .iter()
                .filter(|block| block.block_type == "LINE" && block.page() == page)
                .filter_map(|block| block.text.as_deref())
                .collect();
            (page, lines.join("\n"))
        })
        .filter(|(_, markdown)| !markdown.is_empty())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_multi_page_rejection_names_the_single_page_limit_and_keeps_the_status() {
        let error = error_class(
            r#"{"__type":"UnsupportedDocumentException","Message":"Request has unsupported document format"}"#.into(),
            400,
            vec![("x-amzn-requestid".into(), "abc".into())],
        );

        let Error::Provider {
            status,
            body,
            headers,
        } = error
        else {
            panic!("expected a provider error");
        };
        assert_eq!(status, 400);
        assert!(body.contains("Request has unsupported document format"));
        assert!(body.contains("single-page PDF or TIFF"));
        assert_eq!(headers, vec![("x-amzn-requestid".into(), "abc".into())]);
    }

    #[test]
    fn a_namespaced_exception_type_is_recognized() {
        let Error::Provider { body, .. } = error_class(
            r#"{"__type":"com.amazonaws.textract#UnsupportedDocumentException","message":"bad"}"#
                .into(),
            400,
            Vec::new(),
        ) else {
            panic!("expected a provider error");
        };
        assert!(body.contains("multi-page documents are not supported"));
    }

    #[test]
    fn other_provider_errors_pass_through_untouched() {
        for body in [
            r#"{"__type":"AccessDeniedException","Message":"no"}"#,
            "<html>bad gateway</html>",
        ] {
            let Error::Provider {
                body: reported,
                status,
                ..
            } = error_class(body.into(), 403, Vec::new())
            else {
                panic!("expected a provider error");
            };
            assert_eq!(reported, body);
            assert_eq!(status, 403);
        }
    }
}
