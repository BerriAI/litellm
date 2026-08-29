use crate::CoreResult;
use crate::http_utils::safe_fetch::fetch_url_as_data_uri;

use super::types::OcrDocument;

fn is_url_requiring_fetch(url: &str) -> bool {
    !url.starts_with("data:") && (url.starts_with("http://") || url.starts_with("https://"))
}

pub(super) async fn convert_document_url_to_data_uri(
    document: OcrDocument,
) -> CoreResult<OcrDocument> {
    let (_, url) = document.url_field();
    if !is_url_requiring_fetch(url) {
        return Ok(document);
    }
    let data_uri = fetch_url_as_data_uri(url).await?;

    Ok(match document {
        OcrDocument::DocumentUrl { extra, .. } => OcrDocument::DocumentUrl {
            document_url: data_uri,
            extra,
        },
        OcrDocument::ImageUrl { extra, .. } => OcrDocument::ImageUrl {
            image_url: data_uri,
            extra,
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::CoreError;

    #[tokio::test]
    async fn convert_document_url_rejects_loopback_fetch() {
        let error = convert_document_url_to_data_uri(OcrDocument::ImageUrl {
            image_url: "http://127.0.0.1/image.png".to_string(),
            extra: Default::default(),
        })
        .await
        .unwrap_err();

        assert!(matches!(
            error,
            CoreError::Request(crate::RequestError::InvalidRequest(message))
                if message.contains("SSRF protection")
        ));
    }

    #[tokio::test]
    async fn convert_document_url_leaves_data_uri_untouched() {
        let document = OcrDocument::ImageUrl {
            image_url: "data:image/png;base64,abcd".to_string(),
            extra: Default::default(),
        };

        let transformed = convert_document_url_to_data_uri(document.clone())
            .await
            .unwrap();

        assert_eq!(transformed, document);
    }
}
