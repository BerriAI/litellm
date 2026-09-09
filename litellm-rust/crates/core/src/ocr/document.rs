use data_url::{DataUrl, DataUrlError, forgiving_base64::DecodeError, mime::Mime};

use super::error::OcrRequestError;

pub(crate) struct InlineDocument<'a>(DataUrl<'a>);

impl<'a> InlineDocument<'a> {
    pub(crate) fn parse(source: &'a str) -> Result<Option<Self>, OcrRequestError> {
        match DataUrl::process(source) {
            Ok(url) => Ok(Some(Self(url))),
            Err(DataUrlError::NotADataUrl) => Ok(None),
            Err(DataUrlError::NoComma) => Err(OcrRequestError::InvalidDataUri),
        }
    }

    pub(crate) fn mime_type(&self) -> &Mime {
        self.0.mime_type()
    }

    pub(crate) fn decode(&self, max_bytes: usize) -> Result<Vec<u8>, OcrRequestError> {
        let mut body = Vec::new();
        self.0
            .decode(|bytes| {
                if bytes.len() > max_bytes.saturating_sub(body.len()) {
                    return Err(OcrRequestError::InlineDocumentTooLarge);
                }
                body.extend_from_slice(bytes);
                Ok(())
            })
            .map_err(|error| match error {
                DecodeError::InvalidBase64(_) => OcrRequestError::InvalidDataUri,
                DecodeError::WriteError(error) => error,
            })?;
        Ok(body)
    }
}

#[cfg(test)]
mod tests {
    use super::InlineDocument;
    use crate::ocr::error::OcrRequestError;
    use rstest::rstest;

    #[rstest]
    #[case("data:application/pdf;base64,YWJj", b"abc")]
    #[case("DATA:application/pdf;BASE64,YWI", b"ab")]
    #[case("data:application/pdf;base64,YQ%3D%3D#fragment", b"a")]
    #[case("data:application/pdf;base64,Y W J j", b"abc")]
    #[case("data:,a%20b%00%FF", b"a b\0\xff")]
    #[case("data:;base64,", b"")]
    fn decodes_data_urls(#[case] source: &str, #[case] expected: &[u8]) {
        let document = InlineDocument::parse(source).unwrap().unwrap();
        assert_eq!(document.decode(expected.len()).unwrap(), expected);
        if !expected.is_empty() {
            assert_eq!(
                document.decode(expected.len() - 1),
                Err(OcrRequestError::InlineDocumentTooLarge)
            );
        }
    }

    #[test]
    fn preserves_mime_parameters_and_uses_standard_default() {
        let document = InlineDocument::parse("data:application/pdf;version=1.7;base64,YQ==")
            .unwrap()
            .unwrap();
        assert!(document.mime_type().matches("application", "pdf"));
        assert_eq!(document.mime_type().get_parameter("version"), Some("1.7"));
        let default = InlineDocument::parse("data:,a").unwrap().unwrap();
        assert!(default.mime_type().matches("text", "plain"));
        assert_eq!(
            default.mime_type().get_parameter("charset"),
            Some("US-ASCII")
        );
    }

    #[rstest]
    #[case("data:application/pdf;base64")]
    #[case("data:application/pdf;base64,INVALID!")]
    #[case("data:application/pdf;base64,Y")]
    fn rejects_invalid_input(#[case] source: &str) {
        let result =
            InlineDocument::parse(source).and_then(|document| document.unwrap().decode(100));
        assert_eq!(result, Err(OcrRequestError::InvalidDataUri));
    }

    #[test]
    fn leaves_other_sources_to_the_provider() {
        for source in ["https://example.com/file.pdf", "reducto://file.pdf"] {
            assert!(InlineDocument::parse(source).unwrap().is_none());
        }
    }

    #[test]
    fn limits_total_decoded_bytes_across_chunks() {
        use base64::{Engine, engine::general_purpose::STANDARD};

        let bytes = vec![0xff; 8193];
        let source = format!("data:application/pdf;base64,{}", STANDARD.encode(&bytes));
        let document = InlineDocument::parse(&source).unwrap().unwrap();
        assert_eq!(document.decode(bytes.len()).unwrap(), bytes);
        assert_eq!(
            document.decode(bytes.len() - 1),
            Err(OcrRequestError::InlineDocumentTooLarge)
        );
    }
}
