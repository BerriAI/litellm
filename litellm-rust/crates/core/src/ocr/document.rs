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
#[path = "../../tests/ocr_document.rs"]
mod tests;
