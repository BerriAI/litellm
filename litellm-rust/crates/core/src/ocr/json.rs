use serde::de::{DeserializeOwned, IntoDeserializer};
use serde_json::{Map, Value};

#[derive(Debug)]
pub struct DecodedOcrResponse<T> {
    pub data: T,
    pub native: Option<Map<String, Value>>,
    pub text: String,
}

pub(crate) fn decode_request_value<T: DeserializeOwned>(
    value: Value,
    prefix: &str,
) -> Result<T, crate::ocr::Error> {
    serde_path_to_error::deserialize(value.into_deserializer()).map_err(|error| {
        crate::ocr::Error::RequestField {
            path: format!("{prefix}.{}", error.path()),
        }
    })
}

pub(crate) fn decode_response_value<T: DeserializeOwned>(
    value: Value,
    prefix: &str,
) -> Result<T, crate::ocr::Error> {
    serde_path_to_error::deserialize(value.into_deserializer()).map_err(|error| {
        crate::ocr::Error::ResponseField {
            path: format!("{prefix}.{}", error.path()),
        }
    })
}

pub(crate) fn decode_response<T: DeserializeOwned>(
    bytes: &[u8],
    native: bool,
) -> Result<DecodedOcrResponse<T>, crate::ocr::Error> {
    let mut deserializer = serde_json::Deserializer::from_slice(bytes);
    let data = serde_path_to_error::deserialize(&mut deserializer).map_err(|error| {
        crate::ocr::Error::ResponseField {
            path: error.path().to_string(),
        }
    })?;
    deserializer
        .end()
        .map_err(|_| crate::ocr::Error::ResponseField {
            path: "response".into(),
        })?;
    let native = if native {
        Some(
            serde_json::from_slice(bytes).map_err(|_| crate::ocr::Error::ResponseField {
                path: "response".into(),
            })?,
        )
    } else {
        None
    };
    Ok(DecodedOcrResponse {
        data,
        native,
        text: String::from_utf8_lossy(bytes).into_owned(),
    })
}
