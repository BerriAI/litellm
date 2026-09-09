use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

pub struct OcrAdmissionRequest {
    pub model: String,
    pub custom_llm_provider: Option<String>,
    pub api_key: Option<String>,
    pub api_base: Option<String>,
    pub extra_headers: Vec<(String, String)>,
    pub timeout_seconds: f64,
    pub request_format: Option<String>,
    pub document: OcrDocument,
    pub credentials: litellm_auth::CredentialInputs,
    pub vertex_project: Option<String>,
    pub vertex_location: Option<String>,
    pub stream: bool,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum OcrDocument {
    DocumentUrl {
        document_url: String,
    },
    ImageUrl {
        image_url: String,
    },
    File,
    #[serde(other)]
    Unsupported,
}

impl OcrDocument {
    pub fn source_url(&self) -> Option<&str> {
        match self {
            Self::DocumentUrl { document_url } => Some(document_url),
            Self::ImageUrl { image_url } => Some(image_url),
            Self::File | Self::Unsupported => None,
        }
    }

    pub fn validate(&self, requires_data_uri: bool) -> Result<(), crate::Error> {
        let url = match self {
            Self::DocumentUrl { document_url } => document_url,
            Self::ImageUrl { image_url } => image_url,
            Self::File => return Err(crate::Error::Unsupported("OCR file document preparation")),
            Self::Unsupported => return Err(crate::Error::Unsupported("OCR document type")),
        };
        let parsed = reqwest::Url::parse(url)
            .map_err(|_| crate::Error::Unsupported("OCR local or non-HTTP document preparation"))?;
        match parsed.scheme() {
            "http" | "https" if requires_data_uri => Err(crate::Error::Unsupported(
                "OCR HTTP document URL to data URI conversion",
            )),
            "http" | "https" | "data" => Ok(()),
            _ => Err(crate::Error::Unsupported(
                "OCR local or non-HTTP document preparation",
            )),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OcrDocumentProjection {
    RetainedDocument,
    ShallowCopyDocument,
    Transformed,
}

pub struct OcrPreCallRequest {
    pub endpoint: OcrEndpoint,
    pub headers: Vec<(String, String)>,
    pub body: crate::lifecycle::PreCallBody<Map<String, Value>>,
    pub document_projection: OcrDocumentProjection,
    pub parameter_fields: &'static [&'static str],
}

impl OcrPreCallRequest {
    pub fn request_body_policy(&self) -> crate::lifecycle::RequestBodyPolicy {
        self.body.policy()
    }
}

pub struct OcrEndpoint {
    pub(super) model: String,
    pub(super) custom_llm_provider: String,
    pub(super) url: String,
    pub(super) timeout_seconds: f64,
}

impl OcrEndpoint {
    pub fn model(&self) -> &str {
        &self.model
    }

    pub fn custom_llm_provider(&self) -> &str {
        &self.custom_llm_provider
    }

    pub fn url(&self) -> &str {
        &self.url
    }

    pub fn timeout_seconds(&self) -> f64 {
        self.timeout_seconds
    }

    pub fn settle(
        self,
        headers: Vec<(String, String)>,
        body: crate::lifecycle::PreCallBody<Value>,
    ) -> Result<SettledOcrRequest, crate::Error> {
        let crate::lifecycle::PreCallBody::StructuredAtSend { callback } = body else {
            return Err(crate::Error::Unsupported("OCR request body policy"));
        };
        let wire = crate::lifecycle::WireBody::encode(&callback, "OCR request")?;
        let authorized = crate::lifecycle::AuthorizedBody::new(wire, headers);
        Ok(SettledOcrRequest {
            endpoint: self,
            http: authorized.settle(),
        })
    }
}

pub struct SettledOcrRequest {
    pub(super) endpoint: OcrEndpoint,
    pub(super) http: crate::lifecycle::SettledHttpRequest,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct OcrRequestData {
    pub data: Map<String, Value>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct OcrResponseData {
    pub pages: Vec<Value>,
    pub model: String,
    #[serde(default)]
    pub document_annotation: Option<Value>,
    #[serde(default)]
    pub usage_info: Option<Value>,
    pub object: String,
    #[serde(flatten)]
    pub extra_fields: Map<String, Value>,
    #[serde(default)]
    pub provider_native_response: Option<Value>,
}

impl OcrResponseData {
    pub fn into_json(self) -> Value {
        let mut response = serde_json::json!({
            "pages": self.pages,
            "model": self.model,
            "document_annotation": self.document_annotation,
            "usage_info": self.usage_info,
            "object": self.object,
        });
        if let Value::Object(object) = &mut response {
            object.extend(self.extra_fields);
            if let Some(native_response) = self.provider_native_response {
                object.insert("provider_native_response".to_string(), native_response);
            }
        }
        response
    }
}
