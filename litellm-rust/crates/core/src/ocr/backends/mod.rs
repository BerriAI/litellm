pub(crate) mod azure_ai;
pub(crate) mod azure_document_intelligence;
pub(crate) mod mistral;
pub(crate) mod reducto;

use serde_json::{Map, Value};

use crate::Error;

pub struct PreparedOcrBackend {
    pub url: String,
    pub headers: Vec<(String, String)>,
}

pub trait OcrBackend: Send + Sync + 'static {
    type Config: Clone + std::fmt::Debug + Send + Sync + 'static;
    const PROVIDER: super::registry::OcrProvider;

    fn decode_config(params: &Map<String, Value>) -> Result<Self::Config, Error>;
}
