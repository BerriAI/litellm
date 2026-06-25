use serde_json::{Map, Value};

use crate::CoreResult;

use super::types::{ImageEditInputFile, ImageEditRequestData, ImageEditResponseData};

/// Provider-specific image-edit transforms.
///
/// Implementations stay pure and non-blocking. The route layer owns async HTTP
/// I/O and multipart/JSON transport.
pub trait ImageEditProviderConfig: Send + Sync {
    fn supported_image_edit_params(&self) -> &'static [&'static str];

    fn resolve_api_key(
        &self,
        api_key: Option<&str>,
        env_lookup: &dyn Fn(&str) -> Option<String>,
    ) -> Option<String>;

    fn complete_url(
        &self,
        api_base: Option<&str>,
        env_lookup: &dyn Fn(&str) -> Option<String>,
    ) -> CoreResult<String>;

    fn map_image_edit_params(&self, non_default_params: &Map<String, Value>) -> Map<String, Value> {
        let mut mapped_params = Map::new();
        for (param, value) in non_default_params {
            if self.supported_image_edit_params().contains(&param.as_str()) {
                mapped_params.insert(param.clone(), value.clone());
            }
        }
        mapped_params
    }

    fn transform_image_edit_request(
        &self,
        model: &str,
        images: Vec<ImageEditInputFile>,
        mask: Option<ImageEditInputFile>,
        prompt: Option<&str>,
        optional_params: Map<String, Value>,
    ) -> CoreResult<ImageEditRequestData>;

    fn transform_image_edit_response(
        &self,
        model: &str,
        response_json: Value,
    ) -> CoreResult<ImageEditResponseData>;
}
