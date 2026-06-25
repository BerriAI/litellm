use litellm_core::error::{json_type_name, CoreError, CoreResult};
use litellm_core::image_edit::transformation::ImageEditProviderConfig;
use litellm_core::image_edit::types::{
    ImageEditInputFile, ImageEditMultipartPart, ImageEditRequestData, ImageEditRequestFormat,
    ImageEditResponseData,
};
use serde_json::{Map, Value};

const SUPPORTED_IMAGE_EDIT_PARAMS: &[&str] = &[
    "background",
    "input_fidelity",
    "mask",
    "n",
    "quality",
    "response_format",
    "size",
    "user",
    "imageConfig",
];

pub const VLLM_API_BASE_ENV: &str = "VLLM_API_BASE";
pub const VLLM_API_KEY_ENV: &str = "VLLM_API_KEY";

pub const MISSING_API_BASE_MESSAGE: &str = "VLLM_API_BASE is not set. Please set the environment variable, to use VLLM's image edit endpoint.";

pub fn complete_url(
    api_base: Option<&str>,
    env_lookup: &dyn Fn(&str) -> Option<String>,
) -> CoreResult<String> {
    let base = api_base
        .map(str::trim)
        .filter(|base| !base.is_empty())
        .map(str::to_string)
        .or_else(|| env_lookup(VLLM_API_BASE_ENV).filter(|base| !base.trim().is_empty()))
        .ok_or_else(|| CoreError::Auth(MISSING_API_BASE_MESSAGE.to_string()))?;

    let base = base.trim_end_matches('/');
    if base.ends_with("/v1") {
        Ok(format!("{base}/images/edits"))
    } else {
        Ok(format!("{base}/v1/images/edits"))
    }
}

pub fn resolve_api_key(
    api_key: Option<&str>,
    env_lookup: &dyn Fn(&str) -> Option<String>,
) -> Option<String> {
    api_key
        .map(str::trim)
        .filter(|key| !key.is_empty())
        .map(str::to_string)
        .or_else(|| env_lookup(VLLM_API_KEY_ENV).filter(|key| !key.trim().is_empty()))
}

pub struct VllmImageEditConfig;

pub const VLLM_IMAGE_EDIT_CONFIG: VllmImageEditConfig = VllmImageEditConfig;

impl ImageEditProviderConfig for VllmImageEditConfig {
    fn supported_image_edit_params(&self) -> &'static [&'static str] {
        SUPPORTED_IMAGE_EDIT_PARAMS
    }

    fn resolve_api_key(
        &self,
        api_key: Option<&str>,
        env_lookup: &dyn Fn(&str) -> Option<String>,
    ) -> Option<String> {
        resolve_api_key(api_key, env_lookup)
    }

    fn complete_url(
        &self,
        api_base: Option<&str>,
        env_lookup: &dyn Fn(&str) -> Option<String>,
    ) -> CoreResult<String> {
        complete_url(api_base, env_lookup)
    }

    fn transform_image_edit_request(
        &self,
        model: &str,
        images: Vec<ImageEditInputFile>,
        mask: Option<ImageEditInputFile>,
        prompt: Option<&str>,
        optional_params: Map<String, Value>,
    ) -> CoreResult<ImageEditRequestData> {
        if images.is_empty() {
            return Err(CoreError::MissingField("image"));
        }

        let mut data = Map::new();
        data.insert("model".to_string(), Value::String(model.to_string()));
        if let Some(prompt) = prompt {
            data.insert("prompt".to_string(), Value::String(prompt.to_string()));
        }
        for (param, value) in optional_params {
            if param != "mask" {
                data.insert(param, value);
            }
        }

        let mut files = Vec::new();
        for image in images {
            files.push(ImageEditMultipartPart {
                field_name: "image[]".to_string(),
                filename: image.filename,
                content_type: image.content_type,
                data_base64: image.data_base64,
            });
        }
        if let Some(mask) = mask {
            files.push(ImageEditMultipartPart {
                field_name: "mask".to_string(),
                filename: mask.filename,
                content_type: mask.content_type,
                data_base64: mask.data_base64,
            });
        }

        Ok(ImageEditRequestData {
            data,
            files,
            format: ImageEditRequestFormat::Multipart,
        })
    }

    fn transform_image_edit_response(
        &self,
        _model: &str,
        response_json: Value,
    ) -> CoreResult<ImageEditResponseData> {
        let data = response_json
            .as_object()
            .cloned()
            .ok_or_else(|| CoreError::InvalidType {
                expected: "object",
                actual: json_type_name(&response_json),
            })?;
        Ok(ImageEditResponseData { data })
    }
}

pub fn supported_image_edit_params() -> &'static [&'static str] {
    VLLM_IMAGE_EDIT_CONFIG.supported_image_edit_params()
}

pub fn map_image_edit_params(non_default_params: &Map<String, Value>) -> Map<String, Value> {
    VLLM_IMAGE_EDIT_CONFIG.map_image_edit_params(non_default_params)
}

pub fn transform_image_edit_request(
    model: &str,
    images: Vec<ImageEditInputFile>,
    mask: Option<ImageEditInputFile>,
    prompt: Option<&str>,
    optional_params: Map<String, Value>,
) -> CoreResult<ImageEditRequestData> {
    VLLM_IMAGE_EDIT_CONFIG.transform_image_edit_request(
        model,
        images,
        mask,
        prompt,
        optional_params,
    )
}

pub fn transform_image_edit_response(
    model: &str,
    response_json: Value,
) -> CoreResult<ImageEditResponseData> {
    VLLM_IMAGE_EDIT_CONFIG.transform_image_edit_response(model, response_json)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn image_part() -> ImageEditInputFile {
        ImageEditInputFile {
            filename: "image.png".to_string(),
            content_type: "image/png".to_string(),
            data_base64: "aW1hZ2U=".to_string(),
        }
    }

    #[test]
    fn supported_params_match_openai_compatible_image_edit_subset() {
        assert_eq!(
            supported_image_edit_params(),
            &[
                "background",
                "input_fidelity",
                "mask",
                "n",
                "quality",
                "response_format",
                "size",
                "user",
                "imageConfig",
            ]
        );
    }

    #[test]
    fn map_image_edit_params_drops_unknown_params() {
        let params = json!({
            "quality": "high",
            "unsupported_param": "value",
            "size": "1024x1024"
        });
        let mapped = map_image_edit_params(params.as_object().unwrap());

        assert_eq!(mapped.get("quality"), Some(&json!("high")));
        assert_eq!(mapped.get("size"), Some(&json!("1024x1024")));
        assert!(!mapped.contains_key("unsupported_param"));
    }

    #[test]
    fn transform_request_builds_openai_compatible_multipart_body() {
        let optional_params = json!({
            "quality": "high",
            "size": "1024x1024"
        })
        .as_object()
        .unwrap()
        .clone();

        let request = transform_image_edit_request(
            "qwen-image-edit",
            vec![image_part()],
            Some(ImageEditInputFile {
                filename: "mask.png".to_string(),
                content_type: "image/png".to_string(),
                data_base64: "bWFzaw==".to_string(),
            }),
            Some("make it brighter"),
            optional_params,
        )
        .expect("request transforms");

        assert_eq!(request.format, ImageEditRequestFormat::Multipart);
        assert_eq!(request.data["model"], json!("qwen-image-edit"));
        assert_eq!(request.data["prompt"], json!("make it brighter"));
        assert_eq!(request.data["quality"], json!("high"));
        assert_eq!(request.files[0].field_name, "image[]");
        assert_eq!(request.files[1].field_name, "mask");
    }

    #[test]
    fn transform_request_requires_an_image() {
        let err = transform_image_edit_request("model", Vec::new(), None, None, Map::new())
            .expect_err("empty images rejected");
        assert_eq!(err, CoreError::MissingField("image"));
    }

    #[test]
    fn complete_url_uses_vllm_base_and_dedupes_v1() {
        assert_eq!(
            complete_url(Some("http://localhost:8000"), &|_| None).unwrap(),
            "http://localhost:8000/v1/images/edits"
        );
        assert_eq!(
            complete_url(Some("http://localhost:8000/v1/"), &|_| None).unwrap(),
            "http://localhost:8000/v1/images/edits"
        );
    }
}
