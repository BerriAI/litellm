macro_rules! ocr_format {
    ($vis:vis $name:ident {
        params: $params:ty,
        $($rest:tt)*
    }) => {
        $crate::ocr::formats::ocr_format! {
            $vis $name {
                params: $params => $params,
                map: ::std::result::Result::Ok,
                $($rest)*
            }
        }
    };
    ($vis:vis $name:ident {
        params: $input:ty => $mapped:ty,
        $(validate: $validate:path,)?
        map: $map:path,
        document: $document:ty,
        request: $request:ty => $build:path,
        response: $response:ty => $normalize:path $(,)?
    }) => {
        $vis struct $name;

        impl $crate::ocr::formats::OcrFormat for $name {
            type InputParams = $input;
            type MappedParams = $mapped;
            type PreparedDocument = $document;
            type RequestBody = $request;
            type ResponseBody = $response;

            $(fn validate_input_params(
                params: &::serde_json::Map<String, ::serde_json::Value>,
            ) -> Result<(), $crate::ocr::error::OcrRequestError> {
                $validate(params)
            })?

            #[tracing::instrument(target = "litellm::function_trace", level = "trace", skip_all)]
            fn map_params(
                params: Self::InputParams,
            ) -> Result<Self::MappedParams, $crate::ocr::error::OcrRequestError> {
                $map(params)
            }

            #[tracing::instrument(target = "litellm::function_trace", level = "trace", skip_all)]
            fn transform_request(
                model: &str,
                document: Self::PreparedDocument,
                params: &Self::MappedParams,
            ) -> Result<Self::RequestBody, $crate::ocr::error::OcrRequestError> {
                $build(model, document, params)
            }

            #[tracing::instrument(target = "litellm::function_trace", level = "trace", skip_all)]
            fn transform_response(
                model: &str,
                response: Self::ResponseBody,
                params: &Self::MappedParams,
            ) -> Result<$crate::ocr::types::OcrResponseData, $crate::ocr::error::OcrResponseError> {
                $normalize(model, response, params)
            }
        }
    };
}

pub(crate) use ocr_format;
