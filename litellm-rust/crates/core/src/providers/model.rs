use std::marker::PhantomData;

use serde::{Deserialize, Deserializer, Serialize, Serializer, de::Error as _};
use thiserror::Error;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Error)]
pub(crate) enum ModelNameError {
    #[error("model name cannot be empty")]
    EmptyModel,
    #[error("model namespace must be one non-empty path segment: {0}")]
    InvalidNamespace(&'static str),
}

pub(crate) trait ModelNamespace {
    const NAME: &'static str;
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct RoutedModel<'a>(&'a str);

impl<'a> RoutedModel<'a> {
    pub(crate) fn new(value: &'a str) -> Result<Self, ModelNameError> {
        if value.is_empty() {
            return Err(ModelNameError::EmptyModel);
        }
        Ok(Self(value))
    }

    pub(crate) fn into_provider<N: ModelNamespace>(
        self,
    ) -> Result<ProviderModel<N>, ModelNameError> {
        let namespace = N::NAME;
        if namespace.is_empty() || namespace.contains('/') {
            return Err(ModelNameError::InvalidNamespace(namespace));
        }
        let prefix = format!("{namespace}/");
        let local_model = self.0.trim_start_matches(prefix.as_str());
        if local_model.is_empty() {
            return Err(ModelNameError::EmptyModel);
        }
        Ok(ProviderModel {
            value: format!("{prefix}{local_model}"),
            namespace: PhantomData,
        })
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct ProviderModel<N> {
    value: String,
    namespace: PhantomData<N>,
}

impl<N> ProviderModel<N> {
    #[cfg(test)]
    pub(crate) fn as_str(&self) -> &str {
        &self.value
    }
}

impl<N> Serialize for ProviderModel<N> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        self.value.serialize(serializer)
    }
}

impl<'de, N: ModelNamespace> Deserialize<'de> for ProviderModel<N> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        RoutedModel::new(&value)
            .and_then(RoutedModel::into_provider::<N>)
            .map_err(D::Error::custom)
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[derive(Clone, Debug, Eq, PartialEq)]
    struct DeepSeekAi;

    impl ModelNamespace for DeepSeekAi {
        const NAME: &'static str = "deepseek-ai";
    }

    #[derive(Clone, Debug, Eq, PartialEq)]
    struct FalAi;

    impl ModelNamespace for FalAi {
        const NAME: &'static str = "fal-ai";
    }

    #[test]
    fn qualifies_a_bare_model() {
        let model = RoutedModel::new("deepseek-ocr-maas")
            .and_then(RoutedModel::into_provider::<DeepSeekAi>)
            .unwrap();

        assert_eq!(model.as_str(), "deepseek-ai/deepseek-ocr-maas");
    }

    #[test]
    fn preserves_an_already_qualified_model() {
        let model = RoutedModel::new("deepseek-ai/deepseek-ocr-maas")
            .and_then(RoutedModel::into_provider::<DeepSeekAi>)
            .unwrap();

        assert_eq!(model.as_str(), "deepseek-ai/deepseek-ocr-maas");
    }

    #[test]
    fn collapses_repeated_owned_namespaces() {
        let model = RoutedModel::new("deepseek-ai/deepseek-ai/deepseek-ai/deepseek-ocr-maas")
            .and_then(RoutedModel::into_provider::<DeepSeekAi>)
            .unwrap();

        assert_eq!(model.as_str(), "deepseek-ai/deepseek-ocr-maas");
    }

    #[test]
    fn matches_the_namespace_as_a_complete_segment() {
        let model = RoutedModel::new("deepseek-ai-v2/model")
            .and_then(RoutedModel::into_provider::<DeepSeekAi>)
            .unwrap();

        assert_eq!(model.as_str(), "deepseek-ai/deepseek-ai-v2/model");
    }

    #[test]
    fn preserves_nested_provider_model_paths() {
        let model = RoutedModel::new("publishers/vendor/models/model-v1")
            .and_then(RoutedModel::into_provider::<DeepSeekAi>)
            .unwrap();

        assert_eq!(
            model.as_str(),
            "deepseek-ai/publishers/vendor/models/model-v1"
        );
    }

    #[test]
    fn namespace_markers_select_different_wire_names() {
        let routed = RoutedModel::new("model-v1").unwrap();
        let deepseek = routed.into_provider::<DeepSeekAi>().unwrap();
        let fal = routed.into_provider::<FalAi>().unwrap();

        assert_eq!(deepseek.as_str(), "deepseek-ai/model-v1");
        assert_eq!(fal.as_str(), "fal-ai/model-v1");
    }

    #[test]
    fn rejects_empty_routed_models() {
        assert_eq!(RoutedModel::new(""), Err(ModelNameError::EmptyModel));
    }

    #[test]
    fn rejects_a_namespace_without_a_model() {
        let result =
            RoutedModel::new("deepseek-ai/").and_then(RoutedModel::into_provider::<DeepSeekAi>);

        assert_eq!(result, Err(ModelNameError::EmptyModel));
    }

    #[test]
    fn rejects_invalid_namespace_markers() {
        struct Empty;
        impl ModelNamespace for Empty {
            const NAME: &'static str = "";
        }
        struct MultipleSegments;
        impl ModelNamespace for MultipleSegments {
            const NAME: &'static str = "one/two";
        }

        assert!(matches!(
            RoutedModel::new("model").and_then(RoutedModel::into_provider::<Empty>),
            Err(ModelNameError::InvalidNamespace(""))
        ));
        assert!(matches!(
            RoutedModel::new("model").and_then(RoutedModel::into_provider::<MultipleSegments>),
            Err(ModelNameError::InvalidNamespace("one/two"))
        ));
    }

    #[test]
    fn provider_models_serialize_as_plain_strings() {
        let model = RoutedModel::new("deepseek-ocr-maas")
            .and_then(RoutedModel::into_provider::<DeepSeekAi>)
            .unwrap();

        assert_eq!(
            serde_json::to_value(model).unwrap(),
            json!("deepseek-ai/deepseek-ocr-maas")
        );
    }

    #[test]
    fn deserialization_reestablishes_the_namespace_invariant() {
        let model: ProviderModel<DeepSeekAi> =
            serde_json::from_value(json!("deepseek-ai/deepseek-ai/model-v1")).unwrap();

        assert_eq!(model.as_str(), "deepseek-ai/model-v1");
    }

    #[test]
    fn deserialization_rejects_missing_model_names() {
        let result = serde_json::from_value::<ProviderModel<DeepSeekAi>>(json!("deepseek-ai/"));

        assert!(result.is_err());
    }
}
