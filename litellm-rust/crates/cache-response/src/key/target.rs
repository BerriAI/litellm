use serde::Serialize;

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CacheTarget {
    ModelGroup(String),
    Deployment {
        model: String,
        provider: Option<String>,
        api_base: Option<String>,
    },
}

impl CacheTarget {
    pub fn resolve(
        model_group: Option<&str>,
        model: &str,
        provider: Option<&str>,
        api_base: Option<&str>,
    ) -> Self {
        match model_group {
            Some(group) => Self::ModelGroup(group.to_owned()),
            None => Self::Deployment {
                model: model.to_owned(),
                provider: provider.map(str::to_owned),
                api_base: api_base.map(str::to_owned),
            },
        }
    }
}
