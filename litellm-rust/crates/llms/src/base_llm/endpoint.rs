use litellm_core_utils::{
    ApiUrlError,
    url_utils::{ApiUrl, Complete, EndpointTarget},
};
use reqwest::Method;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResolvedEndpoint {
    method: Method,
    url: ApiUrl<Complete>,
}

impl ResolvedEndpoint {
    pub fn new(method: Method, url: ApiUrl<Complete>) -> Self {
        Self { method, url }
    }
    pub fn parse_exact(method: Method, url: &str) -> Result<Self, ApiUrlError> {
        Ok(Self::new(method, ApiUrl::<Complete>::parse_exact(url)?))
    }
    pub fn url(&self) -> &ApiUrl<Complete> {
        &self.url
    }
    pub fn method(&self) -> &Method {
        &self.method
    }
    pub fn into_parts(self) -> (Method, ApiUrl<Complete>) {
        (self.method, self.url)
    }
}

pub trait ProviderEndpoint {
    fn resolve(&self, target: &EndpointTarget) -> Result<ResolvedEndpoint, ApiUrlError>;
}

#[macro_export]
macro_rules! provider_endpoints {
    ($vis:vis enum $name:ident { $($variant:ident => $method:ident $path:tt $(($typed:expr))?),+ $(,)? }) => {
        #[derive(Clone, Copy, Debug, PartialEq, Eq)]
        $vis enum $name { $($variant),+ }
        impl $name {
            pub fn method(&self) -> reqwest::Method {
                match self { $(Self::$variant => reqwest::Method::$method),+ }
            }
            pub const fn path(&self) -> litellm_llms_types::endpoint::EndpointPath {
                match self { $(Self::$variant => $crate::provider_endpoints!(@path $path $(($typed))?)),+ }
            }
        }
        impl $crate::base_llm::endpoint::ProviderEndpoint for $name {
            fn resolve(&self, target: &litellm_core_utils::url_utils::EndpointTarget)
                -> Result<$crate::base_llm::endpoint::ResolvedEndpoint, litellm_core_utils::ApiUrlError> {
                Ok($crate::base_llm::endpoint::ResolvedEndpoint::new(self.method(), target.resolve(self.path())?))
            }
        }

    };
    (@path path ($typed:expr)) => { $typed };
    (@path $path:literal) => {{
        const PATH: litellm_llms_types::endpoint::EndpointPath = litellm_llms_types::endpoint::EndpointPath::new($path);
        PATH
    }};
}
