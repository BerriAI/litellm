mod aws;
mod vertex;

pub use aws::AwsParams;
pub use vertex::VertexParams;

/// One connection param of Python's `CredentialLiteLLMParams` and every place Python reads
/// it from, in order: the wire names on a call or a deployment, the `litellm.<name>` module
/// global a host folds in, then the environment names.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ParamSpec {
    pub setting: &'static str,
    pub wire: &'static [&'static str],
    pub module_global: Option<&'static str>,
    pub env: &'static [&'static str],
}

impl ParamSpec {
    /// The first non-blank value: each wire name through `params`, then each environment
    /// name through `env`.
    pub fn resolve(
        &self,
        params: &dyn Fn(&str) -> Option<String>,
        env: &dyn Fn(&str) -> Option<String>,
    ) -> Option<String> {
        self.wire
            .iter()
            .map(|name| params(name))
            .chain(self.env.iter().map(|name| env(name)))
            .find_map(|value| value.filter(|value| !value.trim().is_empty()))
    }

    pub fn missing(&'static self, provider: &'static str) -> crate::Error {
        crate::Error::MissingParam {
            provider,
            spec: self,
        }
    }

    pub(crate) fn guidance(&self) -> String {
        let wire = self.wire.join(" or ");
        let env = self.env.join(" or ");
        match self.module_global {
            Some(global) => format!("pass {wire}, set litellm.{global}, or set {env}"),
            None => format!("pass {wire} or set {env}"),
        }
    }
}
