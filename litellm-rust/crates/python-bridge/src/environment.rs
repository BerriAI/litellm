use litellm_core_utils::settings::Lookup;

pub(crate) struct ProcessEnvironment;

impl Lookup for ProcessEnvironment {
    #[allow(clippy::disallowed_methods)] // the Python host's settings live in the process environment
    fn get(&self, name: &str) -> Option<String> {
        std::env::var(name).ok()
    }
}
