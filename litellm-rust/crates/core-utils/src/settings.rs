pub trait Lookup {
    fn get(&self, name: &str) -> Option<String>;
}

impl<F: Fn(&str) -> Option<String>> Lookup for F {
    fn get(&self, name: &str) -> Option<String> {
        self(name)
    }
}

pub struct ProcessEnvironment;

impl Lookup for ProcessEnvironment {
    #[allow(clippy::disallowed_methods)] // the live process environment Lookup for the Python runtimes
    fn get(&self, name: &str) -> Option<String> {
        std::env::var(name).ok()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn env_of(values: &'static [(&'static str, &'static str)]) -> impl Fn(&str) -> Option<String> {
        move |name| {
            values
                .iter()
                .find(|(key, _)| *key == name)
                .map(|(_, value)| value.to_string())
        }
    }

    #[test]
    fn get_keeps_a_present_empty_value_like_os_getenv_with_a_fallback() {
        let env = env_of(&[("EMPTY", "")]);
        assert_eq!(env.get("EMPTY"), Some(String::new()));
        assert_eq!(env.get("ABSENT"), None);
    }
}
