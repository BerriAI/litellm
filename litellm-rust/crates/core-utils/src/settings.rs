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

pub fn resolve_non_empty(
    value: Option<&str>,
    env_lookup: &dyn Fn(&str) -> Option<String>,
    names: &[&str],
) -> Option<String> {
    value
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_string)
        .or_else(|| {
            names
                .iter()
                .find_map(|name| env_lookup(name).filter(|value| !value.trim().is_empty()))
        })
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
