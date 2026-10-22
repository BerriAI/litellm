use std::str::FromStr;

use crate::serde_compat::parse_str_bool;

pub trait Lookup {
    fn get(&self, name: &str) -> Option<String>;

    fn truthy(&self, name: &str) -> Option<String> {
        self.get(name).filter(|value| !value.is_empty())
    }

    fn enabled(&self, name: &str) -> Option<bool> {
        self.get(name)
            .is_some_and(|value| parse_str_bool(&value) == Some(true))
            .then_some(true)
    }

    fn parsed<T: FromStr>(&self, name: &str) -> Option<T>
    where
        Self: Sized,
    {
        self.get(name).and_then(|value| value.trim().parse().ok())
    }
}

impl<F: Fn(&str) -> Option<String>> Lookup for F {
    fn get(&self, name: &str) -> Option<String> {
        self(name)
    }
}

pub struct ProcessEnvironment;

impl Lookup for ProcessEnvironment {
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

    #[test]
    fn truthy_drops_an_empty_value_like_a_python_or_chain() {
        let env = env_of(&[("EMPTY", ""), ("SET", "value")]);
        assert_eq!(env.truthy("EMPTY"), None);
        assert_eq!(env.truthy("SET").as_deref(), Some("value"));
    }

    #[test]
    fn enabled_only_switches_on_for_true_and_never_forces_off() {
        let env = env_of(&[
            ("LOWER", "true"),
            ("PADDED", " True "),
            ("OFF", "false"),
            ("ONE", "1"),
        ]);
        assert_eq!(env.enabled("LOWER"), Some(true));
        assert_eq!(env.enabled("PADDED"), Some(true));
        assert_eq!(env.enabled("OFF"), None);
        assert_eq!(env.enabled("ONE"), None);
        assert_eq!(env.enabled("ABSENT"), None);
    }

    #[test]
    fn parsed_trims_and_skips_values_that_do_not_parse() {
        let env = env_of(&[("PADDED", " 45 "), ("WORD", "soon"), ("FRACTION", "0.5")]);
        assert_eq!(env.parsed::<u32>("PADDED"), Some(45));
        assert_eq!(env.parsed::<u32>("WORD"), None);
        assert_eq!(env.parsed::<f64>("FRACTION"), Some(0.5));
        assert_eq!(env.parsed::<u32>("ABSENT"), None);
    }
}
