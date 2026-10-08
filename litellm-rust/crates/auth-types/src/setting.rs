/// One setting LiteLLM owns: the call arguments that carry it (aliases in precedence order), the
/// environment variables that back it when no argument does, and whether its value is a secret.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Setting {
    pub kwargs: &'static [&'static str],
    pub env: &'static [&'static str],
    pub secret: bool,
}

pub const fn setting(kwargs: &'static [&'static str], env: &'static [&'static str]) -> Setting {
    Setting {
        kwargs,
        env,
        secret: false,
    }
}

pub const fn secret(kwargs: &'static [&'static str], env: &'static [&'static str]) -> Setting {
    Setting {
        kwargs,
        env,
        secret: true,
    }
}

impl Setting {
    pub fn claims(&self, name: &str) -> bool {
        self.kwargs.contains(&name)
    }
}

pub fn kwarg_names(settings: &'static [Setting]) -> impl Iterator<Item = &'static str> {
    settings
        .iter()
        .flat_map(|setting| setting.kwargs.iter().copied())
}

pub fn env_names(settings: &'static [Setting]) -> impl Iterator<Item = &'static str> {
    settings
        .iter()
        .flat_map(|setting| setting.env.iter().copied())
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::{Setting, env_names, kwarg_names, secret, setting};

    const ROWS: &[Setting] = &[
        setting(&["first", "first_alias"], &["FIRST"]),
        secret(&["second"], &["SECOND", "SECOND_ALIAS"]),
        secret(&[], &["ENV_ONLY"]),
    ];

    #[rstest]
    #[case::primary("first", true)]
    #[case::alias("first_alias", true)]
    #[case::other_row("second", false)]
    #[case::env_name_is_not_a_kwarg("FIRST", false)]
    fn claims_matches_any_kwarg_alias_of_the_row(#[case] name: &str, #[case] claimed: bool) {
        assert_eq!(ROWS[0].claims(name), claimed);
    }

    #[test]
    fn kwarg_names_flatten_every_alias_in_declaration_order() {
        assert_eq!(
            kwarg_names(ROWS).collect::<Vec<_>>(),
            ["first", "first_alias", "second"]
        );
    }

    #[test]
    fn env_names_flatten_every_fallback_in_declaration_order() {
        assert_eq!(
            env_names(ROWS).collect::<Vec<_>>(),
            ["FIRST", "SECOND", "SECOND_ALIAS", "ENV_ONLY"]
        );
    }

    #[rstest]
    #[case::plain(ROWS[0], false)]
    #[case::secret(ROWS[1], true)]
    fn constructors_set_the_secret_flag(#[case] row: Setting, #[case] secret: bool) {
        assert_eq!(row.secret, secret);
    }
}
