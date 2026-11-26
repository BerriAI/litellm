use litellm_secret_value::SecretValue;

/// Whether a host looks for a cost map beyond the bundled one
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Remote {
    /// The host serves the bundled document and never fetches
    Off,
    At(SecretValue),
}

impl Remote {
    /// The local override and offline mode each turn the remote document off
    pub fn new(url: SecretValue, local_only: bool, offline: bool) -> Self {
        match local_only || offline {
            true => Self::Off,
            false => Self::At(url),
        }
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    const URL: &str = "https://user:credential-value@example.com/cost.json";

    #[rstest]
    #[case::remote(false, false, Remote::At(URL.into()))]
    #[case::local_override(true, false, Remote::Off)]
    #[case::offline(false, true, Remote::Off)]
    #[case::both(true, true, Remote::Off)]
    fn the_local_override_and_offline_mode_each_turn_the_remote_document_off(
        #[case] local_only: bool,
        #[case] offline: bool,
        #[case] remote: Remote,
    ) {
        assert_eq!(Remote::new(URL.into(), local_only, offline), remote);
    }

    #[rstest]
    fn the_url_stays_out_of_debug_output() {
        assert_eq!(
            format!("{:?}", Remote::new(URL.into(), false, false)),
            "At(SecretValue([REDACTED]))"
        );
    }
}
