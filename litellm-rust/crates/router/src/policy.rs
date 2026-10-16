use std::time::Duration;

use litellm_config::{LiteLlmSettings, RouterSettings, Spelled};

use crate::Error;

const DEFAULT_NUM_RETRIES: u32 = 2;
const DEFAULT_COOLDOWN_TIME: Duration = Duration::from_secs(5);
const FAILURE_RATE_WINDOW: Duration = Duration::from_secs(60);

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, strum::EnumString, strum::VariantArray)]
pub enum RoutingStrategy {
    #[default]
    #[strum(serialize = "simple-shuffle")]
    SimpleShuffle,
    #[strum(serialize = "least-busy")]
    LeastBusy,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CooldownPolicy {
    pub allowed_fails: Option<u32>,
    pub cooldown_time: Duration,
}

impl Default for CooldownPolicy {
    fn default() -> Self {
        Self {
            allowed_fails: None,
            cooldown_time: DEFAULT_COOLDOWN_TIME,
        }
    }
}

impl CooldownPolicy {
    /// Python counts `allowed_fails` in a cache entry that lives for `cooldown_time`, and
    /// its failure-rate rule over the current minute.
    pub(crate) fn window(&self) -> Duration {
        match self.allowed_fails {
            Some(_) => self.cooldown_time,
            None => FAILURE_RATE_WINDOW,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RoutingPolicy {
    pub strategy: RoutingStrategy,
    pub num_retries: u32,
    pub cooldown: CooldownPolicy,
    pub timeout: Option<Duration>,
}

impl Default for RoutingPolicy {
    fn default() -> Self {
        Self {
            strategy: RoutingStrategy::default(),
            num_retries: DEFAULT_NUM_RETRIES,
            cooldown: CooldownPolicy::default(),
            timeout: None,
        }
    }
}

impl RoutingPolicy {
    pub fn from_settings(
        router: &RouterSettings,
        litellm: &LiteLlmSettings,
    ) -> Result<Self, Error> {
        let strategy = match router.routing_strategy.as_deref() {
            None => RoutingStrategy::default(),
            Some(name) => name
                .parse()
                .map_err(|_| Error::UnsupportedStrategy(name.to_owned()))?,
        };
        let cooldown_time = router
            .cooldown_time
            .map(|value| seconds("router_settings.cooldown_time", &Spelled::Value(value)))
            .transpose()?
            .flatten()
            .unwrap_or(DEFAULT_COOLDOWN_TIME);
        Ok(Self {
            strategy,
            num_retries: router
                .num_retries
                .or(litellm.num_retries)
                .map_or(DEFAULT_NUM_RETRIES, saturating_u32),
            cooldown: CooldownPolicy {
                allowed_fails: router.allowed_fails.map(saturating_u32),
                cooldown_time,
            },
            timeout: router
                .timeout
                .map(|value| seconds("router_settings.timeout", &Spelled::Value(value)))
                .transpose()?
                .flatten(),
        })
    }
}

fn saturating_u32(value: u64) -> u32 {
    u32::try_from(value).unwrap_or(u32::MAX)
}

/// Seconds as a config spells them; zero means unset, as Python's `timeout or default` reads it.
pub(crate) fn seconds(
    setting: &'static str,
    value: &Spelled<f64>,
) -> Result<Option<Duration>, Error> {
    let invalid = || Error::InvalidSeconds {
        setting,
        value: match value {
            Spelled::Value(value) => value.to_string(),
            Spelled::Text(text) => text.clone(),
        },
    };
    let number = match value {
        Spelled::Value(value) => *value,
        Spelled::Text(text) => text.trim().parse().map_err(|_| invalid())?,
    };
    let duration = Duration::try_from_secs_f64(number).map_err(|_| invalid())?;
    Ok((!duration.is_zero()).then_some(duration))
}

#[cfg(test)]
mod tests {
    use litellm_config::Config;
    use rstest::rstest;

    use super::*;

    fn policy(yaml: &str) -> Result<RoutingPolicy, Error> {
        let config = Config::from_yaml(yaml).unwrap();
        RoutingPolicy::from_settings(&config.router_settings, &config.litellm_settings)
    }

    #[rstest]
    #[case::unset("{}", Ok(RoutingStrategy::SimpleShuffle))]
    #[case::simple_shuffle(
        "router_settings: {routing_strategy: simple-shuffle}",
        Ok(RoutingStrategy::SimpleShuffle)
    )]
    #[case::least_busy(
        "router_settings: {routing_strategy: least-busy}",
        Ok(RoutingStrategy::LeastBusy)
    )]
    #[case::unsupported(
        "router_settings: {routing_strategy: latency-based-routing}",
        Err(Error::UnsupportedStrategy("latency-based-routing".into()))
    )]
    fn strategy_is_read_from_router_settings(
        #[case] yaml: &str,
        #[case] expected: Result<RoutingStrategy, Error>,
    ) {
        assert_eq!(policy(yaml).map(|policy| policy.strategy), expected);
    }

    #[rstest]
    #[case::python_default("{}", 2)]
    #[case::router_settings("router_settings: {num_retries: 5}", 5)]
    #[case::litellm_settings("litellm_settings: {num_retries: 1}", 1)]
    #[case::router_settings_win(
        "router_settings: {num_retries: 0}\nlitellm_settings: {num_retries: 4}",
        0
    )]
    fn num_retries_falls_back_like_python(#[case] yaml: &str, #[case] expected: u32) {
        assert_eq!(policy(yaml).unwrap().num_retries, expected);
    }

    #[rstest]
    #[case::defaults("{}", None, Duration::from_secs(5))]
    #[case::configured(
        "router_settings: {allowed_fails: 3, cooldown_time: 30}",
        Some(3),
        Duration::from_secs(30)
    )]
    #[case::zero_cooldown_is_default(
        "router_settings: {cooldown_time: 0}",
        None,
        Duration::from_secs(5)
    )]
    #[case::fractional(
        "router_settings: {cooldown_time: 0.5}",
        None,
        Duration::from_millis(500)
    )]
    fn cooldown_settings(
        #[case] yaml: &str,
        #[case] allowed_fails: Option<u32>,
        #[case] cooldown_time: Duration,
    ) {
        assert_eq!(
            policy(yaml).unwrap().cooldown,
            CooldownPolicy {
                allowed_fails,
                cooldown_time
            }
        );
    }

    #[rstest]
    fn negative_cooldown_is_rejected() {
        assert_eq!(
            policy("router_settings: {cooldown_time: -1}"),
            Err(Error::InvalidSeconds {
                setting: "router_settings.cooldown_time",
                value: "-1".into(),
            })
        );
    }

    #[rstest]
    #[case::number(Spelled::Value(90.0), Ok(Some(Duration::from_secs(90))))]
    #[case::text(Spelled::Text(" 1.5 ".into()), Ok(Some(Duration::from_millis(1500))))]
    #[case::zero_is_unset(Spelled::Value(0.0), Ok(None))]
    #[case::unresolved_reference(
        Spelled::Text("os.environ/TIMEOUT".into()),
        Err(Error::InvalidSeconds { setting: "timeout", value: "os.environ/TIMEOUT".into() })
    )]
    fn seconds_reads_numbers_and_numeric_text(
        #[case] value: Spelled<f64>,
        #[case] expected: Result<Option<Duration>, Error>,
    ) {
        assert_eq!(seconds("timeout", &value), expected);
    }
}
