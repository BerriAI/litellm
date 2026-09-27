use std::time::Duration;

use rand::Rng;

use crate::JobName;

#[derive(Clone, Debug)]
pub enum Schedule {
    Every {
        period: Duration,
        jitter: Duration,
    },
    Cron {
        expression: Box<cron::Schedule>,
        jitter: Duration,
    },
}

impl Schedule {
    pub(crate) fn next_delay(&self) -> Duration {
        let (base, jitter) = match self {
            Self::Every { period, jitter } => (*period, *jitter),
            Self::Cron { expression, jitter } => (until_next_fire(expression), *jitter),
        };
        base + random_up_to(jitter)
    }
}

fn until_next_fire(expression: &cron::Schedule) -> Duration {
    expression
        .upcoming(chrono::Utc)
        .next()
        .and_then(|at| (at - chrono::Utc::now()).to_std().ok())
        .unwrap_or(Duration::MAX)
}

fn random_up_to(jitter: Duration) -> Duration {
    if jitter.is_zero() {
        return Duration::ZERO;
    }
    rand::thread_rng().gen_range(Duration::ZERO..=jitter)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Exclusivity {
    EveryPod,
    OnePod { lease_ttl: Duration },
}

#[derive(Clone, Debug)]
pub struct JobSpec {
    pub name: JobName,
    pub schedule: Schedule,
    pub exclusivity: Exclusivity,
}
