use std::time::Duration;

use crate::plan::DeploymentId;

/// What the host currently knows about one deployment. Strategies read it; the router only
/// honors `available`. Learners fill it from the event stream (latency, in-flight, usage),
/// the cooldown table turns `available` off.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Load {
    pub available: bool,
    pub in_flight: u32,
    pub latency: Option<Duration>,
    /// Remaining tokens and requests per minute, when the host tracks usage.
    pub tpm_headroom: Option<u64>,
    pub rpm_headroom: Option<u64>,
}

impl Load {
    pub const fn available() -> Self {
        Self {
            available: true,
            in_flight: 0,
            latency: None,
            tpm_headroom: None,
            rpm_headroom: None,
        }
    }
}

/// Per-deployment signals, shared by every picker. Snapshotted or live is the host's choice.
pub trait Signals: Send + Sync {
    fn load(&self, deployment: DeploymentId) -> Load;
}

/// Every deployment available, nothing known. The SDK path with one deployment.
#[derive(Clone, Copy, Debug, Default)]
pub struct NoSignals;

impl Signals for NoSignals {
    fn load(&self, _: DeploymentId) -> Load {
        Load::available()
    }
}

/// One deployment as a picker sees it: configured weight plus the host's current load.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Candidate {
    pub id: DeploymentId,
    pub weight: u32,
    pub load: Load,
}
