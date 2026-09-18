use std::sync::atomic::{AtomicUsize, Ordering};

use rand::RngCore;

use crate::plan::DeploymentId;
use crate::signals::Candidate;

/// Chooses among the available candidates of the current group. `None` means the group is
/// exhausted and the loop moves to the next one. Unavailable deployments are filtered out
/// before this is called, so a picker never has to know about cooldowns.
pub trait DeploymentPicker: Send + Sync {
    fn pick(&self, candidates: &[Candidate], rng: &mut dyn RngCore) -> Option<DeploymentId>;
}

#[derive(Debug, Default)]
pub struct RoundRobin {
    next: AtomicUsize,
}

impl DeploymentPicker for RoundRobin {
    fn pick(&self, candidates: &[Candidate], _: &mut dyn RngCore) -> Option<DeploymentId> {
        if candidates.is_empty() {
            return None;
        }
        let index = self.next.fetch_add(1, Ordering::Relaxed) % candidates.len();
        Some(candidates[index].id)
    }
}

/// litellm `simple-shuffle`: weighted random by configured weight.
#[derive(Debug, Default)]
pub struct WeightedShuffle;

impl DeploymentPicker for WeightedShuffle {
    fn pick(&self, candidates: &[Candidate], rng: &mut dyn RngCore) -> Option<DeploymentId> {
        let total: u64 = candidates.iter().map(|c| u64::from(c.weight.max(1))).sum();
        if total == 0 {
            return None;
        }
        let mut point = rng.next_u64() % total;
        candidates.iter().find_map(|candidate| {
            let weight = u64::from(candidate.weight.max(1));
            if point < weight {
                return Some(candidate.id);
            }
            point -= weight;
            None
        })
    }
}

/// litellm `latency-based-routing`: lowest observed latency, unknown latency first.
#[derive(Debug, Default)]
pub struct LowestLatency;

impl DeploymentPicker for LowestLatency {
    fn pick(&self, candidates: &[Candidate], _: &mut dyn RngCore) -> Option<DeploymentId> {
        candidates
            .iter()
            .min_by_key(|candidate| candidate.load.latency)
            .map(|candidate| candidate.id)
    }
}

/// litellm `least-busy`: fewest in-flight requests.
#[derive(Debug, Default)]
pub struct LeastBusy;

impl DeploymentPicker for LeastBusy {
    fn pick(&self, candidates: &[Candidate], _: &mut dyn RngCore) -> Option<DeploymentId> {
        candidates
            .iter()
            .min_by_key(|candidate| candidate.load.in_flight)
            .map(|candidate| candidate.id)
    }
}

/// litellm `usage-based-routing-v2`: only deployments with tpm and rpm headroom, lowest
/// usage first; unknown headroom counts as unlimited.
#[derive(Debug, Default)]
pub struct UsageBased;

impl DeploymentPicker for UsageBased {
    fn pick(&self, candidates: &[Candidate], _: &mut dyn RngCore) -> Option<DeploymentId> {
        candidates
            .iter()
            .filter(|c| c.load.tpm_headroom != Some(0) && c.load.rpm_headroom != Some(0))
            .max_by_key(|c| {
                (
                    c.load.tpm_headroom.unwrap_or(u64::MAX),
                    c.load.rpm_headroom.unwrap_or(u64::MAX),
                )
            })
            .map(|c| c.id)
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use rand::SeedableRng;
    use rand::rngs::StdRng;

    use super::*;
    use crate::signals::Load;

    fn candidate(id: u64, weight: u32, load: Load) -> Candidate {
        Candidate {
            id: DeploymentId(id),
            weight,
            load,
        }
    }

    fn with_latency(ms: u64) -> Load {
        Load {
            latency: Some(Duration::from_millis(ms)),
            ..Load::available()
        }
    }

    #[test]
    fn weighted_shuffle_follows_configured_weights() {
        let candidates = [
            candidate(1, 9, Load::available()),
            candidate(2, 1, Load::available()),
        ];
        let mut rng = StdRng::seed_from_u64(3);
        let picks = (0..1000)
            .filter(|_| WeightedShuffle.pick(&candidates, &mut rng) == Some(DeploymentId(1)))
            .count();
        assert!((850..=950).contains(&picks), "{picks}");
        assert_eq!(WeightedShuffle.pick(&[], &mut rng), None);
    }

    #[test]
    fn strategies_read_their_own_signal() {
        let mut rng = StdRng::seed_from_u64(0);
        let latency = [
            candidate(1, 1, with_latency(80)),
            candidate(2, 1, with_latency(20)),
        ];
        assert_eq!(
            LowestLatency.pick(&latency, &mut rng),
            Some(DeploymentId(2))
        );
        let busy = [
            candidate(
                1,
                1,
                Load {
                    in_flight: 4,
                    ..Load::available()
                },
            ),
            candidate(
                2,
                1,
                Load {
                    in_flight: 1,
                    ..Load::available()
                },
            ),
        ];
        assert_eq!(LeastBusy.pick(&busy, &mut rng), Some(DeploymentId(2)));
        let usage = [
            candidate(
                1,
                1,
                Load {
                    tpm_headroom: Some(0),
                    ..Load::available()
                },
            ),
            candidate(
                2,
                1,
                Load {
                    tpm_headroom: Some(500),
                    rpm_headroom: Some(3),
                    ..Load::available()
                },
            ),
            candidate(
                3,
                1,
                Load {
                    tpm_headroom: Some(900),
                    rpm_headroom: Some(0),
                    ..Load::available()
                },
            ),
        ];
        assert_eq!(UsageBased.pick(&usage, &mut rng), Some(DeploymentId(2)));
    }
}
