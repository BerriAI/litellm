use std::sync::Arc;

use litellm_router_types::Spelled;

use crate::{Deployment, RoutingStrategy, health::Health};

#[derive(Debug)]
pub(crate) struct Member {
    /// The deployment's id in shared state: `model_info.id`, or one derived from the config.
    pub id: Box<str>,
    pub deployment: Deployment,
    pub health: Arc<Health>,
    weights: [f64; 3],
}

impl Member {
    pub(crate) fn new(id: Box<str>, deployment: Deployment) -> Self {
        let params = &deployment.litellm_params;
        let weights = [
            params
                .extra
                .get("weight")
                .and_then(|weight| weight.as_f64())
                .unwrap_or_default(),
            metric(params.rpm.as_ref()),
            metric(params.tpm.as_ref()),
        ]
        .map(|weight| weight.max(0.0));
        Self {
            id,
            deployment,
            health: Arc::default(),
            weights,
        }
    }
}

fn metric(value: Option<&Spelled<f64>>) -> f64 {
    match value {
        Some(Spelled::Value(value)) => *value,
        Some(Spelled::Text(text)) => text.trim().parse().unwrap_or_default(),
        None => 0.0,
    }
}

/// The deployments that share a public model name, in config order.
#[derive(Debug)]
pub struct ModelGroup {
    name: Box<str>,
    members: Box<[Member]>,
}

impl ModelGroup {
    pub(crate) fn new(name: Box<str>, members: Box<[Member]>) -> Self {
        Self { name, members }
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn deployments(&self) -> impl ExactSizeIterator<Item = &Deployment> {
        self.members.iter().map(|member| &member.deployment)
    }

    pub(crate) fn len(&self) -> usize {
        self.members.len()
    }

    pub(crate) fn member(&self, index: usize) -> &Member {
        &self.members[index]
    }

    /// Picks a deployment no earlier attempt `tried`, preferring those not cooling down and
    /// falling back to cooling ones rather than failing. `draw` is uniform in `[0, 1)`.
    pub(crate) fn choose(
        &self,
        strategy: RoutingStrategy,
        now: u64,
        tried: &[usize],
        draw: f64,
    ) -> Option<usize> {
        let untried = |index: &usize| !tried.contains(index);
        let healthy = |index: &usize| !self.members[*index].health.cooling(now);
        self.pick(strategy, draw, |index| untried(index) && healthy(index))
            .or_else(|| self.pick(strategy, draw, untried))
    }

    fn pick(
        &self,
        strategy: RoutingStrategy,
        draw: f64,
        eligible: impl Fn(&usize) -> bool + Copy,
    ) -> Option<usize> {
        let candidates = || (0..self.members.len()).filter(eligible);
        let count = candidates().count();
        if count == 0 {
            return None;
        }
        let nth = |count: usize| ((draw * count as f64) as usize).min(count - 1);
        match strategy {
            RoutingStrategy::SimpleShuffle => {
                let weighted = (0..3).find_map(|metric| {
                    let total: f64 = candidates()
                        .map(|index| self.members[index].weights[metric])
                        .sum();
                    (total > 0.0).then_some((metric, total))
                });
                match weighted {
                    Some((metric, total)) => {
                        let target = draw * total;
                        candidates()
                            .scan(0.0, |cumulative, index| {
                                *cumulative += self.members[index].weights[metric];
                                Some((index, *cumulative))
                            })
                            .find(|(_, cumulative)| target < *cumulative)
                            .map(|(index, _)| index)
                            .or_else(|| {
                                candidates()
                                    .rfind(|index| self.members[*index].weights[metric] > 0.0)
                            })
                    }
                    None => candidates().nth(nth(count)),
                }
            }
            RoutingStrategy::LeastBusy => {
                let busy = |index: &usize| self.members[*index].health.in_flight();
                let least = candidates().map(|index| busy(&index)).min()?;
                let idle = || candidates().filter(|index| busy(index) == least);
                idle()
                    .nth(nth(idle().count().max(1)))
                    .or_else(|| candidates().next())
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use litellm_config::Config;
    use rstest::rstest;

    use super::*;
    use crate::{CooldownPolicy, Failure, InFlight};

    const COOLDOWN: CooldownPolicy = CooldownPolicy {
        allowed_fails: Some(0),
        cooldown_time: Duration::from_secs(60),
    };

    fn group(params: &[&str]) -> ModelGroup {
        let yaml = params
            .iter()
            .fold(String::from("model_list:\n"), |yaml, params| {
                yaml + &format!(
                    "  - model_name: group\n    litellm_params: {{model: provider/model{params}}}\n"
                )
            });
        let config = Config::from_yaml(&yaml).unwrap();
        let members = config
            .model_list
            .iter()
            .enumerate()
            .map(|(index, model)| {
                Member::new(
                    index.to_string().into(),
                    Deployment::from_params(&model.litellm_params).unwrap(),
                )
            })
            .collect();
        ModelGroup::new("group".into(), members)
    }

    fn cooling_group(size: usize, cooling: &[usize]) -> ModelGroup {
        let group = group(&vec![""; size]);
        for index in cooling {
            group.members[*index].health.record_failure(
                0,
                Failure::Unavailable,
                &COOLDOWN,
                true,
                None,
            );
        }
        group
    }

    fn busy_guards(group: &ModelGroup, in_flight: &[usize]) -> Vec<InFlight> {
        in_flight
            .iter()
            .enumerate()
            .flat_map(|(index, count)| {
                std::iter::repeat_with(move || InFlight::enter(&group.members[index].health))
                    .take(*count)
            })
            .collect()
    }

    #[rstest]
    #[case::uniform_first(&["", "", "", ""], 0.1, 0)]
    #[case::uniform_third(&["", "", "", ""], 0.6, 2)]
    #[case::uniform_last(&["", "", "", ""], 0.99, 3)]
    #[case::weight_light(&[", weight: 1", ", weight: 3"], 0.2, 0)]
    #[case::weight_heavy(&[", weight: 1", ", weight: 3"], 0.3, 1)]
    #[case::unweighted_skipped_when_others_weigh(&["", ", weight: 2"], 0.0, 1)]
    #[case::weight_beats_rpm(&[", weight: 1, rpm: 1000", ", weight: 9, rpm: 1"], 0.5, 1)]
    #[case::rpm_when_no_weight(&[", rpm: 900", ", rpm: 100"], 0.85, 0)]
    #[case::tpm_when_no_rpm(&[", tpm: 100", ", tpm: 900"], 0.15, 1)]
    #[case::rpm_as_text(&[", rpm: '100'", ", rpm: '300'"], 0.3, 1)]
    fn simple_shuffle_follows_python_weights(
        #[case] params: &[&str],
        #[case] draw: f64,
        #[case] expected: usize,
    ) {
        assert_eq!(
            group(params).choose(RoutingStrategy::SimpleShuffle, 0, &[], draw),
            Some(expected)
        );
    }

    #[rstest]
    #[case::skips_cooling(&[0], &[], 0.0, Some(1))]
    #[case::skips_tried(&[], &[0, 1], 0.0, Some(2))]
    #[case::all_cooling_tries_anyway(&[0, 1, 2], &[], 0.99, Some(2))]
    #[case::cooling_untried_beats_tried(&[2], &[0, 1], 0.0, Some(2))]
    #[case::all_tried(&[], &[0, 1, 2], 0.0, None)]
    fn choice_avoids_cooling_and_tried_deployments(
        #[case] cooling: &[usize],
        #[case] tried: &[usize],
        #[case] draw: f64,
        #[case] expected: Option<usize>,
        #[values(RoutingStrategy::SimpleShuffle, RoutingStrategy::LeastBusy)]
        strategy: RoutingStrategy,
    ) {
        let group = cooling_group(3, cooling);

        assert_eq!(group.choose(strategy, 1, tried, draw), expected);
    }

    #[rstest]
    fn cooldown_ends() {
        let group = cooling_group(2, &[0]);

        assert_eq!(
            group.choose(RoutingStrategy::SimpleShuffle, 60_000, &[], 0.0),
            Some(0)
        );
    }

    #[rstest]
    #[case::least_loaded(&[2, 0, 1], 0.0, 1)]
    #[case::ties_split_by_draw(&[1, 0, 0], 0.9, 2)]
    fn least_busy_picks_the_fewest_in_flight(
        #[case] in_flight: &[usize],
        #[case] draw: f64,
        #[case] expected: usize,
    ) {
        let group = group(&["", "", ""]);
        let _guards = busy_guards(&group, in_flight);

        assert_eq!(
            group.choose(RoutingStrategy::LeastBusy, 0, &[], draw),
            Some(expected)
        );
    }
}
