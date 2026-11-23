//! Which callers used a model in the current window, the proxy's `ActiveTeamRegistry`: one sorted
//! set per model scored by last-seen time, in Redis when there is one and in this process's memory
//! otherwise. The dynamic rate limiter splits a priority class's share of a model among them.

mod keys;
mod lua;
mod types;

use std::{collections::HashMap, sync::LazyLock, time::Duration};

use litellm_python_compat::repr::float_repr;
use litellm_redis_client::{Script, commands};
use parking_lot::Mutex;

use crate::{
    Error,
    memory::{Clock, system_clock},
    remote::{Remote, RemoteSlot},
};

pub use types::ActiveMember;

static TOUCH: LazyLock<Script> = LazyLock::new(|| Script::new(lua::TOUCH));

/// Each model's members in the order they were first seen, with when each was last seen.
type Local = HashMap<String, Vec<(String, f64)>>;

pub struct ActiveTeams {
    redis: RemoteSlot,
    local: Mutex<Local>,
    now: Clock,
}

/// Oldest first, so a member seen under two classes in one window keeps its newest one, and the
/// caller last, so it is listed under the class it just sent with whatever another pod's clock
/// last saw it in.
fn listed(seen: impl IntoIterator<Item = ActiveMember>, caller: ActiveMember) -> Vec<ActiveMember> {
    let mut positions: HashMap<String, usize> = HashMap::new();
    let mut members: Vec<ActiveMember> = Vec::new();
    for active in seen.into_iter().chain([caller]) {
        match positions.get(&active.member) {
            Some(position) => members[*position].priority_class = active.priority_class,
            None => {
                positions.insert(active.member.clone(), members.len());
                members.push(active);
            }
        }
    }
    members
}

impl ActiveTeams {
    pub fn new(remote: Option<Remote>) -> Self {
        Self::with_clock(remote, system_clock())
    }

    pub fn with_clock(remote: Option<Remote>, now: Clock) -> Self {
        Self {
            redis: RemoteSlot::new(remote),
            local: Mutex::new(Local::new()),
            now,
        }
    }

    pub fn attach(&self, remote: Remote) -> bool {
        self.redis.attach(remote)
    }

    pub fn has_redis(&self) -> bool {
        self.redis.get().is_some()
    }

    /// Records `member` as active on `model` under `priority_class` and lists every member active
    /// on it in the last `window`, the caller included. A Redis failure is the caller's.
    pub async fn touch(
        &self,
        model: &str,
        member: &str,
        priority_class: &str,
        window: Duration,
    ) -> Result<Vec<ActiveMember>, Error> {
        let Some(client) = self.redis.client() else {
            return Ok(self.touch_local(model, member, priority_class, window));
        };
        let caller = ActiveMember {
            member: member.to_owned(),
            priority_class: priority_class.to_owned(),
        };
        let now = (self.now)().as_secs_f64();
        let seen: Vec<String> = client
            .send(commands::eval(
                &TOUCH,
                &[keys::key(model).as_str()],
                vec![
                    float_repr(now - window.as_secs_f64()),
                    float_repr(now),
                    caller.spelling(),
                    window.as_secs().to_string(),
                ],
            ))
            .await?;
        Ok(listed(
            seen.iter().filter_map(|member| ActiveMember::read(member)),
            caller,
        ))
    }

    /// `touch` against this process's memory alone. A member leaves once it was last seen a whole
    /// `window` ago, and a model once its newest member has.
    pub fn touch_local(
        &self,
        model: &str,
        member: &str,
        priority_class: &str,
        window: Duration,
    ) -> Vec<ActiveMember> {
        let caller = ActiveMember {
            member: member.to_owned(),
            priority_class: priority_class.to_owned(),
        };
        let spelling = caller.spelling();
        let now = (self.now)().as_secs_f64();
        let cutoff = now - window.as_secs_f64();
        let key = keys::key(model);
        let mut local = self.local.lock();
        local.retain(|other, members| {
            *other == key || members.iter().any(|(_, seen)| *seen > cutoff)
        });
        let members = local.entry(key).or_default();
        members.retain(|(_, seen)| *seen > cutoff);
        match members.iter_mut().find(|(seen, _)| *seen == spelling) {
            Some((_, seen)) => *seen = now,
            None => members.push((spelling, now)),
        }
        let mut oldest_first = members.clone();
        oldest_first.sort_by(|(_, earlier), (_, later)| earlier.total_cmp(later));
        listed(
            oldest_first
                .iter()
                .filter_map(|(member, _)| ActiveMember::read(member)),
            caller,
        )
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::Ordering;

    use rstest::rstest;

    use super::*;
    use crate::test_support::clock;

    const WINDOW: Duration = Duration::from_secs(60);

    fn active(member: &str, priority_class: &str) -> ActiveMember {
        ActiveMember {
            member: member.to_owned(),
            priority_class: priority_class.to_owned(),
        }
    }

    #[rstest]
    fn a_member_is_listed_oldest_first_until_it_was_last_seen_a_window_ago() {
        let (now, seconds) = clock();
        let teams = ActiveTeams::with_clock(None, now);

        teams.touch_local("m", "a", "interactive", WINDOW);
        seconds.fetch_add(59, Ordering::SeqCst);
        let inside = teams.touch_local("m", "b", "batch", WINDOW);
        seconds.fetch_add(1, Ordering::SeqCst);
        let outside = teams.touch_local("m", "b", "batch", WINDOW);

        assert_eq!(
            (inside, outside),
            (
                vec![active("a", "interactive"), active("b", "batch")],
                vec![active("b", "batch")]
            )
        );
    }

    #[rstest]
    fn a_member_seen_again_stays_for_a_window_from_then() {
        let (now, seconds) = clock();
        let teams = ActiveTeams::with_clock(None, now);

        teams.touch_local("m", "a", "interactive", WINDOW);
        seconds.fetch_add(40, Ordering::SeqCst);
        teams.touch_local("m", "a", "interactive", WINDOW);
        seconds.fetch_add(40, Ordering::SeqCst);

        assert_eq!(
            teams.touch_local("m", "b", "batch", WINDOW),
            [active("a", "interactive"), active("b", "batch")]
        );
    }

    #[rstest]
    fn each_model_lists_only_its_own_members() {
        let (now, _) = clock();
        let teams = ActiveTeams::with_clock(None, now);

        teams.touch_local("m1", "a", "interactive", WINDOW);

        assert_eq!(
            teams.touch_local("m2", "b", "batch", WINDOW),
            [active("b", "batch")]
        );
    }

    #[rstest]
    #[case::newest_class_wins(
        &["a\u{1f}interactive", "a\u{1f}batch"],
        active("b", "batch"),
        &[("a", "batch"), ("b", "batch")]
    )]
    #[case::the_caller_keeps_the_class_it_sent_with_when_another_pod_wrote_later(
        &["a\u{1f}interactive", "a\u{1f}batch"],
        active("a", "interactive"),
        &[("a", "interactive")]
    )]
    #[case::an_id_holding_the_separator_stays_whole(
        &["owned\u{1f}extra\u{1f}batch"],
        active("owned\u{1f}extra", "batch"),
        &[("owned\u{1f}extra", "batch")]
    )]
    #[case::a_member_that_names_no_class_is_dropped(
        &["stray"],
        active("a", "batch"),
        &[("a", "batch")]
    )]
    fn the_set_is_read_oldest_first_with_the_caller_under_its_own_class(
        #[case] oldest_first: &[&str],
        #[case] caller: ActiveMember,
        #[case] expected: &[(&str, &str)],
    ) {
        let seen = oldest_first
            .iter()
            .filter_map(|member| ActiveMember::read(member));

        let expected: Vec<ActiveMember> = expected
            .iter()
            .map(|(member, priority_class)| active(member, priority_class))
            .collect();
        assert_eq!(listed(seen, caller), expected);
    }
}
