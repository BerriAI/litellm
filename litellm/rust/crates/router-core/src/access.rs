//! Who may reach a deployment: model access windows (`litellm_core_utils/access_windows.py`,
//! a deployment reserved for some teams during a daily wall-clock window) and model access
//! groups (`_filter_deployments_by_model_access_groups`, a caller allowed only through access
//! groups is kept to its groups' deployments).

use std::{collections::BTreeSet, sync::Arc};

use jiff::{Timestamp, tz::TimeZone};
use serde_json::Value;

use crate::registry::Deployment;

/// One `model_info.access_windows` entry (`ModelAccessWindow`), validated when the deployment
/// was added.
#[derive(Clone, Debug, PartialEq)]
pub struct AccessWindow {
    /// Microseconds after midnight.
    pub start: u64,
    pub end: u64,
    pub timezone: String,
    pub team_ids: Vec<String>,
}

const MICROS_PER_SECOND: u64 = 1_000_000;

/// A `datetime.time` as pydantic reads it: `HH:MM[:SS[.ffffff]]`, or seconds after midnight.
fn wall_clock(value: &Value) -> Option<u64> {
    match value {
        Value::Number(seconds) => {
            let seconds = seconds.as_f64()?;
            (0.0..86_400.0)
                .contains(&seconds)
                .then(|| (seconds * MICROS_PER_SECOND as f64).round() as u64)
        }
        Value::String(text) => {
            let mut parts = text.splitn(3, ':');
            let hour: u64 = parts.next()?.parse().ok()?;
            let minute: u64 = parts.next()?.parse().ok()?;
            let (second, micros) = match parts.next() {
                None => (0, 0),
                Some(rest) => {
                    let (whole, fraction) = rest.split_once('.').unwrap_or((rest, ""));
                    let digits: String = fraction.chars().take(6).collect();
                    let micros = if digits.is_empty() {
                        0
                    } else {
                        format!("{digits:0<6}").parse().ok()?
                    };
                    (whole.parse().ok()?, micros)
                }
            };
            (hour < 24 && minute < 60 && second < 60)
                .then(|| ((hour * 60 + minute) * 60 + second) * MICROS_PER_SECOND + micros)
        }
        _ => None,
    }
}

impl AccessWindow {
    fn parse(value: &Value) -> Option<Self> {
        let window = value.as_object()?;
        Some(Self {
            start: wall_clock(window.get("start")?)?,
            end: wall_clock(window.get("end")?)?,
            timezone: window.get("timezone")?.as_str()?.to_owned(),
            team_ids: window
                .get("team_ids")?
                .as_array()?
                .iter()
                .filter_map(Value::as_str)
                .map(str::to_owned)
                .collect(),
        })
    }

    /// `is_window_active`: the window's wall clock contains `now`, wrapping past midnight when
    /// it ends before it starts.
    pub fn is_active(&self, now: Timestamp) -> bool {
        let Ok(zone) = TimeZone::get(&self.timezone) else {
            return false;
        };
        let local = now.to_zoned(zone).time();
        let micros = ((u64::from(local.hour().unsigned_abs()) * 60
            + u64::from(local.minute().unsigned_abs()))
            * 60
            + u64::from(local.second().unsigned_abs()))
            * MICROS_PER_SECOND
            + u64::from(local.subsec_nanosecond().unsigned_abs()) / 1_000;
        if self.start < self.end {
            self.start <= micros && micros < self.end
        } else {
            micros >= self.start || micros < self.end
        }
    }

    /// The window's end as `{end:%H:%M}`.
    pub fn end_hhmm(&self) -> String {
        let minutes = self.end / MICROS_PER_SECOND / 60;
        format!("{:02}:{:02}", minutes / 60, minutes % 60)
    }
}

/// A deployment's access windows.
pub fn access_windows(deployment: &Deployment) -> Vec<AccessWindow> {
    deployment
        .model_info
        .raw
        .get("access_windows")
        .and_then(Value::as_array)
        .map(|windows| windows.iter().filter_map(AccessWindow::parse).collect())
        .unwrap_or_default()
}

/// `_reservation_blocking_window`: the first active window, unless the caller's team is named
/// by one of the active windows.
fn blocking_window(
    deployment: &Deployment,
    team_id: Option<&str>,
    now: Timestamp,
) -> Option<AccessWindow> {
    let active: Vec<AccessWindow> = access_windows(deployment)
        .into_iter()
        .filter(|window| window.is_active(now))
        .collect();
    if let Some(team) = team_id
        && active
            .iter()
            .any(|window| window.team_ids.iter().any(|id| id == team))
    {
        return None;
    }
    active.into_iter().next()
}

/// `filter_reserved_deployments`: the deployments no active window withholds from the caller,
/// and the first window that withheld one.
pub fn filter_reserved(
    deployments: Vec<Arc<Deployment>>,
    team_id: Option<&str>,
    now: Timestamp,
) -> (Vec<Arc<Deployment>>, Option<AccessWindow>) {
    let mut blocking = None;
    let kept = deployments
        .into_iter()
        .filter(
            |deployment| match blocking_window(deployment, team_id, now) {
                None => true,
                Some(window) => {
                    blocking.get_or_insert(window);
                    false
                }
            },
        )
        .collect();
    (kept, blocking)
}

/// `_filter_deployments_by_model_access_groups`: a caller whose allowed models (`models` plus
/// `team_models`) reach `model` only through access groups is kept to the deployments in those
/// groups. `model_groups` are the access groups the rows listed for `model` carry.
pub fn filter_by_access_groups(
    model: &str,
    deployments: Vec<Arc<Deployment>>,
    allowed_models: &[String],
    model_groups: impl FnOnce() -> BTreeSet<String>,
) -> Vec<Arc<Deployment>> {
    if deployments.is_empty() || allowed_models.is_empty() {
        return deployments;
    }
    let allowed = |name: &str| allowed_models.iter().any(|m| m == name);
    if allowed(model) || allowed("*") || allowed("all-proxy-models") {
        return deployments;
    }
    let groups: BTreeSet<String> = model_groups()
        .into_iter()
        .filter(|group| allowed(group))
        .collect();
    if groups.is_empty() {
        return deployments;
    }
    deployments
        .into_iter()
        .filter(|d| {
            d.model_info
                .access_groups
                .iter()
                .any(|g| groups.contains(g))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use serde_json::json;

    use super::*;

    #[rstest]
    #[case::minutes(json!("09:30"), Some(34_200_000_000))]
    #[case::seconds(json!("09:30:15"), Some(34_215_000_000))]
    #[case::fraction(json!("00:00:00.5"), Some(500_000))]
    #[case::number(json!(60), Some(60_000_000))]
    #[case::bad(json!("25:00"), None)]
    fn wall_clock_reads_pydantic_times(#[case] raw: Value, #[case] expected: Option<u64>) {
        assert_eq!(wall_clock(&raw), expected);
    }

    #[rstest]
    #[case::inside("2026-01-01T10:00:00Z", true)]
    #[case::at_end("2026-01-01T17:00:00Z", false)]
    #[case::before("2026-01-01T08:59:59Z", false)]
    fn a_window_is_active_inside_its_wall_clock(#[case] now: &str, #[case] expected: bool) {
        let window = AccessWindow {
            start: wall_clock(&json!("09:00")).unwrap(),
            end: wall_clock(&json!("17:00")).unwrap(),
            timezone: "UTC".to_owned(),
            team_ids: vec!["t".to_owned()],
        };
        assert_eq!(window.is_active(now.parse().unwrap()), expected);
        assert_eq!(window.end_hhmm(), "17:00");
    }

    #[test]
    fn an_overnight_window_wraps_past_midnight() {
        let window = AccessWindow {
            start: wall_clock(&json!("22:00")).unwrap(),
            end: wall_clock(&json!("02:00")).unwrap(),
            timezone: "UTC".to_owned(),
            team_ids: vec!["t".to_owned()],
        };
        assert!(window.is_active("2026-01-01T23:00:00Z".parse().unwrap()));
        assert!(window.is_active("2026-01-01T01:00:00Z".parse().unwrap()));
        assert!(!window.is_active("2026-01-01T12:00:00Z".parse().unwrap()));
    }
}
