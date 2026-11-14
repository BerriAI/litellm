use std::{
    collections::HashMap,
    sync::Mutex,
    time::{Instant, SystemTime, UNIX_EPOCH},
};

use indexmap::IndexMap;

use crate::tracker::{SeriesLimits, Tracker};

pub const OVERFLOW_LABEL_VALUE: &str = "other";

#[derive(Clone, Copy, Debug, PartialEq, Eq, strum::EnumString, strum::IntoStaticStr)]
pub enum Kind {
    #[strum(serialize = "counter")]
    Counter,
    #[strum(serialize = "gauge")]
    Gauge,
    #[strum(serialize = "histogram")]
    Histogram,
}

#[derive(Clone, Debug)]
pub struct FamilySpec {
    pub declared: String,
    pub documentation: String,
    pub kind: Kind,
    pub labelnames: Vec<String>,
    pub excluded: Vec<String>,
    pub buckets: Vec<f64>,
    pub most_recent: bool,
    pub wrapped: bool,
    pub shares_overflow: bool,
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum StoreError {
    #[error("Buckets not in sorted order")]
    BucketsUnsorted,
    #[error("Must have at least two buckets")]
    TooFewBuckets,
    #[error("Duplicated timeseries in CollectorRegistry: {0}")]
    Duplicate(String),
    #[error("Incorrect label count")]
    LabelCount,
    #[error("Counters can only be incremented by non-negative amounts.")]
    NegativeCounter,
    #[error("{0} must not be used with the mostrecent mode")]
    MostRecent(&'static str),
    #[error("{0} metric is missing label values")]
    MissingLabels(&'static str),
    #[error("no such series")]
    Missing,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Series {
    pub value: f64,
    pub sum: f64,
    pub buckets: Vec<f64>,
    pub created: f64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct FamilyId(usize);

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Target {
    Parent,
    Child(Vec<String>),
    NoOp,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Op {
    Inc(f64),
    Dec(f64),
    Set(f64),
    Observe(f64),
}

struct Family {
    spec: FamilySpec,
    name: String,
    kept_names: Vec<String>,
    unlabeled: Option<Series>,
    series: IndexMap<Vec<String>, Series>,
}

impl Family {
    fn fresh(&self, created: f64) -> Series {
        Series {
            value: 0.0,
            sum: 0.0,
            buckets: vec![0.0; self.spec.buckets.len()],
            created,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct FamilySnapshot {
    pub name: String,
    pub documentation: String,
    pub kind: Kind,
    pub labelnames: Vec<String>,
    pub buckets: Vec<f64>,
    pub series: Vec<(Vec<String>, Series)>,
}

pub trait Clock: Send + Sync {
    fn monotonic(&self) -> f64;
    fn wall(&self) -> f64;
}

pub struct SystemClock(Instant);

impl Default for SystemClock {
    fn default() -> Self {
        Self(Instant::now())
    }
}

impl Clock for SystemClock {
    fn monotonic(&self) -> f64 {
        self.0.elapsed().as_secs_f64()
    }

    fn wall(&self) -> f64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0.0, |elapsed| elapsed.as_secs_f64())
    }
}

#[derive(Default)]
pub struct ManualClock(Mutex<(f64, f64)>);

impl ManualClock {
    pub fn set(&self, monotonic: f64, wall: f64) {
        *self
            .0
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = (monotonic, wall);
    }
}

impl Clock for ManualClock {
    fn monotonic(&self) -> f64 {
        self.0.lock().unwrap_or_else(|p| p.into_inner()).0
    }

    fn wall(&self) -> f64 {
        self.0.lock().unwrap_or_else(|p| p.into_inner()).1
    }
}

struct Inner {
    families: Vec<Family>,
    by_name: HashMap<String, FamilyId>,
    declared: HashMap<String, Vec<String>>,
    caps: Tracker<String>,
    end_users: Tracker<Option<String>>,
}

pub struct Store {
    inner: Mutex<Inner>,
    limits: SeriesLimits,
    clock: Box<dyn Clock>,
}

fn counter_name(declared: &str) -> &str {
    declared.strip_suffix("_total").unwrap_or(declared)
}

fn prepared_buckets(buckets: &[f64]) -> Result<Vec<f64>, StoreError> {
    if buckets.windows(2).any(|pair| pair[0] > pair[1]) {
        return Err(StoreError::BucketsUnsorted);
    }
    let mut prepared = buckets.to_vec();
    if prepared.last().is_some_and(|last| *last != f64::INFINITY) {
        prepared.push(f64::INFINITY);
    }
    if prepared.len() < 2 {
        return Err(StoreError::TooFewBuckets);
    }
    Ok(prepared)
}

fn apply(kind: Kind, bounds: &[f64], series: &mut Series, op: Op) {
    match (kind, op) {
        (Kind::Histogram, Op::Observe(amount)) => {
            series.sum += amount;
            if let Some(index) = bounds.iter().position(|bound| amount <= *bound) {
                series.buckets[index] += 1.0;
            }
        }
        (_, Op::Inc(amount)) => series.value += amount,
        (_, Op::Dec(amount)) => series.value -= amount,
        (_, Op::Set(value)) => series.value = value,
        (_, Op::Observe(_)) => {}
    }
}

impl Store {
    pub fn new(limits: SeriesLimits, clock: Box<dyn Clock>) -> Self {
        Self {
            inner: Mutex::new(Inner {
                families: Vec::new(),
                by_name: HashMap::new(),
                declared: HashMap::new(),
                caps: Tracker::default(),
                end_users: Tracker::default(),
            }),
            limits,
            clock,
        }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Inner> {
        self.inner
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    pub fn limits(&self) -> &SeriesLimits {
        &self.limits
    }

    pub fn register(&self, spec: FamilySpec) -> Result<FamilyId, StoreError> {
        let buckets = match spec.kind {
            Kind::Histogram => prepared_buckets(&spec.buckets)?,
            Kind::Counter | Kind::Gauge => Vec::new(),
        };
        let name = match spec.kind {
            Kind::Counter => counter_name(&spec.declared).to_owned(),
            Kind::Gauge | Kind::Histogram => spec.declared.clone(),
        };
        let kept_names: Vec<String> = spec
            .labelnames
            .iter()
            .filter(|label| !spec.excluded.contains(label))
            .cloned()
            .collect();
        let mut inner = self.lock();
        if inner.families.iter().any(|family| family.name == name) {
            return Err(StoreError::Duplicate(name));
        }
        let id = FamilyId(inner.families.len());
        let mut family = Family {
            spec: FamilySpec { buckets, ..spec },
            name,
            kept_names,
            unlabeled: None,
            series: IndexMap::new(),
        };
        if family.kept_names.is_empty() {
            family.unlabeled = Some(family.fresh(self.clock.wall()));
        }
        inner.by_name.insert(family.spec.declared.clone(), id);
        inner.families.push(family);
        Ok(id)
    }

    pub fn declare_labels(&self, metric: &str, labels: Vec<String>) {
        self.lock().declared.insert(metric.to_owned(), labels);
    }

    pub fn metric_labels(&self, metric: &str) -> Vec<String> {
        let inner = self.lock();
        inner.declared.get(metric).cloned().unwrap_or_else(|| {
            inner
                .by_name
                .get(metric)
                .map(|id| inner.families[id.0].spec.labelnames.clone())
                .unwrap_or_default()
        })
    }

    pub fn family(&self, declared: &str) -> Option<FamilyId> {
        self.lock().by_name.get(declared).copied()
    }

    pub fn labelnames(&self, id: FamilyId) -> Vec<String> {
        self.lock().families[id.0].spec.labelnames.clone()
    }

    pub fn kept_labelnames(&self, id: FamilyId) -> Vec<String> {
        self.lock().families[id.0].kept_names.clone()
    }

    pub fn is_wrapped(&self, id: FamilyId) -> bool {
        self.lock().families[id.0].spec.wrapped
    }

    pub fn labels(&self, id: FamilyId, values: &[String]) -> Result<Target, StoreError> {
        let mut inner = self.lock();
        self.target(&mut inner, id, values)
    }

    fn kept_values(family: &Family, values: &[String]) -> Vec<String> {
        family
            .spec
            .labelnames
            .iter()
            .zip(values)
            .filter(|(name, _)| !family.spec.excluded.contains(name))
            .map(|(_, value)| value.clone())
            .collect()
    }

    fn target(
        &self,
        inner: &mut Inner,
        id: FamilyId,
        values: &[String],
    ) -> Result<Target, StoreError> {
        let family = &inner.families[id.0];
        if family.unlabeled.is_some() && !family.spec.wrapped {
            return Ok(Target::Parent);
        }
        if !family.spec.wrapped {
            return self.child(inner, id, values.to_vec());
        }
        let kept = Self::kept_values(family, values);
        if kept.is_empty() {
            return Ok(Target::Parent);
        }
        if !self.limits.enabled() {
            return self.child(inner, id, kept);
        }
        let now = self.clock.monotonic();
        let name = family.spec.declared.clone();
        let shares = family.spec.shares_overflow;
        let kept_len = kept.len();
        let expired = inner.caps.admit(&name, &kept, now, &self.limits);
        for gone in expired.removed {
            let _ = Self::remove_child(&mut inner.families[id.0], &gone);
        }
        if expired.admitted {
            return self.child(inner, id, kept);
        }
        if shares {
            return self.child(inner, id, vec![OVERFLOW_LABEL_VALUE.to_owned(); kept_len]);
        }
        Ok(Target::NoOp)
    }

    fn child(
        &self,
        inner: &mut Inner,
        id: FamilyId,
        values: Vec<String>,
    ) -> Result<Target, StoreError> {
        let family = &mut inner.families[id.0];
        if values.len() != family.kept_names.len() {
            return Err(StoreError::LabelCount);
        }
        if !family.series.contains_key(&values) {
            let fresh = family.fresh(self.clock.wall());
            family.series.insert(values.clone(), fresh);
        }
        Ok(Target::Child(values))
    }

    pub fn apply(&self, id: FamilyId, target: &Target, op: Op) -> Result<(), StoreError> {
        let mut inner = self.lock();
        Self::apply_locked(&mut inner, id, target, op)
    }

    fn apply_locked(
        inner: &mut Inner,
        id: FamilyId,
        target: &Target,
        op: Op,
    ) -> Result<(), StoreError> {
        let family = &mut inner.families[id.0];
        let kind = family.spec.kind;
        match (kind, op) {
            (Kind::Counter, Op::Inc(amount)) if amount < 0.0 => {
                return Err(StoreError::NegativeCounter);
            }
            (Kind::Gauge, Op::Inc(_)) if family.spec.most_recent => {
                return Err(StoreError::MostRecent("inc"));
            }
            (Kind::Gauge, Op::Dec(_)) if family.spec.most_recent => {
                return Err(StoreError::MostRecent("dec"));
            }
            _ => {}
        }
        let bounds = family.spec.buckets.clone();
        let series = match target {
            Target::NoOp => return Ok(()),
            Target::Parent => match family.unlabeled.as_mut() {
                Some(series) => series,
                None => return Err(StoreError::MissingLabels(kind.into())),
            },
            Target::Child(values) => match family.series.get_mut(values) {
                Some(series) => series,
                None => return Ok(()),
            },
        };
        apply(kind, &bounds, series, op);
        Ok(())
    }

    pub fn record(&self, id: FamilyId, values: &[String], op: Op) -> Result<(), StoreError> {
        let mut inner = self.lock();
        let target = self.target(&mut inner, id, values)?;
        Self::apply_locked(&mut inner, id, &target, op)
    }

    fn remove_child(family: &mut Family, kept: &[String]) -> Result<(), StoreError> {
        if kept.len() != family.kept_names.len() {
            return Err(StoreError::LabelCount);
        }
        family
            .series
            .shift_remove(kept)
            .map(|_| ())
            .ok_or(StoreError::Missing)
    }

    pub fn remove(&self, id: FamilyId, values: &[String]) -> Result<(), StoreError> {
        let mut inner = self.lock();
        Self::remove_locked(&mut inner, id, values)
    }

    fn remove_locked(inner: &mut Inner, id: FamilyId, values: &[String]) -> Result<(), StoreError> {
        let family = &inner.families[id.0];
        if !family.spec.wrapped {
            return Self::remove_child(&mut inner.families[id.0], values);
        }
        let kept = Self::kept_values(family, values);
        let name = family.spec.declared.clone();
        inner.caps.forget(&name, &kept);
        Self::remove_child(&mut inner.families[id.0], &kept)
    }

    pub fn clear(&self, id: FamilyId) {
        self.lock().families[id.0].series.clear();
    }

    pub fn track_end_user(&self, id: FamilyId, values: &[Option<String>], limits: &SeriesLimits) {
        if !limits.enabled() {
            return;
        }
        let now = self.clock.monotonic();
        let mut inner = self.lock();
        let name = inner.families[id.0].spec.declared.clone();
        let evict = |inner: &mut Inner, gone: &[Option<String>]| {
            let values: Vec<String> = gone
                .iter()
                .map(|value| value.clone().unwrap_or_else(|| "None".into()))
                .collect();
            !matches!(
                Self::remove_locked(inner, id, &values),
                Err(StoreError::LabelCount)
            )
        };
        let expired = inner.end_users.track_due(&name, values, now, limits);
        for gone in expired {
            if evict(&mut inner, &gone) {
                inner.end_users.forget(&name, &gone);
            }
        }
        while let Some(oldest) = inner.end_users.over_cap(&name, limits) {
            if !evict(&mut inner, &oldest) {
                break;
            }
            inner.end_users.forget(&name, &oldest);
        }
    }

    pub fn remove_end_user_series(&self, id: FamilyId, values: &[Option<String>]) {
        let mut inner = self.lock();
        let values: Vec<String> = values
            .iter()
            .map(|value| value.clone().unwrap_or_else(|| "None".into()))
            .collect();
        let _ = Self::remove_locked(&mut inner, id, &values);
    }

    pub fn snapshot(&self, id: FamilyId) -> FamilySnapshot {
        let inner = self.lock();
        let family = &inner.families[id.0];
        FamilySnapshot {
            name: family.name.clone(),
            documentation: family.spec.documentation.clone(),
            kind: family.spec.kind,
            labelnames: family.kept_names.clone(),
            buckets: family.spec.buckets.clone(),
            series: match &family.unlabeled {
                Some(series) => vec![(Vec::new(), series.clone())],
                None => family
                    .series
                    .iter()
                    .map(|(values, series)| (values.clone(), series.clone()))
                    .collect(),
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use rstest::{fixture, rstest};

    use super::*;

    fn spec(kind: Kind, declared: &str, labelnames: &[&str], buckets: &[f64]) -> FamilySpec {
        FamilySpec {
            declared: declared.to_owned(),
            documentation: "doc".to_owned(),
            kind,
            labelnames: labelnames.iter().map(|name| (*name).to_owned()).collect(),
            excluded: Vec::new(),
            buckets: buckets.to_vec(),
            most_recent: false,
            wrapped: true,
            shares_overflow: kind != Kind::Gauge,
        }
    }

    fn values(family: &FamilySnapshot) -> Vec<(Vec<String>, f64)> {
        family
            .series
            .iter()
            .map(|(labels, series)| (labels.clone(), series.value))
            .collect()
    }

    fn label(value: &str) -> Vec<String> {
        vec![value.to_owned()]
    }

    #[fixture]
    fn capped() -> Store {
        Store::new(
            SeriesLimits {
                max_series: Some(1),
                ttl_seconds: None,
                cleanup_interval_seconds: None,
            },
            Box::new(ManualClock::default()),
        )
    }

    #[rstest]
    #[case::unsorted(&[2.0, 1.0], Err(StoreError::BucketsUnsorted))]
    #[case::empty(&[], Err(StoreError::TooFewBuckets))]
    #[case::only_infinity(&[f64::INFINITY], Err(StoreError::TooFewBuckets))]
    #[case::infinity_appended(&[1.0], Ok(vec![1.0, f64::INFINITY]))]
    fn histogram_buckets_follow_the_python_client(
        #[case] buckets: &[f64],
        #[case] expected: Result<Vec<f64>, StoreError>,
    ) {
        let store = Store::new(SeriesLimits::default(), Box::new(ManualClock::default()));
        let registered = store
            .register(spec(Kind::Histogram, "h", &["a"], buckets))
            .map(|id| store.snapshot(id).buckets);
        assert_eq!(registered, expected);
    }

    #[rstest]
    fn a_counter_refuses_a_negative_increment() {
        let store = Store::new(SeriesLimits::default(), Box::new(ManualClock::default()));
        let id = store
            .register(spec(Kind::Counter, "c_total", &["a"], &[]))
            .unwrap_or_else(|error| panic!("{error}"));
        assert_eq!(
            store.record(id, &label("x"), Op::Inc(-1.0)),
            Err(StoreError::NegativeCounter)
        );
        assert_eq!(values(&store.snapshot(id)), vec![(label("x"), 0.0)]);
    }

    #[rstest]
    #[case::counter_folds_into_other(Kind::Counter, vec![(label("first"), 1.0), (label(OVERFLOW_LABEL_VALUE), 1.0)])]
    #[case::gauge_drops_the_new_series(Kind::Gauge, vec![(label("first"), 1.0)])]
    fn a_series_past_the_cap_follows_the_python_overflow_rule(
        capped: Store,
        #[case] kind: Kind,
        #[case] expected: Vec<(Vec<String>, f64)>,
    ) {
        let id = capped
            .register(spec(kind, "m", &["a"], &[]))
            .unwrap_or_else(|error| panic!("{error}"));
        for value in ["first", "second"] {
            capped
                .record(id, &label(value), Op::Inc(1.0))
                .unwrap_or_else(|error| panic!("{error}"));
        }
        assert_eq!(values(&capped.snapshot(id)), expected);
    }

    #[rstest]
    fn a_family_snapshot_holds_only_that_family() {
        let store = Store::new(SeriesLimits::default(), Box::new(ManualClock::default()));
        let first = store
            .register(spec(Kind::Counter, "first_total", &["a"], &[]))
            .unwrap_or_else(|error| panic!("{error}"));
        let second = store
            .register(spec(Kind::Counter, "second_total", &["a"], &[]))
            .unwrap_or_else(|error| panic!("{error}"));
        store
            .record(second, &label("x"), Op::Inc(2.0))
            .unwrap_or_else(|error| panic!("{error}"));
        let snapshot = store.snapshot(first);
        assert_eq!(snapshot.name, "first");
        assert!(snapshot.series.is_empty());
        assert_eq!(values(&store.snapshot(second)), vec![(label("x"), 2.0)]);
    }
}
