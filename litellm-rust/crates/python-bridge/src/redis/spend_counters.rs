use std::time::Duration;

use litellm_redis_client::RedisClient;
use litellm_redis_store::spend_counters::{self, Counter, Seed};
use pyo3::{IntoPyObjectExt, exceptions::PyValueError, prelude::*};

use super::{NativeRedisClient, store_call};

#[pyclass(frozen, name = "NativeSpendCounters")]
pub(crate) struct NativeSpendCounters(RedisClient);

#[derive(FromPyObject)]
struct CounterArg {
    scope: String,
    id: String,
    team_id: Option<String>,
    window: Option<String>,
}

impl TryFrom<CounterArg> for Counter {
    type Error = PyErr;

    fn try_from(arg: CounterArg) -> PyResult<Self> {
        Ok(match (arg.scope.as_str(), arg.team_id, arg.window) {
            ("key", None, None) => Self::Key(arg.id),
            ("key", None, Some(duration)) => Self::KeyWindow {
                token: arg.id,
                duration,
            },
            ("team", None, None) => Self::Team(arg.id),
            ("team", None, Some(duration)) => Self::TeamWindow {
                team_id: arg.id,
                duration,
            },
            ("team_member", Some(team_id), None) => Self::TeamMember {
                user_id: arg.id,
                team_id,
            },
            ("user", None, None) => Self::User(arg.id),
            ("end_user", None, None) => Self::EndUser(arg.id),
            ("org", None, None) => Self::Org(arg.id),
            ("tag", None, None) => Self::Tag(arg.id),
            ("project", None, None) => Self::Project(arg.id),
            (scope, ..) => {
                return Err(PyValueError::new_err(format!(
                    "no spend counter for scope {scope:?} with these fields"
                )));
            }
        })
    }
}

fn to_counters(args: Vec<CounterArg>) -> PyResult<Vec<Counter>> {
    args.into_iter().map(Counter::try_from).collect()
}

#[pymethods]
impl NativeSpendCounters {
    #[new]
    fn new(client: &NativeRedisClient) -> Self {
        Self(client.0.clone())
    }

    fn read<'py>(&self, py: Python<'py>, counters: Vec<CounterArg>) -> PyResult<Bound<'py, PyAny>> {
        let client = self.0.clone();
        let counters = to_counters(counters)?;
        store_call(
            py,
            async move { spend_counters::read(&client, &counters).await },
            |py, spend| spend.into_py_any(py),
        )
    }

    #[pyo3(signature = (deltas, *, ttl_seconds))]
    fn add<'py>(
        &self,
        py: Python<'py>,
        deltas: Vec<(CounterArg, f64)>,
        ttl_seconds: u64,
    ) -> PyResult<Bound<'py, PyAny>> {
        let client = self.0.clone();
        let deltas = deltas
            .into_iter()
            .map(|(counter, delta)| Ok((Counter::try_from(counter)?, delta)))
            .collect::<PyResult<Vec<_>>>()?;
        let ttl = Duration::from_secs(ttl_seconds);
        store_call(
            py,
            async move { spend_counters::add(&client, &deltas, ttl).await },
            |py, totals| totals.into_py_any(py),
        )
    }

    #[pyo3(signature = (counter, spend, *, ttl_seconds))]
    fn seed<'py>(
        &self,
        py: Python<'py>,
        counter: CounterArg,
        spend: f64,
        ttl_seconds: u64,
    ) -> PyResult<Bound<'py, PyAny>> {
        let client = self.0.clone();
        let counter = Counter::try_from(counter)?;
        let ttl = Duration::from_secs(ttl_seconds);
        store_call(
            py,
            async move { spend_counters::seed(&client, &counter, spend, ttl).await },
            |py, seeded| match seeded {
                Seed::Seeded => ("seeded", None::<f64>).into_py_any(py),
                Seed::Existing(spend) => ("existing", spend).into_py_any(py),
            },
        )
    }

    #[pyo3(signature = (counter, spend, *, ttl_seconds))]
    fn raise_to<'py>(
        &self,
        py: Python<'py>,
        counter: CounterArg,
        spend: f64,
        ttl_seconds: u64,
    ) -> PyResult<Bound<'py, PyAny>> {
        let client = self.0.clone();
        let counter = Counter::try_from(counter)?;
        let ttl = Duration::from_secs(ttl_seconds);
        store_call(
            py,
            async move { spend_counters::raise_to(&client, &counter, spend, ttl).await },
            |py, spend| spend.into_py_any(py),
        )
    }
}
