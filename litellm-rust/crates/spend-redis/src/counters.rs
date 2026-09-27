use std::{convert::Infallible, time::Duration};

use litellm_spend::{Cost, CounterKey, Counters};
use redis::aio::ConnectionLike;

use crate::{CounterNaming, Error};

#[derive(Clone)]
pub struct RedisCounters<C, N> {
    connection: C,
    naming: N,
    ttl: Option<Duration>,
}

impl<C, N> RedisCounters<C, N>
where
    N: CounterNaming,
{
    pub fn new(connection: C, naming: N, ttl: Option<Duration>) -> Self {
        Self {
            connection,
            naming,
            ttl,
        }
    }

    fn name(&self, key: &CounterKey) -> Result<String, Error<Infallible>> {
        self.naming
            .counter(key)
            .ok_or_else(|| Error::Unnamed(key.clone()))
    }
}

impl<C, N> Counters for RedisCounters<C, N>
where
    C: ConnectionLike + Clone + Send + Sync + 'static,
    N: CounterNaming,
{
    type Error = Error<Infallible>;

    async fn add(&self, increments: &[(CounterKey, Cost)]) -> Result<Vec<Cost>, Self::Error> {
        let named = increments
            .iter()
            .map(|(key, cost)| Ok((self.name(key)?, cost.0)))
            .collect::<Result<Vec<_>, Self::Error>>()?;
        if named.is_empty() {
            return Ok(Vec::new());
        }
        let mut pipeline = redis::pipe();
        for (name, amount) in &named {
            pipeline.cmd("INCRBYFLOAT").arg(name).arg(amount);
            if let Some(ttl) = self.ttl {
                pipeline
                    .cmd("PEXPIRE")
                    .arg(name)
                    .arg(ttl.as_millis().max(1) as u64)
                    .ignore();
            }
        }
        let mut connection = self.connection.clone();
        let totals: Vec<f64> = pipeline.query_async(&mut connection).await?;
        Ok(totals.into_iter().map(Cost).collect())
    }

    async fn current(&self, key: &CounterKey) -> Result<Option<Cost>, Self::Error> {
        let mut connection = self.connection.clone();
        let total: Option<f64> = redis::cmd("GET")
            .arg(self.name(key)?)
            .query_async(&mut connection)
            .await?;
        Ok(total.map(Cost))
    }
}
