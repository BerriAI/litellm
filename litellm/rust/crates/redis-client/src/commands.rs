use std::time::Duration;

use litellm_python_compat::repr::float_repr;
use redis::{
    Cmd, ErrorKind, FromRedisValue, RedisError, RedisWrite, ToRedisArgs, ToSingleRedisArg, Value,
};

use crate::{Error, Namespace, Operation, Script, Ttl, script::ScriptCall};

/// How an increment treats the key's TTL, the two ways `RedisCache` increments with one.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Expiry {
    /// `async_increment(refresh_ttl=True)` and every pipelined increment: `EXPIRE` after each write.
    Refresh(Duration),
    /// `async_increment(refresh_ttl=False)`: read the `TTL` and `EXPIRE` only a key that has none.
    /// Pipelined it is lowered to [`Expiry::Refresh`], like `RedisBatch.increment`.
    KeepExisting(Duration),
}

/// A number written the way redis-py encodes it: an `int` with `str()`, a `float` with `repr()`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Amount {
    Int(i64),
    Float(f64),
}

impl ToRedisArgs for Amount {
    fn write_redis_args<W: ?Sized + RedisWrite>(&self, out: &mut W) {
        match self {
            Self::Int(value) => out.write_arg(value.to_string().as_bytes()),
            Self::Float(value) => out.write_arg(float_repr(*value).as_bytes()),
        }
    }
}

impl ToSingleRedisArg for Amount {}

type Build = Box<dyn FnOnce(Vec<String>) -> Plan + Send>;
type Decode<T> = Box<dyn FnOnce(Value) -> Result<T, Error> + Send>;

/// One Redis command as a value: what it sends alone and pipelined, and how its reply decodes. Every
/// executor (`RedisClient::send`, `Pipeline::with`, `RedisBatch::add`) applies its namespace to the
/// keys once.
#[must_use]
pub struct Command<T> {
    operation: Operation,
    keys: Vec<String>,
    idempotent: bool,
    build: Build,
    decode: Decode<T>,
}

/// What a command sends once its keys carry the namespace. Only the cases whose alone and pipelined
/// forms differ get a variant of their own.
pub(crate) enum Plan {
    /// Answered without a round trip, like an empty `MGET` or `DEL`.
    Reply(Value),
    /// Sent together; the reply is the first command's, failing if any failed.
    Commands(Vec<Cmd>),
    /// Alone through `invoke_async`, pipelined as `EVALSHA` resent once on `NOSCRIPT`.
    Script(ScriptCall),
    /// Alone the increment reads the `TTL` and sends `EXPIRE` only to a key without one, pipelined it
    /// is the increment then `EXPIRE`.
    KeepingTtl {
        increment: Cmd,
        key: String,
        ttl: Duration,
        trips: Trips,
    },
}

#[derive(Clone, Copy)]
pub(crate) enum Trips {
    /// `async_increment`: the increment and its `TTL` in one round trip.
    Shared,
    /// `increment_cache`: the increment and its `TTL` each on its own round trip.
    Separate,
}

/// A command with the namespace applied, ready for an executor to send.
pub(crate) struct Op {
    pub(crate) operation: Operation,
    pub(crate) key: String,
    pub(crate) idempotent: bool,
    pub(crate) plan: Plan,
}

impl Plan {
    /// What a pipeline sends for this op.
    pub(crate) fn pipelined(&self) -> Vec<Cmd> {
        match self {
            Self::Reply(_) => Vec::new(),
            Self::Commands(commands) => commands.clone(),
            Self::Script(call) => vec![call.script.evalsha(&call.keys, &call.args)],
            Self::KeepingTtl {
                increment,
                key,
                ttl,
                ..
            } => vec![increment.clone(), expire_cmd(key, *ttl)],
        }
    }
}

impl<T> Command<T> {
    fn new(
        operation: Operation,
        keys: Vec<String>,
        idempotent: bool,
        build: impl FnOnce(Vec<String>) -> Plan + Send + 'static,
        decode: impl FnOnce(Value) -> Result<T, Error> + Send + 'static,
    ) -> Self {
        Self {
            operation,
            keys,
            idempotent,
            build: Box::new(build),
            decode: Box::new(decode),
        }
    }

    /// Attaches a codec over the decoded reply, such as a store's value format.
    pub fn map<U: 'static>(self, decode: fn(T) -> Result<U, Error>) -> Command<U>
    where
        T: 'static,
    {
        let inner = self.decode;
        Command {
            operation: self.operation,
            keys: self.keys,
            idempotent: self.idempotent,
            build: self.build,
            decode: Box::new(move |value| inner(value).and_then(decode)),
        }
    }

    pub(crate) fn prepare(self, namespace: &Namespace) -> (Op, Decode<T>) {
        let keys: Vec<String> = self.keys.iter().map(|key| namespace.apply(key)).collect();
        let key = keys.first().cloned().unwrap_or_default();
        let op = Op {
            operation: self.operation,
            key,
            idempotent: self.idempotent,
            plan: (self.build)(keys),
        };
        (op, self.decode)
    }
}

fn decode<T: FromRedisValue>(value: Value) -> Result<T, Error> {
    T::from_redis_value(value).map_err(|error| Error::Decode(error.into()))
}

fn owned(keys: &[&str]) -> Vec<String> {
    keys.iter().map(|key| (*key).to_owned()).collect()
}

fn single(keys: Vec<String>) -> String {
    keys.into_iter().next().unwrap_or_default()
}

fn expire_cmd(key: &str, ttl: Duration) -> Cmd {
    redis::cmd("EXPIRE").arg(key).arg(ttl.as_secs()).to_owned()
}

pub(crate) fn ttl_cmd(key: &str) -> Cmd {
    redis::cmd("TTL").arg(key).to_owned()
}

/// A reply whose shape a codec over it refuses, as `what` describes.
pub fn unexpected_reply(what: &'static str) -> Error {
    Error::Decode(RedisError::from((ErrorKind::UnexpectedReturnType, what)))
}

pub fn get<T: FromRedisValue + 'static>(key: &str) -> Command<Option<T>> {
    Command::new(
        Operation::Command("GET"),
        vec![key.to_owned()],
        true,
        |keys| Plan::Commands(vec![redis::cmd("GET").arg(single(keys)).to_owned()]),
        decode,
    )
}

/// One `MGET` of `keys` in order, repeated keys read once, like `RedisBatch.mget`, answering one
/// value per key of `keys` in their order. No keys is no round trip, since Redis rejects an empty
/// `MGET`. Pipelined, every key must share a hash slot on a cluster, and a batch never pipelines
/// there.
pub fn mget<T: FromRedisValue + 'static>(keys: &[&str]) -> Command<Vec<Option<T>>> {
    let unique = keys.iter().fold(Vec::<String>::new(), |mut unique, key| {
        if !unique.iter().any(|seen| seen == key) {
            unique.push((*key).to_owned());
        }
        unique
    });
    let positions: Vec<usize> = keys
        .iter()
        .filter_map(|key| unique.iter().position(|seen| seen == key))
        .collect();
    Command::new(
        Operation::Command("MGET"),
        unique,
        true,
        |keys| match keys.is_empty() {
            true => Plan::Reply(Value::Array(Vec::new())),
            false => Plan::Commands(vec![redis::cmd("MGET").arg(keys).to_owned()]),
        },
        move |reply| {
            let values: Vec<Value> = decode(reply)?;
            let in_key_order = positions
                .iter()
                .map(|&position| values.get(position).cloned())
                .collect::<Option<Vec<Value>>>()
                .ok_or_else(|| unexpected_reply("MGET answered fewer values than keys"))?;
            decode(Value::Array(in_key_order))
        },
    )
}

// redis-py's `set(ex=...)` takes whole seconds, as `int(ttl)` or a `timedelta`'s `int(total_seconds())`
pub fn set(key: &str, value: impl ToSingleRedisArg, ttl: Duration) -> Command<()> {
    let value = value.to_redis_args().concat();
    Command::new(
        Operation::Command("SET"),
        vec![key.to_owned()],
        true,
        move |keys| {
            Plan::Commands(vec![
                redis::cmd("SET")
                    .arg(single(keys))
                    .arg(value)
                    .arg("EX")
                    .arg(ttl.as_secs())
                    .to_owned(),
            ])
        },
        decode,
    )
}

/// `SET ... EX ... NX`, `EX` before `NX` as redis-py builds it. `true` when this call wrote the key.
pub fn set_nx(key: &str, value: impl ToSingleRedisArg, ttl: Duration) -> Command<bool> {
    let value = value.to_redis_args().concat();
    Command::new(
        Operation::Command("SET"),
        vec![key.to_owned()],
        false,
        move |keys| {
            Plan::Commands(vec![
                redis::cmd("SET")
                    .arg(single(keys))
                    .arg(value)
                    .arg("EX")
                    .arg(ttl.as_secs())
                    .arg("NX")
                    .to_owned(),
            ])
        },
        |value| decode::<Option<String>>(value).map(|reply| reply.is_some()),
    )
}

/// One multi-key `DEL`, no round trip for no keys. Pipelined, every key must share a hash slot.
pub fn del(keys: &[&str]) -> Command<usize> {
    Command::new(
        Operation::Command("DEL"),
        owned(keys),
        true,
        |keys| match keys.is_empty() {
            true => Plan::Reply(Value::Int(0)),
            false => Plan::Commands(vec![redis::cmd("DEL").arg(keys).to_owned()]),
        },
        decode,
    )
}

pub fn expire(key: &str, ttl: Duration) -> Command<bool> {
    Command::new(
        Operation::Command("EXPIRE"),
        vec![key.to_owned()],
        true,
        move |keys| Plan::Commands(vec![expire_cmd(&single(keys), ttl)]),
        decode,
    )
}

pub fn ttl(key: &str) -> Command<Ttl> {
    Command::new(
        Operation::Command("TTL"),
        vec![key.to_owned()],
        true,
        |keys| Plan::Commands(vec![ttl_cmd(&single(keys))]),
        |value| {
            Ok(match decode::<i64>(value)? {
                -2 => Ttl::Missing,
                -1 => Ttl::Persistent,
                seconds => Ttl::Expires(Duration::from_secs(seconds.unsigned_abs())),
            })
        },
    )
}

/// `RedisCache.increment_cache`: `INCRBY`, then `TTL`, then `EXPIRE` only when the key has no TTL,
/// each its own round trip.
pub fn incr_by(key: &str, delta: i64, ttl: Duration) -> Command<i64> {
    Command::new(
        Operation::Command("INCRBY"),
        vec![key.to_owned()],
        false,
        move |keys| {
            let key = single(keys);
            Plan::KeepingTtl {
                increment: redis::cmd("INCRBY").arg(&key).arg(delta).to_owned(),
                key,
                ttl,
                trips: Trips::Separate,
            }
        },
        decode,
    )
}

/// `INCRBYFLOAT`, with the TTL handled as `expiry` says, or left alone without one. Alone it is
/// `RedisCache.async_increment`, pipelined `RedisBatch.increment`.
pub fn incr_by_float(key: &str, amount: Amount, expiry: Option<Expiry>) -> Command<f64> {
    Command::new(
        Operation::Command("INCRBYFLOAT"),
        vec![key.to_owned()],
        false,
        move |keys| {
            let key = single(keys);
            let increment = redis::cmd("INCRBYFLOAT").arg(&key).arg(amount).to_owned();
            match expiry {
                None => Plan::Commands(vec![increment]),
                Some(Expiry::Refresh(ttl)) => {
                    Plan::Commands(vec![increment, expire_cmd(&key, ttl)])
                }
                Some(Expiry::KeepExisting(ttl)) => Plan::KeepingTtl {
                    increment,
                    key,
                    ttl,
                    trips: Trips::Shared,
                },
            }
        },
        decode,
    )
}

/// A registered script, like redis-py's `Script`. Every key it touches must be in `keys`.
pub fn eval<T: FromRedisValue + 'static>(
    script: &Script,
    keys: &[&str],
    args: impl ToRedisArgs,
) -> Command<T> {
    let script = script.clone();
    let args = args.to_redis_args();
    Command::new(
        Operation::Script,
        owned(keys),
        false,
        move |keys| Plan::Script(ScriptCall { script, keys, args }),
        decode,
    )
}

/// A script sent with its source as `EVAL` every time, like redis-py's `Redis.eval`, for the scripts
/// `RedisCache` runs that way rather than through `register_script`.
pub fn eval_source<T: FromRedisValue + 'static>(
    source: &'static str,
    keys: &[&str],
    args: impl ToRedisArgs,
) -> Command<T> {
    let args = args.to_redis_args();
    Command::new(
        Operation::Script,
        owned(keys),
        false,
        move |keys| {
            Plan::Commands(vec![
                redis::cmd("EVAL")
                    .arg(source)
                    .arg(keys.len())
                    .arg(keys)
                    .arg(args)
                    .to_owned(),
            ])
        },
        decode,
    )
}

/// The channel is sent as given, without the namespace, like redis-py's `publish`.
pub fn publish(channel: &str, payload: &[u8]) -> Command<()> {
    let command = redis::cmd("PUBLISH").arg(channel).arg(payload).to_owned();
    Command::new(
        Operation::Command("PUBLISH"),
        Vec::new(),
        false,
        move |_| Plan::Commands(vec![command]),
        |_| Ok(()),
    )
}

pub(crate) fn ping() -> Cmd {
    redis::cmd("PING")
}
