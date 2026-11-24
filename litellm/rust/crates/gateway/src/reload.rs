//! Reads the config the proxy stores in its database again while serving, every
//! `general_settings.proxy_config_reload_interval_seconds`, as Python's `add_deployment` job does,
//! and swaps what [`Live`] serves when it changed.
//!
//! Each poll reads the settings rows, the models and the credentials the way boot reads them
//! (`stored.rs`). Rows equal to the ones last served change nothing, so most polls build nothing.
//! Otherwise the config is layered again from the file boot read and the new rows, a new
//! [`Served`] is built from it over the parts the process keeps (`AppState::next`), and `Live`
//! swaps to it. What changed is logged by model id, credential name and settings key, never by
//! value.
//!
//! Python's guards against a database blip emptying the router hold here too
//! (`_update_llm_router` and `_delete_deployment` in `proxy_server.py`). A poll whose model or
//! credential read fails changes nothing. A poll whose settings read fails keeps the settings it
//! last served and deletes no model, only adding and updating. A config that does not build keeps
//! the current one. The first read that loads lifts `db_config_not_loaded`, since the swap builds
//! `Admit` over [`StoredConfig::Loaded`].

use std::{
    collections::{BTreeMap, BTreeSet},
    pin::Pin,
    sync::Arc,
    time::Duration,
};

use futures_util::stream::BoxStream;
use litellm_at_rest::AtRestKey;
use litellm_config::{Config, Value, db::models::DbModel};
use litellm_config_load::Document;
use litellm_db::Pools;
use litellm_error::Report;
use litellm_redis_store::config_sync::{self, Heard};
use litellm_repositories::Database;
use tokio::{sync::Mutex, task::JoinHandle, time::MissedTickBehavior};

use crate::{
    Error, Live, Served, StoredConfig,
    config_sync::{Pacing, listen, uniform},
    stored::{self, Fetched, Secrets, StoredRows},
};

/// Python falls back to this when `proxy_config_reload_interval_seconds` is not positive.
const DEFAULT_EVERY: Duration = Duration::from_secs(30);

/// How often the stored config is read again, `general_settings.proxy_config_reload_interval_seconds`
/// as boot loaded it.
pub fn reload_every(config: &Config) -> Duration {
    match config.general_settings.proxy_config_reload_interval_seconds {
        0 => DEFAULT_EVERY,
        seconds => Duration::from_secs(seconds),
    }
}

/// Where the stored config is read again: the database boot read it from, over the pool boot
/// opened, which lives as long as the poller.
pub struct Source {
    reading: Reading,
    database: Database,
    pools: Pools,
}

impl Source {
    pub(crate) fn new(
        document: Document,
        key: Option<AtRestKey>,
        secrets: Arc<Secrets>,
        database: Database,
        pools: Pools,
        served: Option<Fetched>,
    ) -> Self {
        Self {
            reading: Reading {
                document,
                key,
                secrets,
                served,
            },
            database,
            pools,
        }
    }

    /// Reads the stored config again every `every`, and soon after each change the proxy
    /// announces on the coordination Redis `live` serves with, until [`Poller::stop`], and swaps
    /// `live` to what `rebuild` builds from each config that changed, given the [`Served`] it
    /// replaces. Both go through one [`Reloader`], so they never read at once and always agree on
    /// what is served. Without a coordination Redis only the poll runs, as Python then neither
    /// publishes nor subscribes.
    pub fn reload_on<R, F>(self, every: Duration, live: Arc<Live>, rebuild: R) -> Poller
    where
        R: Fn(Config, StoredConfig, Arc<Served>) -> F + Send + Sync + 'static,
        F: Future<Output = Result<Served, Error>> + Send + 'static,
    {
        let heard = live
            .current()
            .state
            .redis
            .coordination
            .as_ref()
            .map(|client| config_sync::subscribe(client, client.namespace().as_str()));
        tracing::info!(
            every_seconds = every.as_secs(),
            on_redis_changes = heard.is_some(),
            "reading the config the proxy stores in its database again on an interval"
        );
        let reloader = Arc::new(Reloader {
            reading: Mutex::new(self.reading),
            rows: self.database,
            live,
            rebuild,
        });
        Poller {
            tasks: reloader.spawn(every, heard),
            pools: self.pools,
        }
    }
}

/// The running poll task and Redis listener.
pub struct Poller {
    tasks: Vec<JoinHandle<()>>,
    pools: Pools,
}

impl Poller {
    /// Stops polling and listening, dropping a read that is under way before it swaps, and closes
    /// the pool the reads go through.
    pub async fn stop(self) {
        for task in &self.tasks {
            task.abort();
        }
        for task in self.tasks {
            match task.await {
                Err(error) if !error.is_cancelled() => tracing::warn!(
                    error = %Report::new(&error),
                    "a task that reads the stored config again did not stop cleanly"
                ),
                _ => (),
            }
        }
        self.pools.close().await;
    }
}

/// The one [`Reading`] every trigger reads the stored config through, one read at a time.
pub(crate) struct Reloader<S, R> {
    reading: Mutex<Reading>,
    rows: S,
    live: Arc<Live>,
    rebuild: R,
}

impl<S, R, F> Reloader<S, R>
where
    S: StoredRows,
    R: Fn(Config, StoredConfig, Arc<Served>) -> F,
    F: Future<Output = Result<Served, Error>>,
{
    pub(crate) async fn reload(&self) -> Reloaded {
        self.reading
            .lock()
            .await
            .reload(&self.rows, &self.live, &self.rebuild)
            .await
    }
}

impl<R, F> Reloader<Database, R>
where
    R: Fn(Config, StoredConfig, Arc<Served>) -> F + Send + Sync + 'static,
    F: Future<Output = Result<Served, Error>> + Send + 'static,
{
    /// [`Reloader::reload`] on the heap, since its future is too deep for a spawned task to lay
    /// out inline in a release build.
    fn boxed_reload(&self) -> Pin<Box<dyn Future<Output = Reloaded> + Send + '_>> {
        Box::pin(self.reload())
    }

    /// Starts the poll every `every`, and the listener on `heard` when there is one.
    fn spawn(
        self: Arc<Self>,
        every: Duration,
        heard: Option<BoxStream<'static, Heard>>,
    ) -> Vec<JoinHandle<()>> {
        let polls = self.clone();
        let poll = tokio::spawn(async move {
            let mut ticks = tokio::time::interval_at(tokio::time::Instant::now() + every, every);
            ticks.set_missed_tick_behavior(MissedTickBehavior::Delay);
            loop {
                ticks.tick().await;
                polls.boxed_reload().await;
            }
        });
        let listener = heard.map(|heard| {
            tokio::spawn(listen(heard, Pacing::PYTHON, uniform, move || {
                let reloader = self.clone();
                async move {
                    reloader.boxed_reload().await;
                }
            }))
        });
        std::iter::once(poll).chain(listener).collect()
    }
}

/// What the stored config is layered from on each read, and what the last read the native app
/// serves found.
pub(crate) struct Reading {
    document: Document,
    key: Option<AtRestKey>,
    secrets: Arc<Secrets>,
    /// `None` until a read loads, so while boot could not read the stored config.
    served: Option<Fetched>,
}

/// How one poll ended.
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum Reloaded {
    /// A read failed, so nothing changed.
    Unread,
    /// The rows are the ones served, so nothing was built.
    Unchanged,
    /// The config the rows describe could not be built, so the current one is kept.
    Unbuilt,
    Swapped,
}

impl Reading {
    /// Reads `rows` and swaps `live` to what `rebuild` builds when they changed.
    pub(crate) async fn reload<R, F>(
        &mut self,
        rows: &impl StoredRows,
        live: &Live,
        rebuild: &R,
    ) -> Reloaded
    where
        R: Fn(Config, StoredConfig, Arc<Served>) -> F,
        F: Future<Output = Result<Served, Error>>,
    {
        let Some(fetched) = self.read(rows).await else {
            return Reloaded::Unread;
        };
        let changes = Changes::between(self.served.as_ref(), &fetched);
        if self.served.is_some() && changes.is_empty() {
            return Reloaded::Unchanged;
        }
        changes.log();
        let config = match stored::build(self.document.clone(), &fetched, &*self.secrets).await {
            Ok(config) => config,
            Err(error) => {
                tracing::warn!(
                    error = %Report::new(&error),
                    "the config with the changed stored config layered in does not load, so the gateway keeps serving the one it has"
                );
                return Reloaded::Unbuilt;
            }
        };
        let swapped = live
            .swap(|current| rebuild(config, StoredConfig::Loaded, current))
            .await;
        match swapped {
            Ok(_) => {
                self.served = Some(fetched);
                Reloaded::Swapped
            }
            Err(error) => {
                tracing::warn!(
                    error = %Report::new(&error),
                    "could not build the native app from the changed stored config, so the gateway keeps serving the one it has"
                );
                Reloaded::Unbuilt
            }
        }
    }

    /// The stored config `rows` hold now, `None` when it cannot be read. When only the settings
    /// rows cannot be read, the settings last served stand in for them and no model served is
    /// dropped, as Python skips deletions when its config read fails.
    async fn read(&self, rows: &impl StoredRows) -> Option<Fetched> {
        let key = self.key.as_ref();
        let (settings, deletes) = match (rows.settings().await, &self.served) {
            (Ok(found), _) => (stored::settings_rows(&found, key), true),
            (Err(error), Some(served)) => {
                tracing::warn!(
                    error = %Report::new(&error),
                    "could not read the settings the proxy stores in its database, so this read keeps the settings and every model it last served"
                );
                (served.settings.clone(), false)
            }
            (Err(error), None) => {
                tracing::warn!(
                    error = %Report::new(&error),
                    "could not read the settings the proxy stores in its database"
                );
                return None;
            }
        };
        let fetched = match stored::fetch(&self.document, settings, rows, key).await {
            Ok(fetched) => fetched,
            Err(error) => {
                tracing::warn!(
                    error = %Report::new(&error),
                    "could not read the models the proxy stores in its database, so the gateway keeps serving the ones it has"
                );
                return None;
            }
        };
        match (deletes, &self.served) {
            (false, Some(served)) => Some(without_deletes(fetched, &served.models)),
            _ => Some(fetched),
        }
    }
}

/// `fetched` with every model of `served` it no longer holds put back.
fn without_deletes(fetched: Fetched, served: &[DbModel]) -> Fetched {
    let read: BTreeSet<&str> = fetched
        .models
        .iter()
        .map(|model| model.model_id.as_str())
        .collect();
    let kept: Vec<DbModel> = served
        .iter()
        .filter(|model| !read.contains(model.model_id.as_str()))
        .cloned()
        .collect();
    Fetched {
        models: fetched.models.iter().cloned().chain(kept).collect(),
        ..fetched
    }
}

/// What differs between two reads of the stored config, by name only.
#[derive(Debug, Default, PartialEq, Eq)]
pub(crate) struct Changes {
    added: Vec<String>,
    updated: Vec<String>,
    removed: Vec<String>,
    credentials: Vec<String>,
    settings: Vec<String>,
}

impl Changes {
    /// From what the native app serves, nothing when boot could not read it, to `now`.
    fn between(served: Option<&Fetched>, now: &Fetched) -> Self {
        let empty = Fetched::default();
        let before = served.unwrap_or(&empty);
        let models = |fetched: &Fetched| -> BTreeMap<String, Model> {
            fetched
                .models
                .iter()
                .map(|model| (model.model_id.clone(), Model(model.clone())))
                .collect()
        };
        let (old, new) = (models(before), models(now));
        let credentials = |fetched: &Fetched| -> BTreeMap<String, BTreeMap<String, Value>> {
            fetched
                .credentials
                .iter()
                .map(|credential| {
                    (
                        credential.credential_name.clone(),
                        (*credential.credential_values).clone(),
                    )
                })
                .collect()
        };
        let (was, is) = (&before.settings, &now.settings);
        let settings = [
            (
                "general_settings",
                changed(&*was.general_settings, &*is.general_settings),
            ),
            (
                "router_settings",
                changed(&*was.router_settings, &*is.router_settings),
            ),
            (
                "litellm_settings",
                changed(&*was.litellm_settings, &*is.litellm_settings),
            ),
            (
                "environment_variables",
                changed(&was.environment_variables, &is.environment_variables),
            ),
        ]
        .into_iter()
        .flat_map(|(section, keys)| keys.into_iter().map(move |key| format!("{section}.{key}")))
        .collect();
        Self {
            added: new
                .keys()
                .filter(|id| !old.contains_key(*id))
                .cloned()
                .collect(),
            updated: changed(&old, &new)
                .into_iter()
                .filter(|id| old.contains_key(id) && new.contains_key(id))
                .collect(),
            removed: old
                .keys()
                .filter(|id| !new.contains_key(*id))
                .cloned()
                .collect(),
            credentials: changed(&credentials(before), &credentials(now)),
            settings,
        }
    }

    fn is_empty(&self) -> bool {
        *self == Self::default()
    }

    fn log(&self) {
        tracing::info!(
            added = self.added.join(", "),
            updated = self.updated.join(", "),
            removed = self.removed.join(", "),
            credentials = self.credentials.join(", "),
            settings = self.settings.join(", "),
            "the config the proxy stores in its database changed"
        );
    }
}

/// A stored model compared by everything the config is built from.
struct Model(DbModel);

impl PartialEq for Model {
    fn eq(&self, other: &Self) -> bool {
        let (this, that) = (&self.0, &other.0);
        this.model_name == that.model_name
            && *this.litellm_params == *that.litellm_params
            && *this.model_info == *that.model_info
            && this.blocked == that.blocked
    }
}

/// The keys `before` and `after` hold different values under, one of them none.
fn changed<V: PartialEq>(before: &BTreeMap<String, V>, after: &BTreeMap<String, V>) -> Vec<String> {
    let keys: BTreeSet<&String> = before.keys().chain(after.keys()).collect();
    keys.into_iter()
        .filter(|key| before.get(*key) != after.get(*key))
        .cloned()
        .collect()
}

#[cfg(test)]
mod tests {
    use std::{convert::Infallible, io::Write};

    use axum::{body::Body, extract::Request, response::Response};
    use litellm_config::Environment;
    use rstest::rstest;

    use super::*;
    use crate::{
        Admit, AppState,
        stored::layered,
        test_support::{Rows, credential, environment, model, secrets, setting},
    };

    const FILE: &str = "model_list:\n  - model_name: file\n    litellm_params:\n      model: anthropic/claude-haiku-5-5\n      api_key: file-key\ngeneral_settings:\n  master_key: gateway-key\n  store_model_in_db: true\n  supported_db_objects: [models]\n";

    fn params(api_key: &str) -> String {
        format!(r#"{{"model": "anthropic/claude-haiku-5-5", "api_key": "{api_key}"}}"#)
    }

    /// The stored rows: `models` by name with the api key each is called with, a stored variable
    /// and a credential.
    fn stored(models: &[(&str, &str)], variable: &str, credential_key: &str) -> Rows {
        Rows {
            settings: Some(vec![setting(
                "environment_variables",
                &format!(r#"{{"STORED_VARIABLE": "{variable}"}}"#),
            )]),
            models: Some(
                models
                    .iter()
                    .map(|(name, api_key)| model(name, &params(api_key)))
                    .collect(),
            ),
            credentials: Some(vec![credential(
                "shared",
                &format!(r#"{{"api_key": "{credential_key}"}}"#),
            )]),
        }
    }

    fn serving(models: &[(&str, &str)]) -> Rows {
        stored(models, "value", "credential-key")
    }

    fn native()
    -> impl tower::Service<Request, Response = Response, Error = Infallible, Future: Send + 'static>
    + Clone
    + Send
    + Sync
    + 'static {
        tower::service_fn(|_: Request| async { Ok::<_, Infallible>(Response::new(Body::empty())) })
    }

    async fn rebuild(
        config: Config,
        stored: StoredConfig,
        current: Arc<Served>,
    ) -> Result<Served, Error> {
        let environment = Environment::default();
        let state = current.state.next(&config, &environment).await?;
        let admit = Admit::new(&config, &environment, &state, stored);
        Ok(Served::new(state, Some(Arc::new(admit)), native()))
    }

    fn document() -> Document {
        let mut file = tempfile::Builder::new().suffix(".yaml").tempfile().unwrap();
        file.write_all(FILE.as_bytes()).unwrap();
        Document::read(file.path()).unwrap()
    }

    /// What the gateway serves once boot read `rows`, and the reading the poller goes on from.
    async fn booted(rows: &Rows) -> (Arc<Live>, Reading) {
        let document = document();
        let config = document.clone().resolve(secrets(&[])()).await.unwrap();
        let (loaded, served) = layered(
            document.clone(),
            config,
            &environment(&[]),
            rows,
            None,
            &secrets(&[]),
        )
        .await;
        let state = AppState::build(&loaded.config, &Environment::default())
            .await
            .unwrap();
        let admit = Admit::new(
            &loaded.config,
            &Environment::default(),
            &state,
            loaded.stored,
        );
        let live = Live::new(Served::new(state, Some(Arc::new(admit)), native()));
        let reading = Reading {
            document,
            key: None,
            secrets: Arc::new(secrets(&[])),
            served,
        };
        (Arc::new(live), reading)
    }

    /// Why `live` leaves a master key caller's `/v1/messages` request for `model` to Python.
    async fn refused_for(live: &Arc<Live>, model: &str) -> Option<&'static str> {
        let request = Request::post("/v1/messages")
            .header("authorization", "Bearer gateway-key")
            .body(Body::from(format!(
                r#"{{"model":"{model}","max_tokens":8,"messages":[]}}"#
            )))
            .unwrap();
        live.clone()
            .admitted(request)
            .await
            .err()
            .map(|(_, why)| why.into())
    }

    /// The api key the router calls the stored model `name` with.
    fn api_key(live: &Live, name: &str) -> Option<String> {
        live.current()
            .state
            .inference
            .router
            .registry()
            .deployment(&format!("{name}-id"))
            .and_then(|deployment| deployment.params.api_key.clone())
            .map(|key| key.expose().to_owned())
    }

    async fn fetched(rows: &Rows) -> Fetched {
        let settings = stored::settings_rows(&rows.settings.clone().unwrap(), None);
        stored::fetch(&document(), settings, rows, None)
            .await
            .unwrap()
    }

    fn changes(
        added: &[&str],
        updated: &[&str],
        removed: &[&str],
        credentials: &[&str],
        settings: &[&str],
    ) -> Changes {
        let owned = |names: &[&str]| names.iter().map(|name| (*name).to_owned()).collect();
        Changes {
            added: owned(added),
            updated: owned(updated),
            removed: owned(removed),
            credentials: owned(credentials),
            settings: owned(settings),
        }
    }

    #[rstest]
    #[case::the_same_rows(serving(&[("a", "key-1")]), Changes::default())]
    #[case::a_model_added(
        serving(&[("a", "key-1"), ("b", "key-1")]),
        changes(&["b-id"], &[], &[], &[], &[])
    )]
    #[case::a_model_key_rotated(serving(&[("a", "key-2")]), changes(&[], &["a-id"], &[], &[], &[]))]
    #[case::a_model_removed(serving(&[]), changes(&[], &[], &["a-id"], &[], &[]))]
    #[case::a_credential_rotated(
        stored(&[("a", "key-1")], "value", "rotated"),
        changes(&[], &[], &[], &["shared"], &[])
    )]
    #[case::a_stored_variable_changed(
        stored(&[("a", "key-1")], "changed", "credential-key"),
        changes(&[], &[], &[], &[], &["environment_variables.STORED_VARIABLE"])
    )]
    #[tokio::test]
    async fn a_read_names_what_changed_by_model_id_credential_and_settings_key(
        #[case] now: Rows,
        #[case] expected: Changes,
    ) {
        let served = fetched(&serving(&[("a", "key-1")])).await;

        let changed = Changes::between(Some(&served), &fetched(&now).await);

        assert_eq!(changed, expected);
    }

    #[rstest]
    #[tokio::test]
    async fn a_read_that_finds_the_rows_served_builds_nothing_and_swaps_nothing() {
        let rows = serving(&[("a", "key-1")]);
        let (live, mut reading) = booted(&rows).await;
        let before = live.current();

        let reloaded = reading.reload(&rows, &live, &rebuild).await;

        assert_eq!(reloaded, Reloaded::Unchanged);
        assert!(Arc::ptr_eq(&before, &live.current()));
    }

    #[rstest]
    #[case::a_model_added(
        serving(&[("a", "key-1"), ("b", "key-1")]),
        [None, None],
        [Some("key-1"), Some("key-1")]
    )]
    #[case::a_model_key_rotated(
        serving(&[("a", "key-2")]),
        [None, Some("model")],
        [Some("key-2"), None]
    )]
    #[case::a_model_removed(
        serving(&[]),
        [Some("model"), Some("model")],
        [None, None]
    )]
    #[tokio::test]
    async fn a_read_that_finds_changed_rows_swaps_to_a_config_built_from_them(
        #[case] now: Rows,
        #[case] refused: [Option<&str>; 2],
        #[case] keys: [Option<&str>; 2],
    ) {
        let (live, mut reading) = booted(&serving(&[("a", "key-1")])).await;

        let reloaded = reading.reload(&now, &live, &rebuild).await;

        assert_eq!(reloaded, Reloaded::Swapped);
        assert_eq!(
            [
                refused_for(&live, "a").await,
                refused_for(&live, "b").await,
                refused_for(&live, "file").await
            ],
            [refused[0], refused[1], None]
        );
        assert_eq!(
            [api_key(&live, "a"), api_key(&live, "b")],
            keys.map(|key| key.map(str::to_owned))
        );
    }

    #[rstest]
    #[case::the_models_cannot_be_read(Rows { models: None, ..serving(&[]) })]
    #[case::the_credentials_cannot_be_read(Rows { credentials: None, ..serving(&[]) })]
    #[tokio::test]
    async fn a_read_that_fails_keeps_serving_every_model(#[case] now: Rows) {
        let (live, mut reading) = booted(&serving(&[("a", "key-1")])).await;
        let before = live.current();

        let reloaded = reading.reload(&now, &live, &rebuild).await;

        assert_eq!(reloaded, Reloaded::Unread);
        assert!(Arc::ptr_eq(&before, &live.current()));
        assert_eq!(refused_for(&live, "a").await, None);
    }

    #[rstest]
    #[tokio::test]
    async fn a_read_whose_settings_cannot_be_read_adds_and_updates_models_but_removes_none() {
        let (live, mut reading) = booted(&serving(&[("a", "key-1"), ("b", "key-1")])).await;
        let now = Rows {
            settings: None,
            ..serving(&[("b", "key-2"), ("c", "key-1")])
        };

        let reloaded = reading.reload(&now, &live, &rebuild).await;

        assert_eq!(reloaded, Reloaded::Swapped);
        assert_eq!(
            [
                refused_for(&live, "a").await,
                refused_for(&live, "b").await,
                refused_for(&live, "c").await
            ],
            [None, None, None]
        );
        assert_eq!(api_key(&live, "b").as_deref(), Some("key-2"));
    }

    #[rstest]
    #[case::the_database_still_cannot_be_read(
        Rows::default(),
        Reloaded::Unread,
        Some("db_config_not_loaded")
    )]
    #[case::the_database_can_be_read_again(serving(&[("a", "key-1")]), Reloaded::Swapped, None)]
    #[case::the_database_holds_nothing(serving(&[]), Reloaded::Swapped, None)]
    #[tokio::test]
    async fn the_first_read_that_loads_lifts_db_config_not_loaded(
        #[case] now: Rows,
        #[case] expected: Reloaded,
        #[case] refused: Option<&str>,
    ) {
        let (live, mut reading) = booted(&Rows::default()).await;
        let before = refused_for(&live, "file").await;

        let reloaded = reading.reload(&now, &live, &rebuild).await;

        assert_eq!(
            (before, reloaded, refused_for(&live, "file").await),
            (Some("db_config_not_loaded"), expected, refused)
        );
    }

    #[rstest]
    #[case::unset("model_list: []\n", 30)]
    #[case::not_positive("general_settings:\n  proxy_config_reload_interval_seconds: 0\n", 30)]
    #[case::set("general_settings:\n  proxy_config_reload_interval_seconds: 5\n", 5)]
    fn the_stored_config_is_read_again_on_the_interval_the_config_sets(
        #[case] yaml: &str,
        #[case] every: u64,
    ) {
        let config = Config::from_yaml(yaml).unwrap();

        assert_eq!(reload_every(&config), Duration::from_secs(every));
    }

    #[rstest]
    #[tokio::test]
    async fn stopping_the_poller_ends_it_even_while_a_read_waits_on_the_database() {
        let config = Config::from_yaml(
            "general_settings:\n  database_url: postgresql://user:password@127.0.0.1:9/litellm\n",
        )
        .unwrap();
        let resources = crate::Resources::build(&config).unwrap();
        let (pools, database) = crate::db::connect(&config, &resources.secrets)
            .await
            .unwrap()
            .unwrap();
        let (live, reading) = booted(&Rows::default()).await;
        let before = live.current();
        let source = Source {
            reading,
            database,
            pools,
        };
        let poller = source.reload_on(Duration::from_millis(10), live.clone(), rebuild);
        tokio::time::sleep(Duration::from_millis(50)).await;

        let stopped = tokio::time::timeout(Duration::from_secs(5), poller.stop()).await;

        assert!(stopped.is_ok());
        assert!(Arc::ptr_eq(&before, &live.current()));
    }

    /// `rows` whose model read takes a while, counting how many reads are inside it at once.
    struct Slow {
        rows: Rows,
        inside: std::sync::atomic::AtomicUsize,
        most: std::sync::atomic::AtomicUsize,
    }

    impl StoredRows for Slow {
        async fn settings(
            &self,
        ) -> Result<Vec<litellm_repositories::ConfigParam>, litellm_db::Error> {
            self.rows.settings().await
        }

        async fn models(&self) -> Result<Vec<litellm_repositories::ProxyModel>, litellm_db::Error> {
            use std::sync::atomic::Ordering;
            let now = self.inside.fetch_add(1, Ordering::SeqCst) + 1;
            self.most.fetch_max(now, Ordering::SeqCst);
            tokio::time::sleep(Duration::from_millis(20)).await;
            self.inside.fetch_sub(1, Ordering::SeqCst);
            self.rows.models().await
        }

        async fn credentials(
            &self,
        ) -> Result<Vec<litellm_repositories::Credential>, litellm_db::Error> {
            self.rows.credentials().await
        }
    }

    #[rstest]
    #[tokio::test]
    async fn the_poll_and_a_redis_change_never_read_at_once_and_agree_on_what_is_served() {
        let (live, reading) = booted(&serving(&[("a", "key-1")])).await;
        let reloader = Reloader {
            reading: Mutex::new(reading),
            rows: Slow {
                rows: serving(&[("a", "key-2")]),
                inside: Default::default(),
                most: Default::default(),
            },
            live: live.clone(),
            rebuild,
        };

        let (polled, heard) = tokio::join!(reloader.reload(), reloader.reload());

        assert_eq!(
            reloader.rows.most.load(std::sync::atomic::Ordering::SeqCst),
            1
        );
        assert_eq!((polled, heard), (Reloaded::Swapped, Reloaded::Unchanged));
        assert_eq!(api_key(&live, "a").as_deref(), Some("key-2"));
    }
}
