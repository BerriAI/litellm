//! The model list a `store_model_in_db` proxy serves, built on the config document after its
//! `os.environ/` references are resolved.
//!
//! A stored row does not go through that walk. Python reads a reference in a row only where
//! `_resolve_db_litellm_param` does: a top-level string of `litellm_params`, unless the key holds a
//! secret pointer. `model_name`, `model_info` and anything nested stay as stored, since a team
//! admin can write `os.environ/` text there and the proxy must never read its own environment for
//! it. So a row is merged only as a [`ResolvedModel`], which [`DbModel::resolve`] makes with exactly
//! that scope, and stored credential values stay literal, as `decrypt_credentials` leaves them.
//!
//! It is what `ProxyConfig.add_deployment` leaves in Python's router after a reconcile. The file's
//! models are set first, then each `LiteLLM_ProxyModelTable` row is upserted by its `model_id`
//! (`_add_deployment`), which takes out a file model declaring that id and adds the row at the
//! end. A file model that declares no id keeps none here, since router-core gives it Python's
//! stable `deployment_id` hash when it builds the list.
//!
//! Python reads `litellm_credential_name` per call (`_stored_credential_values` in `utils.py`).
//! Here it is filled into `litellm_params` once, so the router only sees plain params: a value
//! the params already hold wins, and a credential row in the DB wins over the file's
//! `credential_list` entry of the same name (`CredentialAccessor.upsert_credentials`).

use std::collections::{BTreeMap, HashMap, HashSet};

use litellm_router_types::holds_secret_pointer;
use serde::Deserialize;
use serde_yaml_ng::Mapping;
use veil::Redact;

use crate::{Error, Object, Value};

const MODEL_LIST: &str = "model_list";
const CREDENTIAL_LIST: &str = "credential_list";
const CREDENTIAL_NAME: &str = "litellm_credential_name";
const ENVIRON_PREFIX: &str = "os.environ/";
/// `CredentialAccessor.get_credential_values` never hands this value to a call.
const WITHHELD_CREDENTIAL_VALUE: &str = "github_copilot_app_private_key";

/// One `LiteLLM_ProxyModelTable` row, its `litellm_params` already decrypted. A row whose
/// `litellm_params` is not an object is skipped by Python and must be skipped by the caller.
#[derive(Clone, Debug)]
pub struct DbModel {
    pub model_id: String,
    pub model_name: String,
    pub litellm_params: Object,
    pub model_info: Object,
    pub blocked: bool,
}

/// A stored row whose `litellm_params` references are read the way Python reads them, the only
/// form [`merge`] takes a row in.
#[derive(Clone, Debug)]
pub struct ResolvedModel(DbModel);

impl DbModel {
    /// The row with each top-level `litellm_params` string that starts with `os.environ/` replaced
    /// by what `read` returns for it, except under a key that holds a secret pointer. Nothing else
    /// in the row is read. `read` gets the whole reference, `os.environ/` included.
    pub async fn resolve<E, F, Fut>(self, mut read: F) -> Result<ResolvedModel, E>
    where
        F: FnMut(String) -> Fut,
        Fut: Future<Output = Result<Value, E>>,
    {
        let mut litellm_params = BTreeMap::new();
        for (key, value) in self.litellm_params.iter() {
            let value = match value {
                Value::String(text)
                    if text.starts_with(ENVIRON_PREFIX) && !holds_secret_pointer(key) =>
                {
                    read(text.clone()).await?
                }
                value => value.clone(),
            };
            litellm_params.insert(key.clone(), value);
        }
        Ok(ResolvedModel(DbModel {
            litellm_params: Object::from_fields(litellm_params),
            ..self
        }))
    }
}

/// One `LiteLLM_CredentialsTable` row, its `credential_values` already decrypted.
#[derive(Clone, Debug)]
pub struct DbCredential {
    pub credential_name: String,
    pub credential_values: Object,
}

/// A model naming a `litellm_credential_name` that neither source holds. Python warns and calls
/// the provider without it, and the model here keeps its params as written.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MissingCredential {
    pub model_name: String,
    pub credential_name: String,
}

/// The document with its final `model_list`, which holds decrypted credentials.
#[derive(Clone, Redact)]
pub struct Merged {
    #[redact(with = "[REDACTED]")]
    pub document: Value,
    pub missing_credentials: Box<[MissingCredential]>,
}

#[derive(Deserialize)]
struct FileCredential {
    credential_name: String,
    credential_values: Object,
}

/// `document` (the config file with its includes merged and its `os.environ/` references already
/// resolved) with `models` added to its `model_list` and every model's `litellm_credential_name`
/// filled from `credentials` and the document's `credential_list`.
pub fn merge(
    document: Value,
    models: impl IntoIterator<Item = ResolvedModel>,
    credentials: impl IntoIterator<Item = DbCredential>,
) -> Result<Merged, Error> {
    let mut document: Mapping = match document {
        Value::Null => Mapping::new(),
        document => serde_yaml_ng::from_value(document)?,
    };
    let credentials = credential_values(document.get(CREDENTIAL_LIST), credentials)?;
    let written: Vec<Value> = match document.get_mut(MODEL_LIST).map(std::mem::take) {
        None | Some(Value::Null) => Vec::new(),
        Some(list) => serde_yaml_ng::from_value(list)?,
    };
    let models: Vec<DbModel> = models
        .into_iter()
        .map(|ResolvedModel(model)| model)
        .collect();
    let db_ids: HashSet<String> = models.iter().map(|model| model.model_id.clone()).collect();
    let mut missing_credentials = Vec::new();
    let model_list = written
        .into_iter()
        .filter(|entry| declared_id(entry).is_none_or(|id| !db_ids.contains(&id)))
        .chain(models.into_iter().map(db_entry))
        .map(|entry| with_credential(entry, &credentials, &mut missing_credentials))
        .collect();
    document.insert(MODEL_LIST.into(), Value::Sequence(model_list));
    Ok(Merged {
        document: Value::Mapping(document),
        missing_credentials: missing_credentials.into(),
    })
}

/// Each credential's values by name: the first file entry of a name, replaced by a DB row of it.
fn credential_values(
    file: Option<&Value>,
    db: impl IntoIterator<Item = DbCredential>,
) -> Result<HashMap<String, Object>, Error> {
    let file: Vec<FileCredential> = match file {
        None | Some(Value::Null) => Vec::new(),
        Some(list) => serde_yaml_ng::from_value(list.clone())?,
    };
    let mut values = HashMap::new();
    for credential in file {
        values
            .entry(credential.credential_name)
            .or_insert(credential.credential_values);
    }
    values.extend(
        db.into_iter()
            .map(|credential| (credential.credential_name, credential.credential_values)),
    );
    Ok(values)
}

/// `model_info.id` as router-core keys a declared id, the key a DB row is upserted by.
fn declared_id(entry: &Value) -> Option<String> {
    match entry.get("model_info")?.get("id")? {
        Value::String(id) => Some(id.clone()),
        Value::Number(id) => Some(id.to_string()),
        _ => None,
    }
}

/// The row as `_add_deployment` upserts it: `model_info` from `get_model_info_with_id`, which
/// overrides the stored `id`, `db_model` and `blocked`.
fn db_entry(model: DbModel) -> Value {
    let mut model_info: Mapping = model
        .model_info
        .iter()
        .map(|(key, value)| (Value::from(key.as_str()), value.clone()))
        .collect();
    model_info.insert("id".into(), Value::String(model.model_id));
    model_info.insert("db_model".into(), Value::Bool(true));
    model_info.insert("blocked".into(), Value::Bool(model.blocked));
    let litellm_params: Mapping = model
        .litellm_params
        .iter()
        .map(|(key, value)| (Value::from(key.as_str()), value.clone()))
        .collect();
    Value::Mapping(Mapping::from_iter([
        ("model_name".into(), Value::String(model.model_name)),
        ("litellm_params".into(), Value::Mapping(litellm_params)),
        ("model_info".into(), Value::Mapping(model_info)),
    ]))
}

/// `entry` with the values of the credential it names added under the params it leaves unset.
fn with_credential(
    mut entry: Value,
    credentials: &HashMap<String, Object>,
    missing: &mut Vec<MissingCredential>,
) -> Value {
    let model_name = entry
        .get("model_name")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_owned();
    let Some(Value::Mapping(params)) = entry.get_mut("litellm_params") else {
        return entry;
    };
    let Some(name) = params
        .get(CREDENTIAL_NAME)
        .and_then(Value::as_str)
        .filter(|name| !name.is_empty())
        .map(str::to_owned)
    else {
        return entry;
    };
    let Some(values) = credentials.get(&name) else {
        missing.push(MissingCredential {
            model_name,
            credential_name: name,
        });
        return entry;
    };
    for (key, value) in values.iter() {
        if key == WITHHELD_CREDENTIAL_VALUE {
            continue;
        }
        let key = Value::from(key.as_str());
        if params.get(&key).is_none_or(Value::is_null) {
            params.insert(key, value.clone());
        }
    }
    entry
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    fn yaml(text: &str) -> Value {
        serde_yaml_ng::from_str(text).unwrap()
    }

    fn object(text: &str) -> Object {
        serde_yaml_ng::from_str(text).unwrap()
    }

    fn db_model(model_id: &str, model_name: &str, params: &str, model_info: &str) -> DbModel {
        DbModel {
            model_id: model_id.to_owned(),
            model_name: model_name.to_owned(),
            litellm_params: object(params),
            model_info: object(model_info),
            blocked: false,
        }
    }

    fn db_credential(name: &str, values: &str) -> DbCredential {
        DbCredential {
            credential_name: name.to_owned(),
            credential_values: object(values),
        }
    }

    /// What `read` hands back for each reference, as the config-load walk reads it from a test
    /// environment holding `LITELLM_MASTER_KEY` and `B_KEY`.
    async fn from_environment(reference: String) -> Result<Value, std::convert::Infallible> {
        Ok(match reference.strip_prefix(ENVIRON_PREFIX) {
            Some("LITELLM_MASTER_KEY") => Value::from("sk-master"),
            Some("B_KEY") => Value::from("b-from-env"),
            _ => Value::Null,
        })
    }

    /// Polls a future that never waits, which every `read` in these tests is.
    fn now<T>(future: impl Future<Output = T>) -> T {
        let mut future = std::pin::pin!(future);
        let mut context = std::task::Context::from_waker(std::task::Waker::noop());
        match future.as_mut().poll(&mut context) {
            std::task::Poll::Ready(value) => value,
            std::task::Poll::Pending => panic!("a test read never waits"),
        }
    }

    fn resolved(model: DbModel) -> ResolvedModel {
        let Ok(model) = now(model.resolve(from_environment));
        model
    }

    fn model_list(document: &str, models: Vec<DbModel>, credentials: Vec<DbCredential>) -> Value {
        merge(
            yaml(document),
            models.into_iter().map(resolved),
            credentials,
        )
        .unwrap()
        .document[MODEL_LIST]
            .clone()
    }

    #[rstest]
    #[case::file_only(
        "model_list:\n- model_name: a\n  litellm_params: {model: openai/a}\n  model_info: {id: file-a}\n",
        vec![],
        "- model_name: a\n  litellm_params: {model: openai/a}\n  model_info: {id: file-a}\n"
    )]
    #[case::db_only(
        "general_settings: {store_model_in_db: true}\n",
        vec![db_model("db-1", "b", "{model: openai/b, api_key: os.environ/B_KEY}", "{}")],
        "- model_name: b\n  litellm_params: {model: openai/b, api_key: b-from-env}\n  model_info: {id: db-1, db_model: true, blocked: false}\n"
    )]
    #[case::db_models_follow_file_models(
        "model_list:\n- model_name: a\n  litellm_params: {model: openai/a}\n",
        vec![
            db_model("db-1", "b", "{model: openai/b}", "{}"),
            db_model("db-2", "a", "{model: azure/a}", "{}"),
        ],
        "- model_name: a\n  litellm_params: {model: openai/a}\n- model_name: b\n  litellm_params: {model: openai/b}\n  model_info: {id: db-1, db_model: true, blocked: false}\n- model_name: a\n  litellm_params: {model: azure/a}\n  model_info: {id: db-2, db_model: true, blocked: false}\n"
    )]
    #[case::id_less_file_models_stay_id_less_for_router_core_to_hash(
        "model_list:\n- model_name: a\n  litellm_params: {model: openai/a}\n- model_name: a\n  litellm_params: {model: openai/a}\n",
        vec![db_model("db-1", "a", "{model: openai/a}", "{}")],
        "- model_name: a\n  litellm_params: {model: openai/a}\n- model_name: a\n  litellm_params: {model: openai/a}\n- model_name: a\n  litellm_params: {model: openai/a}\n  model_info: {id: db-1, db_model: true, blocked: false}\n"
    )]
    #[case::a_db_row_replaces_the_file_model_declaring_its_id(
        "model_list:\n- model_name: a\n  litellm_params: {model: openai/a}\n  model_info: {id: shared}\n- model_name: c\n  litellm_params: {model: openai/c}\n",
        vec![db_model("shared", "a", "{model: openai/a-from-db}", "{}")],
        "- model_name: c\n  litellm_params: {model: openai/c}\n- model_name: a\n  litellm_params: {model: openai/a-from-db}\n  model_info: {id: shared, db_model: true, blocked: false}\n"
    )]
    #[case::a_numeric_file_id_is_matched_as_its_text(
        "model_list:\n- model_name: a\n  litellm_params: {model: openai/a}\n  model_info: {id: 7}\n",
        vec![db_model("7", "a", "{model: openai/a-from-db}", "{}")],
        "- model_name: a\n  litellm_params: {model: openai/a-from-db}\n  model_info: {id: '7', db_model: true, blocked: false}\n"
    )]
    #[case::the_row_id_overrides_the_stored_model_info(
        "{}",
        vec![db_model("db-1", "b", "{model: openai/b}", "{id: stale, db_model: false, team_id: t1}")],
        "- model_name: b\n  litellm_params: {model: openai/b}\n  model_info: {id: db-1, db_model: true, team_id: t1, blocked: false}\n"
    )]
    fn models_are_merged_as_the_python_router_holds_them(
        #[case] document: &str,
        #[case] models: Vec<DbModel>,
        #[case] expected: &str,
    ) {
        assert_eq!(model_list(document, models, vec![]), yaml(expected));
    }

    #[rstest]
    fn a_blocked_row_stays_blocked() {
        let model = DbModel {
            blocked: true,
            ..db_model("db-1", "b", "{model: openai/b}", "{}")
        };
        assert_eq!(
            model_list("{}", vec![model], vec![])[0]["model_info"]["blocked"],
            Value::Bool(true)
        );
    }

    #[rstest]
    fn the_merged_document_reads_as_a_config() {
        let merged = merge(
            yaml(NAMED),
            vec![resolved(db_model("db-1", "b", "{model: openai/b}", "{}"))],
            vec![db_credential("shared", "{api_key: db-key}")],
        )
        .unwrap();
        let config = crate::Config::from_value(merged.document).unwrap();
        assert_eq!(config.model_list.len(), 2);
        assert_eq!(
            config.model_list[0]
                .litellm_params
                .api_key
                .as_ref()
                .map(|key| key.expose()),
            Some("db-key")
        );
        assert_eq!(
            config.model_list[1].model_info.get("db_model"),
            Some(&Value::Bool(true))
        );
    }

    #[rstest]
    fn other_sections_are_kept() {
        let merged = merge(
            yaml("litellm_settings: {drop_params: true}\nmodel_list: []\n"),
            vec![],
            vec![],
        )
        .unwrap();
        assert_eq!(
            merged.document,
            yaml("litellm_settings: {drop_params: true}\nmodel_list: []\n")
        );
    }

    const NAMED: &str = "model_list:\n- model_name: a\n  litellm_params: {model: openai/a, litellm_credential_name: shared}\n";

    #[rstest]
    #[case::from_the_file(
        "credential_list:\n- credential_name: shared\n  credential_values: {api_key: os.environ/FILE_KEY, api_base: https://file.test}\n  credential_info: {}\n",
        vec![],
        "{model: openai/a, litellm_credential_name: shared, api_key: os.environ/FILE_KEY, api_base: https://file.test}"
    )]
    #[case::from_the_db(
        "",
        vec![db_credential("shared", "{api_key: db-key}")],
        "{model: openai/a, litellm_credential_name: shared, api_key: db-key}"
    )]
    #[case::the_db_row_wins_over_the_file_entry(
        "credential_list:\n- credential_name: shared\n  credential_values: {api_key: file-key, api_base: https://file.test}\n",
        vec![db_credential("shared", "{api_key: db-key}")],
        "{model: openai/a, litellm_credential_name: shared, api_key: db-key}"
    )]
    #[case::the_first_file_entry_of_a_name_wins(
        "credential_list:\n- credential_name: shared\n  credential_values: {api_key: first}\n- credential_name: shared\n  credential_values: {api_key: second}\n",
        vec![],
        "{model: openai/a, litellm_credential_name: shared, api_key: first}"
    )]
    #[case::the_withheld_app_key_is_not_filled(
        "",
        vec![db_credential("shared", "{github_copilot_app_private_key: pem, api_key: db-key}")],
        "{model: openai/a, litellm_credential_name: shared, api_key: db-key}"
    )]
    fn a_named_credential_fills_the_params(
        #[case] file: &str,
        #[case] credentials: Vec<DbCredential>,
        #[case] expected: &str,
    ) {
        assert_eq!(
            model_list(&format!("{NAMED}{file}"), vec![], credentials)[0]["litellm_params"],
            yaml(expected)
        );
    }

    #[rstest]
    #[case::set(
        "{model: openai/a, litellm_credential_name: shared, api_key: own}",
        "{model: openai/a, litellm_credential_name: shared, api_key: own, api_base: https://db.test}"
    )]
    #[case::null(
        "{model: openai/a, litellm_credential_name: shared, api_key: null}",
        "{model: openai/a, litellm_credential_name: shared, api_key: db-key, api_base: https://db.test}"
    )]
    #[case::empty_name(
        "{model: openai/a, litellm_credential_name: ''}",
        "{model: openai/a, litellm_credential_name: ''}"
    )]
    fn the_params_a_model_sets_win_over_its_credential(
        #[case] params: &str,
        #[case] expected: &str,
    ) {
        let models = vec![db_model("db-1", "a", params, "{}")];
        let credentials = vec![db_credential(
            "shared",
            "{api_key: db-key, api_base: https://db.test}",
        )];
        assert_eq!(
            model_list("{}", models, credentials)[0]["litellm_params"],
            yaml(expected)
        );
    }

    #[rstest]
    fn a_missing_credential_is_reported_and_leaves_the_params_as_written() {
        let merged = merge(
            yaml(NAMED),
            vec![],
            vec![db_credential("other", "{api_key: k}")],
        )
        .unwrap();
        assert_eq!(
            merged.document[MODEL_LIST][0]["litellm_params"],
            yaml("{model: openai/a, litellm_credential_name: shared}")
        );
        assert_eq!(
            &*merged.missing_credentials,
            [MissingCredential {
                model_name: "a".to_owned(),
                credential_name: "shared".to_owned(),
            }]
        );
    }

    #[derive(Deserialize)]
    struct PythonReconcile {
        input: PythonInput,
        expected: serde_json::Value,
    }

    #[derive(Deserialize)]
    struct PythonInput {
        file: Value,
        db_models: Vec<PythonModelRow>,
        db_credentials: Vec<PythonCredentialRow>,
    }

    #[derive(Deserialize)]
    struct PythonModelRow {
        model_id: String,
        model_name: String,
        litellm_params: Object,
        model_info: Object,
        blocked: bool,
    }

    #[derive(Deserialize)]
    struct PythonCredentialRow {
        credential_name: String,
        credential_values: Object,
    }

    /// An entry in the shape the fixture records Python's router deployments in.
    fn as_recorded(entry: &Value) -> serde_json::Value {
        let info = &entry["model_info"];
        serde_json::json!({
            "model_name": entry["model_name"],
            "id": info.get("id"),
            "db_model": info.get("db_model").and_then(Value::as_bool).unwrap_or(false),
            "blocked": info.get("blocked").and_then(Value::as_bool).unwrap_or(false),
            "litellm_params": entry["litellm_params"],
        })
    }

    #[rstest]
    fn the_list_matches_what_python_reconciles_into_its_router() {
        let fixture: PythonReconcile =
            serde_json::from_str(include_str!("../../fixtures/db_models_python.json")).unwrap();
        let models = fixture.input.db_models.into_iter().map(|row| DbModel {
            model_id: row.model_id,
            model_name: row.model_name,
            litellm_params: row.litellm_params,
            model_info: row.model_info,
            blocked: row.blocked,
        });
        let credentials = fixture
            .input
            .db_credentials
            .into_iter()
            .map(|row| DbCredential {
                credential_name: row.credential_name,
                credential_values: row.credential_values,
            });
        let merged = merge(fixture.input.file, models.map(resolved), credentials).unwrap();
        let recorded: Vec<serde_json::Value> = merged.document[MODEL_LIST]
            .as_sequence()
            .unwrap()
            .iter()
            .map(as_recorded)
            .collect();
        assert_eq!(serde_json::Value::from(recorded), fixture.expected);
    }

    #[rstest]
    #[case::top_level_param(
        "os.environ/LITELLM_MASTER_KEY",
        "{model: openai/a, api_key: os.environ/B_KEY}",
        "{}",
        "- model_name: os.environ/LITELLM_MASTER_KEY\n  litellm_params: {model: openai/a, api_key: b-from-env}\n  model_info: {id: db-1, db_model: true, blocked: false}\n"
    )]
    #[case::unset_top_level_param_is_null(
        "a",
        "{model: openai/a, api_base: os.environ/UNSET}",
        "{}",
        "- model_name: a\n  litellm_params: {model: openai/a, api_base: null}\n  model_info: {id: db-1, db_model: true, blocked: false}\n"
    )]
    #[case::model_info_and_nested_params_stay_literal(
        "a",
        "{model: openai/a, litellm_metadata: {nested: os.environ/LITELLM_MASTER_KEY}, tags: [os.environ/LITELLM_MASTER_KEY]}",
        "{team_public_model_name: os.environ/LITELLM_MASTER_KEY, nested: {value: os.environ/LITELLM_MASTER_KEY}}",
        "- model_name: a\n  litellm_params: {model: openai/a, litellm_metadata: {nested: os.environ/LITELLM_MASTER_KEY}, tags: [os.environ/LITELLM_MASTER_KEY]}\n  model_info: {team_public_model_name: os.environ/LITELLM_MASTER_KEY, nested: {value: os.environ/LITELLM_MASTER_KEY}, id: db-1, db_model: true, blocked: false}\n"
    )]
    #[case::secret_pointer_stays_literal(
        "a",
        "{model: openai/a, anthropic_issuer_signing_key_ref: os.environ/LITELLM_MASTER_KEY}",
        "{}",
        "- model_name: a\n  litellm_params: {model: openai/a, anthropic_issuer_signing_key_ref: os.environ/LITELLM_MASTER_KEY}\n  model_info: {id: db-1, db_model: true, blocked: false}\n"
    )]
    fn a_stored_row_reads_only_its_top_level_params_from_the_environment(
        #[case] model_name: &str,
        #[case] params: &str,
        #[case] model_info: &str,
        #[case] expected: &str,
    ) {
        let models = vec![db_model("db-1", model_name, params, model_info)];
        assert_eq!(model_list("{}", models, vec![]), yaml(expected));
    }

    #[rstest]
    fn a_stored_credential_value_stays_literal() {
        let models = vec![db_model(
            "db-1",
            "a",
            "{model: openai/a, litellm_credential_name: shared}",
            "{}",
        )];
        let credentials = vec![db_credential(
            "shared",
            "{api_key: os.environ/LITELLM_MASTER_KEY}",
        )];
        assert_eq!(
            model_list("{}", models, credentials)[0]["litellm_params"],
            yaml(
                "{model: openai/a, litellm_credential_name: shared, api_key: os.environ/LITELLM_MASTER_KEY}"
            )
        );
    }

    #[derive(Deserialize)]
    struct PythonEnvironRead {
        environment: HashMap<String, String>,
        row: PythonModelRow,
        router_holds: PythonDeployment,
    }

    #[derive(Deserialize)]
    struct PythonDeployment {
        model_name: String,
        litellm_params: Object,
        model_info: Object,
    }

    #[rstest]
    fn a_stored_row_reads_the_environment_where_python_does() {
        let fixture: PythonEnvironRead =
            serde_json::from_str(include_str!("../../fixtures/db_model_environ_python.json"))
                .unwrap();
        let environment = fixture.environment;
        let read = |reference: String| {
            let value = reference
                .strip_prefix(ENVIRON_PREFIX)
                .and_then(|name| environment.get(name))
                .map_or(Value::Null, |value| Value::from(value.as_str()));
            std::future::ready(Ok::<_, std::convert::Infallible>(value))
        };
        let row = fixture.row;
        let model = DbModel {
            model_id: row.model_id,
            model_name: row.model_name,
            litellm_params: row.litellm_params,
            model_info: row.model_info,
            blocked: row.blocked,
        };
        let Ok(model) = now(model.resolve(read));

        let merged = merge(yaml("{}"), [model], vec![]).unwrap();

        let entry = &merged.document[MODEL_LIST][0];
        assert_eq!(
            entry["model_name"],
            Value::from(fixture.router_holds.model_name.as_str())
        );
        let held = |section: &str, expected: &Object| -> Vec<(String, Value, Value)> {
            expected
                .iter()
                .map(|(key, value)| {
                    (
                        key.clone(),
                        entry[section][key.as_str()].clone(),
                        value.clone(),
                    )
                })
                .filter(|(_, ours, python)| ours != python)
                .collect()
        };
        assert_eq!(
            held("litellm_params", &fixture.router_holds.litellm_params),
            vec![]
        );
        assert_eq!(held("model_info", &fixture.router_holds.model_info), vec![]);
    }

    #[rstest]
    #[case::credential_without_values("credential_list:\n- credential_name: shared\n")]
    #[case::model_list_not_a_list("model_list: {model_name: a}\n")]
    fn a_malformed_section_is_an_error(#[case] document: &str) {
        assert!(matches!(
            merge(yaml(document), vec![], vec![]),
            Err(Error::Parse(_))
        ));
    }
}
