use std::{
    collections::HashMap,
    fs::{self, OpenOptions},
    io::Write,
    path::{Path, PathBuf},
    sync::Mutex,
};

use litellm_cache::Error;
use rand::RngCore;
use rusqlite::{Connection, OptionalExtension, params, types::Value};

use crate::{DiskStore, StoredValue};

const MODE_RAW: i64 = 1;
const MODE_BINARY: i64 = 2;
const MODE_TEXT: i64 = 3;
const MODE_PICKLE: i64 = 4;

const DEFAULT_DISK_MIN_FILE_SIZE: i64 = 2_i64.pow(15);
const DEFAULT_SIZE_LIMIT: i64 = 2_i64.pow(30);
const DEFAULT_CULL_LIMIT: i64 = 10;

pub struct DiskcacheSqliteStore {
    directory: PathBuf,
    connection: Mutex<Connection>,
    min_file_size: usize,
    eviction_policy: String,
    size_limit: i64,
    cull_limit: i64,
    statistics: bool,
}

struct StoredColumns {
    size: i64,
    mode: i64,
    filename: Option<String>,
    value: Option<Value>,
}

struct Row {
    rowid: i64,
    mode: i64,
    filename: Option<String>,
    value: Value,
}

impl DiskcacheSqliteStore {
    pub fn open(directory: impl AsRef<Path>) -> Result<Self, Error> {
        let directory = directory.as_ref().to_path_buf();
        fs::create_dir_all(&directory).map_err(|_| Error::Unavailable)?;
        let directory = std::path::absolute(&directory).map_err(|_| Error::Unavailable)?;
        let database = directory.join("cache.db");
        let connection = Connection::open(database).map_err(|_| Error::Unavailable)?;
        connection
            .busy_timeout(std::time::Duration::from_secs(60))
            .map_err(|_| Error::Unavailable)?;

        let mut settings = read_settings(&connection)?;
        for (key, value) in default_settings() {
            settings.entry(key).or_insert(value);
        }
        for (key, value) in settings
            .iter()
            .filter(|(key, _)| key.starts_with("sqlite_"))
        {
            apply_pragma(&connection, key, value)?;
        }

        connection
            .execute_batch(
                "CREATE TABLE IF NOT EXISTS Settings (
                    key TEXT NOT NULL UNIQUE,
                    value
                )",
            )
            .map_err(|_| Error::Unavailable)?;
        for (key, value) in &settings {
            if !matches!(key.as_str(), "count" | "size" | "hits" | "misses") {
                connection
                    .execute(
                        "INSERT OR REPLACE INTO Settings VALUES (?, ?)",
                        params![key, value],
                    )
                    .map_err(|_| Error::Unavailable)?;
            }
        }
        for (key, value) in [
            ("count", Value::Integer(0)),
            ("size", Value::Integer(0)),
            ("hits", Value::Integer(0)),
            ("misses", Value::Integer(0)),
        ] {
            connection
                .execute(
                    "INSERT OR IGNORE INTO Settings VALUES (?, ?)",
                    params![key, value],
                )
                .map_err(|_| Error::Unavailable)?;
        }
        connection
            .execute_batch(
                "CREATE TABLE IF NOT EXISTS Cache (
                    rowid INTEGER PRIMARY KEY,
                    key BLOB,
                    raw INTEGER,
                    store_time REAL,
                    expire_time REAL,
                    access_time REAL,
                    access_count INTEGER DEFAULT 0,
                    tag BLOB,
                    size INTEGER DEFAULT 0,
                    mode INTEGER DEFAULT 0,
                    filename TEXT,
                    value BLOB
                );
                CREATE UNIQUE INDEX IF NOT EXISTS Cache_key_raw ON Cache(key, raw);
                CREATE INDEX IF NOT EXISTS Cache_expire_time ON Cache(expire_time);",
            )
            .map_err(|_| Error::Unavailable)?;

        let eviction_policy = setting_string(&settings, "eviction_policy")
            .unwrap_or_else(|| "least-recently-stored".to_string());
        match eviction_policy.as_str() {
            "none" => {}
            "least-recently-stored" => {
                connection
                    .execute_batch(
                        "CREATE INDEX IF NOT EXISTS Cache_store_time ON Cache(store_time)",
                    )
                    .map_err(|_| Error::Unavailable)?;
            }
            "least-recently-used" => {
                connection
                    .execute_batch(
                        "CREATE INDEX IF NOT EXISTS Cache_access_time ON Cache(access_time)",
                    )
                    .map_err(|_| Error::Unavailable)?;
            }
            "least-frequently-used" => {
                connection
                    .execute_batch(
                        "CREATE INDEX IF NOT EXISTS Cache_access_count ON Cache(access_count)",
                    )
                    .map_err(|_| Error::Unavailable)?;
            }
            _ => return Err(Error::Unavailable),
        }
        connection
            .execute_batch(
                "CREATE TRIGGER IF NOT EXISTS Settings_count_insert
                    AFTER INSERT ON Cache FOR EACH ROW BEGIN
                    UPDATE Settings SET value = value + 1
                    WHERE key = \"count\"; END;
                 CREATE TRIGGER IF NOT EXISTS Settings_count_delete
                    AFTER DELETE ON Cache FOR EACH ROW BEGIN
                    UPDATE Settings SET value = value - 1
                    WHERE key = \"count\"; END;
                 CREATE TRIGGER IF NOT EXISTS Settings_size_insert
                    AFTER INSERT ON Cache FOR EACH ROW BEGIN
                    UPDATE Settings SET value = value + NEW.size
                    WHERE key = \"size\"; END;
                 CREATE TRIGGER IF NOT EXISTS Settings_size_update
                    AFTER UPDATE ON Cache FOR EACH ROW BEGIN
                    UPDATE Settings
                    SET value = value + NEW.size - OLD.size
                    WHERE key = \"size\"; END;
                 CREATE TRIGGER IF NOT EXISTS Settings_size_delete
                    AFTER DELETE ON Cache FOR EACH ROW BEGIN
                    UPDATE Settings SET value = value - OLD.size
                    WHERE key = \"size\"; END;",
            )
            .map_err(|_| Error::Unavailable)?;

        let min_file_size = setting_i64(&settings, "disk_min_file_size")
            .unwrap_or(DEFAULT_DISK_MIN_FILE_SIZE)
            .try_into()
            .map_err(|_| Error::Unavailable)?;
        let size_limit = setting_i64(&settings, "size_limit").unwrap_or(DEFAULT_SIZE_LIMIT);
        let cull_limit = setting_i64(&settings, "cull_limit").unwrap_or(DEFAULT_CULL_LIMIT);
        let statistics = setting_i64(&settings, "statistics").unwrap_or_default() != 0;

        Ok(Self {
            directory,
            connection: Mutex::new(connection),
            min_file_size,
            eviction_policy,
            size_limit,
            cull_limit,
            statistics,
        })
    }

    fn set_locked(
        &self,
        connection: &Connection,
        key: &str,
        columns: StoredColumns,
        expire_time: Option<f64>,
        now: f64,
    ) -> Result<Vec<String>, Error> {
        let mut cleanup = Vec::new();
        if let Some(old_filename) = connection
            .query_row(
                "SELECT filename FROM Cache WHERE key = ? AND raw = 1",
                params![key],
                |row| row.get::<_, Option<String>>(0),
            )
            .optional()
            .map_err(|_| Error::Unavailable)?
            .flatten()
        {
            cleanup.push(old_filename);
        }
        let (size, mode, filename, value) =
            (columns.size, columns.mode, columns.filename, columns.value);
        let rowid = connection
            .query_row(
                "SELECT rowid FROM Cache WHERE key = ? AND raw = 1",
                params![key],
                |row| row.get::<_, i64>(0),
            )
            .optional()
            .map_err(|_| Error::Unavailable)?;
        if let Some(rowid) = rowid {
            connection
                .execute(
                    "UPDATE Cache SET store_time = ?, expire_time = ?, access_time = ?,
                        access_count = 0, tag = NULL, size = ?, mode = ?, filename = ?, value = ?
                        WHERE rowid = ?",
                    params![now, expire_time, now, size, mode, filename, value, rowid],
                )
                .map_err(|_| Error::Unavailable)?;
        } else {
            connection
                .execute(
                    "INSERT INTO Cache(
                        key, raw, store_time, expire_time, access_time, access_count,
                        tag, size, mode, filename, value
                    ) VALUES (?, 1, ?, ?, ?, 0, NULL, ?, ?, ?, ?)",
                    params![key, now, expire_time, now, size, mode, filename, value],
                )
                .map_err(|_| Error::Unavailable)?;
        }
        cleanup.extend(self.cull(connection, now)?);
        Ok(cleanup)
    }

    fn cull(&self, connection: &Connection, now: f64) -> Result<Vec<String>, Error> {
        if self.cull_limit <= 0 {
            return Ok(Vec::new());
        }
        let mut cleanup = Vec::new();
        let expired = connection
            .prepare(
                "SELECT rowid, filename FROM Cache
                 WHERE expire_time IS NOT NULL AND expire_time < ?
                 ORDER BY expire_time LIMIT ?",
            )
            .map_err(|_| Error::Unavailable)?
            .query_map(params![now, self.cull_limit], |row| {
                Ok((row.get::<_, i64>(0)?, row.get::<_, Option<String>>(1)?))
            })
            .map_err(|_| Error::Unavailable)?
            .collect::<Result<Vec<_>, _>>()
            .map_err(|_| Error::Unavailable)?;
        for (_, filename) in &expired {
            if let Some(filename) = filename {
                cleanup.push(filename.clone());
            }
        }
        for (rowid, _) in &expired {
            connection
                .execute("DELETE FROM Cache WHERE rowid = ?", params![rowid])
                .map_err(|_| Error::Unavailable)?;
        }
        let remaining = self.cull_limit - i64::try_from(expired.len()).unwrap_or(self.cull_limit);
        if remaining <= 0 || self.volume(connection)? < self.size_limit {
            return Ok(cleanup);
        }
        let order = match self.eviction_policy.as_str() {
            "none" => return Ok(cleanup),
            "least-recently-stored" => "store_time",
            "least-recently-used" => "access_time",
            "least-frequently-used" => "access_count",
            _ => return Err(Error::Unavailable),
        };
        let rows = connection
            .prepare(&format!(
                "SELECT rowid, filename FROM Cache ORDER BY {order} LIMIT ?"
            ))
            .map_err(|_| Error::Unavailable)?
            .query_map(params![remaining], |row| {
                Ok((row.get::<_, i64>(0)?, row.get::<_, Option<String>>(1)?))
            })
            .map_err(|_| Error::Unavailable)?
            .collect::<Result<Vec<_>, _>>()
            .map_err(|_| Error::Unavailable)?;
        for (_, filename) in &rows {
            if let Some(filename) = filename {
                cleanup.push(filename.clone());
            }
        }
        for (rowid, _) in rows {
            connection
                .execute("DELETE FROM Cache WHERE rowid = ?", params![rowid])
                .map_err(|_| Error::Unavailable)?;
        }
        Ok(cleanup)
    }

    fn volume(&self, connection: &Connection) -> Result<i64, Error> {
        let page_count: i64 = connection
            .query_row("PRAGMA page_count", [], |row| row.get(0))
            .map_err(|_| Error::Unavailable)?;
        let page_size: i64 = connection
            .query_row("PRAGMA page_size", [], |row| row.get(0))
            .map_err(|_| Error::Unavailable)?;
        let size: i64 = connection
            .query_row("SELECT value FROM Settings WHERE key = 'size'", [], |row| {
                row.get(0)
            })
            .map_err(|_| Error::Unavailable)?;
        Ok(page_count.saturating_mul(page_size).saturating_add(size))
    }
}

impl DiskStore for DiskcacheSqliteStore {
    fn directory(&self) -> &Path {
        &self.directory
    }

    fn get(&self, key: &str, now: f64) -> Result<Option<StoredValue>, Error> {
        let connection = self.connection.lock().map_err(|_| Error::Unavailable)?;
        let select = "SELECT rowid, expire_time, mode, filename, value FROM Cache
            WHERE key = ? AND raw = 1 AND (expire_time IS NULL OR expire_time > ?)";
        let row = connection
            .query_row(select, params![key, now], row_from_query)
            .optional()
            .map_err(|_| Error::Unavailable)?;
        if !self.statistics && !has_get_update(&self.eviction_policy) {
            return row
                .map(|row| fetch_row(&self.directory, row))
                .transpose()
                .map(|value| value.flatten());
        }
        transactional(&connection, |connection| {
            let row = connection
                .query_row(select, params![key, now], row_from_query)
                .optional()
                .map_err(|_| Error::Unavailable)?;
            let Some(row) = row else {
                if self.statistics {
                    connection
                        .execute(
                            "UPDATE Settings SET value = value + 1 WHERE key = 'misses'",
                            [],
                        )
                        .map_err(|_| Error::Unavailable)?;
                }
                return Ok(None);
            };
            let rowid = row.rowid;
            let value = fetch_row(&self.directory, row);
            let hit = value.as_ref().is_ok_and(Option::is_some);
            if hit && self.statistics {
                connection
                    .execute(
                        "UPDATE Settings SET value = value + 1 WHERE key = 'hits'",
                        [],
                    )
                    .map_err(|_| Error::Unavailable)?;
            } else if !hit && self.statistics {
                connection
                    .execute(
                        "UPDATE Settings SET value = value + 1 WHERE key = 'misses'",
                        [],
                    )
                    .map_err(|_| Error::Unavailable)?;
            }
            if has_get_update(&self.eviction_policy) && hit {
                let update = match self.eviction_policy.as_str() {
                    "least-recently-used" => "UPDATE Cache SET access_time = ? WHERE rowid = ?",
                    "least-frequently-used" => {
                        "UPDATE Cache SET access_count = access_count + 1 WHERE rowid = ?"
                    }
                    _ => return Err(Error::Unavailable),
                };
                if self.eviction_policy == "least-recently-used" {
                    connection
                        .execute(update, params![now, rowid])
                        .map_err(|_| Error::Unavailable)?;
                } else {
                    connection
                        .execute(update, params![rowid])
                        .map_err(|_| Error::Unavailable)?;
                }
            }
            value
        })
    }

    fn set(
        &self,
        key: &str,
        value: StoredValue,
        expire_time: Option<f64>,
        now: f64,
    ) -> Result<(), Error> {
        let columns = store_value(&self.directory, self.min_file_size, value)?;
        let new_filename = columns.filename.clone();
        let connection = self.connection.lock().map_err(|_| Error::Unavailable)?;
        let result = transactional(&connection, |connection| {
            self.set_locked(connection, key, columns, expire_time, now)
        });
        match result {
            Ok(cleanup) => {
                cleanup_files(&self.directory, cleanup);
                Ok(())
            }
            Err(error) => {
                if let Some(filename) = new_filename {
                    remove_file(&self.directory, &filename);
                }
                Err(error)
            }
        }
    }

    fn pop(&self, key: &str, now: f64) -> Result<Option<StoredValue>, Error> {
        let connection = self.connection.lock().map_err(|_| Error::Unavailable)?;
        let selected = transactional(&connection, |connection| {
            let row = connection
                .query_row(
                    "SELECT rowid, expire_time, mode, filename, value FROM Cache
                     WHERE key = ? AND raw = 1
                     AND (expire_time IS NULL OR expire_time > ?)",
                    params![key, now],
                    row_from_query,
                )
                .optional()
                .map_err(|_| Error::Unavailable)?;
            let Some(row) = row else {
                return Ok(None);
            };
            connection
                .execute("DELETE FROM Cache WHERE rowid = ?", params![row.rowid])
                .map_err(|_| Error::Unavailable)?;
            Ok(Some(row))
        })?;
        let Some(row) = selected else {
            return Ok(None);
        };
        let filename = row.filename.clone();
        let result = fetch_row(&self.directory, row)?;
        if let Some(filename) = filename {
            remove_file(&self.directory, &filename);
        }
        Ok(result)
    }

    fn clear(&self) -> Result<(), Error> {
        let connection = self.connection.lock().map_err(|_| Error::Unavailable)?;
        let mut last_rowid = 0_i64;
        loop {
            let batch = transactional(&connection, |connection| {
                let rows = connection
                    .prepare(
                        "SELECT rowid, filename FROM Cache
                         WHERE rowid > ? ORDER BY rowid LIMIT 100",
                    )
                    .map_err(|_| Error::Unavailable)?
                    .query_map(params![last_rowid], |row| {
                        Ok((row.get::<_, i64>(0)?, row.get::<_, Option<String>>(1)?))
                    })
                    .map_err(|_| Error::Unavailable)?
                    .collect::<Result<Vec<_>, _>>()
                    .map_err(|_| Error::Unavailable)?;
                if rows.is_empty() {
                    return Ok(rows);
                }
                let ids = rows
                    .iter()
                    .map(|(rowid, _)| rowid.to_string())
                    .collect::<Vec<_>>()
                    .join(",");
                connection
                    .execute(&format!("DELETE FROM Cache WHERE rowid IN ({ids})"), [])
                    .map_err(|_| Error::Unavailable)?;
                Ok(rows)
            })?;
            if batch.is_empty() {
                return Ok(());
            }
            last_rowid = batch.last().map(|(rowid, _)| *rowid).unwrap_or(last_rowid);
            cleanup_files(
                &self.directory,
                batch
                    .into_iter()
                    .filter_map(|(_, filename)| filename)
                    .collect(),
            );
        }
    }

    fn update(
        &self,
        key: &str,
        now: f64,
        apply: &mut dyn FnMut(Option<StoredValue>) -> Result<(StoredValue, Option<f64>), Error>,
    ) -> Result<(), Error> {
        let connection = self.connection.lock().map_err(|_| Error::Unavailable)?;
        let mut created_filename = None;
        let result = transactional(&connection, |connection| {
            let current = connection
                .query_row(
                    "SELECT rowid, expire_time, mode, filename, value FROM Cache
                     WHERE key = ? AND raw = 1
                     AND (expire_time IS NULL OR expire_time > ?)",
                    params![key, now],
                    row_from_query,
                )
                .optional()
                .map_err(|_| Error::Unavailable)?
                .map(|row| fetch_row(&self.directory, row))
                .transpose()?
                .flatten();
            let (value, expire_time) = apply(current)?;
            let columns = store_value(&self.directory, self.min_file_size, value)?;
            created_filename = columns.filename.clone();
            let cleanup = self.set_locked(connection, key, columns, expire_time, now)?;
            Ok(cleanup)
        });
        match result {
            Ok(cleanup) => {
                cleanup_files(&self.directory, cleanup);
                Ok(())
            }
            Err(error) => {
                if let Some(filename) = created_filename {
                    remove_file(&self.directory, &filename);
                }
                Err(error)
            }
        }
    }
}

fn default_settings() -> HashMap<String, Value> {
    HashMap::from([
        ("statistics".to_string(), Value::Integer(0)),
        ("tag_index".to_string(), Value::Integer(0)),
        (
            "eviction_policy".to_string(),
            Value::Text("least-recently-stored".to_string()),
        ),
        ("size_limit".to_string(), Value::Integer(DEFAULT_SIZE_LIMIT)),
        ("cull_limit".to_string(), Value::Integer(DEFAULT_CULL_LIMIT)),
        ("sqlite_auto_vacuum".to_string(), Value::Integer(1)),
        ("sqlite_cache_size".to_string(), Value::Integer(8192)),
        (
            "sqlite_journal_mode".to_string(),
            Value::Text("wal".to_string()),
        ),
        (
            "sqlite_mmap_size".to_string(),
            Value::Integer(2_i64.pow(26)),
        ),
        ("sqlite_synchronous".to_string(), Value::Integer(1)),
        (
            "disk_min_file_size".to_string(),
            Value::Integer(DEFAULT_DISK_MIN_FILE_SIZE),
        ),
        ("disk_pickle_protocol".to_string(), Value::Integer(5)),
    ])
}

fn read_settings(connection: &Connection) -> Result<HashMap<String, Value>, Error> {
    let mut statement = match connection.prepare("SELECT key, value FROM Settings") {
        Ok(statement) => statement,
        Err(_) => return Ok(HashMap::new()),
    };
    statement
        .query_map([], |row| Ok((row.get(0)?, row.get(1)?)))
        .map_err(|_| Error::Unavailable)?
        .collect::<Result<HashMap<_, _>, _>>()
        .map_err(|_| Error::Unavailable)
}

fn apply_pragma(connection: &Connection, key: &str, value: &Value) -> Result<(), Error> {
    let pragma = key.strip_prefix("sqlite_").ok_or(Error::Unavailable)?;
    match value {
        Value::Integer(value) => connection
            .pragma_update(None, pragma, value)
            .map_err(|_| Error::Unavailable),
        Value::Text(value) => connection
            .pragma_update(None, pragma, value)
            .map_err(|_| Error::Unavailable),
        _ => Err(Error::Unavailable),
    }
}

fn setting_i64(settings: &HashMap<String, Value>, key: &str) -> Option<i64> {
    match settings.get(key) {
        Some(Value::Integer(value)) => Some(*value),
        _ => None,
    }
}

fn setting_string(settings: &HashMap<String, Value>, key: &str) -> Option<String> {
    match settings.get(key) {
        Some(Value::Text(value)) => Some(value.clone()),
        _ => None,
    }
}

fn has_get_update(policy: &str) -> bool {
    matches!(policy, "least-recently-used" | "least-frequently-used")
}

fn row_from_query(row: &rusqlite::Row<'_>) -> rusqlite::Result<Row> {
    Ok(Row {
        rowid: row.get(0)?,
        mode: row.get(2)?,
        filename: row.get(3)?,
        value: row.get(4)?,
    })
}

fn fetch_row(directory: &Path, row: Row) -> Result<Option<StoredValue>, Error> {
    match row.mode {
        MODE_RAW => match row.value {
            Value::Blob(value) => Ok(Some(StoredValue::Bytes(value))),
            Value::Text(value) => Ok(Some(StoredValue::Text(value))),
            Value::Integer(value) => Ok(Some(StoredValue::Integer(value))),
            Value::Real(value) => Ok(Some(StoredValue::Float(value))),
            Value::Null => Err(Error::InvalidEntry),
        },
        MODE_BINARY | MODE_PICKLE => {
            let bytes = match row.value {
                Value::Blob(value) => value,
                Value::Null => {
                    let Some(value) = read_file(directory, row.filename.as_deref())? else {
                        return Ok(None);
                    };
                    value
                }
                _ => return Err(Error::InvalidEntry),
            };
            Ok(Some(if row.mode == MODE_BINARY {
                StoredValue::Bytes(bytes)
            } else {
                StoredValue::Pickle(bytes)
            }))
        }
        MODE_TEXT => {
            let bytes = match row.value {
                Value::Null => {
                    let Some(value) = read_file(directory, row.filename.as_deref())? else {
                        return Ok(None);
                    };
                    value
                }
                Value::Blob(value) => value,
                Value::Text(value) => value.into_bytes(),
                _ => return Err(Error::InvalidEntry),
            };
            Ok(Some(StoredValue::Text(
                String::from_utf8(bytes).map_err(|_| Error::InvalidEntry)?,
            )))
        }
        _ => Err(Error::InvalidEntry),
    }
}

fn read_file(directory: &Path, filename: Option<&str>) -> Result<Option<Vec<u8>>, Error> {
    let Some(filename) = filename else {
        return Err(Error::InvalidEntry);
    };
    match fs::read(directory.join(filename)) {
        Ok(value) => Ok(Some(value)),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(_) => Err(Error::Unavailable),
    }
}

fn store_value(
    directory: &Path,
    min_file_size: usize,
    value: StoredValue,
) -> Result<StoredColumns, Error> {
    match value {
        StoredValue::Integer(value) => Ok(StoredColumns {
            size: 0,
            mode: MODE_RAW,
            filename: None,
            value: Some(Value::Integer(value)),
        }),
        StoredValue::Float(value) => Ok(StoredColumns {
            size: 0,
            mode: MODE_RAW,
            filename: None,
            value: Some(Value::Real(value)),
        }),
        StoredValue::Text(value) if value.chars().count() < min_file_size => Ok(StoredColumns {
            size: 0,
            mode: MODE_RAW,
            filename: None,
            value: Some(Value::Text(value)),
        }),
        StoredValue::Text(value) => {
            let bytes = value.into_bytes();
            let filename = write_file(directory, &bytes)?;
            Ok(StoredColumns {
                size: i64::try_from(bytes.len()).map_err(|_| Error::Unavailable)?,
                mode: MODE_TEXT,
                filename: Some(filename),
                value: None,
            })
        }
        StoredValue::Bytes(value) if value.len() < min_file_size => Ok(StoredColumns {
            size: 0,
            mode: MODE_RAW,
            filename: None,
            value: Some(Value::Blob(value)),
        }),
        StoredValue::Bytes(value) => {
            let filename = write_file(directory, &value)?;
            Ok(StoredColumns {
                size: i64::try_from(value.len()).map_err(|_| Error::Unavailable)?,
                mode: MODE_BINARY,
                filename: Some(filename),
                value: None,
            })
        }
        StoredValue::Pickle(value) if value.len() < min_file_size => Ok(StoredColumns {
            size: 0,
            mode: MODE_PICKLE,
            filename: None,
            value: Some(Value::Blob(value)),
        }),
        StoredValue::Pickle(value) => {
            let filename = write_file(directory, &value)?;
            Ok(StoredColumns {
                size: i64::try_from(value.len()).map_err(|_| Error::Unavailable)?,
                mode: MODE_PICKLE,
                filename: Some(filename),
                value: None,
            })
        }
    }
}

fn write_file(directory: &Path, bytes: &[u8]) -> Result<String, Error> {
    let mut random = [0_u8; 16];
    rand::rngs::OsRng.fill_bytes(&mut random);
    let hex = random
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();
    let filename = format!("{}/{}/{}.val", &hex[..2], &hex[2..4], &hex[4..]);
    let path = directory.join(&filename);
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|_| Error::Unavailable)?;
    }
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(|_| Error::Unavailable)?;
    file.write_all(bytes).map_err(|_| Error::Unavailable)?;
    Ok(filename)
}

fn cleanup_files(directory: &Path, filenames: Vec<String>) {
    for filename in filenames {
        remove_file(directory, &filename);
    }
}

fn remove_file(directory: &Path, filename: &str) {
    let path = directory.join(filename);
    let _ = fs::remove_file(&path);
}

fn transactional<T>(
    connection: &Connection,
    operation: impl FnOnce(&Connection) -> Result<T, Error>,
) -> Result<T, Error> {
    connection
        .execute_batch("BEGIN IMMEDIATE")
        .map_err(|_| Error::Unavailable)?;
    match operation(connection) {
        Ok(value) => {
            connection
                .execute_batch("COMMIT")
                .map_err(|_| Error::Unavailable)?;
            Ok(value)
        }
        Err(error) => {
            let _ = connection.execute_batch("ROLLBACK");
            Err(error)
        }
    }
}
