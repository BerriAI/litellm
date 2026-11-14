use std::fs::{self, File};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime};

use sha2::{Digest, Sha256};
use tempfile::NamedTempFile;

use crate::attempt::MAX_BODY_BYTES;

/// How long after it was stored or last revalidated a cached document is still served
pub const CACHE_MAX_AGE: Duration = Duration::from_secs(24 * 60 * 60);

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Cached {
    pub body: Vec<u8>,
    pub etag: Option<String>,
}

/// One source's document kept between processes: a file holding the ETag on its first line, then the document
/// bytes. The file is named by a hash of the URL, so a URL that carries credentials never reaches a file name
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Cache {
    path: PathBuf,
}

impl Cache {
    pub fn new(directory: &Path, url: &str) -> Self {
        Self {
            path: directory.join(format!("model_cost_map_{:x}", Sha256::digest(url))),
        }
    }

    /// The kept document, unless it is missing, unreadable, or was last revalidated more than `CACHE_MAX_AGE`
    /// before `now`. The caller still checks the document like a fetched one
    pub fn load(&self, now: SystemTime) -> Option<Cached> {
        let metadata = fs::metadata(&self.path).ok()?;
        let age = now.duration_since(metadata.modified().ok()?).ok()?;
        if age > CACHE_MAX_AGE || metadata.len() > MAX_BODY_BYTES {
            return None;
        }
        let mut etag = fs::read(&self.path).ok()?;
        let body = etag.split_off(etag.iter().position(|byte| *byte == b'\n')? + 1);
        etag.pop();
        let etag = String::from_utf8(etag).ok()?;
        Some(Cached {
            body,
            etag: (!etag.is_empty()).then_some(etag),
        })
    }

    /// Replaces the kept document in one rename, so a reader sees the old file or the new one, never a part of either
    pub fn store(&self, body: &[u8], etag: Option<&str>) -> io::Result<()> {
        let directory = self.path.parent().unwrap_or(Path::new("."));
        fs::create_dir_all(directory)?;
        let mut file = NamedTempFile::new_in(directory)?;
        file.write_all(etag.unwrap_or_default().as_bytes())?;
        file.write_all(b"\n")?;
        file.write_all(body)?;
        file.persist(&self.path)?;
        Ok(())
    }

    /// Records that the source confirmed the kept document at `now`
    pub fn revalidated(&self, now: SystemTime) -> io::Result<()> {
        File::options()
            .write(true)
            .open(&self.path)?
            .set_modified(now)
    }
}

#[cfg(test)]
mod tests {
    use rstest::{fixture, rstest};
    use tempfile::TempDir;

    use super::*;

    const URL: &str = "https://example.com/cost.json";

    #[fixture]
    fn directory() -> TempDir {
        tempfile::tempdir().unwrap()
    }

    fn stored_at(cache: &Cache, at: SystemTime) {
        cache.store(b"{\"a\": 1}", Some("\"v1\"")).unwrap();
        cache.revalidated(at).unwrap();
    }

    #[rstest]
    #[case::with_an_etag(Some("W/\"v1\""))]
    #[case::without_an_etag(None)]
    fn a_stored_document_loads_with_its_etag(directory: TempDir, #[case] etag: Option<&str>) {
        let cache = Cache::new(directory.path(), URL);
        cache.store(b"{\n\"a\": 1\n}", etag).unwrap();
        assert_eq!(
            cache.load(SystemTime::now()),
            Some(Cached {
                body: b"{\n\"a\": 1\n}".to_vec(),
                etag: etag.map(str::to_owned),
            })
        );
    }

    #[rstest]
    #[case::just_stored(Duration::ZERO, true)]
    #[case::at_the_limit(CACHE_MAX_AGE, true)]
    #[case::past_the_limit(CACHE_MAX_AGE + Duration::from_secs(1), false)]
    fn a_document_is_served_until_it_is_older_than_the_limit(
        directory: TempDir,
        #[case] age: Duration,
        #[case] served: bool,
    ) {
        let cache = Cache::new(directory.path(), URL);
        let stored = SystemTime::UNIX_EPOCH + Duration::from_secs(1_700_000_000);
        stored_at(&cache, stored);
        assert_eq!(cache.load(stored + age).is_some(), served);
    }

    #[rstest]
    fn a_document_dated_after_now_is_not_served(directory: TempDir) {
        let cache = Cache::new(directory.path(), URL);
        let stored = SystemTime::UNIX_EPOCH + Duration::from_secs(1_700_000_000);
        stored_at(&cache, stored);
        assert_eq!(cache.load(stored - Duration::from_secs(1)), None);
    }

    #[rstest]
    fn a_revalidated_document_is_served_for_another_full_limit(directory: TempDir) {
        let cache = Cache::new(directory.path(), URL);
        let stored = SystemTime::UNIX_EPOCH + Duration::from_secs(1_700_000_000);
        stored_at(&cache, stored);
        let revalidated = stored + CACHE_MAX_AGE + Duration::from_secs(60);
        assert_eq!(cache.load(revalidated), None);
        cache.revalidated(revalidated).unwrap();
        assert!(cache.load(revalidated + CACHE_MAX_AGE).is_some());
    }

    #[rstest]
    fn a_store_replaces_the_document_and_leaves_one_file(directory: TempDir) {
        let cache = Cache::new(directory.path(), URL);
        cache.store(b"{\"a\": 1}", Some("\"v1\"")).unwrap();
        cache.store(b"{\"a\": 2}", Some("\"v2\"")).unwrap();
        assert_eq!(
            cache.load(SystemTime::now()),
            Some(Cached {
                body: b"{\"a\": 2}".to_vec(),
                etag: Some("\"v2\"".into()),
            })
        );
        assert_eq!(fs::read_dir(directory.path()).unwrap().count(), 1);
    }

    #[rstest]
    fn each_url_keeps_its_own_document(directory: TempDir) {
        Cache::new(directory.path(), URL)
            .store(b"{\"a\": 1}", Some("\"v1\""))
            .unwrap();
        let other = Cache::new(directory.path(), "https://example.com/other.json");
        assert_eq!(other.load(SystemTime::now()), None);
    }

    #[rstest]
    fn a_url_never_reaches_the_file_name(directory: TempDir) {
        Cache::new(
            directory.path(),
            "https://user:secret@example.com/cost.json?token=secret",
        )
        .store(b"{}", None)
        .unwrap();
        let names: Vec<String> = fs::read_dir(directory.path())
            .unwrap()
            .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        assert!(
            matches!(names.as_slice(), [name] if !name.contains("secret") && !name.contains("example")),
            "{names:?}"
        );
    }

    #[rstest]
    fn a_missing_directory_is_created_on_store_and_holds_nothing_before(directory: TempDir) {
        let cache = Cache::new(&directory.path().join("nested/cache"), URL);
        assert_eq!(cache.load(SystemTime::now()), None);
        assert!(cache.revalidated(SystemTime::now()).is_err());
        cache.store(b"{}", None).unwrap();
        assert!(cache.load(SystemTime::now()).is_some());
    }

    #[rstest]
    fn a_directory_that_cannot_be_written_stores_nothing(directory: TempDir) {
        let blocker = directory.path().join("not-a-directory");
        fs::write(&blocker, b"").unwrap();
        let cache = Cache::new(&blocker.join("cache"), URL);
        assert!(cache.store(b"{}", None).is_err());
        assert_eq!(cache.load(SystemTime::now()), None);
    }

    #[rstest]
    #[case::no_etag_line(b"{}".as_slice())]
    #[case::etag_line_is_not_text(b"\xff\xfe\n{}".as_slice())]
    fn a_file_that_is_not_a_kept_document_is_not_served(
        directory: TempDir,
        #[case] contents: &[u8],
    ) {
        let cache = Cache::new(directory.path(), URL);
        cache.store(b"{}", None).unwrap();
        fs::write(&cache.path, contents).unwrap();
        assert_eq!(cache.load(SystemTime::now()), None);
    }
}
