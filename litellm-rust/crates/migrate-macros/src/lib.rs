mod error;

use std::path::{Path, PathBuf};

use error::Error;
use proc_macro::TokenStream;
use quote::quote;
use syn::LitStr;

struct Entry {
    version: u64,
    description: String,
    path: PathBuf,
}

fn resolve(dir: &Path) -> Result<Vec<Entry>, Error> {
    let mut entries = Vec::new();
    let files = std::fs::read_dir(dir).map_err(|source| Error::ReadDirectory {
        path: dir.display().to_string(),
        source,
    })?;
    for file in files {
        let file = file.map_err(|source| Error::ReadDirectory {
            path: dir.display().to_string(),
            source,
        })?;
        let path = file.path();
        let name = path
            .file_name()
            .and_then(|name| name.to_str())
            .ok_or_else(|| Error::NonUtf8Path {
                path: path.display().to_string(),
            })?
            .to_owned();
        let invalid = || Error::InvalidName { name: name.clone() };
        let stem = name
            .strip_suffix(".sql")
            .filter(|_| file.file_type().is_ok_and(|kind| kind.is_file()))
            .and_then(|stem| stem.split_once('_'))
            .filter(|(version, description)| {
                !version.is_empty()
                    && version.bytes().all(|b| b.is_ascii_digit())
                    && !description.is_empty()
                    && description
                        .bytes()
                        .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'_')
            })
            .ok_or_else(invalid)?;
        let version = stem.0.parse::<u64>().map_err(|_| invalid())?;
        entries.push(Entry {
            version,
            description: stem.1.to_owned(),
            path,
        });
    }
    if entries.is_empty() {
        return Err(Error::Empty {
            path: dir.display().to_string(),
        });
    }
    entries.sort_by_key(|entry| entry.version);
    for pair in entries.windows(2) {
        if pair[0].version == pair[1].version {
            return Err(Error::DuplicateVersion {
                version: pair[0].version,
            });
        }
    }
    Ok(entries)
}

fn resolve_input(lit: &LitStr) -> Result<Vec<Entry>, Error> {
    let root = std::env::var("CARGO_MANIFEST_DIR")
        .map(PathBuf::from)
        .unwrap_or_default();
    let dir = root.join(lit.value());
    let dir = dir.canonicalize().map_err(|source| Error::ReadDirectory {
        path: dir.display().to_string(),
        source,
    })?;
    if dir.to_str().is_none() {
        return Err(Error::NonUtf8Path {
            path: dir.display().to_string(),
        });
    }
    resolve(&dir)
}

#[proc_macro]
pub fn migrate(input: TokenStream) -> TokenStream {
    let lit = syn::parse_macro_input!(input as LitStr);
    match resolve_input(&lit) {
        Ok(entries) => {
            let migrations = entries.iter().map(|entry| {
                let version = entry.version;
                let description = &entry.description;
                let path = entry
                    .path
                    .to_str()
                    .expect("canonical migration path is UTF-8");
                quote! {
                    ::litellm_migrate::Migration {
                        version: #version,
                        description: #description,
                        sql: ::core::include_str!(#path),
                    }
                }
            });
            quote! { &[#(#migrations),*] }.into()
        }
        Err(err) => syn::Error::new(lit.span(), err).to_compile_error().into(),
    }
}

#[cfg(test)]
mod tests {
    use std::fs;

    use rstest::rstest;
    use tempfile::TempDir;

    use super::{Error, resolve};

    fn migrations_dir(files: &[&str]) -> TempDir {
        let dir = TempDir::new().expect("tempdir");
        for file in files {
            fs::write(dir.path().join(file), "SELECT 1").expect("write fixture");
        }
        dir
    }

    #[rstest]
    fn orders_versions_numerically() {
        let dir = migrations_dir(&["10_tenth.sql", "2_second.sql", "1_first.sql"]);
        let entries = resolve(dir.path()).expect("resolves");
        let versions: Vec<u64> = entries.iter().map(|entry| entry.version).collect();
        let descriptions: Vec<&str> = entries
            .iter()
            .map(|entry| entry.description.as_str())
            .collect();
        assert_eq!(versions, [1, 2, 10]);
        assert_eq!(descriptions, ["first", "second", "tenth"]);
    }

    #[rstest]
    #[case::dash_in_version(&["0001-dash.sql"])]
    #[case::not_sql(&["notes.txt"])]
    #[case::empty_description(&["0001_.sql"])]
    #[case::non_digit_version(&["x_name.sql"])]
    #[case::uppercase_description(&["0001_Upper.sql"])]
    #[case::no_underscore(&["0001.sql"])]
    #[case::plus_sign_version(&["+10_add.sql"])]
    fn rejects_invalid_names(#[case] files: &[&str]) {
        let dir = migrations_dir(files);
        assert!(matches!(
            resolve(dir.path()),
            Err(Error::InvalidName { .. })
        ));
    }

    #[rstest]
    fn rejects_subdirectories() {
        let dir = migrations_dir(&["0001_a.sql"]);
        fs::create_dir(dir.path().join("0002_b.sql")).expect("subdir");
        assert!(matches!(
            resolve(dir.path()),
            Err(Error::InvalidName { .. })
        ));
    }

    #[cfg(unix)]
    #[rstest]
    fn rejects_symlinks() {
        let dir = migrations_dir(&["0001_a.sql"]);
        let target = TempDir::new().expect("tempdir");
        let target_file = target.path().join("real.sql");
        fs::write(&target_file, "SELECT 2").expect("write fixture");
        std::os::unix::fs::symlink(&target_file, dir.path().join("0002_b.sql")).expect("symlink");
        assert!(matches!(
            resolve(dir.path()),
            Err(Error::InvalidName { .. })
        ));
    }

    #[rstest]
    fn rejects_duplicate_versions() {
        let dir = migrations_dir(&["0001_a.sql", "1_b.sql"]);
        assert!(matches!(
            resolve(dir.path()),
            Err(Error::DuplicateVersion { version: 1 })
        ));
    }

    #[rstest]
    fn rejects_empty_directory() {
        let dir = migrations_dir(&[]);
        assert!(matches!(resolve(dir.path()), Err(Error::Empty { .. })));
    }
}
