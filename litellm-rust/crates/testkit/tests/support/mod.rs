use std::collections::HashMap;
use std::io::Write;
use std::sync::atomic::{AtomicUsize, Ordering};

use litellm_testkit::{Error, Fetch};
use sha2::{Digest, Sha256};

pub struct FakeFetch {
    routes: HashMap<String, Vec<u8>>,
    calls: AtomicUsize,
}

impl FakeFetch {
    pub fn new(routes: impl IntoIterator<Item = (String, Vec<u8>)>) -> Self {
        Self {
            routes: routes.into_iter().collect(),
            calls: AtomicUsize::new(0),
        }
    }

    pub fn calls(&self) -> usize {
        self.calls.load(Ordering::SeqCst)
    }
}

impl Fetch for FakeFetch {
    async fn get(&self, url: &str) -> Result<Vec<u8>, Error> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        self.routes.get(url).cloned().ok_or_else(|| Error::Status {
            url: url.to_owned(),
            status: 404,
        })
    }
}

impl Fetch for &FakeFetch {
    async fn get(&self, url: &str) -> Result<Vec<u8>, Error> {
        (*self).get(url).await
    }
}

pub fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

pub fn script_printing(output: &str) -> Vec<u8> {
    format!("#!/bin/sh\necho '{output}'\n").into_bytes()
}

pub fn tar_gz(member: &str, contents: &[u8]) -> Vec<u8> {
    let mut builder = tar::Builder::new(Vec::new());
    let mut header = tar::Header::new_gnu();
    header.set_size(contents.len() as u64);
    header.set_mode(0o755);
    header.set_cksum();
    builder.append_data(&mut header, member, contents).unwrap();
    let tarball = builder.into_inner().unwrap();
    let mut encoder = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
    encoder.write_all(&tarball).unwrap();
    encoder.finish().unwrap()
}

pub fn zip_archive(member: &str, contents: &[u8]) -> Vec<u8> {
    let mut writer = zip::ZipWriter::new(std::io::Cursor::new(Vec::new()));
    writer
        .start_file(member, zip::write::SimpleFileOptions::default())
        .unwrap();
    writer.write_all(contents).unwrap();
    writer.finish().unwrap().into_inner()
}
