use std::io::{Cursor, Read};

use flate2::read::GzDecoder;
use sha2::{Digest, Sha256};

use crate::Error;
use crate::release::Packaging;

pub(crate) fn verify_sha256(asset: &str, expected: &str, bytes: &[u8]) -> Result<(), Error> {
    let actual = format!("{:x}", Sha256::digest(bytes));
    if actual.eq_ignore_ascii_case(expected) {
        return Ok(());
    }
    Err(Error::ChecksumMismatch {
        asset: asset.to_owned(),
        expected: expected.to_owned(),
        actual,
    })
}

pub(crate) fn extract_binary(packaging: &Packaging, bytes: &[u8]) -> Result<Vec<u8>, Error> {
    match packaging {
        Packaging::Bare => Ok(bytes.to_vec()),
        Packaging::TarGz { member } => extract_tar_gz(member, bytes),
        Packaging::Zip { member } => extract_zip(member, bytes),
    }
}

fn extract_tar_gz(member: &str, bytes: &[u8]) -> Result<Vec<u8>, Error> {
    let mut archive = tar::Archive::new(GzDecoder::new(bytes));
    for entry in archive.entries().map_err(Error::Archive)? {
        let mut entry = entry.map_err(Error::Archive)?;
        let path = entry.path().map_err(Error::Archive)?;
        if path.file_name().is_some_and(|name| name == member) {
            let mut binary = Vec::new();
            entry.read_to_end(&mut binary).map_err(Error::Archive)?;
            return Ok(binary);
        }
    }
    Err(Error::ArchiveMemberNotFound(member.to_owned()))
}

fn extract_zip(member: &str, bytes: &[u8]) -> Result<Vec<u8>, Error> {
    let mut archive = zip::ZipArchive::new(Cursor::new(bytes))?;
    let mut file = archive.by_name(member).map_err(|error| match error {
        zip::result::ZipError::FileNotFound => Error::ArchiveMemberNotFound(member.to_owned()),
        other => Error::Zip(other),
    })?;
    let mut binary = Vec::new();
    file.read_to_end(&mut binary).map_err(Error::Archive)?;
    Ok(binary)
}
