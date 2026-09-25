use std::env::consts::{ARCH, OS};

use crate::Error;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Os {
    Macos,
    Linux,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Arch {
    Aarch64,
    X86_64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Platform {
    pub os: Os,
    pub arch: Arch,
}

impl Platform {
    pub fn current() -> Result<Self, Error> {
        let os = match OS {
            "macos" => Os::Macos,
            "linux" => Os::Linux,
            _ => return Err(Error::UnsupportedPlatform { os: OS, arch: ARCH }),
        };
        let arch = match ARCH {
            "aarch64" => Arch::Aarch64,
            "x86_64" => Arch::X86_64,
            _ => return Err(Error::UnsupportedPlatform { os: OS, arch: ARCH }),
        };
        Ok(Self { os, arch })
    }

    pub(crate) const fn os_name(self) -> &'static str {
        match self.os {
            Os::Macos => "darwin",
            Os::Linux => "linux",
        }
    }

    pub(crate) const fn arch_name(self) -> &'static str {
        match self.arch {
            Arch::Aarch64 => "arm64",
            Arch::X86_64 => "x64",
        }
    }

    pub(crate) const fn rust_triple(self) -> &'static str {
        match (self.os, self.arch) {
            (Os::Macos, Arch::Aarch64) => "aarch64-apple-darwin",
            (Os::Macos, Arch::X86_64) => "x86_64-apple-darwin",
            (Os::Linux, Arch::Aarch64) => "aarch64-unknown-linux-musl",
            (Os::Linux, Arch::X86_64) => "x86_64-unknown-linux-musl",
        }
    }
}
