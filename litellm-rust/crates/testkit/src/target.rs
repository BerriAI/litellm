use target_lexicon::{Architecture, Environment, OperatingSystem, Triple};

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
pub struct Target {
    pub os: Os,
    pub arch: Arch,
    pub musl: bool,
}

impl Target {
    pub fn host() -> Result<Self, Error> {
        Self::try_from(&Triple::host())
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

    pub(crate) const fn musl_suffix(self) -> &'static str {
        if self.musl { "-musl" } else { "" }
    }
}

impl TryFrom<&Triple> for Target {
    type Error = Error;

    fn try_from(triple: &Triple) -> Result<Self, Error> {
        let unsupported = || Error::UnsupportedTarget(triple.to_string());
        let os = match triple.operating_system {
            OperatingSystem::Darwin(_) | OperatingSystem::MacOSX(_) => Os::Macos,
            OperatingSystem::Linux => Os::Linux,
            _ => return Err(unsupported()),
        };
        let arch = match triple.architecture {
            Architecture::Aarch64(_) => Arch::Aarch64,
            Architecture::X86_64 => Arch::X86_64,
            _ => return Err(unsupported()),
        };
        Ok(Self {
            os,
            arch,
            musl: triple.environment == Environment::Musl,
        })
    }
}
