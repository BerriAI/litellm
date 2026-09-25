mod archive;
mod client;
mod error;
mod fetch;
mod gateway;
mod install;
mod platform;
mod release;

pub use client::Client;
pub use error::Error;
pub use fetch::{Fetch, HttpFetch};
pub use gateway::{Gateway, LaunchSpec, launch_spec};
pub use install::{Installed, Installer};
pub use platform::{Arch, Os, Platform};
pub use release::{Packaging, Release, resolve};
