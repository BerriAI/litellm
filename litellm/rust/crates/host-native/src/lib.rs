//! The Rust driver for hosted calls: it answers host services and interceptors with Rust handlers
//! and hands stream deliveries to whichever consumer sits on top, HTTP body polling or an
//! in-process stream consumer.

mod driver;
pub mod in_process;
pub mod services;

pub use driver::{Boundary, Driver};
