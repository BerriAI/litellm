/*
 * Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use criterion::{criterion_group, criterion_main, Criterion};
use hmac::digest::FixedOutput;
use hmac::{Hmac, KeyInit, Mac};
use sha2::Sha256;

pub fn hmac(c: &mut Criterion) {
    c.bench_function("hmac", |b| {
        b.iter(|| {
            let mut mac = Hmac::<Sha256>::new_from_slice(b"secret").unwrap();

            mac.update(b"hello, world");
            mac.finalize_fixed()
        })
    });
}

criterion_group! {
    name = benches;

    config = Criterion::default();

    targets = hmac
}

criterion_main!(benches);
