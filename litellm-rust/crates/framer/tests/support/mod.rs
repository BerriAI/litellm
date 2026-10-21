#![allow(dead_code)]

use std::{error::Error, io};

use bytes::{Bytes, BytesMut};
use futures_util::{Stream, stream};
use tokio_util::codec::Encoder;

pub fn encode_all<C, I>(mut codec: C, items: impl IntoIterator<Item = I>) -> Vec<u8>
where
    C: Encoder<I>,
    C::Error: std::fmt::Debug,
{
    let mut wire = BytesMut::new();
    for item in items {
        codec.encode(item, &mut wire).unwrap();
    }
    wire.to_vec()
}

pub fn cut_at(bytes: &[u8], offsets: impl IntoIterator<Item = usize>) -> Vec<Bytes> {
    let mut sorted: Vec<usize> = offsets
        .into_iter()
        .filter(|offset| *offset <= bytes.len())
        .collect();
    sorted.sort_unstable();
    sorted.dedup();
    let bounds = std::iter::once(0)
        .chain(sorted)
        .chain(std::iter::once(bytes.len()))
        .collect::<Vec<_>>();
    bounds
        .windows(2)
        .map(|pair| Bytes::copy_from_slice(&bytes[pair[0]..pair[1]]))
        .collect()
}

pub fn every(bytes: &[u8], size: usize) -> Vec<Bytes> {
    bytes
        .chunks(size.max(1))
        .map(Bytes::copy_from_slice)
        .collect()
}

pub fn input(pieces: Vec<Bytes>) -> impl Stream<Item = Result<Bytes, io::Error>> + Send {
    stream::iter(pieces.into_iter().map(Ok))
}

pub fn body_cause<T: Error + 'static>(body: &io::Error) -> Option<&T> {
    body.get_ref()?.downcast_ref::<T>()
}

pub fn runtime() -> tokio::runtime::Runtime {
    tokio::runtime::Builder::new_current_thread()
        .build()
        .unwrap()
}
