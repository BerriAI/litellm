use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::task::{Context, Poll};

use bytes::Bytes;
use futures_util::{FutureExt, Stream, stream};
use litellm_python_interop::{
    AsyncByteStream, ByteStreamReader, PythonByteStream, PythonStreamCompletion, to_py_bytes,
};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytesMethods, PyDict, PyList};

struct TrackedSource {
    inner: PythonByteStream,
    drops: Arc<AtomicUsize>,
}

impl Stream for TrackedSource {
    type Item = <PythonByteStream as Stream>::Item;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        self.inner.as_mut().poll_next(cx)
    }
}

impl Drop for TrackedSource {
    fn drop(&mut self) {
        self.drops.fetch_add(1, Ordering::SeqCst);
    }
}

fn tracked(inner: PythonByteStream, drops: &Arc<AtomicUsize>) -> PythonByteStream {
    Box::pin(TrackedSource {
        inner,
        drops: drops.clone(),
    })
}

fn completion(count: &Arc<AtomicUsize>, fails: bool) -> PythonStreamCompletion {
    let count = count.clone();
    Box::pin(async move {
        count.fetch_add(1, Ordering::SeqCst);
        if fails {
            Err(PyValueError::new_err("completion"))
        } else {
            Ok(())
        }
    })
}

#[tokio::test]
async fn eof_and_repeated_close_release_source_and_complete_once() {
    Python::initialize();
    for early in [false, true] {
        let drops = Arc::new(AtomicUsize::new(0));
        let completions = Arc::new(AtomicUsize::new(0));
        let source = Box::pin(stream::iter([Ok(vec![0, 255].into())]));
        let reader =
            ByteStreamReader::new(tracked(source, &drops), completion(&completions, false));
        if !early {
            assert_eq!(
                reader.next_chunk().await.unwrap().unwrap().as_ref(),
                &[0, 255]
            );
            assert!(reader.next_chunk().await.unwrap().is_none());
        }
        reader.aclose().await.unwrap();
        reader.aclose().await.unwrap();
        assert!(reader.next_chunk().await.unwrap().is_none());
        assert_eq!(drops.load(Ordering::SeqCst), 1);
        assert_eq!(completions.load(Ordering::SeqCst), 1);
    }
}

#[tokio::test]
async fn reader_preserves_allocation_and_slice_boundaries() {
    let allocation = Bytes::from(vec![7; 65536]);
    let slices = [allocation.slice(13..79), allocation.slice(1024..4096)];
    let reader = ByteStreamReader::new(
        Box::pin(stream::iter(slices.clone().map(Ok))),
        Box::pin(async { Ok(()) }),
    );
    for expected in slices {
        let actual = reader.next_chunk().await.unwrap().unwrap();
        assert_eq!(actual.as_ptr(), expected.as_ptr());
        assert_eq!(actual.len(), expected.len());
        assert_eq!(actual, expected);
    }
    assert!(reader.next_chunk().await.unwrap().is_none());
}

#[tokio::test]
async fn concurrent_poll_is_rejected_and_close_wakes_the_pending_reader() {
    Python::initialize();
    let drops = Arc::new(AtomicUsize::new(0));
    let completions = Arc::new(AtomicUsize::new(0));
    let reader = ByteStreamReader::new(
        tracked(Box::pin(stream::pending()), &drops),
        completion(&completions, false),
    );
    let mut pending = Box::pin(reader.next_chunk());
    assert!(pending.as_mut().now_or_never().is_none());
    let error = reader.next_chunk().await.unwrap_err();
    Python::attach(|py| assert!(error.is_instance_of::<PyRuntimeError>(py)));
    reader.close();
    assert!(
        tokio::time::timeout(std::time::Duration::from_secs(1), pending)
            .await
            .unwrap()
            .unwrap()
            .is_none()
    );
    reader.aclose().await.unwrap();
    assert_eq!(drops.load(Ordering::SeqCst), 1);
    assert_eq!(completions.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn dropping_a_pending_read_cancels_and_releases_its_source() {
    Python::initialize();
    let drops = Arc::new(AtomicUsize::new(0));
    let completions = Arc::new(AtomicUsize::new(0));
    let reader = ByteStreamReader::new(
        tracked(Box::pin(stream::pending()), &drops),
        completion(&completions, false),
    );
    let mut pending = Box::pin(reader.next_chunk());
    assert!(pending.as_mut().now_or_never().is_none());
    drop(pending);
    assert_eq!(drops.load(Ordering::SeqCst), 1);
    reader.aclose().await.unwrap();
    assert!(reader.next_chunk().await.unwrap().is_none());
    assert_eq!(completions.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn stream_and_completion_errors_preserve_precedence() {
    Python::initialize();
    for stream_fails in [false, true] {
        for completion_fails in [false, true] {
            let drops = Arc::new(AtomicUsize::new(0));
            let completions = Arc::new(AtomicUsize::new(0));
            let source: PythonByteStream = if stream_fails {
                Box::pin(stream::iter([Err(PyRuntimeError::new_err("source"))]))
            } else {
                Box::pin(stream::empty())
            };
            let reader = ByteStreamReader::new(
                tracked(source, &drops),
                completion(&completions, completion_fails),
            );
            let result = reader.next_chunk().await;
            Python::attach(|py| {
                if completion_fails {
                    assert!(result.unwrap_err().is_instance_of::<PyValueError>(py));
                } else if stream_fails {
                    assert!(result.unwrap_err().is_instance_of::<PyRuntimeError>(py));
                } else {
                    assert!(result.unwrap().is_none());
                }
            });
            reader.aclose().await.unwrap();
            assert_eq!(drops.load(Ordering::SeqCst), 1);
            assert_eq!(completions.load(Ordering::SeqCst), 1);
        }
    }
}

#[tokio::test]
async fn cancelled_eof_wait_resumes_the_same_completion_on_close() {
    let starts = Arc::new(AtomicUsize::new(0));
    let release = Arc::new(tokio::sync::Notify::new());
    let reader = ByteStreamReader::new(Box::pin(stream::empty()), {
        let starts = starts.clone();
        let release = release.clone();
        Box::pin(async move {
            starts.fetch_add(1, Ordering::SeqCst);
            release.notified().await;
            Ok(())
        })
    });
    let mut eof = Box::pin(reader.next_chunk());
    assert!(eof.as_mut().now_or_never().is_none());
    drop(eof);
    let mut close = Box::pin(reader.aclose());
    assert!(close.as_mut().now_or_never().is_none());
    release.notify_one();
    close.await.unwrap();
    reader.aclose().await.unwrap();
    assert_eq!(starts.load(Ordering::SeqCst), 1);
}

#[test]
fn python_iteration_returns_owned_bytes_and_exhausts_normally() {
    Python::initialize();
    let payloads = vec![vec![], vec![42], vec![0, 255, 128], vec![7; 65536]];
    let drops = Arc::new(AtomicUsize::new(0));
    let completions = Arc::new(AtomicUsize::new(0));
    let source = Box::pin(stream::iter(
        payloads.clone().into_iter().map(|bytes| Ok(bytes.into())),
    ));
    Python::attach(|py| {
        let scope = PyDict::new(py);
        scope
            .set_item(
                "source",
                Py::new(
                    py,
                    AsyncByteStream::new(tracked(source, &drops), completion(&completions, false)),
                )
                .unwrap(),
            )
            .unwrap();
        let expected =
            PyList::new(py, payloads.iter().map(|bytes| to_py_bytes(py, bytes))).unwrap();
        scope.set_item("expected", &expected).unwrap();
        py.run(
            c"
import asyncio

async def exercise():
    assert source.__aiter__() is source
    retained = [chunk async for chunk in source]
    assert all(type(chunk) is bytes for chunk in retained)
    assert retained == expected
    await source.aclose()
    await source.aclose()
    try:
        await anext(source)
    except StopAsyncIteration:
        pass
    else:
        raise AssertionError('expected exhaustion')
    return retained

retained = asyncio.run(exercise())
del source
assert retained == expected
",
            Some(&scope),
            Some(&scope),
        )
        .unwrap();
        assert_eq!(drops.load(Ordering::SeqCst), 1);
        assert_eq!(completions.load(Ordering::SeqCst), 1);
        for payload in &payloads {
            assert_eq!(to_py_bytes(py, payload).as_bytes(), payload);
        }
    });
}
