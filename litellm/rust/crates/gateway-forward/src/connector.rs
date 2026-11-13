use std::{
    io::{self, IoSlice},
    pin::Pin,
    task::{Context, Poll, ready},
};

use axum::http::Uri;
use hyper::rt::{Read, ReadBufCursor, Write};
use hyper_util::{
    client::legacy::connect::{Connected, Connection, HttpConnector},
    rt::TokioIo,
};
use tokio::net::TcpStream;

/// Connects to Python like `HttpConnector`, but keeps hyper reading Python's answer after Python
/// stops reading the request. uvicorn rejects an upload (413, 401) as soon as it reads the head
/// and closes with the body unread, so writing the rest fails, and hyper would fail the request
/// and drop the answer it already received. Like curl and httpcore, the forwarder discards the
/// rest of the request instead and reads on. A connection that never answered still fails, on the
/// read, and no request is ever sent twice.
#[derive(Clone)]
pub(crate) struct Connector(pub HttpConnector);

impl tower_service::Service<Uri> for Connector {
    type Response = AnswerAfterRefusal;
    type Error = <HttpConnector as tower_service::Service<Uri>>::Error;
    type Future = Pin<Box<dyn Future<Output = Result<Self::Response, Self::Error>> + Send>>;

    fn poll_ready(&mut self, context: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        self.0.poll_ready(context)
    }

    fn call(&mut self, uri: Uri) -> Self::Future {
        let connecting = self.0.call(uri);
        Box::pin(async move {
            Ok(AnswerAfterRefusal {
                stream: connecting.await?,
                refused: false,
            })
        })
    }
}

pub(crate) struct AnswerAfterRefusal {
    stream: TokioIo<TcpStream>,
    refused: bool,
}

impl AnswerAfterRefusal {
    fn written(&mut self, written: io::Result<usize>, length: usize) -> io::Result<usize> {
        match written {
            Err(error)
                if matches!(
                    error.kind(),
                    io::ErrorKind::BrokenPipe | io::ErrorKind::ConnectionReset
                ) =>
            {
                self.refused = true;
                Ok(length)
            }
            written => written,
        }
    }
}

impl Read for AnswerAfterRefusal {
    fn poll_read(
        mut self: Pin<&mut Self>,
        context: &mut Context<'_>,
        buffer: ReadBufCursor<'_>,
    ) -> Poll<io::Result<()>> {
        Pin::new(&mut self.stream).poll_read(context, buffer)
    }
}

impl Write for AnswerAfterRefusal {
    fn poll_write(
        mut self: Pin<&mut Self>,
        context: &mut Context<'_>,
        buffer: &[u8],
    ) -> Poll<io::Result<usize>> {
        if self.refused {
            return Poll::Ready(Ok(buffer.len()));
        }
        let written = ready!(Pin::new(&mut self.stream).poll_write(context, buffer));
        Poll::Ready(self.written(written, buffer.len()))
    }

    fn poll_write_vectored(
        mut self: Pin<&mut Self>,
        context: &mut Context<'_>,
        buffers: &[IoSlice<'_>],
    ) -> Poll<io::Result<usize>> {
        let length = buffers.iter().map(|buffer| buffer.len()).sum();
        if self.refused {
            return Poll::Ready(Ok(length));
        }
        let written = ready!(Pin::new(&mut self.stream).poll_write_vectored(context, buffers));
        Poll::Ready(self.written(written, length))
    }

    fn is_write_vectored(&self) -> bool {
        self.stream.is_write_vectored()
    }

    fn poll_flush(mut self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<io::Result<()>> {
        if self.refused {
            return Poll::Ready(Ok(()));
        }
        Pin::new(&mut self.stream).poll_flush(context)
    }

    fn poll_shutdown(mut self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<io::Result<()>> {
        Pin::new(&mut self.stream).poll_shutdown(context)
    }
}

impl Connection for AnswerAfterRefusal {
    fn connected(&self) -> Connected {
        self.stream.connected()
    }
}
