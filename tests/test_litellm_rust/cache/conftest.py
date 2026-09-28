import threading
from collections.abc import Generator
from typing import Final

import fakeredis
import pytest

from tests.test_litellm_rust.support.s3_stub import S3Stub


@pytest.fixture
def redis_url() -> Generator[str]:
    server: Final = fakeredis.TcpFakeServer(("127.0.0.1", 0), server_type="redis")
    worker: Final = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        yield f"redis://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=5)


@pytest.fixture
def s3_stub() -> Generator[S3Stub]:
    stub: Final = S3Stub()
    try:
        yield stub
    finally:
        stub.close()
