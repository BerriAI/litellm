import asyncio
import math
from typing import Final

from _s3_v2_support import BUCKET, PREFIX, RecordingS3Sink
from integration._support.wire import wire_server

from litellm.integrations.lens.content import S3V2ContentStore

_BODY: Final = '{"messages":"[\\"界\\"]","response":"done"}'


async def test_the_active_s3_v2_logger_stores_and_reads_lens_content_within_the_reference_size() -> None:
    from litellm.integrations.s3_v2 import S3Logger

    sink: Final = RecordingS3Sink(delay_seconds=0)
    with wire_server(sink.respond) as wire:
        loggers: Final = tuple(
            S3Logger(
                s3_bucket_name=BUCKET,
                s3_region_name="us-east-1",
                s3_endpoint_url=wire.url,
                s3_aws_access_key_id="AKIAIOSFODNN7EXAMPLE",
                s3_aws_secret_access_key="wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
                s3_callback_params_override={},
                s3_path=path,
            )
            for path in (f"{PREFIX}/", None)
        )
        try:
            prefixed, unprefixed = (S3V2ContentStore(resolve=lambda logger=logger: logger) for logger in loggers)
            assert unprefixed.object_key("lens-content/object.json") == "lens-content/object.json"
            key: Final = prefixed.object_key("lens-content/object.json")
            assert key == f"{PREFIX}/lens-content/object.json"
            assert key is not None
            assert await prefixed.upload(key, _BODY) is True
            assert sink.objects()[f"/{BUCKET}/{key}"] == _BODY.encode()
            size: Final = len(_BODY.encode())
            assert await prefixed.download(key, size) == _BODY
            assert await prefixed.download(key, size - 1) is None
            sink.fail_status, sink.fail_code, sink.fail_until = 403, "AccessDenied", math.inf
            assert await prefixed.upload(f"{key}.rejected", _BODY) is False
            assert f"/{BUCKET}/{key}.rejected" not in sink.objects()
        finally:
            tasks: Final = tuple(logger.periodic_flush_task for logger in loggers)
            for task in tasks:
                _ = task.cancel()
            _ = await asyncio.gather(*tasks, return_exceptions=True)
