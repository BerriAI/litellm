import base64
import uuid
from typing import Final

import litellm
import pytest
from integration._support.wire import Reply, Request, wire_server

_KEY: Final = "synthetic-topaz-key"


@pytest.mark.parametrize("trailing", ("", "/"), ids=("no-slash", "slash"))
def test_topaz_image_variation_reaches_one_single_slash_url(trailing: str) -> None:
    marker: Final = f"topaz-{uuid.uuid4().hex}".encode()
    enhanced: Final = b"enhanced " + marker

    def respond(_: Request) -> Reply:
        return Reply(body=enhanced, content_type="image/jpeg")

    with wire_server(respond) as wire:
        response: Final = litellm.image_variation(
            image=b"\x89PNG\r\n\x1a\n" + marker,
            model="topaz/Standard V2",
            api_key=_KEY,
            api_base=f"{wire.url}/tz{trailing}",
        )
        received: Final = wire.drain()
    assert [(request.method, request.target) for request in received] == [("POST", "/tz/image/v1/enhance")]
    assert marker in received[0].body
    assert received[0].headers["x-api-key"] == _KEY, sorted(received[0].headers)
    assert response.data is not None
    assert [image.b64_json for image in response.data] == [base64.b64encode(enhanced).decode()]
