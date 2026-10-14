import json
from email.message import Message
from email.parser import BytesParser
from email.policy import HTTP
from typing import Final

from integration._support.client import Gateway
from integration._support.wire import Reply, Request, wire_server

from litellm.types.llms.anthropic_skills import Skill

_API_KEY: Final = "synthetic-anthropic-key"
_TITLE: Final = "Synthetic Skill"
_SKILL_MD: Final = b"---\nname: synthetic-skill\ndescription: A synthetic skill.\n---\nUse the helper.\n"
_HELPER: Final = b"print('synthetic helper')\n"
_UPSTREAM_SKILL: Final = {
    "id": "skill_synthetic_01",
    "type": "skill",
    "display_title": _TITLE,
    "latest_version": "1759178010641129",
    "source": "custom",
    "created_at": "2025-10-02T00:00:00Z",
    "updated_at": "2025-10-02T00:00:00Z",
}


def _multipart_parts(request: Request) -> tuple[Message, ...]:
    envelope: Final = f"content-type: {request.headers['content-type']}\r\n\r\n".encode() + request.body
    parsed: Final = BytesParser(policy=HTTP).parsebytes(envelope)
    assert parsed.is_multipart(), request.headers["content-type"]
    return tuple(parsed.iter_parts())


def _part_summary(part: Message) -> tuple[str, str | None, bytes]:
    return (
        str(part.get_param("name", header="content-disposition")),
        part.get_filename(),
        bytes(part.get_payload(decode=True)),
    )


def test_anthropic_skill_create_forwards_repeated_files_as_file_parts(gateway: Gateway) -> None:
    def respond(request: Request) -> Reply:
        assert request.method == "POST" and request.target == "/v1/skills", request.target
        assert request.headers["x-api-key"] == _API_KEY, request.headers
        assert request.headers["anthropic-version"] == "2023-06-01", request.headers
        assert request.headers["anthropic-beta"] == "skills-2025-10-02", request.headers
        assert "authorization" not in request.headers, request.headers
        assert tuple(_part_summary(part) for part in _multipart_parts(request)) == (
            ("display_title", None, _TITLE.encode()),
            ("files[]", "synthetic-skill/SKILL.md", _SKILL_MD),
            ("files[]", "synthetic-skill/helper.py", _HELPER),
        ), request.body
        return Reply(body=json.dumps(_UPSTREAM_SKILL).encode())

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model="anthropic/claude-sonnet-4-6", api_base=wire.url, api_key=_API_KEY)
        response: Final = gateway.client.post(
            "/v1/skills?beta=true",
            data={"display_title": _TITLE},
            files=[
                ("files[]", ("synthetic-skill/SKILL.md", _SKILL_MD, "text/markdown")),
                ("files[]", ("synthetic-skill/helper.py", _HELPER, "text/x-python")),
            ],
            headers={"Authorization": f"Bearer {gateway.key}", "x-litellm-model": model},
        )
        assert response.status_code == 200, response.text
        assert Skill.model_validate_json(response.content) == Skill.model_validate(_UPSTREAM_SKILL), response.text
        assert [(request.method, request.target) for request in wire.drain()] == [("POST", "/v1/skills")]
