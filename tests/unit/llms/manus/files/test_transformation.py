import time
from unittest.mock import Mock

import httpx
import pytest
from pydantic import ValidationError

from litellm.llms.manus.files.transformation import ManusFilesConfig
from litellm.types.llms.openai import OpenAIFileObject


def _list_files(body: object) -> list[OpenAIFileObject]:
    return ManusFilesConfig().transform_list_files_response(
        raw_response=httpx.Response(200, json=body), logging_obj=Mock(), litellm_params={}
    )


def test_delete_file_response_is_read_into_a_file_deleted_object():
    deleted = ManusFilesConfig().transform_delete_file_response(
        raw_response=httpx.Response(200, json={"id": "file-1", "deleted": True, "object": "file", "region": "eu"}),
        logging_obj=Mock(),
        litellm_params={},
    )

    assert deleted.model_dump() == {"id": "file-1", "deleted": True, "object": "file", "region": "eu"}


@pytest.mark.parametrize("body", [b'["secret-file"]', b'"secret-file"', b"7", b"null"])
def test_delete_file_response_rejects_a_body_that_is_not_an_object_without_echoing_it(body: bytes):
    with pytest.raises(ValidationError) as exc_info:
        ManusFilesConfig().transform_delete_file_response(
            raw_response=httpx.Response(200, content=body), logging_obj=Mock(), litellm_params={}
        )

    assert "secret-file" not in str(exc_info.value)


def test_delete_file_response_requires_the_file_deleted_fields():
    with pytest.raises(ValidationError, match="FileDeleted"):
        ManusFilesConfig().transform_delete_file_response(
            raw_response=httpx.Response(200, json={"id": "file-1"}), logging_obj=Mock(), litellm_params={}
        )


def test_list_files_response_maps_every_listed_file():
    files = _list_files(
        {
            "object": "list",
            "data": [
                {
                    "id": "file-1",
                    "bytes": 12,
                    "filename": "a.pdf",
                    "purpose": "batch",
                    "status": "processed",
                    "status_details": "done",
                    "created_at": "2024-01-02T03:04:05Z",
                },
                {"id": "file-2", "created_at": "2024-01-02T03:04:05.123456+00:00"},
            ],
        }
    )

    created_at = int(time.mktime(time.strptime("2024-01-02T03:04:05", "%Y-%m-%dT%H:%M:%S")))
    assert [file.model_dump() for file in files] == [
        {
            "id": "file-1",
            "bytes": 12,
            "created_at": created_at,
            "filename": "a.pdf",
            "object": "file",
            "purpose": "batch",
            "status": "processed",
            "expires_at": None,
            "status_details": "done",
        },
        {
            "id": "file-2",
            "bytes": 0,
            "created_at": created_at,
            "filename": "",
            "object": "file",
            "purpose": "assistants",
            "status": "uploaded",
            "expires_at": None,
            "status_details": None,
        },
    ]


@pytest.mark.parametrize("created_at", ["not-a-date", "", None, 0])
def test_list_files_response_falls_back_to_now_for_an_unusable_created_at(created_at: object):
    before = int(time.time())

    (file,) = _list_files({"data": [{"id": "file-1", "created_at": created_at}]})

    assert before <= file.created_at <= int(time.time())


@pytest.mark.parametrize("body", [{}, {"data": []}])
def test_list_files_response_is_empty_without_listed_files(body: dict[str, object]):
    assert _list_files(body) == []


@pytest.mark.parametrize(
    "body",
    [
        ["secret-file"],
        "secret-file",
        {"data": None},
        {"data": 7},
        {"data": "secret-file"},
        {"data": ["secret-file"]},
        {"data": [{"id": "file-1"}, ["secret-file"]]},
        {"data": [{"id": "file-1", "created_at": ["secret-file"]}]},
        {"data": [{"id": "file-1", "created_at": 1700000000}]},
    ],
)
def test_list_files_response_rejects_malformed_listings_without_echoing_them(body: object):
    with pytest.raises(ValidationError) as exc_info:
        _list_files(body)

    assert "secret-file" not in str(exc_info.value)


def test_list_files_response_rejects_a_file_the_file_object_cannot_hold():
    with pytest.raises(ValidationError, match="OpenAIFileObject"):
        _list_files({"data": [{"id": "file-1", "purpose": "not-a-purpose"}]})
