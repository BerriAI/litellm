"""Harness coverage for canonical request identity (LIT-5741).

No proxy and no ``e2e`` marker: pure functions over ``RecordedRequest``. Pins
the two failure modes match keys must avoid: keying on volatile material so
nothing ever matches (markers, virtual keys, ids, timestamps, volatile
headers), and keying on too little so different requests collide and a test
silently asserts against another request's response.
"""

from __future__ import annotations

import pytest
from pydantic import JsonValue

from fixture_bundle import RecordedRequest
from fixture_canonical import CanonicalRequest, canonical_string, canonicalize, is_secret_field


def request(
    method: str = "post",
    path: str = "/chat/completions",
    *,
    headers: dict[str, str] | None = None,
    params: dict[str, str] | None = None,
    body: JsonValue | None = None,
    form: dict[str, str] | None = None,
    file_name: str | None = None,
    file_sha256: str | None = None,
    file_bytes: int | None = None,
) -> RecordedRequest:
    return RecordedRequest(
        method=method,
        path=path,
        headers=headers or {},
        params=params or {},
        body=body,
        form=form,
        file_name=file_name,
        file_sha256=file_sha256,
        file_bytes=file_bytes,
    )


class TestPlaceholders:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("Reply ok. 4d5152a995b7", "Reply ok. <marker>"),
            ("e2e-chat-stream-4d5152a995b7", "e2e-chat-stream-<marker>"),
            ("sk-3mCXCTGmYuEEIU2i2qmVE3Xq6tSK1O0X6ZIRP1Lpw8ZlbNjt", "<key>"),
            ("9f1c8a2e-4b3d-4f6a-8f2f-0a1b2c3d4e5f", "<uuid>"),
            ("z" * 64, "z" * 64),
            ("0123456789abcdef" * 4, "<sha256>"),
            ("2026-08-19T20:57:13.363499+00:00", "<timestamp>"),
            ("2026-08-19", "<date>"),
            ("chatcmpl-C0LO6rRkfJlpJ2mqW9BHYo4Sm8FWl", "<id>"),
            ("batch_688a8b7f9a08819096e0f7c88fcd07c5", "<id>"),
            ("file-XyZ12345abc", "<id>"),
            ("gpt-4o-mini", "gpt-4o-mini"),
            ("max_tokens", "max_tokens"),
            ("sk-1234", "sk-1234"),
        ],
    )
    def test_rewrites_exactly_the_volatile_shapes(self, raw: str, expected: str) -> None:
        assert canonical_string(raw) == expected


class TestSecretFields:
    @pytest.mark.parametrize(
        ("name", "secret"),
        [
            ("api_key", True),
            ("openai_api_key", True),
            ("aws_secret_access_key", True),
            ("aws_session_token", True),
            ("vertex_credentials", True),
            ("static_headers", True),
            ("langfuse_secret_key", True),
            ("model", False),
            ("max_completion_tokens", False),
            ("api_base", False),
        ],
    )
    def test_names_that_carry_credentials(self, name: str, secret: bool) -> None:
        assert is_secret_field(name) is secret


class TestKeyStability:
    def test_volatile_material_does_not_change_the_key(self) -> None:
        """Acceptance: a suite recorded on one machine (fresh keys, that day's
        dates, that run's markers) replays on another with no misses."""
        first = request(
            headers={"authorization": "Bearer sk-run-one-aaaaaaaaaaaaaaaa", "x-request-id": "req-1"},
            params={"start_date": "2026-08-18"},
            body={
                "model": "e2e-chat-4d5152a995b7",
                "messages": [{"role": "user", "content": "Reply ok. 4d5152a995b7"}],
                "api_key": "sk-live-one-aaaaaaaaaaaaaaaa",
            },
        )
        second = request(
            headers={"authorization": "Bearer sk-run-two-bbbbbbbbbbbbbbbb", "x-request-id": "req-2"},
            params={"start_date": "2026-08-19"},
            body={
                "model": "e2e-chat-1a2b3c4d5e6f",
                "messages": [{"role": "user", "content": "Reply ok. 1a2b3c4d5e6f"}],
                "api_key": "os.environ/OPENAI_API_KEY",
            },
        )
        assert canonicalize(first).key == canonicalize(second).key

    def test_serialization_order_is_not_identity(self) -> None:
        ordered = request(body={"model": "m", "stream": True})
        reversed_order = request(body={"stream": True, "model": "m"})
        assert canonicalize(ordered).key == canonicalize(reversed_order).key

    def test_generated_ids_in_the_path_do_not_change_the_key(self) -> None:
        first = request("get", "/v1/batches/batch_688a8b7f9a08819096e0f7c88fcd07c5")
        second = request("get", "/v1/batches/batch_770b9c8f0b19920107f1f8d99fde18d6")
        assert canonicalize(first).key == canonicalize(second).key


class TestKeyDistinctness:
    def test_requests_differing_only_inside_canonicalized_fields_stay_distinct(self) -> None:
        """Acceptance: a naive verb+path hash collides these; the content key
        must not, or one test silently asserts against the other's response."""
        first = request(body={"messages": [{"content": "Reply ok. 4d5152a995b7"}]})
        second = request(body={"messages": [{"content": "Count to three. 4d5152a995b7"}]})
        naive = (first.method, first.path)
        assert naive == (second.method, second.path)
        assert canonicalize(first).key != canonicalize(second).key

    def test_a_kept_header_is_identity(self) -> None:
        first = request(headers={"x-litellm-tags": "prod"})
        second = request(headers={"x-litellm-tags": "shadow"})
        assert canonicalize(first).key != canonicalize(second).key

    def test_a_volatile_header_is_not_identity(self) -> None:
        first = request(headers={"traceparent": "00-aa-bb-01", "x-api-key": "one"})
        second = request(headers={"traceparent": "00-cc-dd-01", "x-api-key": "two"})
        assert canonicalize(first).key == canonicalize(second).key

    def test_query_params_are_identity(self) -> None:
        first = request("get", "/v1/vector_stores", params={"limit": "100"})
        second = request("get", "/v1/vector_stores", params={"limit": "10"})
        assert canonicalize(first).key != canonicalize(second).key

    def test_secret_set_versus_unset_stays_distinct(self) -> None:
        with_key = request(body={"api_key": "sk-live-aaaaaaaaaaaaaaaa"})
        without_key = request(body={"api_key": None})
        assert canonicalize(with_key).key != canonicalize(without_key).key

    def test_form_fields_are_identity(self) -> None:
        first = request("upload", "/v1/files", form={"purpose": "assistants"}, file_sha256="a" * 64)
        second = request("upload", "/v1/files", form={"purpose": "batch"}, file_sha256="a" * 64)
        assert canonicalize(first).key != canonicalize(second).key

    def test_file_content_is_identity(self) -> None:
        first = request(
            "upload", "/v1/files", file_name="batch.jsonl", file_sha256="a" * 64, file_bytes=10
        )
        second = request(
            "upload", "/v1/files", file_name="batch.jsonl", file_sha256="b" * 64, file_bytes=10
        )
        assert canonicalize(first).key != canonicalize(second).key


class TestKeyShape:
    def test_key_names_method_path_and_digest(self) -> None:
        canonical = canonicalize(request("post", "/model/new", body={"model_name": "m"}))
        assert isinstance(canonical, CanonicalRequest)
        assert canonical.key.startswith("post /model/new #")
        assert len(canonical.key.rsplit("#", 1)[1]) == 16
