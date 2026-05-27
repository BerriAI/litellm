"""
Regression tests for LIT-3382: large Gemini batch JSONL uploads must not
materialize ~5 simultaneous copies of the payload (raw bytes, decoded
string, splitlines list, parsed-openai-dicts list, parsed-vertex-dicts list,
joined string). That caused uvicorn workers on 12 GB pods to OOM on ~1 GB
inputs and crash with "Child process died" (Tempus customer report).

After the streaming refactor, peak memory should stay roughly within
``input_size + output_size`` (~2-3x input), not ~5-8x input.

These tests guard all three call sites that handle the JSONL transform:

1. ``VertexAIJsonlFilesTransformation.transform_openai_file_content_to_vertex_ai_file_content``
   — used by ``VertexAIFilesHandler.async_create_file`` (legacy GCS-direct path).
2. ``VertexAIFilesConfig.transform_create_file_request``
   — used by ``litellm.files.main.create_file`` (modern ``/v1/files`` route).
3. ``VertexAIFilesConfig.get_object_name``
   — called inside ``get_complete_file_url``.
"""

import gc
import io
import json
import tracemalloc
from typing import List, Tuple

import pytest

from litellm.llms.vertex_ai.files.transformation import (
    VertexAIFilesConfig,
    VertexAIJsonlFilesTransformation,
)


def _build_batch_jsonl(num_rows: int, prompt_chars: int = 2000) -> bytes:
    prompt = "x" * prompt_chars
    lines: List[str] = []
    for i in range(num_rows):
        lines.append(
            json.dumps(
                {
                    "custom_id": f"req-{i}",
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": "gemini-1.5-pro",
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": 8,
                    },
                }
            )
        )
    return ("\n".join(lines) + "\n").encode("utf-8")


def _measure_handler(num_rows: int) -> Tuple[int, int, int]:
    payload = _build_batch_jsonl(num_rows)
    cfg = VertexAIJsonlFilesTransformation()
    gc.collect()
    tracemalloc.start()
    try:
        out, _ = cfg.transform_openai_file_content_to_vertex_ai_file_content(
            io.BytesIO(payload)
        )
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return len(payload), len(out.encode("utf-8")), peak


def _measure_create_file_request(num_rows: int) -> Tuple[int, int, int]:
    payload = _build_batch_jsonl(num_rows)
    cfg = VertexAIFilesConfig()
    d = {"file": ("b.jsonl", payload, "application/jsonl"), "purpose": "batch"}
    gc.collect()
    tracemalloc.start()
    try:
        out = cfg.transform_create_file_request(
            model="gemini-1.5-pro",
            create_file_data=d,
            optional_params={},
            litellm_params={},
        )
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return len(payload), len(out.encode("utf-8")), peak


def _measure_object_name(num_rows: int) -> Tuple[int, int, int]:
    payload = _build_batch_jsonl(num_rows)
    cfg = VertexAIFilesConfig()
    extracted = {"content": payload, "filename": "b.jsonl"}
    gc.collect()
    tracemalloc.start()
    try:
        name = cfg.get_object_name(extracted_file_data=extracted, purpose="batch")
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return len(payload), len(name), peak


class TestLit3382StreamingHandlerPath:
    """Memory + correctness guards for the
    ``VertexAIJsonlFilesTransformation.transform_openai_file_content_to_vertex_ai_file_content``
    code path (legacy ``VertexAIFilesHandler.async_create_file`` route)."""

    def test_peak_under_4x_input_size(self):
        """Peak memory must stay <4x input size for a representative batch.
        Pre-fix this was ~5.24x; post-fix it is ~2.05x. The 4x ceiling
        leaves room for tracemalloc accounting wobble across Python
        versions."""
        in_b, out_b, peak = _measure_handler(num_rows=5000)
        amplification = peak / in_b
        assert 0.5 * in_b <= out_b <= 2 * in_b
        assert amplification < 4.0, (
            f"Peak memory {peak/1e6:.1f} MB is {amplification:.2f}x the "
            f"input {in_b/1e6:.1f} MB — should be <4x."
        )

    def test_scales_linearly(self):
        """Doubling input must not super-linearly increase peak."""
        _, _, peak_small = _measure_handler(num_rows=2000)
        _, _, peak_big = _measure_handler(num_rows=8000)
        ratio = peak_big / peak_small
        assert ratio < 5.5, (
            f"Peak grew {ratio:.2f}x for a 4x input increase "
            f"({peak_small/1e6:.1f}MB -> {peak_big/1e6:.1f}MB)."
        )

    def test_output_is_valid_jsonl_with_request_wrapper(self):
        """Streaming must not change the wire format."""
        payload = _build_batch_jsonl(num_rows=5, prompt_chars=50)
        cfg = VertexAIJsonlFilesTransformation()
        out, object_name = cfg.transform_openai_file_content_to_vertex_ai_file_content(
            io.BytesIO(payload)
        )
        lines = out.splitlines()
        assert len(lines) == 5
        for idx, line in enumerate(lines):
            obj = json.loads(line)
            assert set(obj.keys()) == {"request"}, f"row {idx}"
            assert "contents" in obj["request"]
            assert "labels" in obj["request"]
        assert "publishers/google/models/gemini-1.5-pro" in object_name

    def test_empty_payload_raises_value_error(self):
        cfg = VertexAIJsonlFilesTransformation()
        with pytest.raises(ValueError):
            cfg.transform_openai_file_content_to_vertex_ai_file_content(
                io.BytesIO(b"")
            )

    def test_object_name_uses_first_row_model_only(self):
        """object_name must be derived from the FIRST row's model so we
        never need to keep the full parsed list around just for naming."""
        cfg = VertexAIJsonlFilesTransformation()
        payload = (
            json.dumps(
                {
                    "custom_id": "first",
                    "body": {"model": "gemini-2.0-flash", "messages": []},
                }
            )
            + "\n"
            + json.dumps(
                {
                    "custom_id": "second",
                    "body": {"model": "gemini-1.5-pro", "messages": []},
                }
            )
            + "\n"
        ).encode("utf-8")
        _, name = cfg.transform_openai_file_content_to_vertex_ai_file_content(
            io.BytesIO(payload)
        )
        assert "publishers/google/models/gemini-2.0-flash" in name
        assert "gemini-1.5-pro" not in name


class TestLit3382StreamingFilesApiPath:
    """Memory + correctness guards for
    ``VertexAIFilesConfig.transform_create_file_request`` — the modern
    registered ``/v1/files`` create_file path."""

    def test_peak_under_5x_input_size(self):
        """Pre-fix ~5.24x; post-fix ~3.06x. The 5x ceiling guards against
        re-introducing an intermediate full-payload copy while leaving room
        for the bytes -> str decode step that ``extract_file_data`` performs
        before our transform sees the payload."""
        in_b, out_b, peak = _measure_create_file_request(num_rows=5000)
        amplification = peak / in_b
        assert 0.5 * in_b <= out_b <= 2 * in_b
        assert amplification < 5.0, (
            f"Peak memory {peak/1e6:.1f} MB is {amplification:.2f}x the "
            f"input {in_b/1e6:.1f} MB — should be <5x."
        )

    def test_scales_linearly(self):
        _, _, peak_small = _measure_create_file_request(num_rows=2000)
        _, _, peak_big = _measure_create_file_request(num_rows=8000)
        ratio = peak_big / peak_small
        assert ratio < 5.5, (
            f"Peak grew {ratio:.2f}x for a 4x input increase "
            f"({peak_small/1e6:.1f}MB -> {peak_big/1e6:.1f}MB)."
        )

    def test_output_preserves_request_wrapper_and_labels(self):
        payload = _build_batch_jsonl(num_rows=4, prompt_chars=20)
        cfg = VertexAIFilesConfig()
        d = {"file": ("b.jsonl", payload, "application/jsonl"), "purpose": "batch"}
        out = cfg.transform_create_file_request(
            model="gemini-1.5-pro",
            create_file_data=d,
            optional_params={},
            litellm_params={},
        )
        lines = out.splitlines()
        assert len(lines) == 4
        for line in lines:
            obj = json.loads(line)
            assert set(obj.keys()) == {"request"}
            assert "labels" in obj["request"]

    def test_get_object_name_does_not_load_full_payload(self):
        """``get_object_name`` should only sniff the first row to derive the
        object key — peak memory must stay <3x input."""
        in_b, _, peak = _measure_object_name(num_rows=5000)
        amplification = peak / in_b
        assert amplification < 3.0, (
            f"get_object_name peak {peak/1e6:.1f} MB is {amplification:.2f}x "
            f"the {in_b/1e6:.1f} MB input — must not load full payload."
        )


class TestLit3382IterHelper:
    """Direct tests of the new streaming helper.

    ``_iter_openai_jsonl_entries`` MUST be a generator. If it ever becomes
    a list (e.g. someone writes ``return [...]``), the memory win is lost
    and the OOM regression returns silently."""

    def test_yields_one_entry_at_a_time(self):
        cfg = VertexAIJsonlFilesTransformation()
        payload = _build_batch_jsonl(num_rows=3, prompt_chars=50)
        gen = cfg._iter_openai_jsonl_entries(io.BytesIO(payload))
        assert hasattr(gen, "__next__")
        assert not isinstance(gen, list)
        items = list(gen)
        assert len(items) == 3
        assert items[0]["custom_id"] == "req-0"
        assert items[0]["body"]["model"] == "gemini-1.5-pro"

    def test_accepts_bytes_str_and_tuple_inputs(self):
        """Parity with the legacy ``_get_content_from_openai_file`` helper:
        bytes / str / (filename, content) tuples must all work."""
        cfg = VertexAIJsonlFilesTransformation()
        payload = _build_batch_jsonl(num_rows=2, prompt_chars=20)

        assert len(list(cfg._iter_openai_jsonl_entries(payload))) == 2
        assert (
            len(list(cfg._iter_openai_jsonl_entries(payload.decode("utf-8")))) == 2
        )
        assert (
            len(list(cfg._iter_openai_jsonl_entries(("batch.jsonl", payload))))
            == 2
        )
        assert (
            len(
                list(
                    cfg._iter_openai_jsonl_entries(
                        ("batch.jsonl", io.BytesIO(payload))
                    )
                )
            )
            == 2
        )

    def test_skips_blank_lines(self):
        """Legacy semantics: blank lines must be silently skipped."""
        cfg = VertexAIJsonlFilesTransformation()
        body = (
            json.dumps(
                {
                    "custom_id": "a",
                    "body": {"model": "gemini-1.5-pro", "messages": []},
                }
            )
            + "\n\n   \n"
            + json.dumps(
                {
                    "custom_id": "b",
                    "body": {"model": "gemini-1.5-pro", "messages": []},
                }
            )
            + "\n"
        )
        out = list(cfg._iter_openai_jsonl_entries(body))
        assert [e["custom_id"] for e in out] == ["a", "b"]
