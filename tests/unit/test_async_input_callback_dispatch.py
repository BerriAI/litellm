"""Async input callbacks must run before an async request reaches the provider."""

import pytest

import litellm
from litellm import main as litellm_main


@pytest.mark.asyncio
async def test_async_input_callback_runs_once_with_sync_callbacks(monkeypatch):
    seen = []

    def sync_callback(details):
        seen.append(("sync", details["model"]))

    async def async_callback(details):
        seen.append(("async", details["model"]))

    monkeypatch.setattr(litellm, "input_callback", [])
    monkeypatch.setattr(litellm, "_async_input_callback", [])
    litellm.logging_callback_manager.add_litellm_input_callback(sync_callback)
    litellm.logging_callback_manager.add_litellm_input_callback(async_callback)
    original_provider = litellm_main.mock_completion

    def fake_provider(*args, **kwargs):
        seen.append(("provider", kwargs.get("model", args[0] if args else "gpt-4o-mini")))
        return original_provider(*args, **kwargs)

    monkeypatch.setattr(litellm_main, "mock_completion", fake_provider)
    await litellm.acompletion(model="gpt-4o-mini", messages=[{"role": "user", "content": "hi"}], mock_response="ok")
    assert seen == [("async", "gpt-4o-mini"), ("provider", "gpt-4o-mini"), ("sync", "gpt-4o-mini")]


@pytest.mark.asyncio
async def test_async_input_callback_error_blocks_mock_provider(monkeypatch):
    seen = []

    def sync_callback(details):
        seen.append("sync")

    async def reject(details):
        seen.append("async")
        raise RuntimeError("input denied")

    monkeypatch.setattr(litellm, "input_callback", [])
    monkeypatch.setattr(litellm, "_async_input_callback", [])
    litellm.logging_callback_manager.add_litellm_input_callback(sync_callback)
    litellm.logging_callback_manager.add_litellm_input_callback(reject)

    def provider(*args, **kwargs):
        seen.append("provider")
        raise AssertionError("provider must not be called")

    monkeypatch.setattr(litellm_main, "mock_completion", provider)
    with pytest.raises(Exception, match="input denied"):
        await litellm.acompletion(
            model="gpt-4o-mini", messages=[{"role": "user", "content": "hi"}], mock_response="ok", num_retries=0
        )
    assert seen == ["async"]


def test_sync_completion_never_runs_async_input_callback(monkeypatch):
    seen = []

    async def async_callback(details):
        seen.append("async")

    def sync_callback(details):
        seen.append("sync")

    monkeypatch.setattr(litellm, "input_callback", [])
    monkeypatch.setattr(litellm, "_async_input_callback", [])
    litellm.logging_callback_manager.add_litellm_input_callback(async_callback)
    litellm.logging_callback_manager.add_litellm_input_callback(sync_callback)
    litellm.completion(model="gpt-4o-mini", messages=[{"role": "user", "content": "hi"}], mock_response="ok")
    assert seen == ["sync"]


@pytest.mark.asyncio
async def test_direct_async_input_registration_runs_once(monkeypatch):
    seen = []

    async def async_callback(details):
        seen.append(details["model"])

    monkeypatch.setattr(litellm, "input_callback", [async_callback])
    monkeypatch.setattr(litellm, "_async_input_callback", [])
    await litellm.acompletion(model="gpt-4o-mini", messages=[{"role": "user", "content": "hi"}], mock_response="ok")
    assert seen == ["gpt-4o-mini"]
    assert litellm.input_callback == []
