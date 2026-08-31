"""
Unit tests for the PrivAiTe guardrail provider.

The PrivAiTe engine lives in the external 'privaite' package, which is NOT a
dependency of LiteLLM. These tests therefore install a fake 'privaite' package
into sys.modules (an autouse fixture) so the guardrail's lazy imports resolve to
fakes. That means the suite runs exactly the way LiteLLM CI runs it: with the
real 'privaite' package absent. The fakes let us exercise the guardrail's own
wiring end to end (registry, engine caching, pre/post/streaming hooks) without
re-testing PrivAiTe's detection internals.
"""

import json
import sys
import types

import pytest

# fake placeholder <-> original, the way PrivAiTe maps values.
_FAKES = {"<PERSON_1>": "Marie Dupont", "<EMAIL_ADDRESS_1>": "marie@acme.com"}
_REALS = {original: fake for fake, original in _FAKES.items()}


class _FakeMapping:
    """Stand-in for privaite.pii.mapping.PIIMapping."""

    def __init__(self, fakes=None):
        self._fakes = dict(fakes or {})

    @property
    def is_empty(self):
        return not self._fakes

    def get_all_fakes(self):
        return dict(self._fakes)

    def add(self, original, fake, _entity_type):
        self._fakes[fake] = original


class _FakePIIBlockedError(Exception):
    """Stand-in for privaite.pii.engine.PIIBlockedError. Names TYPES, not values."""

    def __init__(self, entity_types):
        self.entity_types = sorted(entity_types)
        super().__init__("request blocked: contains disallowed PII type(s): " + ", ".join(self.entity_types))


class _FakeEngine:
    """Stand-in for privaite.pii.engine.PIIEngine.

    Scrubs the known reals on the way out, restores the known fakes on the way
    back. ``raise_oserror_until`` lets a test force the spaCy-download retry path
    in the guardrail's _engine_for by failing the first N initialize() calls. If
    ``config.block_entities`` names a type present in the text, process_request
    raises _FakePIIBlockedError, exactly as the real engine would.
    """

    raise_oserror_until = 0
    oserror_message = "[E050] Can't find model 'en_core_web_lg'."
    init_calls = 0

    def __init__(self, config):
        self.config = config

    async def initialize(self):
        type(self).init_calls += 1
        if self.init_calls <= type(self).raise_oserror_until:
            raise OSError(type(self).oserror_message)

    def _scrub(self, content, mapping):
        if isinstance(content, str):
            for real, fake in _REALS.items():
                if real in content:
                    content = content.replace(real, fake)
                    mapping.add(real, fake, "PII")
            return content
        if isinstance(content, list):
            out = []
            for part in content:
                if isinstance(part, str):
                    out.append(self._scrub(part, mapping))
                elif isinstance(part, dict) and isinstance(part.get("text"), str):
                    out.append({**part, "text": self._scrub(part["text"], mapping)})
                else:
                    out.append(part)
            return out
        return content

    async def process_request(self, messages):
        mapping = _FakeMapping()
        anonymized = []
        for message in messages:
            new_message = dict(message)
            new_message["content"] = self._scrub(message.get("content", ""), mapping)
            anonymized.append(new_message)
        blocked = set(getattr(self.config, "block_entities", None) or [])
        if blocked:
            # placeholder "<EMAIL_ADDRESS_1>" -> type "EMAIL_ADDRESS"
            detected = {fake.strip("<>").rsplit("_", 1)[0] for fake in mapping.get_all_fakes()}
            hit = detected & blocked
            if hit:
                raise _FakePIIBlockedError(hit)
        return anonymized, mapping

    async def process_response(self, text, _mapping):
        for fake, original in _FAKES.items():
            text = text.replace(fake, original)
        return text


class _FakeStreamingDeAnonymizer:
    """Stand-in for privaite.streaming.buffer.StreamingDeAnonymizer.

    Stateful like the real one: it buffers a trailing "<...."-without-">" fragment
    so a placeholder split across chunks reassembles, which is what makes per-choice
    buffer isolation observable in tests.
    """

    def __init__(self, mapping):
        self._fakes = mapping.get_all_fakes()
        self._buf = ""

    def feed(self, text):
        self._buf += text
        for fake, original in self._fakes.items():
            self._buf = self._buf.replace(fake, original)
        lt = self._buf.rfind("<")
        if lt != -1 and ">" not in self._buf[lt:]:
            emit, self._buf = self._buf[:lt], self._buf[lt:]
        else:
            emit, self._buf = self._buf, ""
        return emit

    def flush(self):
        out, self._buf = self._buf, ""
        return out


class _Config:
    """Generic stand-in for the privaite.config.schema dataclasses."""

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class _PIIConfig(_Config):
    """PIIConfig stand-in. ``model_fields`` mirrors a modern privaite that declares
    block_entities, so the guardrail's fail-closed guard sees the field as present.
    A test can override model_fields to simulate an older privaite."""

    model_fields = {"preset": None, "languages": None, "block_entities": None}


@pytest.fixture(autouse=True)
def _fake_privaite_package(monkeypatch):
    """Install a fake 'privaite' (and 'spacy') package tree so the guardrail's
    lazy imports resolve without the real packages installed, mirroring CI."""
    _FakeEngine.raise_oserror_until = 0
    _FakeEngine.oserror_message = "[E050] Can't find model 'en_core_web_lg'."
    _FakeEngine.init_calls = 0

    def _module(name, **attrs):
        mod = types.ModuleType(name)
        for key, value in attrs.items():
            setattr(mod, key, value)
        return mod

    schema = _module(
        "privaite.config.schema",
        AnonymizationConfig=_Config,
        DeanonymizationConfig=_Config,
        DetectorsConfig=_Config,
        PIIConfig=_PIIConfig,
        PresidioDetectorConfig=_Config,
    )
    mapping = _module("privaite.pii.mapping", PIIMapping=_FakeMapping)
    engine = _module("privaite.pii.engine", PIIEngine=_FakeEngine, PIIBlockedError=_FakePIIBlockedError)
    buffer = _module("privaite.streaming.buffer", StreamingDeAnonymizer=_FakeStreamingDeAnonymizer)
    config_pkg = _module("privaite.config", schema=schema)
    pii_pkg = _module("privaite.pii", mapping=mapping, engine=engine)
    streaming_pkg = _module("privaite.streaming", buffer=buffer)
    privaite_pkg = _module("privaite", config=config_pkg, pii=pii_pkg, streaming=streaming_pkg)
    spacy_cli = _module("spacy.cli", download=lambda _model: None)
    spacy_pkg = _module("spacy", cli=spacy_cli)

    for name, mod in {
        "privaite": privaite_pkg,
        "privaite.config": config_pkg,
        "privaite.config.schema": schema,
        "privaite.pii": pii_pkg,
        "privaite.pii.mapping": mapping,
        "privaite.pii.engine": engine,
        "privaite.streaming": streaming_pkg,
        "privaite.streaming.buffer": buffer,
        "spacy": spacy_pkg,
        "spacy.cli": spacy_cli,
    }.items():
        monkeypatch.setitem(sys.modules, name, mod)
    yield


def _make_guardrail(**kwargs):
    from litellm.proxy.guardrails.guardrail_hooks.privaite import PrivaiteGuardrail

    return PrivaiteGuardrail(guardrail_name="privaite-test", **kwargs)


async def _collect(aiterator):
    return [chunk async for chunk in aiterator]


def test_registry_and_enum_wiring():
    from litellm.proxy.guardrails.guardrail_hooks.privaite import (
        PrivaiteGuardrail,
        guardrail_class_registry,
        guardrail_initializer_registry,
    )
    from litellm.types.guardrails import SupportedGuardrailIntegrations

    assert SupportedGuardrailIntegrations.PRIVAITE.value == "privaite"
    assert "privaite" in guardrail_initializer_registry
    assert guardrail_class_registry["privaite"] is PrivaiteGuardrail


def test_config_model_ui_name():
    from litellm.types.proxy.guardrails.guardrail_hooks.privaite import (
        PrivaiteGuardrailConfigModel,
    )

    assert PrivaiteGuardrailConfigModel.ui_friendly_name() == "PrivAiTe"
    fields = PrivaiteGuardrailConfigModel.model_fields
    assert fields["preset"].default == "onnx"
    assert fields["languages"].default == "en,fr"
    assert fields["deanonymize"].default is True


def test_initialize_guardrail_registers_callback(monkeypatch):
    import litellm
    from litellm.proxy.guardrails.guardrail_hooks.privaite import (
        PrivaiteGuardrail,
        initialize_guardrail,
    )

    added = []
    monkeypatch.setattr(
        litellm.logging_callback_manager,
        "add_litellm_callback",
        lambda cb: added.append(cb),
    )

    litellm_params = types.SimpleNamespace(
        mode="pre_call",
        default_on=False,
        preset="light",
        languages="en",
        deanonymize=True,
    )
    callback = initialize_guardrail(litellm_params, {"guardrail_name": "privaite"})

    assert isinstance(callback, PrivaiteGuardrail)
    assert callback.preset == "light"
    assert added == [callback]


def test_init_normalizes_params():
    # invalid preset falls back to onnx; "false"-ish strings disable deanonymize.
    gr = _make_guardrail(preset="bogus", languages="en, fr ,", deanonymize="false")
    assert gr.preset == "onnx"
    assert gr.deanonymize is False
    assert gr._languages() == ["en", "fr"]

    # an explicit valid preset and bool are kept; empty languages fall back to
    # the __init__ default of "en,fr".
    gr2 = _make_guardrail(preset="light", languages="", deanonymize=True)
    assert gr2.preset == "light"
    assert gr2.deanonymize is True
    assert gr2._languages() == ["en", "fr"]

    # a non-empty but content-free languages string falls back to ["en"].
    assert _make_guardrail(languages=", ,")._languages() == ["en"]


def test_event_hook_always_includes_pre_and_post():
    # A `mode: post_call` config must NOT disable pre_call anonymization.
    gr = _make_guardrail(event_hook="post_call")
    assert "pre_call" in gr.event_hook and "post_call" in gr.event_hook
    # A list missing post_call gets it added (and vice-versa).
    gr2 = _make_guardrail(event_hook=["pre_call"])
    assert "pre_call" in gr2.event_hook and "post_call" in gr2.event_hook
    # The default (no event_hook configured) still runs both hooks.
    gr3 = _make_guardrail()
    assert "pre_call" in gr3.event_hook and "post_call" in gr3.event_hook


@pytest.mark.asyncio
async def test_engine_is_cached():
    gr = _make_guardrail()
    engine_a = await gr._engine_for(["en"])
    engine_b = await gr._engine_for(["en"])
    assert engine_a is engine_b


@pytest.mark.asyncio
async def test_engine_downloads_spacy_models_on_oserror():
    _FakeEngine.raise_oserror_until = 1  # first initialize() raises, retry succeeds
    gr = _make_guardrail(languages="en,fr")
    engine = await gr._engine_for(["en", "fr"])
    assert isinstance(engine, _FakeEngine)
    assert _FakeEngine.init_calls == 2  # failed once, then succeeded after download


@pytest.mark.asyncio
async def test_engine_does_not_retry_non_model_oserror():
    # Only the missing-spaCy-model OSError may trigger the download retry;
    # a disk/permission/network OSError must propagate untouched.
    _FakeEngine.raise_oserror_until = 1
    _FakeEngine.oserror_message = "disk full"
    gr = _make_guardrail()
    with pytest.raises(OSError, match="disk full"):
        await gr._engine_for(["en"])
    assert _FakeEngine.init_calls == 1  # no retry, no download


@pytest.mark.asyncio
async def test_pre_call_no_messages_is_passthrough():
    gr = _make_guardrail()
    data = {"messages": []}
    assert await gr.async_pre_call_hook(None, None, data, "completion") is data


@pytest.mark.asyncio
async def test_pre_call_anonymizes_text_and_stashes_map():
    gr = _make_guardrail()
    messages = [{"role": "user", "content": "Email Marie Dupont at marie@acme.com"}]
    data = {"messages": messages}
    out = await gr.async_pre_call_hook(None, None, data, "completion")

    # Mutated IN PLACE (same list object), so the proxy's shallow body snapshot
    # ends up pointing at the anonymized messages, not the original raw-PII ones.
    assert out["messages"] is messages
    content = messages[0]["content"]
    assert "Marie Dupont" not in content
    assert "marie@acme.com" not in content
    assert "<PERSON_1>" in content and "<EMAIL_ADDRESS_1>" in content
    assert out["metadata"]["privaite_map"] == _FAKES


@pytest.mark.asyncio
async def test_pre_call_without_deanonymize_does_not_stash():
    gr = _make_guardrail(deanonymize=False)
    data = {"messages": [{"role": "user", "content": "Hi Marie Dupont"}]}
    out = await gr.async_pre_call_hook(None, None, data, "completion")
    assert "<PERSON_1>" in out["messages"][0]["content"]
    assert "privaite_map" not in out.get("metadata", {})


@pytest.mark.asyncio
async def test_pre_call_without_pii_does_not_stash():
    gr = _make_guardrail()
    data = {"messages": [{"role": "user", "content": "no personal data here"}]}
    out = await gr.async_pre_call_hook(None, None, data, "completion")
    assert "privaite_map" not in out.get("metadata", {})


@pytest.mark.asyncio
async def test_pre_call_background_request_anonymizes_but_does_not_stash():
    # Background results are fetched by a later poll the post-call hooks never
    # see, so the fake->original map must not persist in request state.
    gr = _make_guardrail()
    data = {
        "input": "Email Marie Dupont at marie@acme.com",
        "background": True,
    }
    out = await gr.async_pre_call_hook(None, None, data, "aresponses")
    assert "Marie Dupont" not in out["input"]
    assert "<PERSON_1>" in out["input"]
    assert "privaite_map" not in out.get("metadata", {})


@pytest.mark.asyncio
async def test_pre_call_clears_client_supplied_map():
    # metadata is caller-controlled: a client-supplied privaite_map must be
    # dropped, so it can never drive post-call restoration of model output.
    gr = _make_guardrail()
    data = {
        "messages": [{"role": "user", "content": "no personal data here"}],
        "metadata": {"privaite_map": {"<PERSON_1>": "attacker-chosen text"}},
    }
    out = await gr.async_pre_call_hook(None, None, data, "completion")
    assert "privaite_map" not in out["metadata"]


@pytest.mark.asyncio
async def test_post_call_restores_text_and_tool_call_args():
    gr = _make_guardrail()
    data = {"metadata": {"privaite_map": _FAKES}}

    tool_call = types.SimpleNamespace(function=types.SimpleNamespace(arguments='{"to": "<EMAIL_ADDRESS_1>"}'))
    # a tool_call with no function must be skipped, not crash.
    empty_tool_call = types.SimpleNamespace(function=None)
    message = types.SimpleNamespace(
        content="Sending to <PERSON_1>",
        tool_calls=[tool_call, empty_tool_call],
        function_call=types.SimpleNamespace(arguments='{"name": "<PERSON_1>"}'),
    )
    response = types.SimpleNamespace(
        choices=[
            types.SimpleNamespace(message=None),  # choice with no message is skipped
            types.SimpleNamespace(message=message),
        ]
    )

    out = await gr.async_post_call_success_hook(data, None, response)

    msg = out.choices[1].message
    assert msg.content == "Sending to Marie Dupont"
    assert msg.tool_calls[0].function.arguments == '{"to": "marie@acme.com"}'
    assert msg.function_call.arguments == '{"name": "Marie Dupont"}'
    # the map is consumed (popped) so it cannot be persisted to spend logs.
    assert "privaite_map" not in data["metadata"]


@pytest.mark.asyncio
async def test_post_call_deanonymize_false_skips_restore():
    gr = _make_guardrail(deanonymize=False)
    data = {"metadata": {"privaite_map": _FAKES}}
    message = types.SimpleNamespace(content="Sending to <PERSON_1>", tool_calls=None, function_call=None)
    response = types.SimpleNamespace(choices=[types.SimpleNamespace(message=message)])

    out = await gr.async_post_call_success_hook(data, None, response)
    assert out.choices[0].message.content == "Sending to <PERSON_1>"


@pytest.mark.asyncio
async def test_post_call_without_map_is_passthrough():
    gr = _make_guardrail()
    response = types.SimpleNamespace(choices=[])
    assert await gr.async_post_call_success_hook({}, None, response) is response


@pytest.mark.asyncio
async def test_streaming_restores_content():
    gr = _make_guardrail()
    request_data = {"metadata": {"privaite_map": _FAKES}}

    async def _source():
        # a choice with no delta must be skipped, not crash.
        yield types.SimpleNamespace(choices=[types.SimpleNamespace(delta=None, finish_reason=None)])
        yield types.SimpleNamespace(
            choices=[
                types.SimpleNamespace(
                    delta=types.SimpleNamespace(content="Hi <PERSON_1>"),
                    finish_reason=None,
                )
            ]
        )
        yield types.SimpleNamespace(
            choices=[
                types.SimpleNamespace(
                    delta=types.SimpleNamespace(content=""),
                    finish_reason="stop",
                )
            ]
        )

    chunks = await _collect(gr.async_post_call_streaming_iterator_hook(None, _source(), request_data))
    assert chunks[1].choices[0].delta.content == "Hi Marie Dupont"


@pytest.mark.asyncio
async def test_streaming_terminal_none_content_stays_none():
    # A provider finish chunk commonly carries finish_reason with content=None;
    # with nothing held back, restoration must not mutate that None into "".
    gr = _make_guardrail()
    request_data = {"metadata": {"privaite_map": _FAKES}}

    async def _source():
        yield types.SimpleNamespace(
            choices=[
                types.SimpleNamespace(
                    delta=types.SimpleNamespace(content="Hi <PERSON_1>"),
                    finish_reason=None,
                )
            ]
        )
        yield types.SimpleNamespace(
            choices=[
                types.SimpleNamespace(
                    delta=types.SimpleNamespace(content=None),
                    finish_reason="stop",
                )
            ]
        )

    chunks = await _collect(gr.async_post_call_streaming_iterator_hook(None, _source(), request_data))
    assert chunks[0].choices[0].delta.content == "Hi Marie Dupont"
    assert chunks[1].choices[0].delta.content is None


@pytest.mark.asyncio
async def test_streaming_passthrough_when_disabled():
    gr = _make_guardrail(deanonymize=False)

    async def _source():
        yield types.SimpleNamespace(choices=[])

    chunks = await _collect(gr.async_post_call_streaming_iterator_hook(None, _source(), {}))
    assert len(chunks) == 1


@pytest.mark.asyncio
async def test_streaming_isolates_buffers_per_choice():
    # With n>1 the provider interleaves single-choice chunks for different
    # indices. A placeholder split across choice 0's chunks must not be
    # corrupted by choice 1's interleaved bytes (one buffer per choice index).
    gr = _make_guardrail()
    request_data = {"metadata": {"privaite_map": _FAKES}}

    def _chunk(index, content, finish=None):
        return types.SimpleNamespace(
            choices=[
                types.SimpleNamespace(
                    index=index,
                    delta=types.SimpleNamespace(content=content),
                    finish_reason=finish,
                )
            ]
        )

    async def _source():
        yield _chunk(0, "Hi <PER")  # placeholder starts, held in choice-0 buffer
        yield _chunk(1, "Bye")  # interleaved choice-1 content
        yield _chunk(0, "SON_1>", finish="stop")  # completes choice-0 placeholder
        yield _chunk(1, "", finish="stop")

    chunks = await _collect(gr.async_post_call_streaming_iterator_hook(None, _source(), request_data))

    restored = {}
    for chunk in chunks:
        for choice in chunk.choices:
            restored.setdefault(choice.index, "")
            if choice.delta.content:
                restored[choice.index] += choice.delta.content

    assert restored[0] == "Hi Marie Dupont"
    assert restored[1] == "Bye"


@pytest.mark.asyncio
async def test_streaming_restores_reasoning_content():
    # a reasoning model echoes a placeholder in its streamed reasoning trace; it
    # must be de-anonymized like content.
    gr = _make_guardrail()
    request_data = {"metadata": {"privaite_map": _FAKES}}

    def _chunk(reasoning, finish=None):
        return types.SimpleNamespace(
            choices=[
                types.SimpleNamespace(
                    index=0,
                    delta=types.SimpleNamespace(
                        content=None, tool_calls=None, function_call=None, reasoning_content=reasoning
                    ),
                    finish_reason=finish,
                )
            ]
        )

    async def _source():
        yield _chunk("the user is <PERSON_1>")
        yield _chunk(None, finish="stop")

    chunks = await _collect(gr.async_post_call_streaming_iterator_hook(None, _source(), request_data))
    trace = "".join(
        getattr(choice.delta, "reasoning_content", None) or "" for chunk in chunks for choice in chunk.choices
    )
    assert "Marie Dupont" in trace
    assert "<PERSON_1>" not in trace


@pytest.mark.asyncio
async def test_streaming_restores_tool_and_function_call_arguments():
    # Streamed tool-call and legacy function_call argument fragments must be
    # de-anonymized too, with a placeholder split across chunks reassembled per
    # tool_call index. A tool_call with no function is skipped, not crashed.
    gr = _make_guardrail()
    request_data = {"metadata": {"privaite_map": _FAKES}}

    async def _source():
        yield types.SimpleNamespace(
            choices=[
                types.SimpleNamespace(
                    index=0,
                    delta=types.SimpleNamespace(
                        content=None,
                        tool_calls=[
                            types.SimpleNamespace(
                                index=0,
                                function=types.SimpleNamespace(arguments='{"to": "<EMAIL_ADDRESS'),
                            ),
                            types.SimpleNamespace(index=1, function=None),
                        ],
                        function_call=types.SimpleNamespace(arguments='{"n": "<PER'),
                    ),
                    finish_reason=None,
                )
            ]
        )
        yield types.SimpleNamespace(
            choices=[
                types.SimpleNamespace(
                    index=0,
                    delta=types.SimpleNamespace(
                        content=None,
                        tool_calls=[
                            types.SimpleNamespace(
                                index=0,
                                function=types.SimpleNamespace(arguments='_1>"}'),
                            )
                        ],
                        function_call=types.SimpleNamespace(arguments='SON_1>"}'),
                    ),
                    finish_reason="stop",
                )
            ]
        )

    chunks = await _collect(gr.async_post_call_streaming_iterator_hook(None, _source(), request_data))

    tool_args = "".join(
        tc.function.arguments
        for chunk in chunks
        for choice in chunk.choices
        for tc in (choice.delta.tool_calls or [])
        if tc.function is not None and tc.function.arguments
    )
    fc_args = "".join(
        choice.delta.function_call.arguments
        for chunk in chunks
        for choice in chunk.choices
        if choice.delta.function_call is not None and choice.delta.function_call.arguments
    )
    assert tool_args == '{"to": "marie@acme.com"}'
    assert fc_args == '{"n": "Marie Dupont"}'


# --- Responses API (/v1/responses) ---


@pytest.mark.asyncio
async def test_pre_call_anonymizes_responses_string_input_and_fixes_snapshot():
    gr = _make_guardrail()
    # the proxy takes a shallow snapshot of the body before this hook runs
    body = {"input": "Email Marie Dupont at marie@acme.com"}
    data = {
        "input": "Email Marie Dupont at marie@acme.com",
        "proxy_server_request": {"body": body},
    }
    out = await gr.async_pre_call_hook(None, None, data, "aresponses")

    assert "Marie Dupont" not in out["input"]
    assert "<PERSON_1>" in out["input"]
    # the string-rebind would otherwise leave raw PII in the snapshot body
    assert body["input"] == out["input"]
    assert "Marie Dupont" not in body["input"]
    assert out["metadata"]["privaite_map"] == _FAKES


@pytest.mark.asyncio
async def test_pre_call_anonymizes_responses_role_message_list_in_place():
    gr = _make_guardrail()
    input_value = [{"role": "user", "content": "Hi Marie Dupont"}]
    data = {"input": input_value}
    out = await gr.async_pre_call_hook(None, None, data, "aresponses")
    # mutated in place (same list object)
    assert out["input"] is input_value
    assert "<PERSON_1>" in input_value[0]["content"]
    assert "Marie Dupont" not in input_value[0]["content"]


@pytest.mark.asyncio
async def test_pre_call_anonymizes_responses_content_parts_in_place():
    gr = _make_guardrail()
    input_value = [{"type": "input_text", "text": "Hi Marie Dupont"}]
    data = {"input": input_value}
    out = await gr.async_pre_call_hook(None, None, data, "aresponses")
    assert out["input"] is input_value
    assert input_value[0]["text"] == "Hi <PERSON_1>"


@pytest.mark.asyncio
async def test_pre_call_anonymizes_both_messages_and_input():
    # a crafted /v1/responses body with decoy messages + PII in input: BOTH must
    # be anonymized under one shared mapping (input must not be skipped).
    gr = _make_guardrail()
    messages = [{"role": "user", "content": "Email Marie Dupont"}]
    data = {"messages": messages, "input": "also reach marie@acme.com"}
    out = await gr.async_pre_call_hook(None, None, data, "aresponses")

    assert "Marie Dupont" not in messages[0]["content"]
    assert "<PERSON_1>" in messages[0]["content"]
    assert "marie@acme.com" not in out["input"]
    assert "<EMAIL_ADDRESS_1>" in out["input"]
    # one consistent map covering both sources
    assert out["metadata"]["privaite_map"] == _FAKES


@pytest.mark.asyncio
async def test_pre_call_unhandled_input_shape_is_passthrough():
    # an unexpected input shape (neither str nor list) is left untouched.
    gr = _make_guardrail()
    data = {"input": {"unexpected": "shape"}}
    out = await gr.async_pre_call_hook(None, None, data, "aresponses")
    assert "privaite_map" not in out.get("metadata", {})


@pytest.mark.asyncio
async def test_pre_call_scans_mixed_responses_input_list():
    # an agentic turn: a role message + a function_call_output + a bare string.
    # the old homogeneity check wrapped the whole list as one content and left
    # the non-message items raw; every item must be scanned.
    gr = _make_guardrail()
    data = {
        "input": [
            {"role": "user", "content": "I am Marie Dupont"},
            {"type": "function_call_output", "call_id": "c1", "output": "reach marie@acme.com"},
            "also Marie Dupont",
        ]
    }
    out = await gr.async_pre_call_hook(None, None, data, "aresponses")
    serialized = str(out["input"])

    assert "Marie Dupont" not in serialized
    assert "marie@acme.com" not in serialized
    assert out["input"][1]["type"] == "function_call_output"  # structure kept
    assert out["input"][1]["call_id"] == "c1"
    assert out["metadata"]["privaite_map"]


@pytest.mark.asyncio
async def test_pre_call_block_gate_fires_on_mixed_responses_input():
    # a blocked type inside a function_call_output must reject the request; the
    # gate is only reachable because the item is scanned in the first place.
    from fastapi import HTTPException

    gr = _make_guardrail(block_entities=["EMAIL_ADDRESS"])
    data = {"input": [{"type": "function_call_output", "output": "reach marie@acme.com"}]}
    with pytest.raises(HTTPException) as ei:
        await gr.async_pre_call_hook(None, None, data, "aresponses")
    assert ei.value.status_code == 400


@pytest.mark.asyncio
async def test_post_call_restores_responses_output_dict_and_object():
    gr = _make_guardrail()
    data = {"metadata": {"privaite_map": _FAKES}}

    # output item as a dict (message with output_text content)
    dict_item = {
        "type": "message",
        "content": [{"type": "output_text", "text": "Hello <PERSON_1>"}],
    }
    # output item as an object (function_call with arguments)
    obj_item = types.SimpleNamespace(
        type="function_call",
        content=None,
        arguments='{"to": "<EMAIL_ADDRESS_1>"}',
    )
    response = types.SimpleNamespace(output=[dict_item, obj_item])

    out = await gr.async_post_call_success_hook(data, None, response)

    assert out.output[0]["content"][0]["text"] == "Hello Marie Dupont"
    assert out.output[1].arguments == '{"to": "marie@acme.com"}'
    assert "privaite_map" not in data["metadata"]


@pytest.mark.asyncio
async def test_post_call_failure_hook_pops_map():
    # the success/streaming hooks never run on failure; the failure hook must
    # still drop the reversible map so it cannot reach a failure spend-log.
    gr = _make_guardrail()
    rd = {"metadata": {"privaite_map": dict(_FAKES), "other": 1}}
    assert await gr.async_post_call_failure_hook(rd, RuntimeError("x"), None) is None
    assert "privaite_map" not in rd["metadata"]
    assert rd["metadata"]["other"] == 1
    # missing metadata and a non-dict request_data are safe no-ops
    assert await gr.async_post_call_failure_hook({}, RuntimeError("x"), None) is None
    assert await gr.async_post_call_failure_hook(None, RuntimeError("x"), None) is None


def test_block_entities_config_model_field():
    from litellm.types.proxy.guardrails.guardrail_hooks.privaite import (
        PrivaiteGuardrailConfigModel,
    )

    assert PrivaiteGuardrailConfigModel.model_fields["block_entities"].default is None


def test_block_entities_parsed_from_string_and_list():
    assert _make_guardrail(block_entities="US_SSN, CREDIT_CARD").block_entities == [
        "US_SSN",
        "CREDIT_CARD",
    ]
    assert _make_guardrail(block_entities=["EMAIL_ADDRESS"]).block_entities == ["EMAIL_ADDRESS"]
    assert _make_guardrail().block_entities == []


def test_initialize_guardrail_passes_block_entities():
    from litellm.proxy.guardrails.guardrail_hooks.privaite import initialize_guardrail

    litellm_params = types.SimpleNamespace(
        mode="pre_call",
        default_on=False,
        preset="light",
        languages="en",
        deanonymize=True,
        block_entities=["US_SSN"],
    )
    callback = initialize_guardrail(litellm_params, {"guardrail_name": "privaite"})
    assert callback.block_entities == ["US_SSN"]


@pytest.mark.asyncio
async def test_block_entities_rejects_with_400():
    from fastapi import HTTPException

    gr = _make_guardrail(block_entities=["EMAIL_ADDRESS"])
    data = {"messages": [{"role": "user", "content": "reach marie@acme.com"}]}
    with pytest.raises(HTTPException) as ei:
        await gr.async_pre_call_hook(None, None, data, "completion")

    assert ei.value.status_code == 400
    detail = ei.value.detail
    msg = detail["error"] if isinstance(detail, dict) else str(detail)
    assert "EMAIL_ADDRESS" in msg
    assert "marie@acme.com" not in msg  # the value never leaks into the error
    assert "privaite_map" not in (data.get("metadata") or {})  # nothing forwarded


@pytest.mark.asyncio
async def test_block_entities_ignores_types_not_present():
    # a blocked type that is absent must not disturb a normal request: the other
    # PII is still masked and the request goes through with a restore map.
    gr = _make_guardrail(block_entities=["US_SSN"])
    data = {"messages": [{"role": "user", "content": "I am Marie Dupont, marie@acme.com"}]}
    out = await gr.async_pre_call_hook(None, None, data, "completion")

    serialized = str(out["messages"])
    assert "Marie Dupont" not in serialized
    assert "marie@acme.com" not in serialized
    assert out.get("metadata", {}).get("privaite_map")


@pytest.mark.asyncio
async def test_block_entities_fails_closed_when_privaite_too_old(monkeypatch):
    # simulate an older privaite whose PIIConfig has no block_entities field:
    # extra="allow" would swallow it silently, so the guardrail must refuse.
    schema = sys.modules["privaite.config.schema"]
    monkeypatch.setattr(schema.PIIConfig, "model_fields", {"preset": None, "languages": None})

    gr = _make_guardrail(block_entities=["EMAIL_ADDRESS"])
    data = {"messages": [{"role": "user", "content": "marie@acme.com"}]}
    with pytest.raises(RuntimeError, match="block_entities"):
        await gr.async_pre_call_hook(None, None, data, "completion")


# --- Responses API: tool output list-of-parts, typed carriers, prompt variables ---


@pytest.mark.asyncio
async def test_pre_call_scans_custom_tool_call_output_list_of_parts():
    # a shell/custom tool that read a file returns its bytes as a list of
    # {type, text} parts (not a bare string); every text leaf must be scrubbed
    # while the binary image part is relayed whole (base64 carries nothing a
    # text detector can find, and rewriting it would corrupt the payload).
    gr = _make_guardrail()
    output = [
        {"type": "output_text", "text": "the file says reach marie@acme.com"},
        {"type": "input_image", "image_url": "data:image/png;base64,QUJD"},
    ]
    data = {"input": [{"type": "custom_tool_call_output", "call_id": "c1", "output": output}]}
    out = await gr.async_pre_call_hook(None, None, data, "aresponses")

    scrubbed = out["input"][0]["output"]
    assert scrubbed[0]["text"] == "the file says reach <EMAIL_ADDRESS_1>"
    assert scrubbed[1]["image_url"] == "data:image/png;base64,QUJD"  # binary relayed whole
    assert out["input"][0]["call_id"] == "c1"
    assert out["metadata"]["privaite_map"]


@pytest.mark.asyncio
async def test_pre_call_scans_typed_action_carrier():
    # a file_search_call carries user data in its `queries`/`results` fields, not
    # in `content`/`output`; the old field scan never looked there.
    gr = _make_guardrail()
    data = {
        "input": [
            {
                "type": "file_search_call",
                "id": "fs1",
                "queries": ["records for Marie Dupont", "orders by marie@acme.com"],
                "results": [{"text": "found Marie Dupont", "score": 0.9}],
            }
        ]
    }
    out = await gr.async_pre_call_hook(None, None, data, "aresponses")
    item = out["input"][0]
    assert item["queries"] == ["records for <PERSON_1>", "orders by <EMAIL_ADDRESS_1>"]
    assert item["results"][0]["text"] == "found <PERSON_1>"
    assert item["results"][0]["score"] == 0.9  # non-text leaf untouched
    assert item["id"] == "fs1"


@pytest.mark.asyncio
async def test_pre_call_scans_prompt_variables():
    # Responses prompt-template variables carry user data; the template id/version
    # do not. A variable can be a bare string or a typed content part.
    gr = _make_guardrail()
    data = {
        "prompt": {
            "id": "pmpt_123",
            "version": "2",
            "variables": {
                "customer": "Marie Dupont",
                "note": {"type": "input_text", "text": "email marie@acme.com"},
            },
        }
    }
    out = await gr.async_pre_call_hook(None, None, data, "aresponses")
    variables = out["prompt"]["variables"]
    assert variables["customer"] == "<PERSON_1>"
    assert variables["note"]["text"] == "email <EMAIL_ADDRESS_1>"
    assert out["prompt"]["id"] == "pmpt_123"  # template id is not user data
    assert out["prompt"]["version"] == "2"
    assert out["metadata"]["privaite_map"]


@pytest.mark.asyncio
async def test_pre_call_block_gate_fires_on_prompt_variables():
    # the block gate must reach a blocked type sitting only in prompt.variables;
    # it is only reachable because the variable is scanned in the first place.
    from fastapi import HTTPException

    gr = _make_guardrail(block_entities=["EMAIL_ADDRESS"])
    data = {"prompt": {"id": "pmpt_1", "variables": {"to": "marie@acme.com"}}}
    with pytest.raises(HTTPException) as ei:
        await gr.async_pre_call_hook(None, None, data, "aresponses")
    assert ei.value.status_code == 400


# --- Auxiliary request fields: completions prompt/suffix, prediction, user_location ---


@pytest.mark.asyncio
async def test_pre_call_scans_completions_prompt_string_and_suffix_and_fixes_snapshot():
    gr = _make_guardrail()
    body = {"prompt": "Complete for Marie Dupont", "suffix": "signed marie@acme.com"}
    data = {
        "prompt": "Complete for Marie Dupont",
        "suffix": "signed marie@acme.com",
        "proxy_server_request": {"body": body},
    }
    out = await gr.async_pre_call_hook(None, None, data, "text_completion")

    assert out["prompt"] == "Complete for <PERSON_1>"
    assert out["suffix"] == "signed <EMAIL_ADDRESS_1>"
    # the detached snapshot copy must be overwritten too, or raw PII leaks through it
    assert body["prompt"] == out["prompt"]
    assert body["suffix"] == out["suffix"]
    assert "Marie Dupont" not in body["prompt"]
    assert out["metadata"]["privaite_map"]


@pytest.mark.asyncio
async def test_pre_call_scans_completions_prompt_batch_list_and_skips_tokens():
    # the /v1/completions batch shape: string leaves are scrubbed; a tokenized
    # (integer-array) prompt passes through unscanned as documented.
    gr = _make_guardrail()
    prompt = ["reach marie@acme.com", [1, 15, 27]]
    data = {"prompt": prompt}
    out = await gr.async_pre_call_hook(None, None, data, "text_completion")
    assert out["prompt"][0] == "reach <EMAIL_ADDRESS_1>"
    assert out["prompt"][1] == [1, 15, 27]  # token array untouched


@pytest.mark.asyncio
async def test_pre_call_scans_prediction_content():
    # chat predicted outputs carry the client's current document verbatim.
    gr = _make_guardrail()
    prediction = {"type": "content", "content": "draft addressed to Marie Dupont"}
    data = {"messages": [{"role": "user", "content": "edit this"}], "prediction": prediction}
    out = await gr.async_pre_call_hook(None, None, data, "completion")
    # prediction dict is aliased by the body snapshot, so it is rewritten in place
    assert prediction["content"] == "draft addressed to <PERSON_1>"
    assert out["metadata"]["privaite_map"]


@pytest.mark.asyncio
async def test_pre_call_scans_web_search_user_location():
    gr = _make_guardrail()
    web_search = {"user_location": {"type": "approximate", "approximate": {"city": "Marie Dupont"}}}
    data = {"messages": [{"role": "user", "content": "weather?"}], "web_search_options": web_search}
    await gr.async_pre_call_hook(None, None, data, "completion")
    assert web_search["user_location"]["approximate"]["city"] == "<PERSON_1>"


@pytest.mark.asyncio
async def test_pre_call_only_aux_field_is_not_skipped():
    # a request whose ONLY user text is an auxiliary field (no messages/input)
    # must not be short-circuited by the pre-call early return.
    gr = _make_guardrail()
    data = {"suffix": "signed marie@acme.com"}
    out = await gr.async_pre_call_hook(None, None, data, "text_completion")
    assert out["suffix"] == "signed <EMAIL_ADDRESS_1>"
    assert out["metadata"]["privaite_map"]


# --- Restore parity: refusal and audio transcript (non-streaming + streaming) ---


@pytest.mark.asyncio
async def test_post_call_restores_refusal_and_audio_transcript():
    gr = _make_guardrail()
    data = {"metadata": {"privaite_map": _FAKES}}

    # audio as an object (transcript attribute) and refusal quoting the request.
    obj_msg = types.SimpleNamespace(
        content=None,
        tool_calls=None,
        function_call=None,
        refusal="I won't email <PERSON_1>",
        audio=types.SimpleNamespace(transcript="calling <PERSON_1> now"),
    )
    # audio as a dict on a second choice.
    dict_msg = types.SimpleNamespace(
        content=None,
        tool_calls=None,
        function_call=None,
        audio={"transcript": "reach <EMAIL_ADDRESS_1>"},
    )
    response = types.SimpleNamespace(
        choices=[types.SimpleNamespace(message=obj_msg), types.SimpleNamespace(message=dict_msg)]
    )
    out = await gr.async_post_call_success_hook(data, None, response)

    assert out.choices[0].message.refusal == "I won't email Marie Dupont"
    assert out.choices[0].message.audio.transcript == "calling Marie Dupont now"
    assert out.choices[1].message.audio["transcript"] == "reach marie@acme.com"


@pytest.mark.asyncio
async def test_streaming_restores_refusal():
    gr = _make_guardrail()
    request_data = {"metadata": {"privaite_map": _FAKES}}

    def _chunk(refusal, finish=None):
        return types.SimpleNamespace(
            choices=[
                types.SimpleNamespace(
                    index=0,
                    delta=types.SimpleNamespace(content=None, tool_calls=None, function_call=None, refusal=refusal),
                    finish_reason=finish,
                )
            ]
        )

    async def _source():
        yield _chunk("I won't reach <PER")  # placeholder split across chunks
        yield _chunk("SON_1>", finish="stop")

    chunks = await _collect(gr.async_post_call_streaming_iterator_hook(None, _source(), request_data))
    refusal = "".join(getattr(c.choices[0].delta, "refusal", None) or "" for c in chunks)
    assert refusal == "I won't reach Marie Dupont"


@pytest.mark.asyncio
async def test_streaming_restores_audio_transcript():
    # streamed audio transcript fragments are de-anonymized with a placeholder
    # split across chunks reassembled in the audio-segment buffer.
    gr = _make_guardrail()
    request_data = {"metadata": {"privaite_map": _FAKES}}

    def _chunk(transcript, finish=None):
        return types.SimpleNamespace(
            choices=[
                types.SimpleNamespace(
                    index=0,
                    delta=types.SimpleNamespace(
                        content=None,
                        tool_calls=None,
                        function_call=None,
                        audio=types.SimpleNamespace(transcript=transcript),
                    ),
                    finish_reason=finish,
                )
            ]
        )

    async def _source():
        yield _chunk("Hi <PER")
        yield _chunk("SON_1>", finish="stop")

    chunks = await _collect(gr.async_post_call_streaming_iterator_hook(None, _source(), request_data))
    transcript = "".join(c.choices[0].delta.audio.transcript or "" for c in chunks if c.choices[0].delta.audio)
    assert transcript == "Hi Marie Dupont"


def _bare_delta(**fields):
    """A streamed delta carrying only the given fields (the others None), the
    shape of the finish chunk most providers send: delta {} + finish_reason."""
    base = dict(content=None, tool_calls=None, function_call=None)
    base.update(fields)
    return types.SimpleNamespace(**base)


def _choice_chunk(delta, finish=None, index=0):
    return types.SimpleNamespace(choices=[types.SimpleNamespace(index=index, delta=delta, finish_reason=finish)])


def _get(obj, *path):
    """Walk dict-or-object attributes; the guardrail may create a carrier as a
    plain dict on a finish delta that had none."""
    for key in path:
        if obj is None:
            return None
        obj = obj.get(key) if isinstance(obj, dict) else getattr(obj, key, None)
    return obj


# The restorer holds back a tail that could begin a placeholder ("<PER" here),
# so a channel ending on one loses it unless the end of the stream flushes it.


@pytest.mark.parametrize("field", ["reasoning_content", "reasoning", "refusal"])
async def test_streaming_flushes_text_field_tail_onto_a_bare_finish_chunk(field):
    gr = _make_guardrail()
    request_data = {"metadata": {"privaite_map": _FAKES}}

    async def _source():
        yield _choice_chunk(_bare_delta(**{field: "Considering <PERSON_1>, then <PER"}))
        yield _choice_chunk(_bare_delta(**{field: None}), finish="stop")

    chunks = await _collect(gr.async_post_call_streaming_iterator_hook(None, _source(), request_data))
    out = "".join(getattr(choice.delta, field, None) or "" for chunk in chunks for choice in chunk.choices)
    assert out == "Considering Marie Dupont, then <PER"


async def test_streaming_flushes_tool_tail_onto_a_bare_finish_chunk():
    # A provider that stops mid-argument (finish_reason "length") sends the
    # finish on a chunk with no tool_calls slot. The held fragment must still
    # go out, on that chunk, under the same tool_call index: the client
    # reassembles arguments per index and must see every byte it was sent.
    gr = _make_guardrail()
    request_data = {"metadata": {"privaite_map": _FAKES}}

    async def _source():
        call = types.SimpleNamespace(index=2, function=types.SimpleNamespace(arguments='{"to": "<EMAIL'))
        yield _choice_chunk(_bare_delta(tool_calls=[call]))
        yield _choice_chunk(_bare_delta(), finish="length")

    chunks = await _collect(gr.async_post_call_streaming_iterator_hook(None, _source(), request_data))
    args = {}
    for chunk in chunks:
        for choice in chunk.choices:
            for tc in choice.delta.tool_calls or []:
                slot = _get(tc, "index")
                args[slot] = args.get(slot, "") + (_get(tc, "function", "arguments") or "")
    assert args == {2: '{"to": "<EMAIL'}
    assert chunks[-1].choices[0].finish_reason == "length"
    assert chunks[-1].choices[0].delta.tool_calls


async def test_streaming_flushes_function_call_tail_onto_a_bare_finish_chunk():
    gr = _make_guardrail()
    request_data = {"metadata": {"privaite_map": _FAKES}}

    async def _source():
        yield _choice_chunk(_bare_delta(function_call=types.SimpleNamespace(arguments='{"n": "<PER')))
        yield _choice_chunk(_bare_delta(), finish="length")

    chunks = await _collect(gr.async_post_call_streaming_iterator_hook(None, _source(), request_data))
    out = "".join(
        _get(choice.delta, "function_call", "arguments") or "" for chunk in chunks for choice in chunk.choices
    )
    assert out == '{"n": "<PER'


async def test_streaming_drains_held_tails_when_the_stream_ends_without_finish_reason():
    # Some providers close the stream without ever sending finish_reason. What
    # the buffers still hold goes out as trailing chunks instead of vanishing
    # with the generator: one per held channel, cloned from the last chunk of
    # its choice, without that chunk's usage and without a finish_reason.
    gr = _make_guardrail()
    request_data = {"metadata": {"privaite_map": _FAKES}}

    async def _source():
        first = _choice_chunk(_bare_delta(content="Ask <PERSON_1> or <PER", audio={"transcript": "Call <EMAIL"}))
        first.usage = types.SimpleNamespace(total_tokens=3)
        yield first
        call = types.SimpleNamespace(index=0, function=types.SimpleNamespace(arguments='{"to": "<EMAIL'))
        yield _choice_chunk(_bare_delta(tool_calls=[call]), index=1)

    chunks = await _collect(gr.async_post_call_streaming_iterator_hook(None, _source(), request_data))
    content = audio = args = ""
    for chunk in chunks:
        for choice in chunk.choices:
            assert choice.finish_reason is None
            if choice.index == 0:
                content += choice.delta.content or ""
                audio += _get(choice.delta, "audio", "transcript") or ""
            else:
                for tc in choice.delta.tool_calls or []:
                    args += _get(tc, "function", "arguments") or ""
    assert content == "Ask Marie Dupont or <PER"
    assert audio == "Call <EMAIL"
    assert args == '{"to": "<EMAIL'
    assert chunks[0].usage.total_tokens == 3
    assert all(getattr(chunk, "usage", None) is None for chunk in chunks[2:])
    assert len(chunks) == 5  # 2 from the provider + content, audio and tool tails


async def test_streaming_finish_chunk_with_empty_slots_gets_the_tails_under_them():
    # A finish chunk may carry the tool_call slot and the function_call holder
    # with empty arguments. Their restore skips an empty fragment, so the sweep
    # appends the held tail under the existing holder: no duplicate slot.
    gr = _make_guardrail()
    request_data = {"metadata": {"privaite_map": _FAKES}}

    def _slots(args, fc_args):
        return _bare_delta(
            tool_calls=[types.SimpleNamespace(index=2, function=types.SimpleNamespace(arguments=args))],
            function_call=types.SimpleNamespace(arguments=fc_args),
        )

    async def _source():
        yield _choice_chunk(_slots('{"to": "<EMAIL', '{"n": "<PER'))
        yield _choice_chunk(_slots("", ""), finish="length")

    chunks = await _collect(gr.async_post_call_streaming_iterator_hook(None, _source(), request_data))
    finish = chunks[-1].choices[0].delta
    assert len(finish.tool_calls) == 1
    assert finish.tool_calls[0].index == 2
    assert finish.tool_calls[0].function.arguments == "<EMAIL"
    assert finish.function_call.arguments == "<PER"


async def test_streaming_tails_on_real_litellm_chunks_stay_serializable():
    # The same two shapes on litellm's own chunk types: a carrier created on a
    # bare finish Delta and a drained clone must both still serialize the way
    # the proxy does it (model_dump_json).
    from litellm.types.utils import ChatCompletionDeltaToolCall, Delta, Function, ModelResponseStream, StreamingChoices

    gr = _make_guardrail()

    def _chunk(delta, finish=None):
        choice = StreamingChoices(index=0, delta=delta, finish_reason=finish)
        return ModelResponseStream(id="chunk", model="m", choices=[choice])

    async def _source():
        call = ChatCompletionDeltaToolCall(index=0, type="function", function=Function(arguments='{"to": "<EMAIL'))
        yield _chunk(Delta(content=None, tool_calls=[call]))
        yield _chunk(Delta(content=None), finish="length")

    async def _no_finish():
        yield _chunk(Delta(content=None, reasoning_content="then <PER"))

    request_data = {"metadata": {"privaite_map": _FAKES}}
    chunks = await _collect(gr.async_post_call_streaming_iterator_hook(None, _source(), request_data))
    payloads = [json.loads(chunk.model_dump_json(exclude_none=True)) for chunk in chunks]
    finish = payloads[-1]["choices"][0]
    assert finish["finish_reason"] == "length"
    assert finish["delta"]["tool_calls"][0]["index"] == 0
    assert finish["delta"]["tool_calls"][0]["function"]["arguments"] == "<EMAIL"

    request_data = {"metadata": {"privaite_map": _FAKES}}
    chunks = await _collect(gr.async_post_call_streaming_iterator_hook(None, _no_finish(), request_data))
    payloads = [json.loads(chunk.model_dump_json(exclude_none=True)) for chunk in chunks]
    assert [p["choices"][0]["delta"].get("reasoning_content") for p in payloads] == ["then ", "<PER"]
    assert "finish_reason" not in payloads[-1]["choices"][0]
    assert payloads[-1]["id"] == "chunk"


# An original carrying the characters a JSON string literal must escape.
_SPECIALS_ORIGINAL = 'Jean "JJ" O\'Neil\\n\tnext'


async def test_post_call_restores_tool_arguments_as_valid_json(monkeypatch):
    # A restored original holding a quote, a backslash or a newline must be
    # spliced JSON-escaped, on every argument carrier: tool_calls, the legacy
    # function_call and a Responses function_call output item. A plain
    # substitution would leave the client's json.loads failing on the quote.
    monkeypatch.setitem(_FAKES, "<PERSON_2>", _SPECIALS_ORIGINAL)
    gr = _make_guardrail()
    data = {"metadata": {"privaite_map": _FAKES}}
    message = types.SimpleNamespace(
        content=None,
        tool_calls=[
            types.SimpleNamespace(function=types.SimpleNamespace(arguments='{"name": "<PERSON_2>", "n": 1}')),
            # not JSON: the plain string restore is all there is, byte for byte
            types.SimpleNamespace(function=types.SimpleNamespace(arguments="name=<PERSON_2>")),
        ],
        function_call=types.SimpleNamespace(arguments='["<PERSON_2>", {"k": "<PERSON_1>"}]'),
    )
    response = types.SimpleNamespace(
        choices=[types.SimpleNamespace(message=message)],
        output=[{"type": "function_call", "arguments": '{"who": "<PERSON_2>"}'}],
    )

    out = await gr.async_post_call_success_hook(data, None, response)

    msg = out.choices[0].message
    assert json.loads(msg.tool_calls[0].function.arguments) == {"name": _SPECIALS_ORIGINAL, "n": 1}
    assert msg.tool_calls[1].function.arguments == f"name={_SPECIALS_ORIGINAL}"
    assert json.loads(msg.function_call.arguments) == [_SPECIALS_ORIGINAL, {"k": "Marie Dupont"}]
    assert json.loads(out.output[0]["arguments"]) == {"who": _SPECIALS_ORIGINAL}


async def test_post_call_untouched_arguments_are_kept_byte_for_byte():
    # No placeholder in the arguments: the string is returned as received, not
    # re-encoded (key order, spacing and escapes stay the provider's).
    gr = _make_guardrail()
    data = {"metadata": {"privaite_map": _FAKES}}
    raw = '{"b":1,\n "a": "x\\u00e9"}'
    message = types.SimpleNamespace(
        content=None,
        tool_calls=[types.SimpleNamespace(function=types.SimpleNamespace(arguments=raw))],
        function_call=None,
    )
    response = types.SimpleNamespace(choices=[types.SimpleNamespace(message=message)])
    out = await gr.async_post_call_success_hook(data, None, response)
    assert out.choices[0].message.tool_calls[0].function.arguments == raw


async def test_streaming_tool_arguments_stay_valid_json_with_special_chars(monkeypatch):
    # Streamed argument fragments are JSON source text: the original spliced in
    # must be its JSON-escaped form, so the reassembled arguments still parse.
    # Content keeps the raw original. The placeholder is split across chunks.
    monkeypatch.setitem(_FAKES, "<PERSON_2>", _SPECIALS_ORIGINAL)
    gr = _make_guardrail()
    request_data = {"metadata": {"privaite_map": _FAKES}}

    def _delta(args, content=None):
        call = types.SimpleNamespace(index=0, function=types.SimpleNamespace(arguments=args))
        return _bare_delta(content=content, tool_calls=[call], function_call=types.SimpleNamespace(arguments=args))

    async def _source():
        yield _choice_chunk(_delta('{"name": "<PERS', content="Hi <PERS"))
        yield _choice_chunk(_delta('ON_2>"}', content="ON_2>"), finish="stop")

    chunks = await _collect(gr.async_post_call_streaming_iterator_hook(None, _source(), request_data))
    tool_args = "".join(
        tc.function.arguments for chunk in chunks for choice in chunk.choices for tc in (choice.delta.tool_calls or [])
    )
    fc_args = "".join(choice.delta.function_call.arguments for chunk in chunks for choice in chunk.choices)
    content = "".join(choice.delta.content or "" for chunk in chunks for choice in chunk.choices)
    assert json.loads(tool_args) == {"name": _SPECIALS_ORIGINAL}
    assert json.loads(fc_args) == {"name": _SPECIALS_ORIGINAL}
    assert content == f"Hi {_SPECIALS_ORIGINAL}"
