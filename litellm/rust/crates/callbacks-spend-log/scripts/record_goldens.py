import contextvars
import copy
import hashlib
import json
import re
import sys
from collections.abc import Callable, Mapping
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING
from unittest import mock

if TYPE_CHECKING:
    from litellm.proxy.spend_tracking.spend_tracking_utils import AgentNameLookup

CRATE = Path(__file__).resolve().parents[1]
REPO = CRATE.parents[3]
GOLDENS = CRATE / "tests" / "goldens"
TEST_FILES = (
    "tests/unit/proxy/spend_tracking/test_spend_tracking_utils.py",
    "tests/unit/proxy/db/test_db_spend_update_writer.py",
    "tests/unit/accounting",
    "tests/unit/proxy/hooks/test_proxy_track_cost_callback.py",
)
FIXED_NOW = 1763337600.123456
FIXED_SESSION = "7d1c9f9e-1d61-4cf4-9a43-c1b5e5d6c0f1"
GOLDEN_DELIVERY_ID = "golden-delivery-id"
MINIMUM_CUSTOM_KEY_LENGTH = 16

HELPERS = (
    "_redact_prompt_leaks_in_error_string",
    "_masked_ip_text",
    "_scrub_raw_model_from_error_text",
    "_sanitize_request_body_for_spend_logs_payload",
)

_records: dict[str, list[dict]] = {"rows": [], "ledger_rows": [], **{helper: [] for helper in HELPERS}}
_seen: set[tuple[str, str]] = set()
_current_test = ["<collection>"]
_inside: contextvars.ContextVar[bool] = contextvars.ContextVar("inside", default=False)


class Unencodable(Exception):
    pass


def _jsonable(value: object, *, lenient: bool = False) -> object:
    if isinstance(value, datetime):
        return value.isoformat()
    if value is None or type(value) in (bool, int, float):
        return value
    if type(value) is str:
        return value
    if lenient and isinstance(value, str):
        return str.__str__(value)
    if type(value) is dict:
        if not all(type(key) is str for key in value):
            raise Unencodable(repr(value)[:200])
        return {key: _jsonable(item, lenient=lenient) for key, item in value.items()}
    if type(value) is list or (lenient and type(value) is tuple):
        return [_jsonable(item, lenient=lenient) for item in value]
    if lenient and isinstance(value, Mapping):
        return {str(key): _jsonable(item, lenient=lenient) for key, item in value.items()}
    raise Unencodable(f"{type(value).__name__}: {repr(value)[:200]}")


def _row_text(row: Mapping[str, object]) -> str:
    return json.dumps(_jsonable(dict(row), lenient=True))


def _add(kind: str, record: dict) -> None:
    digest = hashlib.sha256(json.dumps({**record, "case": None}, sort_keys=True).encode()).hexdigest()
    if (kind, digest) in _seen:
        return
    _seen.add((kind, digest))
    _records[kind].append({"case": _current_test[0], **record})


def _outermost(function: Callable) -> Callable:
    def call(*args: object, **kwargs: object):
        token = _inside.set(True)
        try:
            return function(*args, **kwargs)
        finally:
            _inside.reset(token)

    return call


def _ledger_row(payload: Mapping[str, object]) -> dict[str, object]:
    return {key: value for key, value in dict(payload).items() if not key.startswith("accounting_")}


def _record_ledger(payload, *, billable_cost, delivery_id, entities, model_access_groups, windows, router, output):
    from litellm.accounting.snapshot import _savings_or_zero
    from litellm.proxy.spend_tracking.rust_spend_log_rows import _ROW_METADATA

    try:
        record = {
            "row": _jsonable(_ledger_row(payload), lenient=True),
            "ledger": {
                "billable_cost": billable_cost,
                "delivery_id": delivery_id,
                "entities": [[key, value] for key, value in entities.items()],
                "model_access_groups": list(model_access_groups),
                "windows": _jsonable([dict(window) for window in windows], lenient=True),
                "savings": _savings_or_zero(
                    payload, _ROW_METADATA.validate_json(payload["metadata"]), router
                )._asdict(),
            },
            "output": json.dumps(_jsonable(output, lenient=True)),
        }
    except Unencodable:
        return
    _add("ledger_rows", record)


def _install() -> None:
    from litellm.accounting import snapshot
    from litellm.proxy.db import db_spend_update_writer
    from litellm.proxy.spend_tracking import legacy_spend_log_rows, rust_spend_log_rows, spend_tracking_utils

    original_payload = spend_tracking_utils.get_logging_payload
    original_accounting = snapshot.build_accounting_payload

    def recording_payload(
        kwargs, response_obj, start_time, end_time, llm_router=None, agent_names: "AgentNameLookup | None" = None
    ):
        record: dict = {}
        try:
            config = rust_spend_log_rows.spend_log_row_config()
            row_input = rust_spend_log_rows.spend_log_row_input(
                copy.copy(kwargs) if kwargs is not None else {},
                response_obj,
                start_time,
                end_time,
                llm_router,
                FIXED_NOW,
                FIXED_SESSION,
                agent_names,
            )
            record = {"config": config._asdict(), "input": json.loads(rust_spend_log_rows.encode_row_input(row_input))}
        except Exception as error:  # noqa: BLE001
            record = {"input_error": f"{type(error).__name__}: {error}"}
        fixed_uuid = mock.Mock(return_value=FIXED_SESSION)
        with (
            mock.patch("time.time", return_value=FIXED_NOW),
            mock.patch.object(spend_tracking_utils_uuid(), "uuid4", fixed_uuid),
            mock.patch.object(spend_tracking_utils, "uses_spend_logs_ledger", return_value=False),
        ):
            result = original_payload(kwargs, response_obj, start_time, end_time, llm_router, agent_names)
        if "input" in record:
            try:
                record["output"] = _row_text(result)
            except Unencodable as error:
                record = {"output_error": str(error)}
        _add("rows", record)
        return result

    def python_ledger_columns(payload, ledger):
        row = dict(payload)
        row["spend"] = ledger.billable_cost or 0.0
        accounting = _outermost(original_accounting)(
            row,
            entities=ledger.entities,
            model_access_groups=ledger.model_access_groups,
            windows=ledger.windows,
            router=ledger.router,
        )
        output = {
            "spend": row["spend"],
            "accounting_delivery_id": None
            if db_spend_update_writer._is_settle_once_cost_row(row)
            else ledger.delivery_id,
            "accounting_payload": accounting,
        }
        _record_ledger(
            row,
            billable_cost=ledger.billable_cost,
            delivery_id=ledger.delivery_id,
            entities=ledger.entities,
            model_access_groups=ledger.model_access_groups,
            windows=ledger.windows,
            router=ledger.router,
            output=output,
        )
        return output

    def recording_accounting(payload, *, entities, model_access_groups, windows, router):
        accounting = original_accounting(
            payload, entities=entities, model_access_groups=model_access_groups, windows=windows, router=router
        )
        spend = payload.get("spend")
        if not _inside.get() and type(spend) in (int, float):
            row = {key: value.isoformat() if isinstance(value, datetime) else value for key, value in payload.items()}
            _record_ledger(
                row,
                billable_cost=spend,
                delivery_id=GOLDEN_DELIVERY_ID,
                entities=entities,
                model_access_groups=model_access_groups,
                windows=windows,
                router=router,
                output={
                    "spend": spend,
                    "accounting_delivery_id": None
                    if db_spend_update_writer._is_settle_once_cost_row(row)
                    else GOLDEN_DELIVERY_ID,
                    "accounting_payload": accounting,
                },
            )
        return accounting

    def recording_helper(name: str, function: Callable) -> Callable:
        def call(*args: object, **kwargs: object):
            outermost = not _inside.get()
            result = _outermost(function)(*args, **kwargs)
            if outermost:
                try:
                    record = {
                        "args": _jsonable(list(args), lenient=name == "_scrub_raw_model_from_error_text"),
                        "kwargs": _jsonable(kwargs),
                        **(
                            {
                                "max_string_length_prompt_in_db": spend_tracking_utils._get_max_string_length_prompt_in_db(),
                                "redact_tokens": _general_settings().get("redact_tokens_in_spend_logs") is True,
                            }
                            if name == "_sanitize_request_body_for_spend_logs_payload"
                            else {}
                        ),
                        "output": _jsonable(result),
                    }
                except (Unencodable, AttributeError, RecursionError):
                    return result
                _add(name, record)
            return result

        return call

    spend_tracking_utils.get_logging_payload = recording_payload
    rust_spend_log_rows.ledger_columns = python_ledger_columns
    snapshot.build_accounting_payload = recording_accounting
    for helper in HELPERS:
        owner = spend_tracking_utils if hasattr(spend_tracking_utils, helper) else legacy_spend_log_rows
        recorder = recording_helper(helper, getattr(owner, helper))
        # The ledger-off builder holds its own reference to a helper it shares with spend_tracking_utils.
        for module in (spend_tracking_utils, legacy_spend_log_rows):
            if hasattr(module, helper):
                setattr(module, helper, recorder)


def _general_settings() -> Mapping[str, object]:
    from litellm.proxy.proxy_server import general_settings_view

    return general_settings_view()


def spend_tracking_utils_uuid():
    from litellm import _uuid

    return _uuid.uuid


def _constants() -> dict[str, list[str]]:
    from litellm.accounting import snapshot
    from litellm.litellm_core_utils import batch_utils, redact_messages
    from litellm.proxy import _types
    from litellm.proxy.spend_tracking import legacy_spend_log_rows, spend_tracking_utils
    from litellm.types import utils as types_utils
    from litellm.types.llms import vertex_ai
    from litellm.types.videos import utils as video_utils

    ordered = {
        "SpendLogsMetadata": list(_types.SpendLogsMetadata.__annotations__),
        "_CustomerIdentityMetadata": list(spend_tracking_utils._CustomerIdentityMetadata.__annotations__),
        "_REDACTED_REQUEST_FIELDS": list(redact_messages._REDACTED_REQUEST_FIELDS),
        "_NESTED_REQUEST_FIELDS": list(redact_messages._NESTED_REQUEST_FIELDS),
        "_WEB_ACTION_TEXT_FIELDS": list(redact_messages._WEB_ACTION_TEXT_FIELDS),
        "_WEB_ACTION_LIST_FIELDS": list(redact_messages._WEB_ACTION_LIST_FIELDS),
        "VERTEX_AI_PROVIDER_METADATA_FIELDS": list(vertex_ai.VERTEX_AI_PROVIDER_METADATA_FIELDS),
        "BEDROCK_GUARDRAIL_TRACE_FIELDS": list(redact_messages.BEDROCK_GUARDRAIL_TRACE_FIELDS),
        "MESSAGE_REDACTION_ENABLE_HEADERS": list(redact_messages.MESSAGE_REDACTION_ENABLE_HEADERS),
    }
    sets = {
        "_STAMPED_METADATA_KEYS": spend_tracking_utils._STAMPED_METADATA_KEYS,
        "_NON_SECRET_KEY_ALIASES": spend_tracking_utils._NON_SECRET_KEY_ALIASES,
        "_BATCH_TRACE_CALL_TYPES": legacy_spend_log_rows._BATCH_TRACE_CALL_TYPES,
        "BATCH_COST_REQUEST_ID_SUFFIX": {batch_utils.BATCH_COST_REQUEST_ID_SUFFIX},
        "VIDEO_COST_REQUEST_ID_SUFFIX": {video_utils.VIDEO_COST_REQUEST_ID_SUFFIX},
        "VIDEO_STATUS_CALL_TYPES": video_utils.VIDEO_STATUS_CALL_TYPES,
        "_REQUEST_BODY_CREDENTIAL_MASKER": spend_tracking_utils._REQUEST_BODY_CREDENTIAL_MASKER.sensitive_patterns,
        "_TOOL_INPUT_BLOCK_TYPES": spend_tracking_utils._TOOL_INPUT_BLOCK_TYPES,
        "_TOOL_OUTPUT_BLOCK_TYPES": spend_tracking_utils._TOOL_OUTPUT_BLOCK_TYPES,
        "PROMPT_CARRYING_GUARDRAIL_FIELDS": types_utils.PROMPT_CARRYING_GUARDRAIL_FIELDS,
        "_NUMERIC_COMPRESSION_STAT_KEYS": redact_messages._NUMERIC_COMPRESSION_STAT_KEYS,
        "_SCRUBBED_ERROR_TEXT_FIELDS": spend_tracking_utils._SCRUBBED_ERROR_TEXT_FIELDS,
        "_CLIENT_IP_HEADERS": legacy_spend_log_rows._CLIENT_IP_HEADERS,
        "_MATCHED_TEXT_KEYS": redact_messages._MATCHED_TEXT_KEYS,
        "accounting._METADATA_FIELDS": snapshot._METADATA_FIELDS,
        "accounting._GUARDRAIL_METADATA_FIELDS": snapshot._GUARDRAIL_METADATA_FIELDS,
        "accounting._ACCOUNTING_FIELDS": snapshot._ACCOUNTING_FIELDS,
        "accounting._GUARDRAIL_ACCOUNTING_FIELDS": snapshot._GUARDRAIL_ACCOUNTING_FIELDS,
    }
    return {**ordered, **{name: sorted(str(item) for item in values) for name, values in sets.items()}}


def _token_word_ranges() -> list[list[int]]:
    characters = "".join(map(chr, range(128, sys.maxunicode + 1)))
    return [[match.start() + 128, match.end() + 127] for match in re.finditer(r"\w+", characters)]


# Secret scanners flag provider-token shapes, and the masking tests feed the helpers realistic
# fake tokens. Each token-shaped run is written with TOKEN_SPLIT after its first character, so no
# committed golden matches a scanner, and the Rust readers remove every TOKEN_SPLIT on load.
TOKEN_SPLIT = "#~#"
_TOKEN_SHAPES = re.compile(
    r"(?:hf_|lin_api_|sk-|sk_live_|sk_test_|rk_live_|rk_test_|pk_live_|gh[opsur]_|github_pat_"
    r"|xox[abeoprs]-|AKIA|ASIA|AIza|glpat-|-----BEGIN|eyJ|npm_|pypi-|SG\.|xai-|gsk_|r8_|dop_v1_"
    r"|shpat_|sq0atp-)[A-Za-z0-9_-]*"
)


def _without_token_shapes(line: str) -> str:
    if TOKEN_SPLIT in line:
        raise ValueError(f"recorded text already holds {TOKEN_SPLIT!r}")
    return _TOKEN_SHAPES.sub(lambda match: match[0][0] + TOKEN_SPLIT + match[0][1:], line)


def pytest_configure(config):
    _install()


def pytest_runtest_setup(item):
    _current_test[0] = item.nodeid


def pytest_sessionfinish(session, exitstatus):
    GOLDENS.mkdir(parents=True, exist_ok=True)
    (GOLDENS / "helpers").mkdir(exist_ok=True)
    for kind, records in _records.items():
        path = GOLDENS / ("helpers" if kind in HELPERS else ".") / f"{kind.lstrip('_')}.jsonl"
        with open(path, "w", encoding="ascii") as out:
            for line in sorted(json.dumps(record) for record in records):
                out.write(_without_token_shapes(line) + "\n")
        sys.stderr.write(f"{kind}: {len(records)} records\n")
    (GOLDENS / "token_word_ranges.json").write_text(json.dumps(_token_word_ranges()) + "\n")
    (GOLDENS / "constants.json").write_text(json.dumps(_constants(), indent=1) + "\n")


if __name__ == "__main__":
    import pytest

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    sys.exit(
        pytest.main(
            [
                "-q",
                "-p",
                "no:cacheprovider",
                "-p",
                "no:xdist",
                "-p",
                "record_goldens",
                str(Path(__file__).resolve().parent / "golden_scenarios.py"),
                *(str(REPO / p) for p in TEST_FILES),
            ]
        )
    )
