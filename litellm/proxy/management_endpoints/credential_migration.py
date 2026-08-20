"""
At-rest credential re-encryption.

Two passes share one set of walkers, selected by a :class:`ReencryptPolicy`:

* **algorithm** (:data:`ALGORITHM_POLICY`) switches every encrypted-at-rest value
  from the legacy XSalsa20-Poly1305 (nacl) format to the versioned AES-256-GCM
  (``v2:gcm:``) format produced by ``encrypt_decrypt_utils`` when
  ``general_settings.encryption_algorithm`` is set to ``aes-256-gcm``. The key is
  unchanged.
* **salt key** (:data:`SALT_KEY_POLICY`) re-encrypts every value that still
  decrypts only under a retired salt key (``LITELLM_SALT_KEY_PREVIOUS``) under the
  active ``LITELLM_SALT_KEY``. This is what makes a leaked salt key recoverable
  without regenerating virtual keys: those are SHA-256 hashes, never salt-key
  ciphertext.

The two are independent axes, and a key-only rotation must not move the algorithm
one. Each salt-key rewrite therefore reproduces the algorithm of the value it
replaces (``encrypt_value_in_format_of``) rather than writing through the
configured write algorithm, which would downgrade an AES-256-GCM value whenever
that setting sits at its legacy default.

Design properties (see case 2026-06-24 fix plan):

* **Same key, new algorithm.** The migration does not change the encryption key;
  it re-encrypts existing ciphertext under the same derived key but in the new
  AES format. This is achieved by decrypting with the format-detecting reader and
  re-encrypting through ``encrypt_value_helper`` with the AES gate enabled.
* **Idempotent.** A value already carrying the ``v2:gcm:`` prefix is recognised
  and left untouched, so re-running the migration is a no-op on migrated rows.
* **Resumable.** Walkers commit per row (or per small table), so an interrupted
  run leaves a clean mixed state that a re-run completes.
* **Skip-on-undecryptable.** A value that cannot be decrypted is never
  overwritten — corrupt rows are preserved and reported, never destroyed.
* **Attestable.** :func:`check_encryption` is a read-only scan that classifies
  every value as ``migrated`` / ``legacy`` / ``plaintext`` / ``undecryptable``.
  A residual ``legacy == 0`` with an empty ``unreadable_locations`` is the
  compliance attestation. A store the scan could not open reports zero of
  everything, which means unknown rather than clean, so it is named there
  instead of quietly passing.

Coverage. The covered tables (model table, credentials table, MCP credential/env
tables, SSO identity assertions, config ``environment_variables``) already have a
re-encryption path in ``_rotate_master_key``; this module delegates to it in
*same-key* mode and adds walkers for the locations that had no rotation path: team
/ verification-token ``callback_vars`` metadata, the ``vantage_settings`` / ``cloudzero_settings``
config rows, and the SSO / cache / config-override settings rows. Every one of them is
scanned by :func:`check_encryption`, so a store whose rotation failed (that path
logs and carries on rather than aborting) surfaces as residual legacy instead of
being attested clean.
"""

import json
from collections.abc import AsyncIterator, Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Literal, assert_never, cast

from pydantic import TypeAdapter

from litellm._logging import verbose_proxy_logger

if TYPE_CHECKING:
    from litellm.proxy._types import UserAPIKeyAuth
    from litellm.proxy.utils import PrismaClient
from litellm.proxy.common_utils.encrypt_decrypt_utils import (
    _ALGO_AES_GCM,
    _ENCRYPTION_ALGORITHM_SETTING,
    _V2_GCM_PREFIX,
    _get_salt_key,
    decrypt_value_helper,
    encrypt_value_helper,
    encrypt_value_in_format_of,
    get_previous_salt_keys,
    try_decrypt_with_key,
)

_MAYBE_STR: Final = TypeAdapter(str | None)

ValueClass = Literal["migrated", "legacy", "plaintext", "undecryptable", "not-a-string"]


@dataclass(frozen=True, slots=True)
class ReencryptPolicy:
    """What "already re-encrypted" means for one pass over the stored values.

    ``is_current`` decides whether a decryptable value is left alone (counted as
    ``already_v2``) or rewritten (counted as ``migrated``). Everything a pass
    would still rewrite is residual ``legacy``. ``encrypt`` takes the recovered
    plaintext and the value as stored, and produces the replacement ciphertext:
    the algorithm pass writes through the configured algorithm, while the
    salt-key pass keeps the stored value's own algorithm.
    """

    name: Literal["algorithm", "salt_key"]
    is_current: Callable[[str], bool]
    encrypt: Callable[[str, str], str]


def _is_target_algorithm(value: str) -> bool:
    return value.startswith(_V2_GCM_PREFIX)


def _is_under_active_salt_key(value: str) -> bool:
    primary: Final = _get_salt_key()
    return primary is not None and try_decrypt_with_key(value=value, signing_key=primary) is not None


def _encrypt_under_configured_algorithm(plaintext: str, _stored: str) -> str:
    return encrypt_value_helper(plaintext)


ALGORITHM_POLICY: Final = ReencryptPolicy(
    name="algorithm",
    is_current=_is_target_algorithm,
    encrypt=_encrypt_under_configured_algorithm,
)
SALT_KEY_POLICY: Final = ReencryptPolicy(
    name="salt_key",
    is_current=_is_under_active_salt_key,
    encrypt=encrypt_value_in_format_of,
)

ReencryptMode = Literal["algorithm", "salt-key"]


def policy_for_mode(mode: ReencryptMode) -> ReencryptPolicy:
    """Map the wire-level ``mode`` of the migration endpoints onto its policy."""
    match mode:
        case "algorithm":
            return ALGORITHM_POLICY
        case "salt-key":
            return SALT_KEY_POLICY
        case _:
            assert_never(mode)


@dataclass
class LocationReport:
    """Per-location counters for one migration / check pass."""

    location: str
    scanned: int = 0
    migrated: int = 0  # values rewritten this run
    already_v2: int = 0  # values already in the target shape (skipped)
    plaintext: int = 0  # legacy-plaintext values (no ciphertext to migrate)
    undecryptable: int = 0  # could not decrypt — preserved, not overwritten

    # Used by --check (read-only classification):
    legacy: int = 0  # ciphertext still awaiting re-encryption

    # The store could not be read at all, so its zero counts mean "unknown",
    # never "clean". Kept out of the counters so it can never be mistaken for one.
    unreadable: bool = False

    def absorb(self, other: "LocationReport") -> None:
        """Fold another report's counters into this one, for per-row accumulation."""
        self.scanned += other.scanned
        self.migrated += other.migrated
        self.already_v2 += other.already_v2
        self.plaintext += other.plaintext
        self.undecryptable += other.undecryptable
        self.legacy += other.legacy
        self.unreadable = self.unreadable or other.unreadable

    def as_dict(self) -> dict[str, int]:
        return {
            "scanned": self.scanned,
            "migrated": self.migrated,
            "already_v2": self.already_v2,
            "plaintext": self.plaintext,
            "undecryptable": self.undecryptable,
            "legacy": self.legacy,
        }


@dataclass
class MigrationReport:
    """Aggregate report across all locations."""

    locations: list[LocationReport] = field(default_factory=list)

    def add(self, report: LocationReport) -> None:
        self.locations.append(report)

    @property
    def residual_legacy(self) -> int:
        """Total legacy ciphertext still un-migrated (the TRO attestation number)."""
        return sum(loc.legacy for loc in self.locations)

    @property
    def total_undecryptable(self) -> int:
        return sum(loc.undecryptable for loc in self.locations)

    @property
    def unreadable_locations(self) -> tuple[str, ...]:
        """Stores that could not be read, so their zero counts prove nothing.

        ``residual_legacy == 0`` only attests a clean rotation while this is
        empty: a store the scan could not open may still hold values under the
        retired key, and dropping that key would then make them unreadable.
        """
        return tuple(loc.location for loc in self.locations if loc.unreadable)

    def as_dict(self) -> dict[str, object]:
        return {
            "residual_legacy": self.residual_legacy,
            "total_undecryptable": self.total_undecryptable,
            "unreadable_locations": list(self.unreadable_locations),
            "locations": {loc.location: loc.as_dict() for loc in self.locations},
        }


# ---------------------------------------------------------------------------
# Pure engine — no DB I/O, fully unit-testable.
# ---------------------------------------------------------------------------


def is_migrated(value: object) -> bool:
    """True if ``value`` is already an AES-256-GCM (``v2:gcm:``) ciphertext."""
    return isinstance(value, str) and value.startswith(_V2_GCM_PREFIX)


def classify_value(value: object, key: str = "scan", policy: ReencryptPolicy = ALGORITHM_POLICY) -> ValueClass:
    """Classify a stored value for the residual scanner.

    * ``not-a-string`` — not a string (numbers/bools/None left as-is on disk).
    * ``migrated`` — already in the shape ``policy`` targets (``v2:gcm:`` format,
      or readable under the active salt key).
    * ``legacy`` — decrypts (under any configured salt key) but not in the target
      shape, so this pass would rewrite it.
    * ``plaintext`` — a non-empty string that does not decrypt at all; treated as
      legacy plaintext (nothing to migrate).
    * ``undecryptable`` — reserved for callers that already know a value is
      ciphertext but cannot decrypt it; ``classify_value`` itself cannot tell a
      corrupt ciphertext from plaintext, so it returns ``plaintext`` for both.
    """
    if not isinstance(value, str):
        return "not-a-string"
    if value == "":
        return "plaintext"
    if policy.is_current(value):
        return "migrated"
    decrypted: Final = decrypt_value_helper(value=value, key=key, exception_type="debug", return_original_value=False)
    if decrypted is None:
        # Did not decrypt under any configured salt key: legacy plaintext.
        return "plaintext"
    return "legacy"


def reencrypt_string(value: str, key: str = "migrate", policy: ReencryptPolicy = ALGORITHM_POLICY) -> str:
    """Re-encrypt one stored string into the shape ``policy`` targets.

    Returns the value unchanged if it already matches the policy or cannot be
    decrypted (skip-on-undecryptable). Otherwise decrypts under the format- and
    key-detecting reader and re-encrypts through the policy's writer, always
    under the active salt key.
    """
    if value == "" or policy.is_current(value):
        return value  # idempotent: already migrated
    decrypted: Final = _MAYBE_STR.validate_python(
        decrypt_value_helper(value=value, key=key, exception_type="debug", return_original_value=False)
    )
    if decrypted is None:
        # Either legacy plaintext (no ciphertext to migrate) or corrupt. Either
        # way, do not overwrite — preserve the value as stored.
        return value
    return policy.encrypt(decrypted, value)


def reencrypt_value(value: object, key: str = "migrate", policy: ReencryptPolicy = ALGORITHM_POLICY) -> object:
    """:func:`reencrypt_string` over a stored value of unknown type.

    Non-strings (numbers/bools/None) are left on disk exactly as they are.
    """
    if not isinstance(value, str):
        return value
    return reencrypt_string(value, key=key, policy=policy)


def reencrypt_selective_dict(
    data: dict[str, object],
    sensitive_keys: list[str],
    policy: ReencryptPolicy = ALGORITHM_POLICY,
) -> dict[str, object]:
    """Return a copy of ``data`` with only ``sensitive_keys`` re-encrypted.

    Non-sensitive fields (e.g. ``base_url``, ``connection_id``) are left as-is.
    Null/missing fields are skipped.
    """
    out: Final = dict(data)
    for k in sensitive_keys:
        v = out.get(k)
        if v is None:
            continue
        out[k] = reencrypt_value(v, key=k, policy=policy)
    return out


def _configured_algorithm() -> object:
    from litellm.proxy.proxy_server import general_settings

    return general_settings.get(_ENCRYPTION_ALGORITHM_SETTING)


def _aes_gate_enabled() -> bool:
    """Whether new writes land in AES-256-GCM (``general_settings``, read live)."""
    algo: Final = _configured_algorithm()
    return isinstance(algo, str) and algo.lower() == _ALGO_AES_GCM


def _assert_aes_gate_enabled() -> None:
    """Fail fast if the AES algorithm gate is not enabled.

    Running the migration with the gate off would decrypt then re-encrypt right
    back into the legacy format — a no-op that silently fails the migration.
    """
    if not _aes_gate_enabled():
        raise RuntimeError(
            "Encryption migration requires general_settings.encryption_algorithm: "
            f"'{_ALGO_AES_GCM}'. Current value: {_configured_algorithm()!r}. Set it before "
            "migrating so re-encrypted values are written in the AES-256-GCM format."
        )


# ---------------------------------------------------------------------------
# Walkers for the locations with no pre-existing rotation path.
# Each walker delegates the structural transform to the existing, tested helper
# for that table and only adds the per-row re-encrypt + commit + counters.
# ---------------------------------------------------------------------------


def _reencrypt_settings_fields(
    settings: Mapping[str, object],
    fields: tuple[str, ...],
    dry_run: bool,
    policy: ReencryptPolicy,
    location: str,
) -> tuple[Mapping[str, object], LocationReport]:
    """Re-encrypt ``fields`` of a settings dict, with the counters for what it did.

    A dry run counts what it would rewrite as residual ``legacy`` and returns the
    dict unchanged, so ``--check`` never reports something as migrated that was
    never written.
    """
    report: Final = LocationReport(location=location)
    out: Final = dict(settings)
    for fld in fields:
        v = out.get(fld)
        if v is None:
            continue
        report.scanned += 1
        cls = classify_value(v, key=fld, policy=policy)
        if cls == "migrated":
            report.already_v2 += 1
        elif cls != "legacy":
            report.plaintext += 1
        elif dry_run:
            report.legacy += 1
        else:
            new_v = reencrypt_value(v, key=fld, policy=policy)
            if new_v == v:
                report.legacy += 1  # did not re-encrypt, so it is still residual
            else:
                out[fld] = new_v
                report.migrated += 1
    return out, report


def _encrypted_string_fields(settings: Mapping[str, object]) -> tuple[str, ...]:
    """Field names holding a non-empty string, the only shape that can be ciphertext."""
    return tuple(k for k, v in settings.items() if isinstance(v, str) and v != "")


def _parse_settings_column(raw: object) -> Mapping[str, object] | None:
    """A settings column as a dict, or ``None`` when it is absent, unparseable, or not one."""
    if not isinstance(raw, str):
        return raw if isinstance(raw, dict) else None
    try:
        parsed: Final = json.loads(raw)
    except (ValueError, TypeError):
        return None
    return parsed if isinstance(parsed, dict) else None


async def _migrate_config_settings_row(
    prisma_client: object,
    param_name: str,
    sensitive_fields: list[str],
    dry_run: bool,
    policy: ReencryptPolicy = ALGORITHM_POLICY,
) -> LocationReport:
    """Migrate a single ``LiteLLM_Config`` row whose ``param_value`` is a JSON
    dict with selected sensitive fields (vantage_settings / cloudzero_settings).
    """
    report: Final = LocationReport(location=param_name)
    record: Final = await prisma_client.db.litellm_config.find_unique(where={"param_name": param_name})
    if record is None or record.param_value is None:
        return report

    settings: Final = _parse_settings_column(record.param_value)
    if settings is None:
        return report

    rewritten, fields_report = _reencrypt_settings_fields(
        settings, tuple(sensitive_fields), dry_run, policy, param_name
    )
    report.absorb(fields_report)
    if rewritten != settings:
        await prisma_client.db.litellm_config.update(
            where={"param_name": param_name},
            data={"param_value": json.dumps(rewritten)},
        )
    return report


# (location, prisma db attribute, primary-key field, JSON column of encrypted values).
# Every value in these rows is written through ``_encrypt_env_variables`` (or the
# SSO save path, which is the same code), so each field is salt-key ciphertext
# with no marker. None of them has a master-key rotation path, so a salt-key
# rotation that skipped them would leave live credentials (an IdP client secret, a
# Redis password, a Vault token) readable only under the retired key while the
# attestation reported a clean run.
_SETTINGS_ROW_SPECS: Final = (
    ("sso_config", "litellm_ssoconfig", "id", "sso_settings"),
    ("cache_config", "litellm_cacheconfig", "id", "cache_settings"),
    ("config_overrides", "litellm_configoverrides", "config_type", "config_value"),
)


async def _migrate_settings_rows(
    prisma_client: object,
    location: str,
    db_attr: str,
    pk: str,
    column: str,
    dry_run: bool,
    policy: ReencryptPolicy = ALGORITHM_POLICY,
) -> LocationReport:
    """Re-encrypt every encrypted field of one settings table's rows."""
    report: Final = LocationReport(location=location)
    table: Final = getattr(prisma_client.db, db_attr, None)
    if table is None:
        return report
    try:
        rows: Final = await table.find_many()
    except Exception as e:  # noqa: BLE001  # any driver failure means this store is unknown, not clean
        verbose_proxy_logger.warning("migrate: %s could not be read: %s", location, str(e))
        report.unreadable = True
        return report

    for row in rows or []:
        settings = _parse_settings_column(getattr(row, column, None))
        if settings is None:
            continue
        rewritten, row_report = _reencrypt_settings_fields(
            settings, _encrypted_string_fields(settings), dry_run, policy, location
        )
        report.absorb(row_report)
        if rewritten != settings:
            await table.update(where={pk: getattr(row, pk)}, data={column: json.dumps(rewritten)})
    return report


async def _migrate_settings_tables(
    prisma_client: object,
    dry_run: bool,
    policy: ReencryptPolicy = ALGORITHM_POLICY,
) -> AsyncIterator[LocationReport]:
    """One report per settings table, walked in order (a table at a time, never all in memory)."""
    for location, db_attr, pk, column in _SETTINGS_ROW_SPECS:
        yield await _migrate_settings_rows(prisma_client, location, db_attr, pk, column, dry_run, policy=policy)


async def _migrate_callback_vars_table(
    prisma_client: object,
    table_name: Literal["team", "verification_token"],
    dry_run: bool,
    policy: ReencryptPolicy = ALGORITHM_POLICY,
) -> LocationReport:
    """Migrate callback-var credentials on the team or verification-token table.

    Covers both shapes ``transform_callback_vars`` understands:
    ``metadata.logging[*].callback_vars.<sensitive>`` and the top-level
    ``metadata.callback_settings.callback_vars.<sensitive>``. Rewrites are
    per value and prefix-marked; legacy plaintext is left alone.
    """
    from litellm.proxy.common_utils.callback_utils import transform_callback_vars

    report: Final = LocationReport(location=f"{table_name}.callback_vars")

    if table_name == "team":
        table = prisma_client.db.litellm_teamtable
        pk = "team_id"
    else:
        table = prisma_client.db.litellm_verificationtoken
        pk = "token"

    rows: Final = await table.find_many()
    for row in rows or []:
        metadata = getattr(row, "metadata", None)
        if not isinstance(metadata, dict) or ("logging" not in metadata and "callback_settings" not in metadata):
            continue

        # Classify every callback-var value directly (strip the litellm_enc::
        # marker, then prefix/decrypt-classify), exactly like the covered-table
        # scanner. Detecting legacy this way is independent of the AES gate, so
        # the check_encryption (dry-run) attestation is correct even when run
        # before the gate is enabled -- a re-encrypt-delta heuristic would read
        # zero residual here with the gate off.
        row_legacy = 0
        for cvs in _iter_callback_var_dicts(metadata):
            for v in cvs.values():
                report.scanned += 1
                cls = _classify_callback_value(v, policy=policy)
                if cls == "migrated":
                    report.already_v2 += 1
                elif cls == "legacy":
                    row_legacy += 1
                else:  # plaintext / not-a-string
                    report.plaintext += 1

        if row_legacy == 0:
            continue  # no legacy ciphertext in this row

        if dry_run:
            # Residual for the attestation; a dry run writes nothing.
            report.legacy += row_legacy
            continue

        # Real run: rewrite value by value, so a row that mixes a retired-key
        # value with an already-current one only touches the former. A whole-row
        # decrypt/re-encrypt would push every value through the configured write
        # algorithm and downgrade active AES-GCM ciphertext during a key-only
        # rotation. Never drop a row on failure.
        try:
            re_encrypted = transform_callback_vars(
                metadata, lambda k, v: _reencrypt_callback_value(v, key=k, policy=policy)
            )
        except Exception as e:  # pragma: no cover - defensive; never drop a row
            verbose_proxy_logger.warning(
                "Skipping %s row %s callback_vars (transform failed): %s",
                table_name,
                getattr(row, pk, "?"),
                str(e),
            )
            report.undecryptable += row_legacy
            continue
        report.migrated += row_legacy
        await table.update(
            where={pk: getattr(row, pk)},
            data={"metadata": json.dumps(re_encrypted)},
        )

    return report


def _iter_callback_var_dicts(metadata: dict[str, object]):
    """Yield each ``callback_vars`` dict in a metadata structure.

    Mirrors ``transform_callback_vars``: credentials live both under
    ``logging[*].callback_vars`` and under the top-level
    ``callback_settings.callback_vars``. Counting only the former would let the
    walker report success while leaving ``callback_settings`` secrets in legacy
    format at rest.
    """
    for entry in metadata.get("logging", []) or []:
        if isinstance(entry, dict):
            cvs = entry.get("callback_vars")
            if isinstance(cvs, dict):
                yield cvs
    callback_settings: Final = metadata.get("callback_settings")
    if isinstance(callback_settings, dict):
        cvs = callback_settings.get("callback_vars")
        if isinstance(cvs, dict):
            yield cvs


def _callback_marker() -> str:
    """The marker that fronts an encrypted callback var (``litellm_enc::``)."""
    from litellm.proxy.common_utils.callback_utils import (
        _CALLBACK_VAR_ENCRYPTED_PREFIX,
    )

    return _CALLBACK_VAR_ENCRYPTED_PREFIX


def _reencrypt_callback_value(value: object, key: str, policy: ReencryptPolicy) -> object:
    """Re-encrypt one stored callback-var value, keeping the ``litellm_enc::`` marker.

    Only marked ciphertext is rewritten: plaintext callback vars are the write
    path's business, and this pass reports them as ``plaintext`` rather than
    silently encrypting them on rows that happen to also hold legacy ciphertext.
    """
    marker: Final = _callback_marker()
    if not isinstance(value, str) or not value.startswith(marker):
        return value
    return marker + reencrypt_string(value.removeprefix(marker), key=key, policy=policy)


def _classify_callback_value(value: object, policy: ReencryptPolicy = ALGORITHM_POLICY) -> ValueClass:
    """Classify one stored callback-var value, independent of the AES gate.

    Encrypted callback vars carry the ``litellm_enc::`` marker in front of the
    ciphertext; strip it, then classify the inner value the same way the
    covered-table scanner does (``v2:gcm:`` prefix -> migrated, nacl-decryptable
    -> legacy, otherwise plaintext). Detecting legacy by decrypt rather than by a
    re-encrypt delta is what makes the ``check_encryption`` attestation correct
    even when run with the AES write gate off.
    """
    if not isinstance(value, str):
        return "not-a-string"
    return classify_value(value.removeprefix(_callback_marker()), key="callback", policy=policy)


# ---------------------------------------------------------------------------
# Read-only scanner for the rotation-covered tables.
#
# ``_rotate_master_key`` re-encrypts these tables but returns no counts, so on
# its own it can neither attest residual legacy nor report how many rows it
# migrated. This scanner reads (never writes) the same encrypted columns the
# rotation path touches and classifies every value, giving both the attestation
# coverage and the pre/post counts the rotation path can't supply itself.
# ---------------------------------------------------------------------------

# (location, prisma db attribute, JSON columns to walk, scalar string columns).
_COVERED_TABLE_SPECS: Final = [
    ("model_table", "litellm_proxymodeltable", ("litellm_params",), ()),
    ("credentials", "litellm_credentialstable", ("credential_values",), ()),
    ("mcp_server", "litellm_mcpservertable", ("credentials", "env_vars"), ()),
    ("mcp_user_credentials", "litellm_mcpusercredentials", (), ("credential_b64",)),
    ("mcp_user_env_vars", "litellm_mcpuserenvvars", (), ("values_b64",)),
    ("sso_identity_assertions", "litellm_ssoidentityassertion", (), ("assertion_b64",)),
    ("mcp_oauth_clients", "litellm_mcpserveroauthclient", ("credentials",), ()),
]


def _iter_encrypted_strings(obj: object):
    """Yield every string leaf in a nested dict/list/scalar structure.

    Iterative (explicit stack) on purpose: recursion here is banned by the
    code-quality recursive-function detector (unbounded nesting has caused CPU
    spikes in the past), and an explicit stack walks arbitrary depth safely.
    """
    stack: Final[list[object]] = [obj]
    while stack:
        cur = stack.pop()
        if isinstance(cur, str):
            yield cur
        elif isinstance(cur, dict):
            stack.extend(cur.values())
        elif isinstance(cur, list):
            stack.extend(cur)


def _classify_into_report(report: LocationReport, value: str, policy: ReencryptPolicy = ALGORITHM_POLICY) -> None:
    """Classify one stored string and bump the matching read-only counter.

    Only genuine nacl ciphertext lands in ``legacy``; non-secret strings (model
    names, base URLs, …) do not decrypt and fall through to ``plaintext``, so
    over-scanning a column is harmless to the residual count.
    """
    report.scanned += 1
    cls: Final = classify_value(value, key="scan", policy=policy)
    if cls == "migrated":
        report.already_v2 += 1
    elif cls == "legacy":
        report.legacy += 1
    else:  # plaintext / not-a-string
        report.plaintext += 1


async def _scan_one_table(
    prisma_client: object,
    location: str,
    db_attr: str,
    json_columns: tuple,
    scalar_columns: tuple,
    policy: ReencryptPolicy = ALGORITHM_POLICY,
) -> LocationReport:
    report: Final = LocationReport(location=location)
    table: Final = getattr(prisma_client.db, db_attr, None)
    if table is None:
        return report
    try:
        rows: Final = await table.find_many()
    except Exception as e:  # noqa: BLE001  # any driver failure means this store is unknown, not clean
        verbose_proxy_logger.warning("scan: %s could not be read: %s", location, str(e))
        report.unreadable = True
        return report
    for row in rows or []:
        for col in json_columns:
            raw = getattr(row, col, None)
            if raw is None:
                continue
            if isinstance(raw, str):
                try:
                    raw = json.loads(raw)
                except (ValueError, TypeError):
                    pass
            for s in _iter_encrypted_strings(raw):
                _classify_into_report(report, s, policy=policy)
        for col in scalar_columns:
            v = getattr(row, col, None)
            if isinstance(v, str):
                _classify_into_report(report, v, policy=policy)
    return report


async def _scan_config_env_vars(prisma_client: object, policy: ReencryptPolicy = ALGORITHM_POLICY) -> LocationReport:
    """Scan the ``environment_variables`` config row (``param_value`` dict)."""
    report: Final = LocationReport(location="config_environment_variables")
    try:
        record: Final = await prisma_client.db.litellm_config.find_unique(where={"param_name": "environment_variables"})
    except Exception as e:  # noqa: BLE001  # any driver failure means this store is unknown, not clean
        verbose_proxy_logger.warning("scan: config env vars could not be read: %s", str(e))
        report.unreadable = True
        return report
    if record is None or record.param_value is None:
        return report
    value = record.param_value
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (ValueError, TypeError):
            value = {}
    for s in _iter_encrypted_strings(value):
        _classify_into_report(report, s, policy=policy)
    return report


async def _scan_covered_tables(
    prisma_client: object,
    policy: ReencryptPolicy = ALGORITHM_POLICY,
) -> list[LocationReport]:
    """Read-only classification of every rotation-covered table. No writes."""
    reports: Final[list[LocationReport]] = []
    for location, db_attr, json_cols, scalar_cols in _COVERED_TABLE_SPECS:
        reports.append(await _scan_one_table(prisma_client, location, db_attr, json_cols, scalar_cols, policy=policy))
    reports.append(await _scan_config_env_vars(prisma_client, policy=policy))
    return reports


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

# vantage_settings / cloudzero_settings sensitive fields (see *_endpoints.py).
_VANTAGE_SENSITIVE: Final = ["api_key", "integration_token"]
_CLOUDZERO_SENSITIVE: Final = ["api_key"]


async def _migrate_covered_tables(
    prisma_client: object,
    user_api_key_dict: object,
    policy: ReencryptPolicy = ALGORITHM_POLICY,
) -> list[LocationReport]:
    """Re-encrypt the tables already covered by ``_rotate_master_key`` (model
    table, credentials, MCP credential/env tables, config environment_variables)
    by running that orchestrator in *same-key* mode. With the AES gate on, the
    re-encrypt writes land in ``v2:`` format.

    ``_rotate_master_key`` returns no counts, so we bracket it with read-only
    scans: the pre-scan's legacy total minus the post-scan's gives the number
    actually migrated per location, and the post-scan supplies the residual /
    already-v2 / scanned figures. Returns one report per covered location.
    """
    from litellm.proxy.management_endpoints.key_management_endpoints import (
        _rotate_master_key,
    )

    pre: Final = MappingProxyType({r.location: r for r in await _scan_covered_tables(prisma_client, policy=policy)})

    current_key: Final = _get_salt_key()
    if current_key is None:
        raise RuntimeError(
            "Cannot migrate covered tables: no salt key / master key is set. Set LITELLM_SALT_KEY before migrating."
        )
    await _rotate_master_key(
        prisma_client=cast("PrismaClient", prisma_client),
        user_api_key_dict=cast("UserAPIKeyAuth", user_api_key_dict),
        current_master_key=current_key,
        new_master_key=current_key,  # writes land under the active salt key
    )

    post: Final = await _scan_covered_tables(prisma_client, policy=policy)
    for post_report in post:
        pre_report = pre.get(post_report.location)
        pre_legacy = pre_report.legacy if pre_report else 0
        # Everything that was legacy before and is no longer legacy now was
        # converted this run.
        post_report.migrated = max(0, pre_legacy - post_report.legacy)
    return post


async def migrate_encryption(
    prisma_client: object,
    user_api_key_dict: object,
    dry_run: bool = False,
    policy: ReencryptPolicy = ALGORITHM_POLICY,
) -> MigrationReport:
    """Run the full at-rest re-encryption pass for ``policy``.

    Under :data:`ALGORITHM_POLICY` this requires
    ``general_settings.encryption_algorithm == 'aes-256-gcm'`` so writes are
    produced in the AES format. Idempotent and resumable: re-running skips
    already-migrated values and finishes any partial run.

    A ``dry_run`` performs no writes: the covered tables are scanned read-only
    (so their residual legacy still counts toward the attestation) and the
    net-new walkers run in dry-run mode.
    """
    if policy is ALGORITHM_POLICY:
        _assert_aes_gate_enabled()
    else:
        _assert_previous_salt_keys_configured()
        if not dry_run:
            await _assert_no_algorithm_downgrade(prisma_client)

    report: Final = MigrationReport()

    # Tables that already have a rotation path (items 1, 2, 5-10). On a real run
    # delegate to the rotation path (with bracketing scans for counts); on a dry
    # run only classify them read-only.
    if dry_run:
        for covered in await _scan_covered_tables(prisma_client, policy=policy):
            report.add(covered)
    else:
        for covered in await _migrate_covered_tables(prisma_client, user_api_key_dict, policy=policy):
            report.add(covered)

    # Net-new walkers (items 3, 4, 11, 12, 13).
    report.add(await _migrate_callback_vars_table(prisma_client, "team", dry_run, policy=policy))
    report.add(await _migrate_callback_vars_table(prisma_client, "verification_token", dry_run, policy=policy))
    report.add(
        await _migrate_config_settings_row(
            prisma_client, "vantage_settings", _VANTAGE_SENSITIVE, dry_run, policy=policy
        )
    )
    report.add(
        await _migrate_config_settings_row(
            prisma_client, "cloudzero_settings", _CLOUDZERO_SENSITIVE, dry_run, policy=policy
        )
    )
    async for settings_report in _migrate_settings_tables(prisma_client, dry_run, policy=policy):
        report.add(settings_report)

    return report


async def check_encryption(prisma_client: object, policy: ReencryptPolicy = ALGORITHM_POLICY) -> MigrationReport:
    """Read-only residual scan across **every** at-rest location. No writes.

    Covers both the rotation-managed tables (model / credentials / MCP credential
    and env-var tables / config ``environment_variables``) and the net-new walker
    locations (team and verification-token ``callback_vars``, vantage / cloudzero
    config rows, SSO / cache / config-override settings). Reports how many values
    are still ``legacy``;
    ``residual_legacy == 0`` across this full scan is the compliance attestation.
    """
    report: Final = MigrationReport()

    # Rotation-covered tables (read-only classification).
    for covered in await _scan_covered_tables(prisma_client, policy=policy):
        report.add(covered)

    # Net-new walker locations, in dry-run (read-only) mode.
    report.add(await _migrate_callback_vars_table(prisma_client, "team", dry_run=True, policy=policy))
    report.add(await _migrate_callback_vars_table(prisma_client, "verification_token", dry_run=True, policy=policy))
    report.add(
        await _migrate_config_settings_row(
            prisma_client, "vantage_settings", _VANTAGE_SENSITIVE, dry_run=True, policy=policy
        )
    )
    report.add(
        await _migrate_config_settings_row(
            prisma_client, "cloudzero_settings", _CLOUDZERO_SENSITIVE, dry_run=True, policy=policy
        )
    )
    async for settings_report in _migrate_settings_tables(prisma_client, dry_run=True, policy=policy):
        report.add(settings_report)
    return report


async def _assert_no_algorithm_downgrade(prisma_client: object) -> None:
    """Fail fast when a salt-key rotation would rewrite AES values in legacy format.

    The rotation-covered tables are re-encrypted by ``_rotate_master_key``, which
    writes through ``general_settings.encryption_algorithm`` and so cannot
    preserve a value's own algorithm. Downgrading AES-256-GCM ciphertext is not
    something a key-only rotation may do silently, so refuse the run instead.
    """
    if _aes_gate_enabled():
        return
    aes_values: Final = sum(r.already_v2 for r in await _scan_covered_tables(prisma_client, policy=ALGORITHM_POLICY))
    if aes_values:
        raise RuntimeError(
            f"Salt-key rotation would rewrite {aes_values} AES-256-GCM value(s) in the legacy "
            f"format, because general_settings.encryption_algorithm is not '{_ALGO_AES_GCM}'. "
            "Set it to that, restart the proxy, then re-run the rotation."
        )


def _assert_previous_salt_keys_configured() -> None:
    """Fail fast when no retired salt key is configured.

    Without ``LITELLM_SALT_KEY_PREVIOUS``, values written under the retired key
    cannot be read at all, so a rotation pass would classify them as plaintext
    and leave them behind while reporting a clean run.
    """
    if not get_previous_salt_keys():
        raise RuntimeError(
            "Salt key rotation requires LITELLM_SALT_KEY_PREVIOUS to list the retired "
            "salt key(s), with LITELLM_SALT_KEY set to the new one. Restart the proxy "
            "with both set, then re-run the rotation."
        )
