"""
At-rest credential re-encryption migration.

Switches every encrypted-at-rest value from the legacy XSalsa20-Poly1305 (nacl)
format to the versioned AES-256-GCM (``v2:gcm:``) format produced by
``encrypt_decrypt_utils`` when ``general_settings.encryption_algorithm`` is set to
``aes-256-gcm``.

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
  A residual ``legacy == 0`` is the compliance attestation.

Coverage. The covered tables (model table, credentials table, MCP credential/env
tables, config ``environment_variables``) already have a re-encryption path in
``_rotate_master_key``; this module delegates to it in *same-key* mode and adds
walkers for the locations that had no rotation path: team / verification-token
``callback_vars`` metadata, the ``vantage_settings`` / ``cloudzero_settings``
config rows, and the SSO config table.
"""

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, cast

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
)

ValueClass = Literal["migrated", "legacy", "plaintext", "undecryptable", "not-a-string"]


@dataclass
class LocationReport:
    """Per-location counters for one migration / check pass."""

    location: str
    scanned: int = 0
    migrated: int = 0  # values rewritten to v2 this run
    already_v2: int = 0  # values already migrated (skipped)
    plaintext: int = 0  # legacy-plaintext values (no ciphertext to migrate)
    undecryptable: int = 0  # could not decrypt — preserved, not overwritten

    # Used by --check (read-only classification):
    legacy: int = 0  # nacl ciphertext still awaiting migration

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

    def as_dict(self) -> dict[str, object]:
        return {
            "residual_legacy": self.residual_legacy,
            "total_undecryptable": self.total_undecryptable,
            "locations": {loc.location: loc.as_dict() for loc in self.locations},
        }


# ---------------------------------------------------------------------------
# Pure engine — no DB I/O, fully unit-testable.
# ---------------------------------------------------------------------------


def is_migrated(value: object) -> bool:
    """True if ``value`` is already an AES-256-GCM (``v2:gcm:``) ciphertext."""
    return isinstance(value, str) and value.startswith(_V2_GCM_PREFIX)


def classify_value(value: object, key: str = "scan") -> ValueClass:
    """Classify a stored value for the residual scanner.

    * ``not-a-string`` — not a string (numbers/bools/None left as-is on disk).
    * ``migrated`` — carries the ``v2:gcm:`` prefix.
    * ``legacy`` — decrypts under the legacy nacl reader (still needs migrating).
    * ``plaintext`` — a non-empty string that does not decrypt and is not v2;
      treated as legacy plaintext (nothing to migrate).
    * ``undecryptable`` — reserved for callers that already know a value is
      ciphertext but cannot decrypt it; ``classify_value`` itself cannot tell a
      corrupt ciphertext from plaintext, so it returns ``plaintext`` for both.
    """
    if not isinstance(value, str):
        return "not-a-string"
    if value == "":
        return "plaintext"
    if value.startswith(_V2_GCM_PREFIX):
        return "migrated"
    decrypted = decrypt_value_helper(value=value, key=key, exception_type="debug", return_original_value=False)
    if decrypted is None:
        # Did not decrypt under nacl and has no v2 marker: legacy plaintext.
        return "plaintext"
    return "legacy"


def reencrypt_value(value: object, key: str = "migrate") -> object:
    """Re-encrypt a single stored string into the configured (AES) format.

    Returns the value unchanged if it is not a string, is already ``v2:``, or
    cannot be decrypted (skip-on-undecryptable). Otherwise decrypts under the
    format-detecting reader and re-encrypts through ``encrypt_value_helper``
    (which writes AES when the gate is on).
    """
    if not isinstance(value, str) or value == "":
        return value
    if value.startswith(_V2_GCM_PREFIX):
        return value  # idempotent: already migrated
    decrypted = decrypt_value_helper(value=value, key=key, exception_type="debug", return_original_value=False)
    if decrypted is None:
        # Either legacy plaintext (no ciphertext to migrate) or corrupt. Either
        # way, do not overwrite — preserve the value as stored.
        return value
    return encrypt_value_helper(decrypted)


def reencrypt_selective_dict(data: dict[str, object], sensitive_keys: list[str]) -> dict[str, object]:
    """Return a copy of ``data`` with only ``sensitive_keys`` re-encrypted.

    Non-sensitive fields (e.g. ``base_url``, ``connection_id``) are left as-is.
    Null/missing fields are skipped.
    """
    out = dict(data)
    for k in sensitive_keys:
        v = out.get(k)
        if v is None:
            continue
        out[k] = reencrypt_value(v, key=k)
    return out


def _assert_aes_gate_enabled() -> None:
    """Fail fast if the AES algorithm gate is not enabled.

    Running the migration with the gate off would decrypt then re-encrypt right
    back into the legacy format — a no-op that silently fails the migration.
    """
    from litellm.proxy.proxy_server import general_settings

    algo = general_settings.get(_ENCRYPTION_ALGORITHM_SETTING)
    if not (isinstance(algo, str) and algo.lower() == _ALGO_AES_GCM):
        raise RuntimeError(
            "Encryption migration requires general_settings.encryption_algorithm: "
            f"'{_ALGO_AES_GCM}'. Current value: {algo!r}. Set it before migrating "
            "so re-encrypted values are written in the AES-256-GCM format."
        )


# ---------------------------------------------------------------------------
# Walkers for the locations with no pre-existing rotation path.
# Each walker delegates the structural transform to the existing, tested helper
# for that table and only adds the per-row re-encrypt + commit + counters.
# ---------------------------------------------------------------------------


async def _migrate_config_settings_row(
    prisma_client: object,
    param_name: str,
    sensitive_fields: list[str],
    dry_run: bool,
) -> LocationReport:
    """Migrate a single ``LiteLLM_Config`` row whose ``param_value`` is a JSON
    dict with selected sensitive fields (vantage_settings / cloudzero_settings).
    """
    report = LocationReport(location=param_name)
    record = await prisma_client.db.litellm_config.find_unique(where={"param_name": param_name})
    if record is None or record.param_value is None:
        return report

    settings = record.param_value
    if isinstance(settings, str):
        settings = json.loads(settings)
    if not isinstance(settings, dict):
        return report

    changed = False
    for fld in sensitive_fields:
        v = settings.get(fld)
        if v is None:
            continue
        report.scanned += 1
        cls = classify_value(v, key=fld)
        if cls == "migrated":
            report.already_v2 += 1
            continue
        if cls == "legacy":
            if dry_run:
                # Residual: would migrate, but a dry run writes nothing, so it
                # stays legacy for the attestation (never counted as migrated).
                report.legacy += 1
                continue
            new_v = reencrypt_value(v, key=fld)
            if new_v != v:
                settings[fld] = new_v
                report.migrated += 1
                changed = True
            else:
                # Defensive: a legacy value that did not re-encrypt is still
                # residual, not migrated.
                report.legacy += 1
        else:  # plaintext / not-a-string — nothing to migrate
            report.plaintext += 1

    if changed and not dry_run:
        await prisma_client.db.litellm_config.update(
            where={"param_name": param_name},
            data={"param_value": json.dumps(settings)},
        )
    return report


async def _migrate_sso_config(prisma_client: object, dry_run: bool) -> LocationReport:
    """Migrate the ``LiteLLM_SSOConfig`` row. All non-null fields are encrypted
    (via the same ``_encrypt_env_variables`` path used on save), so we re-encrypt
    every present string field.
    """
    report = LocationReport(location="sso_config")
    record = await prisma_client.db.litellm_ssoconfig.find_unique(where={"id": "sso_config"})
    if record is None or record.sso_settings is None:
        return report

    settings = record.sso_settings
    if isinstance(settings, str):
        settings = json.loads(settings)
    if not isinstance(settings, dict):
        return report

    new_settings = dict(settings)
    changed = False
    for fld, v in settings.items():
        if not isinstance(v, str) or v == "":
            continue
        report.scanned += 1
        cls = classify_value(v, key=fld)
        if cls == "migrated":
            report.already_v2 += 1
            continue
        if cls == "legacy":
            if dry_run:
                # Residual: would migrate, but a dry run writes nothing, so it
                # stays legacy for the attestation (never counted as migrated).
                report.legacy += 1
                continue
            new_v = reencrypt_value(v, key=fld)
            if new_v != v:
                new_settings[fld] = new_v
                report.migrated += 1
                changed = True
            else:
                # Defensive: a legacy value that did not re-encrypt is still
                # residual, not migrated.
                report.legacy += 1
        else:
            report.plaintext += 1

    if changed and not dry_run:
        await prisma_client.db.litellm_ssoconfig.update(
            where={"id": "sso_config"},
            data={"sso_settings": json.dumps(new_settings)},
        )
    return report


async def _migrate_callback_vars_table(
    prisma_client: object,
    table_name: Literal["team", "verification_token"],
    dry_run: bool,
) -> LocationReport:
    """Migrate callback-var credentials on the team or verification-token table.

    Covers both shapes the ``decrypt_callback_vars`` / ``encrypt_callback_vars``
    transforms understand: ``metadata.logging[*].callback_vars.<sensitive>`` and
    the top-level ``metadata.callback_settings.callback_vars.<sensitive>``. Reuses
    those proven transforms (selective, prefix-marked; legacy plaintext is left
    alone until re-encrypted).
    """
    from litellm.proxy.common_utils.callback_utils import (
        decrypt_callback_vars,
        encrypt_callback_vars,
    )

    report = LocationReport(location=f"{table_name}.callback_vars")

    if table_name == "team":
        table = prisma_client.db.litellm_teamtable
        pk = "team_id"
    else:
        table = prisma_client.db.litellm_verificationtoken
        pk = "token"

    rows = await table.find_many()
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
                cls = _classify_callback_value(v)
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

        # Real run: re-encrypt the legacy ciphertext to AES via the proven
        # selective transforms and persist. Never drop a row on failure.
        try:
            re_encrypted = encrypt_callback_vars(decrypt_callback_vars(metadata))
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

    Mirrors ``_transform_callback_vars``: credentials live both under
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
    callback_settings = metadata.get("callback_settings")
    if isinstance(callback_settings, dict):
        cvs = callback_settings.get("callback_vars")
        if isinstance(cvs, dict):
            yield cvs


def _classify_callback_value(value: object) -> ValueClass:
    """Classify one stored callback-var value, independent of the AES gate.

    Encrypted callback vars carry the ``litellm_enc::`` marker in front of the
    ciphertext; strip it, then classify the inner value the same way the
    covered-table scanner does (``v2:gcm:`` prefix -> migrated, nacl-decryptable
    -> legacy, otherwise plaintext). Detecting legacy by decrypt rather than by a
    re-encrypt delta is what makes the ``check_encryption`` attestation correct
    even when run with the AES write gate off.
    """
    from litellm.proxy.common_utils.callback_utils import (
        _CALLBACK_VAR_ENCRYPTED_PREFIX,
    )

    if not isinstance(value, str):
        return "not-a-string"
    inner = value
    if inner.startswith(_CALLBACK_VAR_ENCRYPTED_PREFIX):
        inner = inner[len(_CALLBACK_VAR_ENCRYPTED_PREFIX) :]
    return classify_value(inner, key="callback")


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
_COVERED_TABLE_SPECS = [
    ("model_table", "litellm_proxymodeltable", ("litellm_params",), ()),
    ("credentials", "litellm_credentialstable", ("credential_values",), ()),
    ("mcp_server", "litellm_mcpservertable", ("credentials", "env_vars"), ()),
    ("mcp_user_credentials", "litellm_mcpusercredentials", (), ("credential_b64",)),
    ("mcp_user_env_vars", "litellm_mcpuserenvvars", (), ("values_b64",)),
]


def _iter_encrypted_strings(obj: object):
    """Yield every string leaf in a nested dict/list/scalar structure.

    Iterative (explicit stack) on purpose: recursion here is banned by the
    code-quality recursive-function detector (unbounded nesting has caused CPU
    spikes in the past), and an explicit stack walks arbitrary depth safely.
    """
    stack: list[object] = [obj]
    while stack:
        cur = stack.pop()
        if isinstance(cur, str):
            yield cur
        elif isinstance(cur, dict):
            stack.extend(cur.values())
        elif isinstance(cur, list):
            stack.extend(cur)


def _classify_into_report(report: LocationReport, value: str) -> None:
    """Classify one stored string and bump the matching read-only counter.

    Only genuine nacl ciphertext lands in ``legacy``; non-secret strings (model
    names, base URLs, …) do not decrypt and fall through to ``plaintext``, so
    over-scanning a column is harmless to the residual count.
    """
    report.scanned += 1
    cls = classify_value(value, key="scan")
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
) -> LocationReport:
    report = LocationReport(location=location)
    table = getattr(prisma_client.db, db_attr, None)
    if table is None:
        return report
    try:
        rows = await table.find_many()
    except Exception as e:  # pragma: no cover - table absent / not migrated
        verbose_proxy_logger.debug("scan: %s unavailable: %s", location, str(e))
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
                _classify_into_report(report, s)
        for col in scalar_columns:
            v = getattr(row, col, None)
            if isinstance(v, str):
                _classify_into_report(report, v)
    return report


async def _scan_config_env_vars(prisma_client: object) -> LocationReport:
    """Scan the ``environment_variables`` config row (``param_value`` dict)."""
    report = LocationReport(location="config_environment_variables")
    try:
        record = await prisma_client.db.litellm_config.find_unique(where={"param_name": "environment_variables"})
    except Exception as e:  # pragma: no cover - defensive
        verbose_proxy_logger.debug("scan: config env vars unavailable: %s", str(e))
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
        _classify_into_report(report, s)
    return report


async def _scan_covered_tables(prisma_client: object) -> list[LocationReport]:
    """Read-only classification of every rotation-covered table. No writes."""
    reports: list[LocationReport] = []
    for location, db_attr, json_cols, scalar_cols in _COVERED_TABLE_SPECS:
        reports.append(await _scan_one_table(prisma_client, location, db_attr, json_cols, scalar_cols))
    reports.append(await _scan_config_env_vars(prisma_client))
    return reports


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

# vantage_settings / cloudzero_settings sensitive fields (see *_endpoints.py).
_VANTAGE_SENSITIVE = ["api_key", "integration_token"]
_CLOUDZERO_SENSITIVE = ["api_key"]


async def _migrate_covered_tables(prisma_client: object, user_api_key_dict: object) -> list[LocationReport]:
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

    pre = {r.location: r for r in await _scan_covered_tables(prisma_client)}

    current_key = _get_salt_key()
    if current_key is None:
        raise RuntimeError(
            "Cannot migrate covered tables: no salt key / master key is set. Set LITELLM_SALT_KEY before migrating."
        )
    await _rotate_master_key(
        prisma_client=cast("PrismaClient", prisma_client),
        user_api_key_dict=cast("UserAPIKeyAuth", user_api_key_dict),
        current_master_key=current_key,
        new_master_key=current_key,  # same key, algorithm-only switch
    )

    post = await _scan_covered_tables(prisma_client)
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
) -> MigrationReport:
    """Run the full at-rest re-encryption migration.

    Requires ``general_settings.encryption_algorithm == 'aes-256-gcm'`` so writes
    are produced in the AES format. Idempotent and resumable: re-running skips
    already-migrated values and finishes any partial run.

    A ``dry_run`` performs no writes: the covered tables are scanned read-only
    (so their residual legacy still counts toward the attestation) and the
    net-new walkers run in dry-run mode.
    """
    _assert_aes_gate_enabled()

    report = MigrationReport()

    # Tables that already have a rotation path (items 1, 2, 5-10). On a real run
    # delegate to the rotation path (with bracketing scans for counts); on a dry
    # run only classify them read-only.
    if dry_run:
        for covered in await _scan_covered_tables(prisma_client):
            report.add(covered)
    else:
        for covered in await _migrate_covered_tables(prisma_client, user_api_key_dict):
            report.add(covered)

    # Net-new walkers (items 3, 4, 11, 12, 13).
    report.add(await _migrate_callback_vars_table(prisma_client, "team", dry_run))
    report.add(await _migrate_callback_vars_table(prisma_client, "verification_token", dry_run))
    report.add(await _migrate_config_settings_row(prisma_client, "vantage_settings", _VANTAGE_SENSITIVE, dry_run))
    report.add(await _migrate_config_settings_row(prisma_client, "cloudzero_settings", _CLOUDZERO_SENSITIVE, dry_run))
    report.add(await _migrate_sso_config(prisma_client, dry_run))

    return report


async def check_encryption(prisma_client: object) -> MigrationReport:
    """Read-only residual scan across **every** at-rest location. No writes.

    Covers both the rotation-managed tables (model / credentials / MCP credential
    and env-var tables / config ``environment_variables``) and the net-new walker
    locations (team and verification-token ``callback_vars``, vantage / cloudzero
    config rows, SSO config). Reports how many values are still ``legacy``;
    ``residual_legacy == 0`` across this full scan is the compliance attestation.
    """
    report = MigrationReport()

    # Rotation-covered tables (read-only classification).
    for covered in await _scan_covered_tables(prisma_client):
        report.add(covered)

    # Net-new walker locations, in dry-run (read-only) mode.
    report.add(await _migrate_callback_vars_table(prisma_client, "team", dry_run=True))
    report.add(await _migrate_callback_vars_table(prisma_client, "verification_token", dry_run=True))
    report.add(await _migrate_config_settings_row(prisma_client, "vantage_settings", _VANTAGE_SENSITIVE, dry_run=True))
    report.add(
        await _migrate_config_settings_row(prisma_client, "cloudzero_settings", _CLOUDZERO_SENSITIVE, dry_run=True)
    )
    report.add(await _migrate_sso_config(prisma_client, dry_run=True))
    return report
