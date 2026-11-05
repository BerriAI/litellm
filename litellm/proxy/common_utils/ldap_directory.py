import importlib
import ssl
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from threading import Event
from typing import Final, Protocol, TypeAlias, runtime_checkable
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, StrictStr, TypeAdapter

from litellm.secret_managers.main import get_secret_str
from litellm.types.proxy.ldap_sync import LDAPSyncSettings

_ITEMS: Final = TypeAdapter(Iterable[object])
_STRINGS: Final = TypeAdapter(tuple[StrictStr, ...])
_GROUP_CLASSES: Final = frozenset({"group", "groupofnames", "groupofuniquenames", "posixgroup"})
_NO_SUCH_OBJECT: Final = 32


@dataclass(frozen=True, slots=True)
class DirectoryGroup:
    team_id: str
    name: str
    identities: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class DirectoryError:
    reason: str
    group_dn: str | None = None


DirectoryResult: TypeAlias = tuple[DirectoryGroup | DirectoryError, ...] | DirectoryError


class _Entry(BaseModel):
    model_config = ConfigDict(frozen=True)

    type: str
    dn: str = ""
    attributes: Mapping[str, object] = {}


_ENTRIES: Final = TypeAdapter(tuple[_Entry, ...])


class _PagedSearch(Protocol):
    def paged_search(
        self,
        *,
        search_base: str,
        search_filter: str,
        search_scope: str,
        attributes: Sequence[str],
        paged_size: int,
        time_limit: int,
        generator: bool,
    ) -> object: ...


class _Extensions(Protocol):
    @property
    def standard(self) -> _PagedSearch: ...


class LDAPConnection(Protocol):
    @property
    def result(self) -> Mapping[str, object]: ...

    @property
    def response(self) -> object: ...

    @property
    def extend(self) -> _Extensions: ...

    def open(self) -> object: ...
    def start_tls(self) -> bool: ...
    def bind(self) -> bool: ...
    def unbind(self) -> bool: ...
    def search(
        self,
        *,
        search_base: str,
        search_filter: str,
        search_scope: str,
        attributes: Sequence[str],
        time_limit: int,
    ) -> bool: ...


@runtime_checkable
class _LDAPLibrary(Protocol):
    def Tls(self, *, validate: ssl.VerifyMode, ca_certs_file: str | None) -> object: ...
    def Server(self, host: str, *, port: int, use_ssl: bool, tls: object, connect_timeout: int) -> object: ...
    def Connection(
        self,
        server: object,
        *,
        user: str | None,
        password: str | None,
        authentication: str,
        auto_referrals: bool,
        raise_exceptions: bool,
        receive_timeout: int,
        check_names: bool,
        return_empty_attributes: bool,
    ) -> LDAPConnection: ...


def _secret(value: str | None) -> str | None:
    if value is None or not value.startswith("os.environ/"):
        return value
    resolved: Final = get_secret_str(value)
    if resolved is None:
        raise ValueError("An LDAP bind credential environment variable is missing")
    return resolved


def open_ldap_connection(settings: LDAPSyncSettings) -> LDAPConnection:
    library: Final = importlib.import_module("ldap3")
    if not isinstance(library, _LDAPLibrary):
        raise TypeError("ldap3 does not expose the required LDAP constructors")
    parsed: Final = urlsplit(settings.server_url)
    tls: Final = library.Tls(validate=ssl.CERT_REQUIRED, ca_certs_file=settings.ca_cert_path)
    server: Final = library.Server(
        parsed.hostname or "",
        port=parsed.port or (636 if parsed.scheme == "ldaps" else 389),
        use_ssl=parsed.scheme == "ldaps",
        tls=tls,
        connect_timeout=settings.timeout_seconds,
    )
    return library.Connection(
        server,
        user=_secret(settings.bind_dn),
        password=_secret(settings.bind_password.get_secret_value()) if settings.bind_password is not None else None,
        authentication="SIMPLE" if settings.bind_dn else "ANONYMOUS",
        auto_referrals=False,
        raise_exceptions=True,
        receive_timeout=settings.timeout_seconds,
        check_names=False,
        return_empty_attributes=True,
    )


def _values(entry: _Entry, attribute: str) -> tuple[str, ...]:
    value: Final = next((value for key, value in entry.attributes.items() if key.lower() == attribute.lower()), ())
    return (value,) if isinstance(value, str) else _STRINGS.validate_python(value)


def _one(entry: _Entry, attribute: str) -> str | None:
    values: Final = _values(entry, attribute)
    return values[0] if len(values) == 1 and values[0].strip() else None


def _entries(raw: object) -> tuple[_Entry, ...] | DirectoryError:
    entries: Final = _ENTRIES.validate_python(raw)
    if any(entry.type != "searchResEntry" for entry in entries):
        return DirectoryError("LDAP search returned a referral or incomplete entry")
    return entries


def _escape_filter(value: str) -> str:
    return "".join(
        {"\\": r"\5c", "*": r"\2a", "(": r"\28", ")": r"\29", "\x00": r"\00"}.get(char, char) for char in value
    )


def _check_cancelled(stop: Event) -> None:
    if stop.is_set():
        raise InterruptedError("LDAP sync cancelled")


def _checked_items(raw: object, stop: Event) -> Iterable[object]:
    items: Final = iter(_ITEMS.validate_python(raw))
    while True:
        _check_cancelled(stop)
        try:
            yield next(items)
        except StopIteration:
            return


def _resolve_member(
    connection: LDAPConnection, settings: LDAPSyncSettings, value: str, stop: Event
) -> str | DirectoryError:
    _check_cancelled(stop)
    usernames: Final = settings.member_attribute.lower() == "memberuid"
    try:
        _ = connection.search(
            search_base=settings.user_search_base if usernames else value,
            search_filter=f"({settings.member_lookup_attribute}={_escape_filter(value)})"
            if usernames
            else "(objectClass=*)",
            search_scope="SUBTREE" if usernames else "BASE",
            attributes=(settings.user_match_attribute, "objectClass"),
            time_limit=settings.timeout_seconds,
        )
    except Exception:  # noqa: BLE001  # ldap3 raises noSuchObject for a deleted member DN
        if usernames or connection.result.get("result") != _NO_SUCH_OBJECT:
            raise
        return DirectoryError("LDAP member entry does not exist")
    if connection.result.get("result") != 0:
        return DirectoryError("LDAP member lookup did not complete successfully")
    entries: Final = _entries(connection.response)
    if isinstance(entries, DirectoryError):
        return entries
    if len(entries) != 1:
        return DirectoryError("LDAP member lookup was missing or ambiguous")
    entry: Final = entries[0]
    if _GROUP_CLASSES.intersection(value.lower() for value in _values(entry, "objectClass")):
        return DirectoryError("Nested LDAP groups are not supported")
    identity: Final = _one(entry, settings.user_match_attribute)
    return identity if identity is not None else DirectoryError("LDAP member identity is missing or multivalued")


def _read_group(
    connection: LDAPConnection, settings: LDAPSyncSettings, entry: _Entry, stop: Event
) -> DirectoryGroup | DirectoryError:
    _check_cancelled(stop)
    team_id: Final = _one(entry, settings.group_id_attribute)
    name: Final = _one(entry, settings.group_name_attribute)
    if team_id is None or name is None:
        return DirectoryError("LDAP group ID or name is missing or multivalued", entry.dn)
    if any(";range=" in key.lower() for key in entry.attributes):
        return DirectoryError("LDAP ranged membership responses are not supported", entry.dn)
    identities: Final = tuple(
        _resolve_member(connection, settings, value, stop) for value in _values(entry, settings.member_attribute)
    )
    error: Final = next((identity for identity in identities if isinstance(identity, DirectoryError)), None)
    if error is not None:
        return DirectoryError(error.reason, entry.dn)
    return DirectoryGroup(team_id, name, tuple(identity for identity in identities if isinstance(identity, str)))


def read_directory(
    settings: LDAPSyncSettings, connection: LDAPConnection | None = None, stop: Event | None = None
) -> DirectoryResult:
    cancellation: Final = stop if stop is not None else Event()
    try:
        _check_cancelled(cancellation)
        client: Final = connection if connection is not None else open_ldap_connection(settings)
        try:
            _ = client.open()
            _check_cancelled(cancellation)
            if settings.start_tls and urlsplit(settings.server_url).scheme == "ldap" and not client.start_tls():
                return DirectoryError("LDAP TLS negotiation failed")
            _check_cancelled(cancellation)
            if not client.bind():
                return DirectoryError("LDAP bind failed")
            _check_cancelled(cancellation)
            raw: Final = client.extend.standard.paged_search(
                search_base=settings.group_search_base,
                search_filter=settings.group_filter,
                search_scope="SUBTREE",
                attributes=(settings.group_id_attribute, settings.group_name_attribute, settings.member_attribute),
                paged_size=500,
                time_limit=settings.timeout_seconds,
                generator=True,
            )
            items: Final = tuple(_checked_items(raw, cancellation))
            _check_cancelled(cancellation)
            if client.result.get("result") != 0:
                return DirectoryError("LDAP group search did not complete successfully")
            entries: Final = _entries(items)
            if isinstance(entries, DirectoryError):
                return entries
            group_ids: Final = tuple(_one(entry, settings.group_id_attribute) for entry in entries)
            if len(group_ids) != len(frozenset(group_ids)):
                return DirectoryError("LDAP group IDs are not unique within the selected groups")
            groups: Final = tuple(_read_group(client, settings, entry, cancellation) for entry in entries)
            _check_cancelled(cancellation)
            return groups
        finally:
            _ = client.unbind()
    except Exception as exc:  # noqa: BLE001  # SDK errors must abort safely without exposing bind credentials
        return DirectoryError(f"LDAP read failed ({type(exc).__name__})")
