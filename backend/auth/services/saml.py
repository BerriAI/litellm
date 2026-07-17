from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

from saml2 import BINDING_HTTP_POST
from saml2.client import Saml2Client
from saml2.config import SPConfig
from scim2_models import Email, Name
from scim2_models import User as ScimUser

from litellm.proxy.auth_v2.config import SAMLConfig

_SINGLE_VALUE_TARGETS = {
    "email",
    "given_name",
    "family_name",
    "user_name",
    "display_name",
}
_MULTI_VALUE_TARGETS = ("groups", "roles")


def map_attributes(ava: Dict[str, Any], attribute_map: Dict[str, str]) -> Dict[str, Any]:
    mapped: Dict[str, Any] = {}
    for saml_attr, target in attribute_map.items():
        if saml_attr not in ava:
            continue
        value = ava[saml_attr]
        if target in _SINGLE_VALUE_TARGETS:
            scalar = value[0] if isinstance(value, list) and value else value
            mapped.setdefault(target, scalar)
        elif target in _MULTI_VALUE_TARGETS and isinstance(value, list):
            mapped[target] = value
    return mapped


def _formatted_name(mapped: Dict[str, Any]) -> Optional[str]:
    if mapped.get("display_name"):
        return mapped["display_name"]
    parts: List[str] = [part for part in (mapped.get("given_name"), mapped.get("family_name")) if part]
    return " ".join(parts) if parts else None


def user_from_mapped(name_id: str, mapped: Dict[str, Any]) -> ScimUser:
    display = _formatted_name(mapped)
    user = ScimUser(
        external_id=name_id,
        user_name=mapped.get("user_name") or mapped.get("email") or name_id,
        display_name=display,
    )
    if mapped.get("given_name") or mapped.get("family_name"):
        user.name = Name(
            given_name=mapped.get("given_name"),
            family_name=mapped.get("family_name"),
            formatted=display,
        )
    if mapped.get("email"):
        user.emails = [Email(value=mapped["email"], primary=True)]
    return user


def claims_from_mapped(mapped: Dict[str, Any]) -> Dict[str, Any]:
    claims: Dict[str, Any] = {}
    if mapped.get("email"):
        claims["email"] = mapped["email"]
    if mapped.get("user_name"):
        claims["preferred_username"] = mapped["user_name"]
    display = _formatted_name(mapped)
    if display:
        claims["name"] = display
    for target in _MULTI_VALUE_TARGETS:
        if mapped.get(target):
            claims[target] = mapped[target]
    return claims


def _metadata_source(idp_metadata: str) -> Dict[str, Any]:
    stripped = idp_metadata.strip()
    if stripped.startswith("<"):
        return {"inline": [idp_metadata]}
    if stripped.startswith("http://") or stripped.startswith("https://"):
        return {"remote": [{"url": idp_metadata}]}
    return {"local": [idp_metadata]}


def _sp_config_dict(config: SAMLConfig) -> Dict[str, Any]:
    cfg: Dict[str, Any] = {
        "entityid": config.entity_id,
        "service": {
            "sp": {
                "endpoints": {"assertion_consumer_service": [(config.acs_url, BINDING_HTTP_POST)]},
                "allow_unsolicited": config.allow_unsolicited,
                "authn_requests_signed": False,
                "want_assertions_signed": True,
                "want_response_signed": False,
            }
        },
        "metadata": _metadata_source(config.idp_metadata),
        "allow_unknown_attributes": True,
    }
    if config.sp_key_file:
        cfg["key_file"] = config.sp_key_file
    if config.sp_cert_file:
        cfg["cert_file"] = config.sp_cert_file
    if config.xmlsec_binary:
        cfg["xmlsec_binary"] = config.xmlsec_binary
    return cfg


def build_sp_client(config: SAMLConfig) -> Saml2Client:
    conf = SPConfig()
    conf.load(_sp_config_dict(config))
    return Saml2Client(config=conf)


class SAMLProtocolStore:
    def __init__(
        self,
        replay_ttl_seconds: int,
        outstanding_ttl_seconds: int = 300,
        max_outstanding: int = 10000,
    ) -> None:
        self._outstanding: Dict[str, Tuple[float, str]] = {}
        self._seen_assertions: Dict[str, float] = {}
        self._replay_ttl = replay_ttl_seconds
        self._outstanding_ttl = outstanding_ttl_seconds
        self._max_outstanding = max_outstanding

    def remember_request(self, request_id: str, relay_state: str) -> None:
        now = time.time()
        self._evict_outstanding(now)
        self._outstanding[request_id] = (now + self._outstanding_ttl, relay_state)

    def outstanding_relays(self) -> Dict[str, str]:
        now = time.time()
        return {rid: relay for rid, (exp, relay) in self._outstanding.items() if exp >= now}

    def consume_request(self, request_id: str) -> Optional[str]:
        entry = self._outstanding.pop(request_id, None)
        if entry is None:
            return None
        expires_at, relay = entry
        return relay if expires_at >= time.time() else None

    def _evict_outstanding(self, now: float) -> None:
        for rid in [r for r, (exp, _) in self._outstanding.items() if exp < now]:
            self._outstanding.pop(rid, None)
        overflow = len(self._outstanding) - self._max_outstanding + 1
        if overflow > 0:
            oldest = sorted(self._outstanding, key=lambda r: self._outstanding[r][0])
            for rid in oldest[:overflow]:
                self._outstanding.pop(rid, None)

    def consume_assertion(self, assertion_id: str) -> bool:
        now = time.time()
        self._seen_assertions = {aid: exp for aid, exp in self._seen_assertions.items() if exp >= now}
        if assertion_id in self._seen_assertions:
            return False
        self._seen_assertions[assertion_id] = now + self._replay_ttl
        return True
