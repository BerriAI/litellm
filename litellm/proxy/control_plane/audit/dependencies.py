from typing import Final

from litellm.proxy.control_plane.audit.log import AuditLog, build_audit_log

_UNBOUND: Final = build_audit_log(capacity=1)


def get_audit_log() -> AuditLog:
    """Overridden in create_app. A throwaway ring by default so a bare router still works in tests."""
    return _UNBOUND
