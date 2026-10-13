from litellm.proxy.control_plane.audit.log import AuditLog


def get_audit_log() -> AuditLog:
    raise RuntimeError("get_audit_log is not bound, build the app with create_app")
