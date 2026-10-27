from litellm.litellm_core_utils.deprecated_package import alias_deprecated_package

__getattr__ = alias_deprecated_package(__name__, "litellm.proxy.mcp_server")
