from types import MappingProxyType
from typing import Final, NamedTuple
from urllib.parse import urlparse

from litellm.constants import DEFAULT_AZURE_AUTHORITY_HOST


class AzureCloud(NamedTuple):
    microsoft_graph_base: str
    azure_monitor_scope: str


_PUBLIC_CLOUD: Final = AzureCloud(
    microsoft_graph_base="https://graph.microsoft.com",
    azure_monitor_scope="https://monitor.azure.com/.default",
)
_US_GOVERNMENT_CLOUD: Final = AzureCloud(
    microsoft_graph_base="https://graph.microsoft.us",
    azure_monitor_scope="https://monitor.azure.us/.default",
)
_CHINA_CLOUD: Final = AzureCloud(
    microsoft_graph_base="https://microsoftgraph.chinacloudapi.cn",
    azure_monitor_scope="https://monitor.azure.cn/.default",
)

_CLOUD_BY_AUTHORITY_HOSTNAME: Final = MappingProxyType(
    {
        "login.microsoftonline.com": _PUBLIC_CLOUD,
        "login.microsoftonline.us": _US_GOVERNMENT_CLOUD,
        "login.partner.microsoftonline.cn": _CHINA_CLOUD,
        "login.chinacloudapi.cn": _CHINA_CLOUD,
    }
)


def normalize_azure_authority_host(authority_host: str) -> str:
    """
    Turn a Microsoft Entra authority into an https origin with no trailing slash.

    Accepts the scheme-qualified form litellm documents ("https://login.microsoftonline.us") and the
    bare-host form the azure-identity AzureAuthorityHosts constants use. Anything other than an https
    origin is rejected because the client secret is posted to this URL.
    """
    stripped: Final = authority_host.strip()
    parsed: Final = urlparse(stripped if "://" in stripped else f"https://{stripped}")
    is_https_origin: Final = (
        parsed.scheme == "https"
        and bool(parsed.hostname)
        and parsed.username is None
        and parsed.path in ("", "/")
        and not parsed.params
        and not parsed.query
        and not parsed.fragment
    )
    if not is_https_origin:
        raise ValueError(
            f"Azure authority host must be an https origin with no path, e.g. {DEFAULT_AZURE_AUTHORITY_HOST}; "
            f"got {authority_host!r}"
        )
    return f"https://{parsed.netloc}"


def get_azure_cloud(authority_host: str) -> AzureCloud:
    """Resolve the cloud an Entra authority belongs to, defaulting to Azure Public Cloud for unknown hosts."""
    hostname: Final = urlparse(authority_host).hostname or ""
    return _CLOUD_BY_AUTHORITY_HOSTNAME.get(hostname, _PUBLIC_CLOUD)
