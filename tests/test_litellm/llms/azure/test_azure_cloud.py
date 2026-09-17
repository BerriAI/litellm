import pytest

from litellm.constants import DEFAULT_AZURE_AUTHORITY_HOST
from litellm.llms.azure.azure_cloud import get_azure_cloud, normalize_azure_authority_host


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("https://login.microsoftonline.us", "https://login.microsoftonline.us"),
        ("https://login.microsoftonline.us/", "https://login.microsoftonline.us"),
        ("  login.microsoftonline.us  ", "https://login.microsoftonline.us"),
        ("login.partner.microsoftonline.cn", "https://login.partner.microsoftonline.cn"),
        ("https://adfs.contoso.example:8443", "https://adfs.contoso.example:8443"),
    ],
)
def test_normalize_azure_authority_host_accepts_scheme_qualified_and_bare_hosts(raw, expected):
    assert normalize_azure_authority_host(raw) == expected


@pytest.mark.parametrize(
    "raw",
    [
        "http://login.microsoftonline.us",
        "https://login.microsoftonline.us/common",
        "https://login.microsoftonline.us/?x=1",
        "https://login.microsoftonline.us#frag",
        "https://user@login.microsoftonline.us",
        "ftp://login.microsoftonline.us",
        "https://",
        "",
    ],
)
def test_normalize_azure_authority_host_rejects_anything_but_an_https_origin(raw):
    with pytest.raises(ValueError, match="https origin with no path"):
        normalize_azure_authority_host(raw)


# Graph and Monitor audiences per cloud: https://management.azure.com/metadata/endpoints?api-version=2022-09-01
# and the usgov / china ARM equivalents, plus Azure Monitor Logs Ingestion docs (checked 2026-09-17)
@pytest.mark.parametrize(
    "authority_host, graph_base, monitor_scope",
    [
        (DEFAULT_AZURE_AUTHORITY_HOST, "https://graph.microsoft.com", "https://monitor.azure.com/.default"),
        ("https://login.microsoftonline.us", "https://graph.microsoft.us", "https://monitor.azure.us/.default"),
        (
            "https://login.partner.microsoftonline.cn",
            "https://microsoftgraph.chinacloudapi.cn",
            "https://monitor.azure.cn/.default",
        ),
        (
            "https://login.chinacloudapi.cn",
            "https://microsoftgraph.chinacloudapi.cn",
            "https://monitor.azure.cn/.default",
        ),
    ],
)
def test_get_azure_cloud_maps_each_entra_authority_to_the_same_cloud_audiences(
    authority_host, graph_base, monitor_scope
):
    cloud = get_azure_cloud(authority_host)

    assert cloud.microsoft_graph_base == graph_base
    assert cloud.azure_monitor_scope == monitor_scope


def test_get_azure_cloud_falls_back_to_public_cloud_for_an_unknown_authority():
    assert get_azure_cloud("https://adfs.contoso.example") == get_azure_cloud(DEFAULT_AZURE_AUTHORITY_HOST)
