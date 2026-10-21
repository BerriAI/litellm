import httpx

from poc.profile_candidates import Candidate, RegistryPacer, _lookup_image


def _candidate() -> Candidate:
    return Candidate(
        task_id="example__repo-1",
        problem_statement="A sanitized problem statement.",
        image="swebench/sweb.eval.x86_64.example_1776_repo-1",
    )


def test_registry_404_is_the_only_manifest_status_recorded_as_unavailable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "auth.docker.io":
            return httpx.Response(200, json={"token": "anonymous"})
        return httpx.Response(404)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        result = _lookup_image(client, RegistryPacer(), _candidate())
    assert result.exists is False
    assert result.error is None
    assert result.manifest_response_received


def test_registry_401_is_an_error_not_an_unavailable_image() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "auth.docker.io":
            return httpx.Response(200, json={"token": "anonymous"})
        return httpx.Response(401)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        result = _lookup_image(client, RegistryPacer(), _candidate())
    assert result.exists is None
    assert result.error == "manifest_head_http_401"
    assert result.manifest_response_received
