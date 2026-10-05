"""SigV4 re-signing for Bedrock traffic routed through the provider edge.

Bedrock is the one provider the edge could never mount. SigV4 signs the Host
header, so rewriting ``api_base`` to point at the edge invalidates the proxy's
signature and Bedrock rejects the call before it reaches a model. The edge
therefore has to drop the proxy's signature and mint its own over the upstream
URL it is actually about to call.

The identity it signs with is the run pod's own, from the EKS Pod Identity
association on ServiceAccount ``buildkite-e2e-run``. That role carries Bedrock
invoke and converse on an allowlist of the Anthropic models the suite registers
and nothing else, so a re-signed call can reach exactly the models the suite
already uses. The proxy's own Bedrock credentials are not involved in a routed
deployment, which is why ``aws_role_name`` deployments stay off the edge: their
whole point is to prove the product's assume-role chain.

Signature headers are excluded from the cache key by the caller, and they have
to be: ``x-amz-date`` is a timestamp, so keying on it would make every Bedrock
request a permanent miss.
"""

from __future__ import annotations

import functools
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Final

from botocore.auth import SigV4Auth
from botocore.awsrequest import AWSRequest
from botocore.credentials import Credentials
from botocore.session import Session
from provider_cache import SIGNATURE_HEADERS

BEDROCK_SERVICE: Final = "bedrock"


class MissingAwsCredentials(RuntimeError):
    """No AWS identity is resolvable, so the edge cannot sign for Bedrock."""


@dataclass(frozen=True, slots=True)
class BedrockSigner:
    region: str
    credentials: Callable[[], Credentials]

    def __call__(self, method: str, url: str, headers: Mapping[str, str], body: bytes | None) -> dict[str, str]:
        unsigned: Final = {
            name: value for name, value in headers.items() if name.lower() not in SIGNATURE_HEADERS
        }
        request: Final = AWSRequest(method=method, url=url, headers=unsigned, data=body or b"")
        SigV4Auth(self.credentials(), BEDROCK_SERVICE, self.region).add_auth(request)
        return dict(request.headers)


@functools.lru_cache(maxsize=1)
def pod_credentials() -> Credentials:
    """The run pod's own identity, resolved once per process through botocore's
    ordinary chain, which reaches Pod Identity at the ``container-role`` link."""
    resolved: Final = Session().get_credentials()
    if resolved is None:  # pyright: ignore[reportUnnecessaryComparison]  # stubs miss the empty-chain None
        raise MissingAwsCredentials(
            "the provider edge is mounted for Bedrock but no AWS credentials resolve; "
            "the run pod gets them from the Pod Identity association on buildkite-e2e-run"
        )
    return resolved


def bedrock_signer(region: str, credentials: Callable[[], Credentials] = pod_credentials) -> BedrockSigner:
    """Credentials are resolved on the first signed request, not here, so a run
    that mounts Bedrock but never calls it needs no AWS identity at all."""
    return BedrockSigner(region, credentials)
