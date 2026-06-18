"""The AWS SigV4 signer factory, isolated behind a port.

`resolve()` delegates the whole `aws_sigv4` arm here: given the server's AWS config, return an
`httpx.Auth` that signs each outbound request with SigV4, or fail closed. The production body is
backed by botocore - it matches on the credential source (static keys / assumed role / ambient
chain), eagerly resolves the credentials so failures surface here rather than mid-request, and
lets botocore refresh temporary (STS) credentials at sign time. Faked in tests.
"""

from __future__ import annotations

from typing import Protocol

import httpx

from ..result import Result
from .types import AwsSigV4Config, CredError


class SignerFactory(Protocol):
    """Builds the per-request SigV4 `httpx.Auth` for an AWS-hosted upstream, or fails closed.

    Returns `misconfigured` when the credentials cannot be resolved (an unassumable role, no
    ambient credentials) and `upstream_unavailable` when STS cannot be reached. The gateway
    signs with its own AWS identity; the caller's identity is never involved.
    """

    async def build(self, config: AwsSigV4Config) -> Result[httpx.Auth, CredError]: ...
