from __future__ import annotations

from dataclasses import dataclass

import httpx
from pydantic import BaseModel


class TurnstileResponse(BaseModel):
    success: bool


@dataclass(frozen=True, slots=True)
class TurnstileVerifier:
    verify_url: str

    async def verify(self, token: str, remote_ip: str | None) -> bool:
        body = {"token": token, **({"remoteip": remote_ip} if remote_ip is not None else {})}
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(self.verify_url, json=body)
        if response.status_code != 200:
            return False
        return TurnstileResponse.model_validate(response.json()).success
