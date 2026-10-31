"""Google Vertex AI Agent Engine session API adapter."""

from collections.abc import Mapping

import httpx


class VertexAgentEngineSessionBackend:
    """List and create sessions through the Agent Engine query API."""

    def __init__(
        self,
        *,
        client: httpx.AsyncClient,
        query_url: str,
        auth_headers: Mapping[str, str],
    ) -> None:
        self.client = client
        self.query_url = query_url
        self.auth_headers = dict(auth_headers)

    async def _query(
        self,
        method: str,
        user_id: str,
    ) -> dict:
        response = await self.client.post(
            self.query_url,
            headers=self.auth_headers,
            json={
                "class_method": method,
                "input": {"user_id": user_id},
            },
        )
        response.raise_for_status()

        result = response.json()
        if not isinstance(result, dict):
            raise ValueError("Invalid Agent Engine response")

        output = result.get("output")
        if not isinstance(output, dict):
            raise ValueError("Missing Agent Engine output")

        return output

    async def list_sessions(self, *, user_id: str) -> list[str]:
        output = await self._query("async_list_sessions", user_id)
        sessions = output.get("sessions", [])

        if not isinstance(sessions, list):
            raise ValueError("Invalid Agent Engine session list")

        session_ids = []
        for session in sessions:
            if not isinstance(session, dict):
                raise ValueError("Invalid Agent Engine session")

            session_id = session.get("id")
            if not isinstance(session_id, str) or not session_id:
                raise ValueError("Missing Agent Engine session ID")

            session_ids.append(session_id)

        return session_ids

    async def create_session(self, *, user_id: str) -> str:
        output = await self._query("async_create_session", user_id)

        session_id = output.get("id")
        if not isinstance(session_id, str) or not session_id:
            raise ValueError("Missing newly created Agent Engine session ID")

        return session_id
