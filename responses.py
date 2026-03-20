from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from functools import wraps
from typing import Any
from unittest.mock import patch

import requests

GET = "GET"
POST = "POST"
PUT = "PUT"
PATCH = "PATCH"
DELETE = "DELETE"


class _Call:
    def __init__(self, request: requests.PreparedRequest):
        self.request = request


class _Calls(Sequence[_Call]):
    def __len__(self) -> int:
        return len(_get_active_mock().calls)

    def __getitem__(self, index: int) -> _Call:
        return _get_active_mock().calls[index]

    def __iter__(self) -> Iterator[_Call]:
        return iter(_get_active_mock().calls)


class _RegisteredResponse:
    def __init__(self, method: str, url: str, **kwargs: Any):
        self.method = method.upper()
        self.url = requests.Request(method=self.method, url=url).prepare().url
        self.status_code = kwargs.pop("status", kwargs.pop("status_code", 200))
        self.json_body = kwargs.pop("json", None)
        self.body = kwargs.pop("body", None)
        self.headers = kwargs.pop("headers", None) or {}
        if kwargs:
            raise TypeError(f"Unsupported responses kwargs: {sorted(kwargs)}")


class RequestsMock:
    def __init__(self) -> None:
        self._registered: list[_RegisteredResponse] = []
        self.calls: list[_Call] = []
        self._patcher: patch | None = None

    def __enter__(self) -> "RequestsMock":
        _ACTIVE_MOCKS.append(self)
        self._patcher = patch.object(
            requests.sessions.Session,
            "request",
            new=self._request,
        )
        self._patcher.start()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if self._patcher is not None:
            self._patcher.stop()
        _ACTIVE_MOCKS.pop()

    def add(self, method: str, url: str, **kwargs: Any) -> None:
        self._registered.append(_RegisteredResponse(method, url, **kwargs))

    def register_uri(self, method: str, url: str, **kwargs: Any) -> None:
        self.add(method, url, **kwargs)

    def _request(
        self,
        method: str,
        url: str,
        **kwargs: Any,
    ) -> requests.Response:
        request = requests.Request(
            method=method,
            url=url,
            headers=kwargs.get("headers"),
            params=kwargs.get("params"),
            data=kwargs.get("data"),
            json=kwargs.get("json"),
        ).prepare()

        registered = self._match(request.method, request.url)
        self.calls.append(_Call(request))

        response = requests.Response()
        response.status_code = registered.status_code
        response.request = request
        response.url = request.url
        response.headers.update(registered.headers)

        if registered.json_body is not None:
            response.headers.setdefault("Content-Type", "application/json")
            response._content = json.dumps(registered.json_body).encode("utf-8")
        elif registered.body is not None:
            body = registered.body
            response._content = body if isinstance(body, bytes) else str(body).encode("utf-8")
        else:
            response._content = b""

        return response

    def _match(self, method: str, url: str) -> _RegisteredResponse:
        for registered in self._registered:
            if registered.method == method.upper() and registered.url == url:
                return registered
        raise AssertionError(f"No mocked response for {method} {url}")


_ACTIVE_MOCKS: list[RequestsMock] = []
calls = _Calls()


def _get_active_mock() -> RequestsMock:
    if not _ACTIVE_MOCKS:
        raise RuntimeError("responses mock is not active")
    return _ACTIVE_MOCKS[-1]


def add(method: str, url: str, **kwargs: Any) -> None:
    _get_active_mock().add(method, url, **kwargs)


def activate(func):
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any):
        with RequestsMock():
            return func(*args, **kwargs)

    return wrapper
