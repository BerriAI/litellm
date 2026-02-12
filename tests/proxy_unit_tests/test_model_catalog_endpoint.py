"""
Tests for the /model_catalog Stripe-style endpoint.
"""

import pytest
from fastapi.testclient import TestClient
from fastapi import FastAPI

from litellm.proxy.model_catalog_endpoint import router


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


class TestModelCatalogList:
    def test_basic_list(self, client):
        resp = client.get("/model_catalog")
        assert resp.status_code == 200
        body = resp.json()
        assert body["object"] == "list"
        assert isinstance(body["data"], list)
        assert body["total_count"] > 0
        assert body["page"] == 1
        assert body["page_size"] == 50
        assert len(body["data"]) <= 50

    def test_pagination(self, client):
        resp1 = client.get("/model_catalog?page=1&page_size=5")
        body1 = resp1.json()
        assert len(body1["data"]) == 5
        assert body1["has_more"] is True

        resp2 = client.get("/model_catalog?page=2&page_size=5")
        body2 = resp2.json()
        assert body2["page"] == 2
        # Pages should have different models
        ids1 = {e["id"] for e in body1["data"]}
        ids2 = {e["id"] for e in body2["data"]}
        assert ids1.isdisjoint(ids2)

    def test_filter_by_provider(self, client):
        resp = client.get("/model_catalog?provider=openai&page_size=10")
        body = resp.json()
        for entry in body["data"]:
            assert entry["provider"] == "openai"

    def test_filter_by_mode(self, client):
        resp = client.get("/model_catalog?mode=chat&page_size=10")
        body = resp.json()
        for entry in body["data"]:
            assert entry["mode"] == "chat"

    def test_filter_by_model_name_substring(self, client):
        resp = client.get("/model_catalog?model=gpt-4o&page_size=10")
        body = resp.json()
        assert body["total_count"] > 0
        for entry in body["data"]:
            assert "gpt-4o" in entry["id"].lower()

    def test_filter_by_model_name_regex(self, client):
        resp = client.get("/model_catalog?model=re:^gpt-4o$&page_size=10")
        body = resp.json()
        assert body["total_count"] >= 1
        for entry in body["data"]:
            assert entry["id"] == "gpt-4o"

    def test_invalid_regex(self, client):
        resp = client.get("/model_catalog?model=re:[invalid")
        assert resp.status_code == 400

    def test_entry_structure(self, client):
        resp = client.get("/model_catalog?model=gpt-4o&mode=chat&provider=openai&page_size=1")
        body = resp.json()
        if body["total_count"] > 0:
            entry = body["data"][0]
            assert entry["object"] == "model_catalog.entry"
            assert "id" in entry
            assert "provider" in entry

    def test_page_size_limit(self, client):
        resp = client.get("/model_catalog?page_size=999")
        assert resp.status_code == 422  # validation error

    def test_filter_supports_vision(self, client):
        resp = client.get("/model_catalog?supports_vision=true&page_size=5")
        body = resp.json()
        for entry in body["data"]:
            assert entry.get("supports_vision") is True


class TestModelCatalogSingle:
    def test_get_existing_model(self, client):
        resp = client.get("/model_catalog/gpt-4o")
        assert resp.status_code == 200
        body = resp.json()
        assert body["id"] == "gpt-4o"
        assert body["object"] == "model_catalog.entry"

    def test_get_nonexistent_model(self, client):
        resp = client.get("/model_catalog/nonexistent-model-xyz")
        assert resp.status_code == 404
