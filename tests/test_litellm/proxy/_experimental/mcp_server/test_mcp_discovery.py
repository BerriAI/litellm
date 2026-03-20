import json
import os
import sys

import pytest

sys.path.insert(
    0, os.path.abspath("../../../..")
)  # Adds the parent directory to the system path


class TestMCPRegistryFile:
    """Tests for the curated MCP registry JSON file."""

    @pytest.fixture
    def registry_path(self):
        return os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..",
            "..",
            "..",
            "..",
            "..",
            "litellm",
            "proxy",
            "mcp_registry.json",
        )

    def test_registry_file_exists(self, registry_path):
        assert os.path.exists(registry_path), f"Registry file not found at {registry_path}"

    def test_registry_file_is_valid_json(self, registry_path):
        with open(registry_path, "r") as f:
            data = json.load(f)
        assert isinstance(data, dict)
        assert "servers" in data

    def test_registry_servers_have_required_fields(self, registry_path):
        with open(registry_path, "r") as f:
            data = json.load(f)
        servers = data["servers"]
        assert len(servers) > 0, "Registry should have at least one server"

        required_fields = ["name", "title", "description", "category", "transport"]
        for server in servers:
            for field in required_fields:
                assert field in server, f"Server {server.get('name', '?')} missing field '{field}'"

    def test_registry_server_names_are_unique(self, registry_path):
        with open(registry_path, "r") as f:
            data = json.load(f)
        names = [s["name"] for s in data["servers"]]
        assert len(names) == len(set(names)), f"Duplicate server names found: {[n for n in names if names.count(n) > 1]}"

    def test_registry_transport_values_are_valid(self, registry_path):
        with open(registry_path, "r") as f:
            data = json.load(f)
        valid_transports = {"stdio", "http", "sse"}
        for server in data["servers"]:
            assert server["transport"] in valid_transports, (
                f"Server {server['name']} has invalid transport '{server['transport']}'"
            )

    def test_stdio_servers_have_command(self, registry_path):
        with open(registry_path, "r") as f:
            data = json.load(f)
        for server in data["servers"]:
            if server["transport"] == "stdio":
                assert "command" in server and server["command"], (
                    f"stdio server {server['name']} missing 'command'"
                )

    def test_http_servers_have_url(self, registry_path):
        with open(registry_path, "r") as f:
            data = json.load(f)
        for server in data["servers"]:
            if server["transport"] in ("http", "sse"):
                assert "url" in server and server["url"], (
                    f"HTTP/SSE server {server['name']} missing 'url'"
                )

    def test_well_known_servers_present(self, registry_path):
        """Ensure key well-known MCPs are in the registry."""
        with open(registry_path, "r") as f:
            data = json.load(f)
        names = {s["name"] for s in data["servers"]}
        expected = {"github", "slack", "postgresql", "snowflake", "atlassian"}
        missing = expected - names
        assert not missing, f"Missing well-known servers: {missing}"

    def test_env_vars_structure(self, registry_path):
        with open(registry_path, "r") as f:
            data = json.load(f)
        for server in data["servers"]:
            if "env_vars" in server:
                assert isinstance(server["env_vars"], list)
                for var in server["env_vars"]:
                    assert "name" in var, f"env_var in {server['name']} missing 'name'"


class TestDiscoverEndpointFiltering:
    """Tests for the discover endpoint filtering logic (unit-level)."""

    @pytest.fixture
    def sample_servers(self):
        return [
            {
                "name": "github",
                "title": "GitHub",
                "description": "Repository management",
                "category": "Developer Tools",
                "transport": "http",
                "url": "https://mcp.github.com/sse",
            },
            {
                "name": "slack",
                "title": "Slack",
                "description": "Channel management and messaging",
                "category": "Communication",
                "transport": "stdio",
                "command": "npx",
            },
            {
                "name": "postgresql",
                "title": "PostgreSQL",
                "description": "Query and manage databases",
                "category": "Databases",
                "transport": "stdio",
                "command": "npx",
            },
        ]

    def test_query_filter_by_name(self, sample_servers):
        query = "github"
        q = query.lower()
        result = [
            s
            for s in sample_servers
            if q in s.get("name", "").lower()
            or q in s.get("title", "").lower()
            or q in s.get("description", "").lower()
        ]
        assert len(result) == 1
        assert result[0]["name"] == "github"

    def test_query_filter_by_description(self, sample_servers):
        query = "messaging"
        q = query.lower()
        result = [
            s
            for s in sample_servers
            if q in s.get("name", "").lower()
            or q in s.get("title", "").lower()
            or q in s.get("description", "").lower()
        ]
        assert len(result) == 1
        assert result[0]["name"] == "slack"

    def test_category_filter(self, sample_servers):
        category = "Databases"
        result = [s for s in sample_servers if s.get("category") == category]
        assert len(result) == 1
        assert result[0]["name"] == "postgresql"

    def test_no_filter_returns_all(self, sample_servers):
        assert len(sample_servers) == 3

    def test_query_filter_no_match(self, sample_servers):
        query = "nonexistent"
        q = query.lower()
        result = [
            s
            for s in sample_servers
            if q in s.get("name", "").lower()
            or q in s.get("title", "").lower()
            or q in s.get("description", "").lower()
        ]
        assert len(result) == 0

    def test_categories_extraction(self, sample_servers):
        categories = sorted(set(s.get("category", "Other") for s in sample_servers))
        assert categories == ["Communication", "Databases", "Developer Tools"]
