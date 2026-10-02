import json
import os
from typing import Final

import pytest



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
        assert os.path.exists(
            registry_path
        ), f"Registry file not found at {registry_path}"

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
                assert (
                    field in server
                ), f"Server {server.get('name', '?')} missing field '{field}'"

    def test_registry_server_names_are_unique(self, registry_path):
        with open(registry_path, "r") as f:
            data = json.load(f)
        names = [s["name"] for s in data["servers"]]
        assert len(names) == len(
            set(names)
        ), f"Duplicate server names found: {[n for n in names if names.count(n) > 1]}"

    def test_registry_transport_values_are_valid(self, registry_path):
        with open(registry_path, "r") as f:
            data = json.load(f)
        valid_transports = {"stdio", "http", "sse"}
        for server in data["servers"]:
            assert (
                server["transport"] in valid_transports
            ), f"Server {server['name']} has invalid transport '{server['transport']}'"

    def test_stdio_servers_have_command(self, registry_path):
        with open(registry_path, "r") as f:
            data = json.load(f)
        for server in data["servers"]:
            if server["transport"] == "stdio":
                assert (
                    "command" in server and server["command"]
                ), f"stdio server {server['name']} missing 'command'"

    def test_http_servers_have_url(self, registry_path):
        with open(registry_path, "r") as f:
            data = json.load(f)
        for server in data["servers"]:
            if server["transport"] in ("http", "sse"):
                assert (
                    "url" in server and server["url"]
                ), f"HTTP/SSE server {server['name']} missing 'url'"

    def test_linear_uses_streamable_http(self, registry_path):
        """Linear's MCP server should default to streamable HTTP at /mcp, not SSE at /sse."""
        with open(registry_path, "r") as f:
            data = json.load(f)
        linear = next(s for s in data["servers"] if s["name"] == "linear")
        assert linear["transport"] == "http"
        assert linear["url"] == "https://mcp.linear.app/mcp"
        assert "/sse" not in linear["url"]

    def test_well_known_servers_present(self, registry_path):
        """Ensure key well-known MCPs are in the registry."""
        with open(registry_path, "r") as f:
            data = json.load(f)
        names = {s["name"] for s in data["servers"]}
        expected = {"github", "slack", "postgresql", "snowflake", "atlassian", "microsoft_365"}
        missing = expected - names
        assert not missing, f"Missing well-known servers: {missing}"

    def test_microsoft_365_is_a_self_hosted_streamable_http_server(self, registry_path):
        """The Graph server runs next to the proxy in org mode, so the entry must be streamable HTTP at /mcp."""
        with open(registry_path, "r") as f:
            data = json.load(f)
        entry: Final = next(s for s in data["servers"] if s["name"] == "microsoft_365")
        assert entry["transport"] == "http"
        assert entry["url"].endswith("/mcp")
        assert entry["category"] == "Productivity"
        assert "ms-365-mcp-server" in entry["registry_url"]

    def test_bundled_icons_exist(self, registry_path):
        """An icon served from the proxy's own assets ships twice, as the built copy the wheel packages and as
        the dashboard source copy every Docker image rebuilds from. Both must exist and match or a card goes blank."""
        with open(registry_path, "r") as f:
            data = json.load(f)
        proxy_dir: Final = os.path.dirname(registry_path)
        built_logos_dir: Final = os.path.join(proxy_dir, "_experimental", "out", "assets", "logos")
        source_logos_dir: Final = os.path.join(
            proxy_dir, "..", "..", "ui", "litellm-dashboard", "public", "assets", "logos"
        )
        bundled: Final = [s for s in data["servers"] if s.get("icon_url", "").startswith("/ui/assets/logos/")]
        assert bundled, "at least one registry entry ships its own icon"
        for server in bundled:
            file_name: Final = os.path.basename(server["icon_url"])
            built: Final = os.path.join(built_logos_dir, file_name)
            source: Final = os.path.join(source_logos_dir, file_name)
            assert os.path.isfile(built), f"{server['name']}: {server['icon_url']} missing from the built dashboard"
            assert os.path.isfile(source), f"{server['name']}: {server['icon_url']} missing from the dashboard source"
            with open(built, "rb") as built_file, open(source, "rb") as source_file:
                same_bytes: Final = built_file.read() == source_file.read()
            assert same_bytes, f"{server['name']}: built and source copies of {file_name} differ"

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
