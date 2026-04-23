"""
Tests for SCIM v2 resource discovery endpoints:
- GET /scim/v2 (base endpoint)
- GET /scim/v2/ResourceTypes
- GET /scim/v2/ResourceTypes/{id}
- GET /scim/v2/Schemas
- GET /scim/v2/Schemas/{uri}
"""

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException

from litellm.proxy.management_endpoints.scim.scim_v2 import (
    _get_resource_types,
    _get_schemas,
    get_resource_type,
    get_resource_types,
    get_schema,
    get_schemas,
    get_scim_base,
)
from litellm.types.proxy.management_endpoints.scim_v2 import (
    SCIMResourceType,
    SCIMSchema,
)


def _make_mock_request(base_url="http://localhost:4000/", url="http://localhost:4000/scim/v2"):
    """Create a mock FastAPI Request object."""
    request = MagicMock()
    request.method = "GET"
    request.url = url
    request.base_url = base_url
    return request


# ---- Helper function tests ----


class TestGetResourceTypes:
    def test_returns_user_and_group(self):
        resource_types = _get_resource_types()
        assert len(resource_types) == 2
        ids = [rt.id for rt in resource_types]
        assert "User" in ids
        assert "Group" in ids

    def test_user_resource_type_fields(self):
        resource_types = _get_resource_types()
        user_rt = next(rt for rt in resource_types if rt.id == "User")
        assert user_rt.name == "User"
        assert user_rt.endpoint == "/Users"
        assert user_rt.schema_ == "urn:ietf:params:scim:schemas:core:2.0:User"
        assert user_rt.schemas == ["urn:ietf:params:scim:schemas:core:2.0:ResourceType"]

    def test_group_resource_type_fields(self):
        resource_types = _get_resource_types()
        group_rt = next(rt for rt in resource_types if rt.id == "Group")
        assert group_rt.name == "Group"
        assert group_rt.endpoint == "/Groups"
        assert group_rt.schema_ == "urn:ietf:params:scim:schemas:core:2.0:Group"

    def test_custom_base_url(self):
        resource_types = _get_resource_types("https://example.com/scim/v2")
        user_rt = next(rt for rt in resource_types if rt.id == "User")
        assert user_rt.meta["location"] == "https://example.com/scim/v2/ResourceTypes/User"

    def test_model_dump_uses_schema_key(self):
        """Ensure model_dump() outputs 'schema' not 'schema_'."""
        resource_types = _get_resource_types()
        dumped = resource_types[0].model_dump()
        assert "schema" in dumped
        assert "schema_" not in dumped


class TestGetSchemas:
    def test_returns_user_and_group_schemas(self):
        schemas = _get_schemas()
        assert len(schemas) == 2
        ids = [s.id for s in schemas]
        assert "urn:ietf:params:scim:schemas:core:2.0:User" in ids
        assert "urn:ietf:params:scim:schemas:core:2.0:Group" in ids

    def test_user_schema_has_required_attributes(self):
        schemas = _get_schemas()
        user_schema = next(
            s for s in schemas if s.id == "urn:ietf:params:scim:schemas:core:2.0:User"
        )
        attr_names = [a.name for a in user_schema.attributes]
        assert "userName" in attr_names
        assert "name" in attr_names
        assert "emails" in attr_names
        assert "active" in attr_names
        assert "groups" in attr_names

    def test_group_schema_has_required_attributes(self):
        schemas = _get_schemas()
        group_schema = next(
            s for s in schemas if s.id == "urn:ietf:params:scim:schemas:core:2.0:Group"
        )
        attr_names = [a.name for a in group_schema.attributes]
        assert "displayName" in attr_names
        assert "members" in attr_names

    def test_schema_meta_fields(self):
        schemas = _get_schemas()
        user_schema = next(
            s for s in schemas if s.id == "urn:ietf:params:scim:schemas:core:2.0:User"
        )
        assert user_schema.meta is not None
        assert user_schema.meta["resourceType"] == "Schema"


# ---- Endpoint tests ----


class TestGetScimBase:
    @pytest.mark.asyncio
    async def test_returns_list_response(self):
        request = _make_mock_request()
        result = await get_scim_base(request)

        assert result["schemas"] == ["urn:ietf:params:scim:api:messages:2.0:ListResponse"]
        assert result["totalResults"] == 2
        assert len(result["Resources"]) == 2

    @pytest.mark.asyncio
    async def test_resources_contain_user_and_group(self):
        request = _make_mock_request()
        result = await get_scim_base(request)

        resource_ids = [r["id"] for r in result["Resources"]]
        assert "User" in resource_ids
        assert "Group" in resource_ids

    @pytest.mark.asyncio
    async def test_resources_have_schema_field(self):
        """Each resource should have 'schema' (not 'schema_') per SCIM spec."""
        request = _make_mock_request()
        result = await get_scim_base(request)

        for resource in result["Resources"]:
            assert "schema" in resource
            assert "schema_" not in resource

    @pytest.mark.asyncio
    async def test_location_uses_base_url(self):
        request = _make_mock_request(base_url="https://proxy.example.com/")
        result = await get_scim_base(request)

        user_resource = next(r for r in result["Resources"] if r["id"] == "User")
        assert user_resource["meta"]["location"] == "https://proxy.example.com/scim/v2/ResourceTypes/User"


class TestGetResourceTypesEndpoint:
    @pytest.mark.asyncio
    async def test_returns_list_response(self):
        request = _make_mock_request()
        result = await get_resource_types(request)

        assert result["schemas"] == ["urn:ietf:params:scim:api:messages:2.0:ListResponse"]
        assert result["totalResults"] == 2

    @pytest.mark.asyncio
    async def test_resources_match_base_endpoint(self):
        """ResourceTypes endpoint should return same data as base endpoint."""
        request = _make_mock_request()
        base_result = await get_scim_base(request)
        rt_result = await get_resource_types(request)

        assert base_result["totalResults"] == rt_result["totalResults"]
        assert len(base_result["Resources"]) == len(rt_result["Resources"])


class TestGetResourceTypeById:
    @pytest.mark.asyncio
    async def test_get_user_resource_type(self):
        request = _make_mock_request()
        result = await get_resource_type(request, resource_type_id="User")

        assert result["id"] == "User"
        assert result["name"] == "User"
        assert result["endpoint"] == "/Users"
        assert result["schema"] == "urn:ietf:params:scim:schemas:core:2.0:User"

    @pytest.mark.asyncio
    async def test_get_group_resource_type(self):
        request = _make_mock_request()
        result = await get_resource_type(request, resource_type_id="Group")

        assert result["id"] == "Group"
        assert result["name"] == "Group"
        assert result["endpoint"] == "/Groups"

    @pytest.mark.asyncio
    async def test_not_found(self):
        request = _make_mock_request()
        with pytest.raises(HTTPException) as exc_info:
            await get_resource_type(request, resource_type_id="NonExistent")
        assert exc_info.value.status_code == 404


class TestGetSchemasEndpoint:
    @pytest.mark.asyncio
    async def test_returns_list_response(self):
        request = _make_mock_request()
        result = await get_schemas(request)

        assert result["schemas"] == ["urn:ietf:params:scim:api:messages:2.0:ListResponse"]
        assert result["totalResults"] == 2

    @pytest.mark.asyncio
    async def test_resources_have_correct_ids(self):
        request = _make_mock_request()
        result = await get_schemas(request)

        schema_ids = [r["id"] for r in result["Resources"]]
        assert "urn:ietf:params:scim:schemas:core:2.0:User" in schema_ids
        assert "urn:ietf:params:scim:schemas:core:2.0:Group" in schema_ids


class TestGetSchemaById:
    @pytest.mark.asyncio
    async def test_get_user_schema(self):
        request = _make_mock_request()
        result = await get_schema(
            request, schema_id="urn:ietf:params:scim:schemas:core:2.0:User"
        )

        assert result["id"] == "urn:ietf:params:scim:schemas:core:2.0:User"
        assert result["name"] == "User"
        assert len(result["attributes"]) > 0

    @pytest.mark.asyncio
    async def test_get_group_schema(self):
        request = _make_mock_request()
        result = await get_schema(
            request, schema_id="urn:ietf:params:scim:schemas:core:2.0:Group"
        )

        assert result["id"] == "urn:ietf:params:scim:schemas:core:2.0:Group"
        assert result["name"] == "Group"

    @pytest.mark.asyncio
    async def test_not_found(self):
        request = _make_mock_request()
        with pytest.raises(HTTPException) as exc_info:
            await get_schema(request, schema_id="urn:nonexistent:schema")
        assert exc_info.value.status_code == 404


class TestSCIMResourceTypeModel:
    """Test the SCIMResourceType Pydantic model itself."""

    def test_model_dump_schema_key(self):
        rt = SCIMResourceType(
            id="Test",
            name="Test",
            endpoint="/Test",
            schema_="urn:test",
        )
        dumped = rt.model_dump()
        assert "schema" in dumped
        assert "schema_" not in dumped
        assert dumped["schema"] == "urn:test"

    def test_no_schema_extensions_omitted(self):
        rt = SCIMResourceType(
            id="Test",
            name="Test",
            endpoint="/Test",
            schema_="urn:test",
        )
        dumped = rt.model_dump()
        assert "schemaExtensions" not in dumped


class TestSCIMSchemaModel:
    """Test the SCIMSchema Pydantic model."""

    def test_basic_schema(self):
        schema = SCIMSchema(
            id="urn:test",
            name="Test",
            description="A test schema",
        )
        assert schema.id == "urn:test"
        assert schema.attributes == []

    def test_sub_attributes_omitted_when_none(self):
        from litellm.types.proxy.management_endpoints.scim_v2 import SCIMSchemaAttribute

        attr = SCIMSchemaAttribute(
            name="test",
            type="string",
        )
        dumped = attr.model_dump()
        assert "subAttributes" not in dumped
