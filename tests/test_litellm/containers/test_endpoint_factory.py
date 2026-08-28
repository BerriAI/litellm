import pytest

from litellm.containers import endpoint_factory
from litellm.containers.endpoint_factory import (
    RESPONSE_TYPES,
    _load_endpoints_config,
    create_sync_endpoint_function,
    generate_container_endpoints,
    get_all_endpoint_names,
    get_async_endpoint_names,
)
from litellm.types.containers.main import (
    ContainerFileListResponse,
    ContainerFileObject,
    DeleteContainerFileResponse,
)

_SYNC_NAMES = [
    "list_container_files",
    "upload_container_file",
    "retrieve_container_file",
    "delete_container_file",
    "retrieve_container_file_content",
]
_ASYNC_NAMES = ["a" + n for n in _SYNC_NAMES]


class TestEndpointsConfig:
    def test_config_exposes_every_declared_endpoint(self):
        config = _load_endpoints_config()
        assert [e["name"] for e in config["endpoints"]] == _SYNC_NAMES

    def test_every_endpoint_declares_the_keys_the_factory_reads(self):
        for endpoint in _load_endpoints_config()["endpoints"]:
            assert set(endpoint) >= {
                "name",
                "async_name",
                "path",
                "method",
                "path_params",
                "response_type",
            }

    def test_async_name_is_the_sync_name_prefixed_with_a(self):
        for endpoint in _load_endpoints_config()["endpoints"]:
            assert endpoint["async_name"] == "a" + endpoint["name"]

    def test_config_is_reread_rather_than_shared_between_callers(self):
        first = _load_endpoints_config()
        first["endpoints"].clear()
        assert len(_load_endpoints_config()["endpoints"]) == len(_SYNC_NAMES)


class TestResponseTypeMapping:
    def test_mapping_resolves_every_named_response_type(self):
        assert RESPONSE_TYPES == {
            "ContainerFileListResponse": ContainerFileListResponse,
            "ContainerFileObject": ContainerFileObject,
            "DeleteContainerFileResponse": DeleteContainerFileResponse,
        }

    @pytest.mark.parametrize(
        "endpoint_name,expected",
        [
            ("list_container_files", ContainerFileListResponse),
            ("upload_container_file", ContainerFileObject),
            ("retrieve_container_file", ContainerFileObject),
            ("delete_container_file", DeleteContainerFileResponse),
        ],
    )
    def test_each_endpoint_maps_to_its_declared_response_type(self, endpoint_name, expected):
        config = next(e for e in _load_endpoints_config()["endpoints"] if e["name"] == endpoint_name)
        assert RESPONSE_TYPES[config["response_type"]] is expected

    def test_raw_response_type_is_deliberately_unmapped(self):
        config = next(
            e for e in _load_endpoints_config()["endpoints"] if e["name"] == "retrieve_container_file_content"
        )
        assert config["response_type"] == "raw"
        assert RESPONSE_TYPES.get(config["response_type"]) is None


class TestGeneratedEndpoints:
    def test_generates_exactly_one_sync_and_one_async_function_per_endpoint(self):
        assert set(generate_container_endpoints()) == set(_SYNC_NAMES) | set(_ASYNC_NAMES)

    def test_every_generated_value_is_callable(self):
        assert all(callable(f) for f in generate_container_endpoints().values())

    def test_sync_and_async_entries_are_distinct_objects(self):
        endpoints = generate_container_endpoints()
        for name in _SYNC_NAMES:
            assert endpoints[name] is not endpoints["a" + name]

    def test_each_call_builds_fresh_functions(self):
        assert (
            generate_container_endpoints()["list_container_files"]
            is not generate_container_endpoints()["list_container_files"]
        )

    def test_module_exports_are_wired_and_not_none(self):
        for name in _SYNC_NAMES + _ASYNC_NAMES:
            assert getattr(endpoint_factory, name) is not None


class TestEndpointNameHelpers:
    def test_all_endpoint_names_interleaves_sync_then_async_per_endpoint(self):
        expected = [n for name in _SYNC_NAMES for n in (name, "a" + name)]
        assert get_all_endpoint_names() == expected

    def test_async_endpoint_names_are_only_the_async_ones(self):
        assert get_async_endpoint_names() == _ASYNC_NAMES

    def test_async_names_are_a_strict_subset_of_all_names(self):
        assert set(get_async_endpoint_names()) < set(get_all_endpoint_names())


class TestSyncEndpointFactory:
    def test_returns_a_callable_for_a_minimal_config(self):
        assert callable(
            create_sync_endpoint_function({"name": "x", "response_type": "ContainerFileObject", "path_params": []})
        )

    def test_missing_path_params_defaults_to_empty_rather_than_raising(self):
        assert callable(create_sync_endpoint_function({"name": "x", "response_type": "ContainerFileObject"}))

    def test_unknown_response_type_is_tolerated_at_build_time(self):
        assert callable(
            create_sync_endpoint_function({"name": "x", "response_type": "NotARealType", "path_params": []})
        )

    def test_missing_name_is_a_build_time_error(self):
        with pytest.raises(KeyError):
            create_sync_endpoint_function({"response_type": "ContainerFileObject"})

    def test_missing_response_type_is_a_build_time_error(self):
        with pytest.raises(KeyError):
            create_sync_endpoint_function({"name": "x"})
