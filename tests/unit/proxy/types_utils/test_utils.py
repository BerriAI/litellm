import io
import re
import sys
import tempfile
from collections.abc import Iterator, Mapping
from pathlib import Path

import boto3
import pytest

from litellm.proxy.types_utils.utils import get_instance_fn

_PLUGIN_SOURCE = """
class Handler:
    label = "plugin-handler"


handler = Handler()
alias = handler
nothing = None
zero = 0
disabled = False
empty = []
label = "plugin"
location = __file__


def build():
    return "built"
"""

_DYNAMIC_PLUGIN_SOURCE = """
def __getattr__(name):
    if name.startswith("make_"):
        return name.removeprefix("make_").upper()
    if name == "needs_dependency":
        raise ImportError("optional dependency missing")
    raise LookupError(f"plugin cannot provide {name}")
"""

_WARNING_PLUGIN_TEMPLATE = """
import warnings


def __getattr__(name):
    warnings.warn(f"{{name}} is deprecated", DeprecationWarning, stacklevel={stacklevel})
    return name.upper()
"""

_PLAIN_ATTRIBUTES = [
    pytest.param("nothing", "NoneType", "None", id="none"),
    pytest.param("zero", "int", "0", id="zero"),
    pytest.param("disabled", "bool", "False", id="false"),
    pytest.param("empty", "list", "[]", id="empty-list"),
    pytest.param("label", "str", "'plugin'", id="string"),
]

_INSTALLED_PLUGIN = "types_utils_installed_plugin"


class _FakeS3:
    def __init__(self, objects: Mapping[tuple[str, str], bytes]) -> None:
        self.objects = objects
        self.requests: list[tuple[str, str]] = []

    def get_object(self, Bucket: str, Key: str) -> Mapping[str, io.BytesIO]:
        self.requests.append((Bucket, Key))
        return {"Body": io.BytesIO(self.objects[(Bucket, Key)])}


@pytest.fixture
def local_config(tmp_path: Path) -> str:
    (tmp_path / "local_plugin.py").write_text(_PLUGIN_SOURCE)
    (tmp_path / "dynamic_plugin.py").write_text(_DYNAMIC_PLUGIN_SOURCE)
    return str(tmp_path / "config.yaml")


@pytest.fixture
def installed_plugin(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[str]:
    site = tmp_path / "site"
    site.mkdir()
    (site / f"{_INSTALLED_PLUGIN}.py").write_text(_PLUGIN_SOURCE)
    monkeypatch.syspath_prepend(str(site))
    yield _INSTALLED_PLUGIN
    sys.modules.pop(_INSTALLED_PLUGIN, None)


@pytest.fixture
def downloads(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    directory = tmp_path / "downloads"
    directory.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(directory))
    return directory


@pytest.fixture
def plugin_bucket(downloads: Path, monkeypatch: pytest.MonkeyPatch) -> _FakeS3:
    bucket = _FakeS3(
        {
            ("plugins-bucket", "loggers/remote_plugin.py"): _PLUGIN_SOURCE.encode(),
            ("plugins-bucket", "loggers/warning_plugin.py"): _WARNING_PLUGIN_TEMPLATE.format(stacklevel=4).encode(),
        }
    )

    def fake_client(
        service_name: str,
        aws_access_key_id: str,
        aws_secret_access_key: str,
        aws_session_token: str | None,
    ) -> _FakeS3:
        assert service_name == "s3"
        return bucket

    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "unit-test")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "unit-test")
    monkeypatch.setenv("AWS_REGION_NAME", "us-east-1")
    monkeypatch.setattr(boto3, "client", fake_client)
    return bucket


@pytest.mark.parametrize(("attribute", "expected_type", "expected_repr"), _PLAIN_ATTRIBUTES)
def test_get_instance_fn_returns_the_attribute_of_the_module_beside_the_config_unchanged(
    local_config: str, attribute: str, expected_type: str, expected_repr: str
):
    result = get_instance_fn(f"local_plugin.{attribute}", config_file_path=local_config)

    assert (type(result).__name__, repr(result)) == (expected_type, expected_repr)


def test_get_instance_fn_returns_instances_and_callables_defined_in_the_local_module(local_config: str):
    handler = get_instance_fn("local_plugin.handler", config_file_path=local_config)
    build = get_instance_fn("local_plugin.build", config_file_path=local_config)

    assert (type(handler).__name__, handler.label) == ("Handler", "plugin-handler")
    assert build() == "built"


def test_get_instance_fn_surfaces_a_name_the_local_module_lacks_as_attribute_error(local_config: str):
    with pytest.raises(AttributeError) as raised:
        get_instance_fn("local_plugin.missing", config_file_path=local_config)

    assert type(raised.value) is AttributeError
    assert (raised.value.obj.__name__, raised.value.name) == ("local_plugin", "missing")
    assert raised.value.__cause__ is None


@pytest.mark.parametrize(
    ("attribute", "expected"),
    [
        pytest.param("make_logger", "LOGGER", id="computed"),
        pytest.param("make_", "", id="computed-empty"),
    ],
)
def test_get_instance_fn_returns_what_a_module_level_getattr_computes(local_config: str, attribute: str, expected: str):
    result = get_instance_fn(f"dynamic_plugin.{attribute}", config_file_path=local_config)

    assert (type(result), result) == (str, expected)


def test_get_instance_fn_lets_an_error_from_a_module_level_getattr_through_unchanged(local_config: str):
    with pytest.raises(LookupError, match=r"^plugin cannot provide other$") as raised:
        get_instance_fn("dynamic_plugin.other", config_file_path=local_config)

    assert type(raised.value) is LookupError


def test_get_instance_fn_rewords_an_import_error_raised_while_reading_the_attribute(local_config: str):
    with pytest.raises(ImportError, match=r"^Could not import needs_dependency from dynamic_plugin$") as raised:
        get_instance_fn("dynamic_plugin.needs_dependency", config_file_path=local_config)

    assert type(raised.value) is ImportError
    assert (type(raised.value.__cause__), str(raised.value.__cause__)) == (ImportError, "optional dependency missing")


def test_get_instance_fn_returns_the_very_object_an_installed_module_holds(installed_plugin: str):
    handler = get_instance_fn(f"{installed_plugin}.handler")
    alias = get_instance_fn(f"{installed_plugin}.alias")

    assert handler is sys.modules[installed_plugin].handler
    assert alias is handler


def test_get_instance_fn_surfaces_a_name_an_installed_module_lacks_as_attribute_error(installed_plugin: str):
    with pytest.raises(AttributeError) as raised:
        get_instance_fn(f"{installed_plugin}.missing")

    assert type(raised.value) is AttributeError
    assert (raised.value.obj, raised.value.name) == (sys.modules[installed_plugin], "missing")
    assert raised.value.__cause__ is None


@pytest.mark.parametrize(("attribute", "expected_type", "expected_repr"), _PLAIN_ATTRIBUTES)
def test_get_instance_fn_loads_the_attribute_from_the_bucket_object_unchanged(
    plugin_bucket: _FakeS3, attribute: str, expected_type: str, expected_repr: str
):
    result = get_instance_fn(
        f"s3://plugins-bucket/loggers/remote_plugin.{attribute}", config_file_path="/etc/litellm/config.yaml"
    )

    assert (type(result).__name__, repr(result)) == (expected_type, expected_repr)
    assert plugin_bucket.requests == [("plugins-bucket", "loggers/remote_plugin.py")]


def test_get_instance_fn_runs_the_bucket_module_from_a_temporary_file_and_removes_it(
    plugin_bucket: _FakeS3, downloads: Path
):
    location = get_instance_fn(
        "s3://plugins-bucket/loggers/remote_plugin.location", config_file_path="/etc/litellm/config.yaml"
    )

    assert (Path(location).parent, Path(location).suffix) == (downloads, ".py")
    assert not Path(location).exists()


def test_get_instance_fn_reports_a_name_the_bucket_module_lacks_as_a_failed_load(plugin_bucket: _FakeS3):
    prefix = "Failed to load custom logger from s3://plugins-bucket/loggers/remote_plugin.missing: "

    with pytest.raises(ImportError, match=f"^{re.escape(prefix)}") as raised:
        get_instance_fn(
            "s3://plugins-bucket/loggers/remote_plugin.missing", config_file_path="/etc/litellm/config.yaml"
        )

    assert type(raised.value) is ImportError
    assert type(raised.value.__cause__) is AttributeError
    assert str(raised.value) == prefix + str(raised.value.__cause__)
    assert (raised.value.__cause__.obj.__name__, raised.value.__cause__.name) == ("loggers/remote_plugin", "missing")
    assert plugin_bucket.requests == [("plugins-bucket", "loggers/remote_plugin.py")]


def test_get_instance_fn_attributes_a_stacklevel_three_warning_from_a_local_plugin_to_its_own_caller(tmp_path: Path):
    (tmp_path / "warning_plugin.py").write_text(_WARNING_PLUGIN_TEMPLATE.format(stacklevel=3))

    with pytest.warns(DeprecationWarning, match=r"^old_handler is deprecated$") as recorded:
        result = get_instance_fn("warning_plugin.old_handler", config_file_path=str(tmp_path / "config.yaml"))

    assert result == "OLD_HANDLER"
    assert [Path(w.filename) for w in recorded if str(w.message) == "old_handler is deprecated"] == [Path(__file__)]


def test_get_instance_fn_attributes_a_stacklevel_four_warning_from_a_bucket_plugin_to_its_own_caller(
    plugin_bucket: _FakeS3,
):
    with pytest.warns(DeprecationWarning, match=r"^old_handler is deprecated$") as recorded:
        result = get_instance_fn(
            "s3://plugins-bucket/loggers/warning_plugin.old_handler", config_file_path="/etc/litellm/config.yaml"
        )

    assert result == "OLD_HANDLER"
    assert [Path(w.filename) for w in recorded if str(w.message) == "old_handler is deprecated"] == [Path(__file__)]
    assert plugin_bucket.requests == [("plugins-bucket", "loggers/warning_plugin.py")]
