"""Config read from an object-store bucket instead of a file, with the store diverted to loopback.

When `LITELLM_CONFIG_BUCKET_NAME` is set, `ProxyConfig.get_config` (`litellm/proxy/proxy_server.py`)
ignores the file path it was given and reads `LITELLM_CONFIG_BUCKET_OBJECT_KEY` from the bucket instead,
through S3 unless `LITELLM_CONFIG_BUCKET_TYPE` is `gcs` (`litellm/proxy/common_utils/load_config_utils.py`).
The object goes through the same `include` merge, with include entries resolved against the object's
key prefix (a leading `/` means the bucket root), and then through the same `os.environ/` walk.

No cloud service is contacted: `s3_bucket` is a loopback HTTP peer that answers path-style
`GET /<bucket>/<key>`, and boto3 is pointed at it with `AWS_ENDPOINT_URL_S3` and fake credentials.
These tests therefore assume a botocore recent enough to honour `AWS_ENDPOINT_URL_S3` (1.31+) and that
it uses path-style addressing for an IP endpoint; if they fail with the bucket never contacted, check that
first. Every config declares `router_settings.num_retries` so the harness does not rewrite the file.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import pytest
import yaml

from tests.integration._support.client import Gateway
from tests.integration.configuration.dec1_cfgload_support import (
    FAKE_AWS,
    refused_with_exit,
    requested_keys,
    s3_bucket,
    s3_environment,
)
from tests.integration.configuration.dec1_config_support import (
    booted,
    chat,
    deployment,
    scripted_upstream,
    sent,
    served_model_names,
    write_config,
)

_FILE_MODEL: Final = "dec1-cfgload-file-model"
_BUCKET_MODEL: Final = "dec1-cfgload-bucket-model"
_INCLUDED_MODEL: Final = "dec1-cfgload-bucket-included"
_ROOT_KEY: Final = "configs/dec1/proxy.yaml"
_RETRIES: Final = {"num_retries": 0}


def _yaml(config: dict[str, object]) -> bytes:
    return yaml.safe_dump(config, sort_keys=False).encode()


def _file_config(directory: Path) -> Path:
    return write_config(
        directory,
        {"model_list": [deployment(_FILE_MODEL, "http://127.0.0.1:9/v1")], "router_settings": _RETRIES},
    )


def _bucket_config(api_base: str = "http://127.0.0.1:9/v1", **extra: object) -> dict[str, object]:
    return {"model_list": [deployment(_BUCKET_MODEL, api_base)], "router_settings": _RETRIES, **extra}


def test_a_bucket_config_replaces_the_file_named_on_the_command_line(gateway: Gateway, tmp_path: Path) -> None:
    """`--config` names a file declaring `dec1-cfgload-file-model`, and the bucket variables name an
    object declaring `dec1-cfgload-bucket-model`. `GET /v1/models` must list the bucket's model and not
    the file's, and the bucket must have been asked for `configs/dec1/proxy.yaml`: the bucket, when
    configured, is the only config source."""
    with s3_bucket({_ROOT_KEY: _yaml(_bucket_config())}) as bucket:
        with booted(gateway, tmp_path, _file_config(tmp_path), environment=s3_environment(bucket, _ROOT_KEY)) as owned:
            names: Final = served_model_names(owned.gateway)
        keys: Final = requested_keys(bucket.drain())
    assert _BUCKET_MODEL in names
    assert _FILE_MODEL not in names
    assert _ROOT_KEY in keys


def test_a_bucket_config_also_wins_over_config_file_path(gateway: Gateway, tmp_path: Path) -> None:
    """`CONFIG_FILE_PATH` names a valid yaml file with `dec1-cfgload-file-model` and the bucket variables
    are also set. `GET /v1/models` must list the bucket's model and not the file's: `get_config` checks
    the bucket before it looks at any path, whichever path the startup chose.
    UNVERIFIED: read from `get_config` and the `CONFIG_FILE_PATH` branch of the startup event."""
    file_path: Final = _file_config(tmp_path)
    with s3_bucket({_ROOT_KEY: _yaml(_bucket_config())}) as bucket:
        environment: Final = {**s3_environment(bucket, _ROOT_KEY), "CONFIG_FILE_PATH": str(file_path)}
        with booted(gateway, tmp_path, file_path, environment=environment) as owned:
            names: Final = served_model_names(owned.gateway)
    assert _BUCKET_MODEL in names
    assert _FILE_MODEL not in names


def test_a_bucket_config_without_a_bucket_type_is_read_from_s3(gateway: Gateway, tmp_path: Path) -> None:
    """The bucket variables are set except `LITELLM_CONFIG_BUCKET_TYPE`. The proxy must still read the
    object from the S3 endpoint and serve its model: any type other than `gcs`, including none, means S3
    (`bucket_object_reader`)."""
    with s3_bucket({_ROOT_KEY: _yaml(_bucket_config())}) as bucket:
        environment: Final = {
            name: value
            for name, value in s3_environment(bucket, _ROOT_KEY).items()
            if name != "LITELLM_CONFIG_BUCKET_TYPE"
        }
        with booted(
            gateway,
            tmp_path,
            _file_config(tmp_path),
            environment=environment,
            remove_environment=("LITELLM_CONFIG_BUCKET_TYPE",),
        ) as owned:
            names: Final = served_model_names(owned.gateway)
    assert _BUCKET_MODEL in names


def test_references_inside_a_bucket_config_are_resolved_from_the_environment(gateway: Gateway, tmp_path: Path) -> None:
    """The bucket object's deployment has `api_key: os.environ/DEC1_BUCKET_PROVIDER_KEY`, set in the
    proxy's environment. A chat completion must reach the scripted upstream with `Bearer
    sk-dec1-bucket-key`: a bucket config gets the same reference walk as a file."""
    with scripted_upstream() as upstream:
        config: Final = {
            "model_list": [
                deployment(_BUCKET_MODEL, f"{upstream.url}/v1", api_key="os.environ/DEC1_BUCKET_PROVIDER_KEY")
            ],
            "router_settings": _RETRIES,
        }
        with s3_bucket({_ROOT_KEY: _yaml(config)}) as bucket:
            environment: Final = {
                **s3_environment(bucket, _ROOT_KEY),
                "DEC1_BUCKET_PROVIDER_KEY": "sk-dec1-bucket-key",
            }
            with booted(gateway, tmp_path, _file_config(tmp_path), environment=environment) as owned:
                assert chat(owned.gateway, _BUCKET_MODEL).status_code == 200
                requests: Final = sent(upstream)
    assert [request.headers["authorization"] for request in requests] == ["Bearer sk-dec1-bucket-key"]


def test_a_bucket_include_is_read_next_to_the_including_object(gateway: Gateway, tmp_path: Path) -> None:
    """The root object `configs/dec1/proxy.yaml` has `include: [models.yaml]` and the bucket holds
    `configs/dec1/models.yaml` with one more model. The proxy must serve both models and the bucket must
    have been asked for `configs/dec1/models.yaml`: an include entry is relative to the including
    object's key prefix (`resolve_include_object_key`)."""
    objects: Final = {
        _ROOT_KEY: _yaml(_bucket_config(include=["models.yaml"])),
        "configs/dec1/models.yaml": _yaml({"model_list": [deployment(_INCLUDED_MODEL, "http://127.0.0.1:9/v1")]}),
    }
    with s3_bucket(objects) as bucket:
        with booted(gateway, tmp_path, _file_config(tmp_path), environment=s3_environment(bucket, _ROOT_KEY)) as owned:
            names: Final = served_model_names(owned.gateway)
        keys: Final = requested_keys(bucket.drain())
    assert {_BUCKET_MODEL, _INCLUDED_MODEL} <= names
    assert "configs/dec1/models.yaml" in keys


def test_a_bucket_include_with_a_leading_slash_is_read_from_the_bucket_root(gateway: Gateway, tmp_path: Path) -> None:
    """The root object `configs/dec1/proxy.yaml` has `include: [/shared/models.yaml]`. The bucket must be
    asked for `shared/models.yaml` (no leading slash, not under `configs/dec1/`) and its model must be
    served: a leading `/` names the bucket root, as an absolute path ignores the including file's
    directory on disk."""
    objects: Final = {
        _ROOT_KEY: _yaml(_bucket_config(include=["/shared/models.yaml"])),
        "shared/models.yaml": _yaml({"model_list": [deployment(_INCLUDED_MODEL, "http://127.0.0.1:9/v1")]}),
    }
    with s3_bucket(objects) as bucket:
        with booted(gateway, tmp_path, _file_config(tmp_path), environment=s3_environment(bucket, _ROOT_KEY)) as owned:
            names: Final = served_model_names(owned.gateway)
        keys: Final = requested_keys(bucket.drain())
    assert _INCLUDED_MODEL in names
    assert "shared/models.yaml" in keys


def test_a_bucket_include_that_is_missing_stops_the_boot_and_names_the_key(gateway: Gateway, tmp_path: Path) -> None:
    """The root object includes `absent.yaml`, which the bucket does not hold. The proxy must exit
    non-zero and log `Included config could not be read from bucket: configs/dec1/absent.yaml`
    (`resolve_bucket_includes`)."""
    with s3_bucket({_ROOT_KEY: _yaml(_bucket_config(include=["absent.yaml"]))}) as bucket:
        refusal: Final = refused_with_exit(gateway, tmp_path, _file_config(tmp_path), s3_environment(bucket, _ROOT_KEY))
    assert refusal.exit_code != 0
    assert "Included config could not be read from bucket: configs/dec1/absent.yaml" in refusal.log, refusal.log[-4000:]


@pytest.mark.parametrize(
    "objects",
    [{}, {_ROOT_KEY: b""}, {_ROOT_KEY: b"model_list: [{model_name: x\n"}],
    ids=["missing_object", "empty_object", "invalid_yaml"],
)
def test_a_bucket_object_that_yields_no_config_stops_the_boot(
    gateway: Gateway, tmp_path: Path, objects: dict[str, bytes]
) -> None:
    """The bucket has no object at the key, an empty object, or an object that is not valid YAML. Each is
    read as no config (the read error is logged and swallowed), so the proxy must exit non-zero and log
    `Unable to load config from given source.`, never fall back to the `--config` file."""
    with s3_bucket(objects) as bucket:
        refusal: Final = refused_with_exit(gateway, tmp_path, _file_config(tmp_path), s3_environment(bucket, _ROOT_KEY))
    assert refusal.exit_code != 0
    assert "Unable to load config from given source." in refusal.log, refusal.log[-4000:]


def test_a_bucket_object_that_is_a_yaml_list_stops_the_boot_and_names_the_key(gateway: Gateway, tmp_path: Path) -> None:
    """The bucket object is a YAML list. The proxy must exit non-zero and log `Config object in bucket is
    not a YAML mapping: configs/dec1/proxy.yaml` (`get_config_from_bucket`)."""
    with s3_bucket({_ROOT_KEY: b"- model_name: dec1-list\n"}) as bucket:
        refusal: Final = refused_with_exit(gateway, tmp_path, _file_config(tmp_path), s3_environment(bucket, _ROOT_KEY))
    assert refusal.exit_code != 0
    assert f"Config object in bucket is not a YAML mapping: {_ROOT_KEY}" in refusal.log, refusal.log[-4000:]


def test_a_bucket_name_without_an_object_key_stops_the_boot(gateway: Gateway, tmp_path: Path) -> None:
    """Only `LITELLM_CONFIG_BUCKET_NAME` is set. The proxy must exit non-zero and log
    `LITELLM_CONFIG_BUCKET_OBJECT_KEY must be set to load the config from a bucket.` before contacting
    any store."""
    environment: Final = {**FAKE_AWS, "LITELLM_CONFIG_BUCKET_NAME": "dec1-cfgload-bucket"}
    refusal: Final = refused_with_exit(
        gateway, tmp_path, _file_config(tmp_path), environment, remove_environment=("LITELLM_CONFIG_BUCKET_OBJECT_KEY",)
    )
    assert refusal.exit_code != 0
    assert "LITELLM_CONFIG_BUCKET_OBJECT_KEY must be set to load the config from a bucket." in refusal.log


def test_a_gcs_bucket_that_cannot_be_reached_stops_the_boot(gateway: Gateway, tmp_path: Path) -> None:
    """`LITELLM_CONFIG_BUCKET_TYPE=gcs` with a bucket name and key, and `GCS_PATH_SERVICE_ACCOUNT` naming
    a file that does not exist. The GCS read fails, is logged and swallowed, so the proxy must exit
    non-zero and log `Unable to load config from given source.`.
    UNVERIFIED: that the GCS client fails on the missing service account file before any network call;
    `litellm/integrations/gcs_bucket/gcs_bucket_base.py` decides."""
    environment: Final = {
        "LITELLM_CONFIG_BUCKET_TYPE": "gcs",
        "LITELLM_CONFIG_BUCKET_NAME": "dec1-cfgload-gcs",
        "LITELLM_CONFIG_BUCKET_OBJECT_KEY": _ROOT_KEY,
        "GCS_PATH_SERVICE_ACCOUNT": str(tmp_path / "dec1-no-such-service-account.json"),
    }
    refusal: Final = refused_with_exit(gateway, tmp_path, _file_config(tmp_path), environment)
    assert refusal.exit_code != 0
    assert "Unable to load config from given source." in refusal.log, refusal.log[-4000:]


def test_the_file_is_used_when_no_bucket_variable_is_set(gateway: Gateway, tmp_path: Path) -> None:
    """Control: the bucket variables are removed from the environment. The `--config` file's model must
    be served, so the bucket tests above prove the bucket path rather than a broken file path."""
    with booted(
        gateway,
        tmp_path,
        _file_config(tmp_path),
        remove_environment=(
            "LITELLM_CONFIG_BUCKET_NAME",
            "LITELLM_CONFIG_BUCKET_OBJECT_KEY",
            "LITELLM_CONFIG_BUCKET_TYPE",
        ),
    ) as owned:
        assert _FILE_MODEL in served_model_names(owned.gateway)
