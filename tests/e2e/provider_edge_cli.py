from __future__ import annotations

import argparse
import os
import signal
import threading
from datetime import datetime, timezone
from pathlib import Path
from types import FrameType
from typing import Final, Literal, cast

from botocore.config import Config
from botocore.session import get_session
from capture_policy import RUN_BYTES, SCENARIO_BYTES, RequestBudget, ScenarioIdentity, canonical_scenario_id
from capture_session import CaptureResult
from capture_store import DynamoCaptureStore, DynamoClient
from fixture_bundle import BundleRecorder, FreshBundle, UnreadableBundle, check_freshness, load_bundle, prepare_bundle
from provider_edge import start_provider_edge
from provider_edge_control import ControlServer, EdgeController
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter


class CredentialHeader(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    variable: str
    prefix: str = ""


class EdgeConfiguration(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    mode: Literal["record", "replay"]
    owner: str = Field(min_length=1)
    bundle_dir: Path
    outcome_file: Path
    identities: tuple[ScenarioIdentity, ...] = Field(min_length=1, max_length=RUN_BYTES // SCENARIO_BYTES)
    mounts: dict[str, str]
    advertise_host: str
    data_port: int = Field(default=8080, gt=0, le=65535)
    control_port: int = Field(default=8081, gt=0, le=65535)
    lease_deadline: int
    attempt_cap: int = Field(default=12, gt=0, le=12)
    table: str | None = None
    region: str | None = None
    credential_env: dict[str, dict[str, CredentialHeader]] = {}
    request_budgets: dict[str, RequestBudget] = {}


def _write_outcomes(path: Path, results: tuple[CaptureResult, ...]) -> None:
    temporary: Final = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(TypeAdapter(tuple[CaptureResult, ...]).dump_json(results))
    os.replace(temporary, path)


def configured_controller(config: EdgeConfiguration) -> EdgeController:
    identities: Final = {canonical_scenario_id(identity.node): identity for identity in config.identities}
    if len(identities) != len(config.identities):
        raise ValueError("duplicate configured scenario identity")
    if config.mode == "replay":
        if config.credential_env or config.table:
            raise ValueError("replay cannot receive capture credentials or a store")
        if not isinstance(
            check_freshness(config.bundle_dir, now=datetime.now(timezone.utc), profile="stateless_v1"), FreshBundle
        ):
            raise ValueError("replay bundle is outside its age limit")
        bundle: Final = load_bundle(config.bundle_dir, profile="stateless_v1")
        if isinstance(bundle, UnreadableBundle):
            raise ValueError("verified replay bundle is unavailable")
        return EdgeController(identities, config.owner, config.lease_deadline, replay_bundle=bundle)
    if not config.table or not config.region:
        raise ValueError("capture requires a configured attempt store")
    if set(config.request_budgets) != set(identities):
        raise ValueError("capture requires a model and request budget for every enrolled scenario")
    if config.bundle_dir.exists():
        raise ValueError("capture requires a new owned bundle directory")
    store: Final = DynamoCaptureStore(
        cast(
            DynamoClient,
            get_session().create_client(
                "dynamodb",
                region_name=config.region,
                config=Config(
                    retries={"total_max_attempts": 1},
                    connect_timeout=5,
                    read_timeout=5,
                ),
            ),
        ),
        config.table,
    )
    run_error: Final = store.create_run(owner=config.owner, cap=config.attempt_cap, expires_at=config.lease_deadline)
    if run_error is not None:
        raise ValueError(run_error.reason)
    recorder: Final = prepare_bundle(config.bundle_dir, profile="stateless_v1")
    assert isinstance(recorder, BundleRecorder)
    return EdgeController(
        identities,
        config.owner,
        config.lease_deadline,
        store=store,
        recorder=recorder,
        outcome_sink=lambda results: _write_outcomes(config.outcome_file, results),
        request_budgets=config.request_budgets,
    )


def serve(config: EdgeConfiguration) -> None:
    controller: Final = configured_controller(config)
    upstream_headers: Final = {
        mount: {
            header.lower(): credential.prefix + os.environ[credential.variable]
            for header, credential in headers.items()
        }
        for mount, headers in config.credential_env.items()
    }
    stopped: Final = threading.Event()

    def stop(signum: int, frame: FrameType | None) -> None:
        stopped.set()

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    control: Final = ControlServer(controller, port=config.control_port)
    running: Final = start_provider_edge(
        controller.backend(upstream_headers),
        mounts=config.mounts,
        bind_host="0.0.0.0",
        advertise_host=config.advertise_host,
        bind_port=config.data_port,
        guard=controller,
        observation=controller,
    )
    thread: Final = threading.Thread(target=control.serve_forever, daemon=True)
    thread.start()
    try:
        stopped.wait()
    finally:
        control.shutdown()
        control.server_close()
        running.shutdown()
        thread.join(timeout=5)


class CommandArguments(argparse.Namespace):
    config: Path


def main() -> None:
    parser: Final = argparse.ArgumentParser(description="Serve an explicitly configured trusted provider edge")
    parser.add_argument("--config", required=True, type=Path)
    arguments: Final = CommandArguments()
    parser.parse_args(namespace=arguments)
    serve(EdgeConfiguration.model_validate_json(arguments.config.read_bytes()))


if __name__ == "__main__":
    main()
