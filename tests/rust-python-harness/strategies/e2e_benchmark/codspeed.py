from __future__ import annotations

import asyncio
import gc
import math
import os
import signal
import subprocess
import sys
import tempfile
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import TYPE_CHECKING, Final, cast

import click
import pytest

from litellm.llms.base_llm.ocr.transformation import OCRResponse

from .constants import PYTHON_SENTINEL
from .models import Backend, Options, Profile, Ready, Route
from .provider import provider_process
from .worker import capture_ready
from .workloads import ocr_workload

if TYPE_CHECKING:
    from pytest_codspeed.plugin import BenchmarkFixture

CASES: Final = tuple(
    (backend, route, profile)
    for profile in Options().profiles
    for route in Options().routes
    for backend in ("python", "rust")
)


@pytest.mark.parametrize(("backend", "route", "profile"), CASES, ids=tuple("-".join(case) for case in CASES))
@pytest.mark.benchmark(min_time=0)
def test_sdk(benchmark: BenchmarkFixture, backend: Backend, route: Route, profile: Profile) -> None:
    import litellm

    assert os.environ.get("LITELLM_RUST") == ("1" if backend == "rust" else "0"), "use the codspeed module CLI"
    workload: Final = ocr_workload(profile)
    with provider_process(workload.response, backend) as url:
        kwargs: Final = {
            "model": workload.model,
            "document": {"type": "document_url", "document_url": workload.document_url},
            "api_key": "benchmark-local-only",
            "api_base": url,
            "timeout": 10,
            "num_retries": 0,
        }
        if route == "ocr":
            sync_call: Final = cast(Callable[..., OCRResponse], litellm.ocr)
            ready: Final = capture_ready(sync_call(**kwargs))

            def call_sync() -> None:
                sync_call(**kwargs)

            gc.collect()
            benchmark(call_sync)
            Path(os.environ["LITELLM_BENCHMARK_READY"]).write_text(ready.model_dump_json())
            return
        async_call: Final = cast(Callable[..., Awaitable[OCRResponse]], litellm.aocr)
        loop: Final = asyncio.new_event_loop()
        try:
            async_ready: Final = capture_ready(loop.run_until_complete(async_call(**kwargs)))

            async def call_async() -> None:
                await async_call(**kwargs)

            def call_on_loop() -> None:
                loop.run_until_complete(call_async())

            gc.collect()
            benchmark(call_on_loop)
            Path(os.environ["LITELLM_BENCHMARK_READY"]).write_text(async_ready.model_dump_json())
        finally:
            loop.run_until_complete(loop.shutdown_asyncgens())
            loop.close()


def run_case(backend: Backend, route: Route, profile: Profile, ready_file: Path, max_time: float) -> Ready:
    root: Final = Path(__file__).resolve().parents[4]
    nodeid: Final = f"{Path(__file__).relative_to(root)}::test_sdk[{backend}-{route}-{profile}]"
    with subprocess.Popen(
        (
            sys.executable,
            "-m",
            "pytest",
            "-p",
            "pytest_codspeed.plugin",
            "-c",
            os.devnull,
            f"--rootdir={root}",
            "--import-mode=importlib",
            "-o",
            "consider_namespace_packages=true",
            "--codspeed",
            "--codspeed-mode=walltime",
            "--codspeed-warmup-time=1",
            f"--codspeed-max-time={max_time}",
            nodeid,
            "-q",
        ),
        cwd=root,
        env={
            **os.environ,
            "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
            "LITELLM_RUST": "1" if backend == "rust" else "0",
            "LITELLM_USER_AGENT": PYTHON_SENTINEL,
            "LITELLM_LOCAL_MODEL_COST_MAP": "True",
            "LITELLM_BENCHMARK_READY": str(ready_file),
            "NO_PROXY": "127.0.0.1,localhost",
            "no_proxy": "127.0.0.1,localhost",
        },
        start_new_session=True,
    ) as child:
        try:
            code: Final = child.wait(timeout=max_time + 120)
            if code:
                raise click.ClickException(f"CodSpeed benchmark failed: {backend}/{route}/{profile}, exit {code}")
        except subprocess.TimeoutExpired as error:
            raise click.ClickException(f"CodSpeed worker timed out: {backend}/{route}/{profile}") from error
        finally:
            try:
                os.killpg(child.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
    return Ready.model_validate_json(ready_file.read_bytes())


def run_pair(route: Route, profile: Profile, directory: Path, max_time: float) -> None:
    pair: Final = tuple(
        run_case(backend, route, profile, directory / f"{backend}.json", max_time) for backend in ("python", "rust")
    )
    if pair[0].response_digest != pair[1].response_digest:
        raise click.ClickException(f"Python/Rust responses differ for {route}/{profile}; run e2e_parity")


@click.command()
@click.option("--profile", "profiles", multiple=True, type=click.Choice(Options().profiles), default=Options().profiles)
@click.option("--route", "routes", multiple=True, type=click.Choice(Options().routes), default=Options().routes)
@click.option("--max-time", type=click.FloatRange(min=0.1, max=60), default=5.0, show_default=True)
def main(profiles: tuple[Profile, ...], routes: tuple[Route, ...], max_time: float) -> None:
    """Run isolated SDK workers with CodSpeed walltime calibration and profiling."""
    if not math.isfinite(max_time):
        raise click.BadParameter("must be finite", param_hint="--max-time")
    with tempfile.TemporaryDirectory(prefix="litellm-codspeed-") as directory:
        for profile in dict.fromkeys(profiles):
            for route in dict.fromkeys(routes):
                run_pair(route, profile, Path(directory), max_time)


if __name__ == "__main__":
    main()
